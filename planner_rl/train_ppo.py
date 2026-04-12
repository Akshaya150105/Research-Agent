"""
rl/train_ppo.py
===============
Real sequential MDP training for the PPO planner policy.

ROOT CAUSE OF PREVIOUS COLLAPSE (and fixes):
  1. review_completeness had a 0.40 floor — writer-first scored 0.17, same as
     a random specialist policy. Policy had no reason to avoid writer-first.
     FIX: review_completeness = context_score (no floor). Writer-first = 0.10,
          optimal = 0.77. Clear 0.67 total range.

  2. First-step advantage was near zero for all actions — gradients vanished.
     FIX: Writer-first advantage is now -0.10, specialists +0.09 to +0.10.
          PPO gets a strong gradient from the very first batch.

  3. Shaping for specialists was too weak; comparator on 1-paper penalised -0.06
     which poisoned gradients in early-regime episodes.
     FIX: Specialist shaping increased to 0.07-0.08; 1-paper comparator = -0.03.

  4. Entropy decayed too fast — policy locked in before learning.
     FIX: ENTROPY stays high for first 40% of training, then decays.
          Controlled by step count in ppo_policy.py.

EXPECTED LEARNING CURVE:
  eps   1-100:  random policy, mean reward 0.30-0.35 (explores all actions)
  eps 100-200:  learns writer-last, reward climbs to 0.45-0.55
  eps 200-350:  learns specialist ordering, reward climbs to 0.60-0.70
  eps 350-500:  refines per-regime ordering, reward stabilises ~0.72-0.80
  Actor loss: starts near 0, goes negative (policy improves), converges ~0
  Critic loss: starts high, decays smoothly as value function learns
  Mean return: rises from ~0.3 to ~0.7 over training

HOW TO RUN:
  python planner_rl/train_ppo.py --episodes 500 --plot
  python planner_rl/train_ppo.py --reset --episodes 500 --plot  # fresh start
"""
from __future__ import annotations
import argparse
import json
import pathlib
import random
import sys
import time
from collections import Counter
import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

_RL_DIR       = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _RL_DIR.parent
for p in [str(_PROJECT_ROOT), str(_RL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from planner_rl.ppo_policy    import PPOPolicy, compute_gae, GAMMA, PPO_EPOCHS
from planner_rl.state_encoder import ACTIONS, ACTION_INDEX, N_ACTIONS

WEIGHTS_PATH   = _RL_DIR / "ppo_weights.npz"
TRAINING_LOG   = _RL_DIR / "ppo_training_log.json"
TRAINING_CURVE = _RL_DIR / "ppo_training_curve.png"

BATCH_SIZE = 32   # episodes per PPO update
MAX_STEPS  = 6   

#simulates research.db evolving during an episode

class WorldState:
    def __init__(self, papers: int, claims: int, entities: int):
        self.papers    = papers
        self.claims    = claims
        self.entities  = entities
        self.contras   = 0
        self.critiques = 0
        self.gaps      = 0
        self.ran: list[str] = []

    def apply_action(self, action: str) -> float:

        shaping = 0.0

        if action == "run_comparator":
            already = self.ran.count("run_comparator")
            if self.papers >= 2:
                base = min(self.papers / 8.0, 1.0) * (0.75 ** already)
                new_c = int(base * self.papers * 1.5 * (0.5 + random.random() * 0.5))
                self.contras += max(new_c, 0)
                shaping = 0.08 * base if already == 0 else -0.05
            else:
              
                shaping = -0.03

        elif action == "run_critic":
            already  = self.ran.count("run_critic")
            comp_ran = "run_comparator" in self.ran
            base     = min(self.claims / 15.0, 1.0) * (0.70 ** already)
            bonus    = 1.35 if comp_ran else 0.80
            new_k    = int(base * bonus * max(self.papers, 1) * (0.6 + random.random() * 0.4))
            self.critiques += max(new_k, 0)
            shaping = 0.07 * base * bonus if already == 0 else -0.05

        elif action == "run_gap_detector":
            already   = self.ran.count("run_gap_detector")
            comp_ran  = "run_comparator" in self.ran
            crit_ran  = "run_critic"     in self.ran
            base      = (0.25 + min(self.papers / 12.0, 0.45)) * (0.65 ** already)
            bonus     = 1.0 + (0.30 if comp_ran else 0.0) + (0.20 if crit_ran else 0.0)
            new_g     = int(base * bonus * (3 + random.random() * 5))
            self.gaps += max(new_g, 0)
            shaping = 0.07 * base * bonus if already == 0 else -0.04

        elif action == "run_writer":
            ctx     = self._context_score()
            shaping = 0.15 * ctx - 0.10 * (1.0 - ctx)

        self.ran.append(action)
        return float(np.clip(shaping, -0.20, 0.20))

    def _context_score(self) -> float:
        comp = "run_comparator"  in self.ran
        crit = "run_critic"      in self.ran
        gap  = "run_gap_detector" in self.ran
        s    = 0.35 * comp + 0.30 * crit + 0.25 * gap
        if comp and crit:
            try:
                if self.ran.index("run_comparator") < self.ran.index("run_critic"):
                    s += 0.05
            except ValueError:
                pass
        if (comp or crit) and gap:
            s += 0.05
        return min(s, 1.0)

    def to_vector(self) -> np.ndarray:
        fired_comp = 1.0 if "run_comparator"   in self.ran else 0.0
        fired_crit = 1.0 if "run_critic"        in self.ran else 0.0
        fired_gap  = 1.0 if "run_gap_detector"  in self.ran else 0.0
        fired_writer = 1.0 if "run_writer" in self.ran else 0.0
        step_fraction = len(self.ran) / 6.0
        raw = np.array([
            self.papers, self.claims, self.entities,
            self.contras, self.critiques, self.gaps,
            fired_comp, fired_crit, fired_gap,
            fired_writer, step_fraction,
        ], dtype=np.float32)
        raw[:6] = np.log1p(raw[:6])
        return raw


def terminal_reward(world: WorldState) -> tuple[float, dict]:
    ran           = world.ran
    writer_called = "run_writer" in ran

    # Contradiction yield
    if world.papers >= 2:
        max_pairs = max(world.papers * (world.papers - 1) / 2, 1)
        cy = min(world.contras / max_pairs, 1.0)
    else:
        cy = 0.05

    # Critique depth
    cd = (min(world.critiques / max(world.papers * 2, 1), 1.0) ** 0.65
          if world.critiques > 0 else 0.0)

    # Gap novelty
    gn = min(world.gaps / 8.0, 1.0) if world.gaps > 0 else 0.0

    # Review completeness 
    rc = world._context_score() if writer_called else 0.0

    # Efficiency
    counts     = Counter(ran)
    duplicates = sum(max(v - 1, 0) for v in counts.values())
    eff        = max(1.0 - duplicates * 0.18, 0.25)

    # Ordering bonus
    seq_bonus = 0.0
    if "run_comparator" in ran and "run_critic" in ran:
        if ran.index("run_comparator") < ran.index("run_critic"):
            seq_bonus += 0.05
    if "run_critic" in ran and "run_gap_detector" in ran:
        if ran.index("run_critic") < ran.index("run_gap_detector"):
            seq_bonus += 0.04
    if "run_gap_detector" in ran and "run_writer" in ran:
        if ran.index("run_gap_detector") < ran.index("run_writer"):
            seq_bonus += 0.04

    w = {
        "contradiction_yield":  0.22,
        "critique_depth":       0.18,
        "gap_novelty":          0.18,
        "review_completeness":  0.32,
        "efficiency":           0.10,
    }
    components = dict(
        contradiction_yield=cy,
        critique_depth=cd,
        gap_novelty=gn,
        review_completeness=rc,
        efficiency=eff,
    )
    reward = sum(w[k] * components[k] for k in w) + seq_bonus
    return float(np.clip(reward, 0.0, 1.0)), components


def sample_initial_state() -> WorldState:
    regime = random.choices(["early", "mid", "late"], weights=[0.35, 0.40, 0.25])[0]
    if regime == "early":
        p = random.randint(1, 3)
        return WorldState(p, random.randint(0, p * 5), random.randint(0, p * 8))
    elif regime == "mid":
        p = random.randint(3, 8)
        return WorldState(p, random.randint(p * 3, p * 10), random.randint(p * 5, p * 15))
    else:
        p = random.randint(8, 20)
        return WorldState(p, random.randint(p * 8, p * 20), random.randint(p * 10, p * 25))



def rollout_episode(policy: PPOPolicy) -> dict:
    """
    Policy chooses freely from all 4 actions each step.
    No forbidden set. Learns from experience that writer-first is bad.
    Episode ends: writer called OR MAX_STEPS hit.
    """
    world = sample_initial_state()
    traj  = dict(states=[], actions=[], rewards=[], log_probs=[], values=[], dones=[])

    for step in range(MAX_STEPS):
        sv         = world.to_vector()
        probs, _   = policy.actor.forward(sv)
        action_idx = int(np.random.choice(N_ACTIONS, p=probs))
        action     = ACTIONS[action_idx]
        log_prob   = float(np.log(probs[action_idx] + 1e-8))
        value      = policy.critic.value(sv)

        shaping    = world.apply_action(action)
        is_term    = (action == "run_writer") or (step == MAX_STEPS - 1)

        if is_term:
            term_r, _ = terminal_reward(world)
            total_r   = shaping + term_r
            done      = True
        else:
            total_r = shaping
            done    = False

        traj["states"].append(sv)
        traj["actions"].append(action_idx)
        traj["rewards"].append(total_r)
        traj["log_probs"].append(log_prob)
        traj["values"].append(value)
        traj["dones"].append(done)

        if is_term:
            break

    final_r, comps = terminal_reward(world)
    traj["final_reward"] = final_r
    traj["sequence"]     = list(world.ran)
    traj["components"]   = comps
    return traj


def ppo_update(policy: PPOPolicy, batch: list[dict]) -> dict:
    all_s, all_a, all_adv, all_ret, all_olp = [], [], [], [], []

    for traj in batch:
        last_done = traj["dones"][-1]
        next_val  = 0.0 if last_done else policy.critic.value(traj["states"][-1])
        adv, ret  = compute_gae(
            traj["rewards"], traj["values"], next_val, traj["dones"])
        all_s.extend(traj["states"])
        all_a.extend(traj["actions"])
        all_adv.extend(adv)
        all_ret.extend(ret)
        all_olp.extend(traj["log_probs"])

    adv_arr = np.array(all_adv)
    if adv_arr.std() > 1e-8:
        adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)

    n = len(all_s)
    actor_losses, critic_losses = [], []
    for _ in range(PPO_EPOCHS):
        for i in np.random.permutation(n):
            _, ac = policy.actor.forward(all_s[i])
            actor_losses.append(
                policy.actor.backward_and_update(
                    ac, all_a[i], float(adv_arr[i]), all_olp[i]))
            _, cc = policy.critic.forward(all_s[i])
            critic_losses.append(
                policy.critic.backward_and_update(cc, all_ret[i]))

    policy._save()
    return {
        "actor_loss":  float(np.mean(actor_losses)),
        "critic_loss": float(np.mean(critic_losses)),
        "mean_return": float(np.mean(all_ret)),
    }



def train(n_episodes: int = 500, verbose: bool = False):
    n_updates = max(n_episodes // BATCH_SIZE, 1)
    total_eps = n_updates * BATCH_SIZE

    print(f"\n{'═'*68}")
    print(f"  PPO Training — Research Agent Planner")
    print(f"  No forced ordering. Policy learns writer-last from reward signal.")
    print(f"  Episodes : {total_eps}  ({n_updates} updates × {BATCH_SIZE} eps/update)")
    print(f"  Max steps/episode : {MAX_STEPS}")
    print(f"  Weights  - {WEIGHTS_PATH}")
    print(f"{'═'*68}")

    policy      = PPOPolicy(save_path=str(WEIGHTS_PATH))
    all_rewards = []
    update_logs = []
    start       = time.time()

    for upd in range(1, n_updates + 1):
        batch      = [rollout_episode(policy) for _ in range(BATCH_SIZE)]
        ep_rewards = [t["final_reward"] for t in batch]
        all_rewards.extend(ep_rewards)

        metrics                = ppo_update(policy, batch)
        metrics["update"]      = upd
        metrics["mean_reward"] = float(np.mean(ep_rewards))
        update_logs.append(metrics)

        if verbose or upd % 5 == 0:
            greedy = _greedy_rollout(policy)
            gs     = " → ".join(a.replace("run_", "") for a in greedy)

            seq_counts = Counter(tuple(t["sequence"]) for t in batch)
            top_seq    = seq_counts.most_common(1)[0][0]
            ts_str     = " → ".join(a.replace("run_", "") for a in top_seq)

            ep_s = (upd-1)*BATCH_SIZE+1; ep_e = upd*BATCH_SIZE
            print(
                f"  upd {upd:3d}/{n_updates}  "
                f"eps {ep_s:3d}-{ep_e:3d}  "
                f"rew={metrics['mean_reward']:.4f}  "
                f"actor={metrics['actor_loss']:+.4f}  "
                f"critic={metrics['critic_loss']:.4f}  "
                f"greedy=[{gs}]"
            )

    elapsed   = time.time() - start
    r_arr     = np.array(all_rewards)
    first_avg = float(np.mean(r_arr[:BATCH_SIZE]))
    last_avg  = float(np.mean(r_arr[-BATCH_SIZE:]))

    print(f"\n{'─'*68}")
    print(f"  Done in {elapsed:.1f}s")
    print(f"  First batch avg reward : {first_avg:.4f}")
    print(f"  Last  batch avg reward : {last_avg:.4f}")
    print(f"  Improvement            : {last_avg - first_avg:+.4f}")
    print(f"{'═'*68}\n")

    TRAINING_LOG.write_text(json.dumps(update_logs, indent=2))
    return all_rewards, update_logs


def _greedy_rollout(policy: PPOPolicy, papers: int = 5, claims: int = 40,
                    entities: int = 60) -> list[str]:
    world = WorldState(papers, claims, entities)
    seq   = []
    for _ in range(MAX_STEPS):
        sv       = world.to_vector()
        probs, _ = policy.actor.forward(sv)
        action   = ACTIONS[int(np.argmax(probs))]
        world.apply_action(action)
        seq.append(action)
        if action == "run_writer":
            break
    return seq


def plot_training(rewards: list[float], update_logs: list[dict]):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib"); return

    n      = len(rewards)
    window = max(BATCH_SIZE, n // 10)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("PPO Training — Research Agent Planner (Agent Ordering)", fontsize=13)

    ax = axes[0]
    ax.plot(rewards, alpha=0.2, color="#4C72B0")
    if n >= window:
        sm = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window-1, n), sm, lw=2.5, color="#DD8452",
                label=f"Rolling avg (w={window})")
    ax.axhline(np.mean(rewards[:BATCH_SIZE]),  ls="--", color="red",   alpha=0.7,
               label=f"First ({np.mean(rewards[:BATCH_SIZE]):.2f})")
    ax.axhline(np.mean(rewards[-BATCH_SIZE:]), ls="--", color="green", alpha=0.7,
               label=f"Last ({np.mean(rewards[-BATCH_SIZE:]):.2f})")
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward [0,1]")
    ax.set_title("Episode Reward"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    upd_nums = [u["update"] for u in update_logs]
    axes[1].plot(upd_nums, [u["actor_loss"]  for u in update_logs],
                 label="Actor",  color="#4C72B0", lw=1.5)
    axes[1].plot(upd_nums, [u["critic_loss"] for u in update_logs],
                 label="Critic", color="#C44E52", lw=1.5)
    axes[1].axhline(0, color="gray", lw=0.8, ls=":")
    axes[1].set_xlabel("Update"); axes[1].set_title("Actor & Critic Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(upd_nums, [u["mean_return"] for u in update_logs],
                 color="#55A868", lw=1.5)
    axes[2].set_xlabel("Update"); axes[2].set_title("Mean Return (Value Target)")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(TRAINING_CURVE), dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {TRAINING_CURVE}")



def main():
    parser = argparse.ArgumentParser(description="Pre-train PPO ordering policy")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--plot",    action="store_true")
    parser.add_argument("--reset",   action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.reset and WEIGHTS_PATH.exists():
        WEIGHTS_PATH.unlink()
        print(f"  Deleted {WEIGHTS_PATH}")

    rewards, logs = train(n_episodes=args.episodes, verbose=args.verbose)

    if args.plot:
        plot_training(rewards, logs)

    policy = PPOPolicy(save_path=str(WEIGHTS_PATH))
    print("Policy ready.")


if __name__ == "__main__":
    main()