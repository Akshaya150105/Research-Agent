import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

_RL_DIR       = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _RL_DIR.parent
for p in [str(_PROJECT_ROOT), str(_RL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from planner_rl.bandit_policy import LinUCBPolicy, ACTIONS, N_ACTIONS, ALPHA
from planner_rl.state_encoder import STATE_DIM, STATE_LABELS

WEIGHTS_PATH     = _RL_DIR / "policy_weights.json"
TRAINING_LOG     = _RL_DIR / "training_log.json"
TRAINING_CURVE   = _RL_DIR / "training_curve.png"


def simulate_reward(
    state_vec:    np.ndarray,
    action_sequence: list[str],
) -> tuple[float, dict[str, float]]:

    # Decode log1p-transformed counts back to approximate raw values
    papers        = int(np.expm1(state_vec[0]))
    claims        = int(np.expm1(state_vec[1]))
    existing_contras = int(np.expm1(state_vec[3]))

    ran_comparator  = "run_comparator"  in action_sequence
    ran_critic      = "run_critic"      in action_sequence
    ran_gap         = "run_gap_detector" in action_sequence
    ran_writer      = "run_writer"      in action_sequence
    n_agents        = len(action_sequence)

    #Contradiction yield
    if ran_comparator and papers >= 2:
        # More papers - more potential contradictions
        base_yield = min(papers / 10, 1.0)
        # If contradictions already existed in DB, diminishing returns
        already_found_penalty = min(existing_contras / 5, 0.5)
        contradiction_yield = base_yield * (1.0 - already_found_penalty * 0.3)
    elif ran_comparator and papers < 2:
        contradiction_yield = 0.05   # nearly useless with <2 papers
    else:
        contradiction_yield = 0.0    # didn't run comparator at all

    #Critique depth
    # Critic needs claims to exist; more claims → deeper critique
    if ran_critic and claims >= 3:
        critique_depth = min(claims / 20, 1.0) * 0.85
        # Critic is more effective if Comparator already ran
        if ran_comparator:
            critique_depth = min(critique_depth * 1.2, 1.0)
    elif ran_critic and claims < 3:
        critique_depth = 0.15   # shallow - not enough to critique
    else:
        critique_depth = 0.0

    #Gap novelty
    # Gap detector is most effective after Comparator found contradictions
    if ran_gap:
        base_novelty = 0.4 + (0.3 if ran_comparator else 0.0)
        # More papers = more entity combinations = more novel gaps
        base_novelty += min(papers / 20, 0.3)
        gap_novelty = min(base_novelty, 1.0)
    else:
        gap_novelty = 0.0

    #Review completeness
    # Writer can only fill sections if the relevant agents ran
    sections_possible = 1  # introduction always possible
    if ran_comparator: sections_possible += 1   # Contradictions section
    if ran_critic:     sections_possible += 1   # Weaknesses section
    if ran_gap:        sections_possible += 1   # Gaps section
    if papers >= 1:    sections_possible += 1   # Conclusion always
    review_completeness = sections_possible / 5 if ran_writer else 0.0

    #Efficiency
    # Penalise for invoking more than needed
    excess = max(n_agents - 4, 0)
    efficiency = max(1.0 - excess / 4, 0.0)

    #Sequence score
    sequence_score = 0.0
    if "run_comparator" in action_sequence and "run_critic" in action_sequence:
        if action_sequence.index("run_comparator") < action_sequence.index("run_critic"):
            sequence_score += 1.0
    if "run_critic" in action_sequence and "run_gap_detector" in action_sequence:
        if action_sequence.index("run_critic") < action_sequence.index("run_gap_detector"):
            sequence_score += 1.0
    if "run_gap_detector" in action_sequence and "run_writer" in action_sequence:
        if action_sequence.index("run_gap_detector") < action_sequence.index("run_writer"):
            sequence_score += 1.0
    sequence_score /= 3.0

    #Weighted sum
    weights = {
        "contradiction_yield": 0.25,
        "critique_depth":      0.20,
        "gap_novelty":         0.20,
        "sequence_score":      0.15,
        "review_completeness": 0.15,
        "efficiency":          0.05,
    }
    components = {
        "contradiction_yield": contradiction_yield,
        "critique_depth":      critique_depth,
        "gap_novelty":         gap_novelty,
        "sequence_score":      sequence_score,
        "review_completeness": review_completeness,
        "efficiency":          efficiency,
    }
    reward = sum(weights[k] * components[k] for k in weights)
    reward = float(np.clip(reward, 0.0, 1.0))

    return reward, components

def sample_state() -> np.ndarray:
    regime = random.choices(
        ["early", "mid", "late"],
        weights=[0.35, 0.40, 0.25]
    )[0]

    if regime == "early":
        papers     = random.randint(1, 3)
        claims     = random.randint(0, papers * 5)
        entities   = random.randint(0, papers * 8)
        contras    = 0
        critiques  = 0
        gaps       = 0

    elif regime == "mid":
        papers     = random.randint(3, 8)
        claims     = random.randint(papers * 3, papers * 10)
        entities   = random.randint(papers * 5, papers * 15)
        contras    = random.randint(0, papers)
        critiques  = random.randint(0, papers * 2)
        gaps       = random.randint(0, 5)

    else:
        papers     = random.randint(8, 20)
        claims     = random.randint(papers * 8, papers * 20)
        entities   = random.randint(papers * 10, papers * 25)
        contras    = random.randint(papers // 2, papers * 2)
        critiques  = random.randint(papers, papers * 3)
        gaps       = random.randint(3, 15)


    fired_comp   = 0.0
    fired_crit   = 0.0
    fired_gap    = 0.0
    fired_writer = 0.0
    step_frac    = random.uniform(0.0, 0.8)

    raw = np.array([
        papers, claims, entities, contras, critiques, gaps,
        fired_comp, fired_crit, fired_gap, fired_writer, step_frac
    ], dtype=np.float32)

    raw[:6] = np.log1p(raw[:6])
    return raw


def simulate_episode(policy: LinUCBPolicy) -> tuple[float, list[str], dict]:

    state         = sample_state()
    action_seq    = []
    states_taken  = []   
    for step in range(5):  
        # LinUCB native exploration via ALPHA term
        chosen, _probs = policy.select_action(state, forbidden=[])

        action_seq.append(chosen)
        states_taken.append((state.copy(), ACTIONS.index(chosen)))

        if chosen == "run_writer":
                break
        # Update fired flags in state vector so next decision sees them
        flag_map = {
            "run_comparator":   6,
            "run_critic":       7,
            "run_gap_detector": 8,
            "run_writer":       9,
        }
        if chosen in flag_map:
            state[flag_map[chosen]] = 1.0
        state[10] = (step + 1) / 5.0

    # Compute reward
    reward, components = simulate_reward(state, action_seq)
    
    reward = float(np.clip(reward, 0.0, 1.0))
    for sv, action_idx in states_taken:
        policy.update(sv, action_idx, reward)

    return reward, action_seq, components


def train(n_episodes: int = 100, verbose: bool = False) -> list[float]:
    print(f"\n{'═'*60}")
    print(f"  LinUCB Pre-Training  |  {n_episodes} episodes")
    print(f"  Weights - {WEIGHTS_PATH}")
    print(f"{'═'*60}")

    policy  = LinUCBPolicy(save_path=str(WEIGHTS_PATH))
    rewards = []
    log     = []

    start = time.time()

    for ep in range(1, n_episodes + 1):
        reward, seq, components = simulate_episode(policy)
        rewards.append(reward)
        log.append({
            "episode":    ep,
            "reward":     round(reward, 4),
            "sequence":   seq,
            "components": {k: round(v, 3) for k, v in components.items()},
        })

        if verbose or ep % 10 == 0:
            seq_short = " → ".join(s.replace("run_", "") for s in seq)
            print(f"  ep {ep:4d}/{n_episodes}  reward={reward:.4f}  [{seq_short}]")

    elapsed = time.time() - start

    TRAINING_LOG.write_text(json.dumps(log, indent=2))

    rewards_arr = np.array(rewards)
    first10_avg = float(np.mean(rewards_arr[:10]))
    last10_avg  = float(np.mean(rewards_arr[-10:]))
    improvement = last10_avg - first10_avg

    print(f"\n{'─'*60}")
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  First 10 episodes avg reward : {first10_avg:.4f}")
    print(f"  Last  10 episodes avg reward : {last10_avg:.4f}")
    print(f"  Improvement                  : {improvement:+.4f}")
    print(f"  Training log - {TRAINING_LOG}")
    print(f"  Policy weights - {WEIGHTS_PATH}")
    print(f"{'═'*60}\n")

    return rewards


def plot_training(rewards: list[float]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. pip install matplotlib")
        return

    n = len(rewards)
    window   = min(50, max(5, n // 10))
    smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("LinUCB Pre-Training — Research Agent Planner", fontsize=13)

    # Left: reward curve
    ax = axes[0]
    ax.plot(rewards, alpha=0.25, color="#4C72B0", label="Episode reward")
    ax.plot(np.arange(window - 1, n), smoothed,
            linewidth=2.5, color="#DD8452", label=f"Rolling avg (w={window})")
    ax.axhline(np.mean(rewards[:10]),  linestyle="--", color="red",
               alpha=0.5, label=f"First-10 avg ({np.mean(rewards[:10]):.2f})")
    ax.axhline(np.mean(rewards[-10:]), linestyle="--", color="green",
               alpha=0.5, label=f"Last-10 avg ({np.mean(rewards[-10:]):.2f})")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Simulated Reward [0, 1]")
    ax.set_title("Reward Over Training")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Right: learned theta heatmap
    ax2    = axes[1]
    policy = LinUCBPolicy(save_path=str(WEIGHTS_PATH))
    theta_matrix = np.array([policy.get_theta(i) for i in range(N_ACTIONS)])
    vmax = max(abs(theta_matrix).max(), 1e-6)
    im   = ax2.imshow(theta_matrix, cmap="RdYlGn", aspect="auto",
                      vmin=-vmax, vmax=vmax)
    ax2.set_yticks(range(N_ACTIONS))
    ax2.set_yticklabels(ACTIONS, fontsize=8)
    ax2.set_xticks(range(STATE_DIM))
    ax2.set_xticklabels(STATE_LABELS, rotation=45, ha="right", fontsize=7)
    ax2.set_title("Learned θ Weights (green=prefer, red=avoid)")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(TRAINING_CURVE), dpi=150, bbox_inches="tight")
    print(f"Training curve saved - {TRAINING_CURVE}")


def print_policy_intuition(policy: LinUCBPolicy) -> None:
    print("\n  What the policy learned (top weights per action):")
    print(f"  {'─'*55}")
    for i, action in enumerate(ACTIONS):
        theta = policy.get_theta(i)
        # Top 3 most positive weights
        top_idx = np.argsort(theta)[::-1][:3]
        top = [(STATE_LABELS[j], round(float(theta[j]), 3)) for j in top_idx
               if theta[j] > 0]
        if top:
            top_str = ", ".join(f"{lbl}({val:+.2f})" for lbl, val in top)
            print(f"  {action:25s} - prefers: {top_str}")
        else:
            print(f"  {action:25s} - no strong positive weights yet")
    print(f"  {'─'*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train the LinUCB planner policy via simulation"
    )
    parser.add_argument(
        "--episodes", type=int, default=100,
        help="Number of simulated training episodes (default: 100)"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save training curve plot to rl/training_curve.png"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing weights and train from scratch"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print every episode (default: every 10)"
    )
    args = parser.parse_args()

    if args.reset and WEIGHTS_PATH.exists():
        WEIGHTS_PATH.unlink()
        print(f"  [OK] Deleted existing weights: {WEIGHTS_PATH}")

    rewards = train(n_episodes=args.episodes, verbose=args.verbose)

    if args.plot:
        plot_training(rewards)

    policy = LinUCBPolicy(save_path=str(WEIGHTS_PATH))
    print_policy_intuition(policy)

    print("  [OK] Policy is ready for real sessions.")


if __name__ == "__main__":
    main()