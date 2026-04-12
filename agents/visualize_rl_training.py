from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time
from collections import defaultdict
from copy import deepcopy
from typing import Optional

import numpy as np

# Import matplotlib with headless backend
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    sys.exit("❌  matplotlib is required.  Run:  pip install matplotlib")

# Import RL components from sibling file
try:
    from gap_selector_rl import (
        GapType, GapPriority,
        GapEnvironment, BanditTrainer, LinUCBDisjoint,
        GapSelectorRL, SessionContext,
        build_feature_vector, compute_reward, evaluate_policy,
        SIM_CANDIDATES, SIM_SELECT_K, SIM_EPISODES, MODEL_PATH,
        FEATURE_DIM, ALPHA,
        W_NOVELTY, W_SPECIFICITY, W_COVERAGE, W_CONSISTENCY, W_CONFIDENCE, W_PRIORITY,
    )
except ImportError as e:
    sys.exit(f"❌  Cannot import gap_selector_rl.py — make sure it's in the same directory.\n    {e}")


# Extends BanditTrainer to capture per-arm and per-component reward history
class InstrumentedTrainer(BanditTrainer):
    """
    Wraps BanditTrainer and records:
      - per-episode average reward
      - per-episode per-arm (gap type) average reward
      - per-episode reward component breakdown (last component snapshot)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.reward_history:     list[float]            = []
        self.arm_history:        dict[str, list[float]] = {
            t.value: [] for t in GapType
        }
        self.component_history:  list[dict[str, float]] = []

    def _run_episode(self) -> float:
        """Override to capture per-arm and per-component breakdowns."""
        candidates, ctx, contradiction_set = self.env.sample_session(
            self.n_candidates
        )
        remaining      = list(candidates)
        selected       = []
        episode_reward = 0.0

        arm_rewards:   dict[str, list[float]] = defaultdict(list)
        comp_snapshot: dict[str, float]       = {
            "novelty": 0.0, "specificity": 0.0,
            "coverage": 0.0, "consistency": 0.0, "confidence": 0.0,
        }

        # Score and select gaps greedily, updating the model after each pick
        for step in range(min(self.k_select, len(remaining))):
            scores, xs = [], []
            for gap in remaining:
                x = build_feature_vector(gap, selected, ctx, len(candidates))
                s = self.model.score(gap, x)
                scores.append(s)
                xs.append(x)

            best_idx = int(np.argmax(scores))
            chosen   = remaining[best_idx]
            x_chosen = xs[best_idx]

            reward = compute_reward(chosen, selected, ctx, contradiction_set)
            self.model.update(chosen, x_chosen, reward)

            arm = chosen.get("gap_type", GapType.COMBINATORIAL.value)
            arm_rewards[arm].append(reward)

            ents        = set(chosen.get("entities_involved", []))
            redundancy  = _redundancy_score_local(chosen, selected)
            n_ents      = len(ents)
            n_papers    = len(set(chosen.get("papers_involved", [])))
            conflict_f  = len(ents & contradiction_set) / max(len(ents), 1)

            comp_snapshot = {
                "novelty":      round(1.0 - redundancy, 4),
                "specificity":  round(min(n_ents / 5.0, 1.0), 4),
                "coverage":     round(min(n_papers / max(ctx.n_papers, 1), 1.0), 4),
                "consistency":  round(1.0 - conflict_f, 4),
                "confidence":   round(float(chosen.get("confidence", 0.5)), 4),
            }

            selected.append(chosen)
            remaining.pop(best_idx)
            episode_reward += reward

        mean_ep = episode_reward / max(len(selected), 1)

        # Store per-arm averages (NaN if arm was not chosen this episode)
        for t in GapType:
            vals = arm_rewards.get(t.value, [])
            self.arm_history[t.value].append(
                sum(vals) / len(vals) if vals else float("nan")
            )

        self.component_history.append(comp_snapshot)
        return mean_ep

    def train(self) -> LinUCBDisjoint:
        t0        = time.time()
        log_every = max(1, self.n_episodes // 10)

        # Run all training episodes and log progress periodically
        for ep in range(1, self.n_episodes + 1):
            avg_r = self._run_episode()
            self.reward_history.append(avg_r)

            if self.verbose and ep % log_every == 0:
                window = self.reward_history[-log_every:]
                print(f"  [train] Episode {ep:>5}/{self.n_episodes}  "
                      f"avg_reward (last {log_every}): {sum(window)/len(window):.4f}")

        elapsed = time.time() - t0
        if self.verbose:
            overall = sum(self.reward_history) / len(self.reward_history)
            late    = sum(self.reward_history[-200:]) / min(200, len(self.reward_history))
            print(f"\n  [train] Done in {elapsed:.1f}s  |  "
                  f"overall avg: {overall:.4f}  |  "
                  f"late-phase avg (last 200): {late:.4f}")
        return self.model


# Computes Jaccard-based redundancy between a gap and already-selected gaps
def _redundancy_score_local(gap: dict, already_selected: list[dict]) -> float:
    """Local copy to avoid circular dependency issues."""
    if not already_selected:
        return 0.0
    my_ents = set(gap.get("entities_involved", []))
    if not my_ents:
        return 0.0
    sims = []
    for sel in already_selected:
        their = set(sel.get("entities_involved", []))
        union = len(my_ents | their)
        sims.append(len(my_ents & their) / union if union else 0.0)
    return max(sims)


# Computes rolling average over a list of values with a given window size
def _rolling(values: list[float], window: int) -> list[float]:
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = [v for v in values[start:i+1] if not math.isnan(v)]
        out.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return out


def plot_training(
    trainer:      InstrumentedTrainer,
    eval_results: dict,
    n_episodes:   int,
    save_path:    pathlib.Path,
):
    """
    Single-panel chart matching the LinUCB pre-training style:
      - Light blue translucent raw episode reward line
      - Thick orange rolling average line
      - Red dashed horizontal line for first-N avg (with value in legend)
      - Green dashed horizontal line for last-N avg (with value in legend)
      - Clean white background, subtle grid
    """
    episodes = list(range(1, n_episodes + 1))
    window   = max(10, n_episodes // 20)

    # Compute rolling stats and first/last window averages
    history   = trainer.reward_history
    rolled    = _rolling(history, window)
    first_avg = sum(history[:window]) / max(len(history[:window]), 1)
    last_avg  = sum(history[-window:]) / max(len(history[-window:]), 1)

    # Set up figure with white background
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    ax.set_facecolor("white")

    fig.suptitle(
        f"LinUCB Pre-Training  —  Gap Selector RL  "
        f"({n_episodes} eps, α={ALPHA}, d={FEATURE_DIM})",
        fontsize=12, fontweight="bold", x=0.5, y=1.01, ha="center",
    )

    # Plot raw episode reward as a thin translucent line
    ax.plot(
        episodes, history,
        color="#A8C4E0",
        linewidth=0.9,
        alpha=0.6,
        label="Episode reward",
        zorder=2,
    )

    # Plot rolling average as a thick line
    ax.plot(
        episodes, rolled,
        color="#D2691E",
        linewidth=2.5,
        label=f"Rolling avg (w={window})",
        zorder=4,
    )

    # Draw first-N and last-N average reference lines
    ax.axhline(
        first_avg,
        color="#E05555",
        linestyle="--",
        linewidth=1.6,
        zorder=3,
        label=f"First-{window} avg ({first_avg:.2f})",
    )

    ax.axhline(
        last_avg,
        color="#4A9A5A",
        linestyle="--",
        linewidth=1.6,
        zorder=3,
        label=f"Last-{window} avg ({last_avg:.2f})",
    )

    # Configure axes labels, limits, and grid
    ax.set_title("Reward Over Training", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Training Episode", fontsize=11)
    ax.set_ylabel("Simulated Reward [0, 1]", fontsize=11)
    ax.set_xlim(0, n_episodes)
    ax.set_ylim(-0.05, 1.05)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))

    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.35, color="#CCCCCC")
    ax.set_axisbelow(True)

    # Add legend and style spines
    legend = ax.legend(
        loc="upper left",
        fontsize=9,
        framealpha=0.92,
        edgecolor="#CCCCCC",
        fancybox=False,
    )
    legend.get_frame().set_linewidth(0.8)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#AAAAAA")

    # Save figure to disk
    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"\n  ✅  Plot saved → {save_path.resolve()}")
    plt.close(fig)


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(
        description="Visualize Gap Selector RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_rl_training.py                   # full 2 000-episode run
  python visualize_rl_training.py --episodes 500    # quick check
  python visualize_rl_training.py --load-only       # eval & plot saved model
  python visualize_rl_training.py --no-save         # train but don't overwrite model
        """,
    )
    parser.add_argument("--episodes",  type=int, default=SIM_EPISODES,
                        help=f"Training episodes (default: {SIM_EPISODES})")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--no-save",   action="store_true",
                        help="Don't overwrite shared_memory/gap_rl_model.json")
    parser.add_argument("--load-only", action="store_true",
                        help="Skip training; load saved model and evaluate only")
    parser.add_argument("--out",       type=str, default="rl_training_report.png",
                        help="Output image path (default: rl_training_report.png)")
    parser.add_argument("--eval-eps",  type=int, default=500,
                        help="Evaluation episodes (default: 500)")
    args = parser.parse_args()

    save_path = pathlib.Path(args.out)

    # Load saved model and evaluate without training
    if args.load_only:
        model_path = MODEL_PATH
        if not model_path.exists():
            sys.exit(f"❌  No saved model at {model_path}. "
                     f"Run without --load-only to train first.")
        print(f"  Loading saved model from {model_path} ...")
        with open(model_path, encoding="utf-8") as f:
            payload = json.load(f)
        model = LinUCBDisjoint.from_dict(payload["model"])
        print(f"  Evaluating on {args.eval_eps} held-out sessions ...")
        eval_results = evaluate_policy(model, n_eval_episodes=args.eval_eps,
                                       seed=args.seed + 1)
        print(f"  RL mean reward      : {eval_results['rl_mean']}")
        print(f"  Heuristic baseline  : {eval_results['heuristic_mean']}")
        print(f"  Random baseline     : {eval_results['random_mean']}")
        print(f"  Lift vs heuristic   : {eval_results['rl_vs_heuristic_lift']:+.2%}")
        print(f"  Lift vs random      : {eval_results['rl_vs_random_lift']:+.2%}")

        dummy_trainer = InstrumentedTrainer(
            n_episodes=1, n_candidates=SIM_CANDIDATES,
            k_select=SIM_SELECT_K, seed=args.seed, verbose=False
        )
        dummy_trainer.model          = model
        dummy_trainer.reward_history = [eval_results["rl_mean"]]
        dummy_trainer.arm_history    = {t.value: [eval_results["rl_mean"]]
                                        for t in GapType}
        dummy_trainer.component_history = [{
            "novelty": 0.0, "specificity": 0.0,
            "coverage": 0.0, "consistency": 0.0, "confidence": 0.0,
        }]
        plot_training(dummy_trainer, eval_results, 1, save_path)
        return

    # Print training configuration header
    print("=" * 62)
    print(f"  Gap Selector RL — Instrumented Training + Visualization")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Candidates : {SIM_CANDIDATES} per episode")
    print(f"  Select K   : {SIM_SELECT_K}")
    print(f"  Features   : {FEATURE_DIM}")
    print(f"  Alpha (UCB): {ALPHA}")
    print(f"  Seed       : {args.seed}")
    print("=" * 62)

    # Run instrumented training
    trainer = InstrumentedTrainer(
        n_episodes   = args.episodes,
        n_candidates = SIM_CANDIDATES,
        k_select     = SIM_SELECT_K,
        seed         = args.seed,
        verbose      = True,
    )
    model = trainer.train()

    # Evaluate trained model against baselines
    print(f"\n  Evaluating on {args.eval_eps} held-out sessions ...")
    eval_results = evaluate_policy(model, n_eval_episodes=args.eval_eps,
                                   seed=args.seed + 1)
    print(f"  RL mean reward      : {eval_results['rl_mean']}")
    print(f"  Heuristic baseline  : {eval_results['heuristic_mean']}")
    print(f"  Random baseline     : {eval_results['random_mean']}")
    print(f"  Lift vs heuristic   : {eval_results['rl_vs_heuristic_lift']:+.2%}")
    print(f"  Lift vs random      : {eval_results['rl_vs_random_lift']:+.2%}")

    # Persist model to disk unless --no-save flag is set
    if not args.no_save:
        selector = GapSelectorRL(model)
        selector.save(MODEL_PATH)
    else:
        print("  [--no-save] Model NOT written to disk.")

    # Generate and save the training plot
    print(f"\n  Generating plot → {save_path} ...")
    plot_training(trainer, eval_results, args.episodes, save_path)


if __name__ == "__main__":
    main()