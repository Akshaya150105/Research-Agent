#Visualise learning progress and policy weights after multiple sessions.


import argparse
import json
import sqlite3
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt


_RL_DIR      = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _RL_DIR.parent
for p in [str(_PROJECT_ROOT), str(_RL_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from planner_rl.bandit_policy import LinUCBPolicy, ACTIONS
from planner_rl.state_encoder import STATE_LABELS

DEFAULT_DB   = _PROJECT_ROOT / "shared_memory" / "research.db"
DEFAULT_WEIGHTS = _RL_DIR / "policy_weights.json"
OUTPUT_PNG   = _RL_DIR / "learning_curve.png"


def load_episode_rewards(db_path: pathlib.Path) -> list[dict]:
    """Return rl_episodes rows ordered by timestamp."""
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    rows = cur.execute(
        "SELECT episode_id, session_id, timestamp, reward, "
        "       paper_count, contradiction_count, critique_count, gap_count, "
        "       reward_breakdown "
        "FROM rl_episodes ORDER BY timestamp ASC"
    ).fetchall()
    conn.close()
    cols = ["episode_id", "session_id", "timestamp", "reward",
            "paper_count", "contradiction_count", "critique_count", "gap_count",
            "reward_breakdown"]
    return [dict(zip(cols, r)) for r in rows]


def load_decision_frequencies(db_path: pathlib.Path) -> dict[str, int]:
    """Count how many times each action was selected across all decisions."""
    conn = sqlite3.connect(str(db_path))
    cur  = conn.cursor()
    rows = cur.execute(
        "SELECT action, COUNT(*) FROM rl_decisions GROUP BY action"
    ).fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


def plot_learning(episodes: list[dict], weights_path: pathlib.Path) -> None:
    rewards = [e["reward"] for e in episodes]
    n       = len(rewards)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Research Agent — RL Planner Learning Diagnostics", fontsize=13)

    #reward over episodes
    ax = axes[0]
    ax.plot(rewards, alpha=0.35, color="#4C72B0", label="Episode reward")

    if n >= 3:
        window   = max(2, n // 5)
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, n),
            smoothed,
            linewidth=2,
            color="#DD8452",
            label=f"Rolling avg (w={window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward [0, 1]")
    ax.set_title("Reward Over Time")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotate first and last reward
    if n >= 2:
        ax.annotate(f"{rewards[0]:.2f}", (0, rewards[0]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.annotate(f"{rewards[-1]:.2f}", (n - 1, rewards[-1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    #theta heatmap
    ax2    = axes[1]
    policy = LinUCBPolicy(save_path=str(weights_path))
    theta_matrix = np.array([policy.get_theta(i) for i in range(len(ACTIONS))])
    vmax = max(abs(theta_matrix.max()), abs(theta_matrix.min()), 1e-6)
    im   = ax2.imshow(
        theta_matrix,
        cmap="RdYlGn",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
    )
    ax2.set_yticks(range(len(ACTIONS)))
    ax2.set_yticklabels(ACTIONS, fontsize=9)
    ax2.set_xticks(range(len(STATE_LABELS)))
    ax2.set_xticklabels(STATE_LABELS, rotation=45, ha="right", fontsize=8)
    ax2.set_title("Learned Policy Weights (θ per action arm)")
    plt.colorbar(im, ax=ax2, shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(OUTPUT_PNG), dpi=150, bbox_inches="tight")
    print(f"\n  📊 Plot saved → {OUTPUT_PNG}")


def print_summary(episodes: list[dict], freqs: dict[str, int]) -> None:
    n = len(episodes)
    print(f"\n{'═'*60}")
    print(f"  RL PLANNER — SESSION SUMMARY  ({n} episodes recorded)")
    print(f"{'═'*60}")

    if n == 0:
        print("  No episodes found yet. Run at least one session.")
        return

    rewards = [e["reward"] for e in episodes]
    print(f"  First reward   : {rewards[0]:.4f}")
    print(f"  Last  reward   : {rewards[-1]:.4f}")
    print(f"  Mean  reward   : {np.mean(rewards):.4f}")
    print(f"  Best  reward   : {max(rewards):.4f}")
    print(f"  Improvement    : {rewards[-1] - rewards[0]:+.4f}")

    # Show reward breakdown for last episode if available
    last_breakdown = episodes[-1].get("reward_breakdown")
    if last_breakdown:
        try:
            bd = json.loads(last_breakdown)
            print(f"\n  Last-episode reward breakdown:")
            for k, v in bd.items():
                print(f"    {k:28s}: {v:.4f}")
        except Exception:
            pass

    print(f"\n  Action selection frequencies:")
    total_decisions = sum(freqs.values()) or 1
    for action in ACTIONS:
        cnt  = freqs.get(action, 0)
        pct  = cnt / total_decisions * 100
        bar  = "█" * int(pct / 5)
        print(f"    {action:25s}: {cnt:4d}  ({pct:5.1f}%)  {bar}")

    print(f"{'═'*60}\n")


def main():
    parser = argparse.ArgumentParser(description="RL Planner Evaluation")
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="Path to research.db")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                        help="Path to policy_weights.json")
    args = parser.parse_args()

    db_path = pathlib.Path(args.db)
    if not db_path.exists():
        print(f"⚠  DB not found: {db_path}")
        sys.exit(1)

    episodes = load_episode_rewards(db_path)
    freqs    = load_decision_frequencies(db_path)

    print_summary(episodes, freqs)
    plot_learning(episodes, pathlib.Path(args.weights))


if __name__ == "__main__":
    main()