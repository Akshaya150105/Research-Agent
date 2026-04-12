import sqlite3
import pathlib
import numpy as np
from typing import Optional

#Reward weights
_WEIGHTS = {
    "contradiction_yield": 0.25,
    "critique_depth":      0.20,
    "gap_novelty":         0.20,
    "sequence_score":      0.15,
    "review_completeness": 0.15,
    "efficiency":          0.05,
}

#Sections the Writer should produce
_EXPECTED_SECTIONS = [
    "## Methods",
    "## Results",
    "## Contradictions",
    "## Gaps",
    "## Conclusion",
]

_IDEAL_INVOCATIONS = 4

def compute_reward(
    db_path:           pathlib.Path | str,
    session_id:        str,
    review_path:       Optional[pathlib.Path | str],
    agent_invocations: int,
    verbose:           bool = False,
) -> tuple[float, dict[str, float]]:
    db_path = pathlib.Path(str(db_path))
    components: dict[str, float] = {}

    #Contradiction yield
    try:
        conn = sqlite3.connect(str(db_path))
        cur  = conn.cursor()

        paper_count = cur.execute(
            "SELECT COUNT(*) FROM papers WHERE action='read'"
        ).fetchone()[0]

        n_contradictions = cur.execute(
            "SELECT COUNT(*) FROM comparisons "
            "WHERE CAST(severity AS REAL) >= 0.5 "
            "OR severity IN ('HIGH', 'MEDIUM')"
        ).fetchone()[0]

        max_pairs = max(paper_count * (paper_count - 1) / 2, 1)
        components["contradiction_yield"] = min(n_contradictions / 3.0, 1.0)

        #Critique depth
        critique_rows = cur.execute(
            "SELECT severity FROM critiques"
        ).fetchall() if _table_exists(cur, "critiques") else []

        if critique_rows:
            sev_map = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}
            scores  = []
            for (s,) in critique_rows:
                try:
                    scores.append(float(s))
                except (ValueError, TypeError):
                    scores.append(sev_map.get(str(s).upper(), 0.3))
            components["critique_depth"] = float(np.mean(scores) ** 2)
        else:
            components["critique_depth"] = 0.0

        #Gap novelty
        if _table_exists(cur, "gaps"):
            current_gaps = cur.execute(
                "SELECT entity_combination FROM gaps WHERE session_id=?",
                (session_id,)
            ).fetchall()
            prior_combos = {
                r[0] for r in cur.execute(
                    "SELECT entity_combination FROM gaps WHERE session_id!=?",
                    (session_id,)
                ).fetchall()
            }
            if current_gaps:
                novel = sum(1 for (ec,) in current_gaps if ec not in prior_combos)
                components["gap_novelty"] = min(novel / len(current_gaps), 1.0)
            else:
                components["gap_novelty"] = 0.0
        else:
            components["gap_novelty"] = 0.0

        #Sequence score
        sequence_score = 0.0
        if _table_exists(cur, "agent_actions"):
            actions_rows = cur.execute(
                "SELECT action_name FROM agent_actions WHERE session_id=? ORDER BY timestamp",
                (session_id,)
            ).fetchall()
            actions = [a[0] for a in actions_rows]
            
            if "run_comparator" in actions and "run_critic" in actions:
                if actions.index("run_comparator") < actions.index("run_critic"):
                    sequence_score += 1.0
            if "run_critic" in actions and "run_gap_detector" in actions:
                if actions.index("run_critic") < actions.index("run_gap_detector"):
                    sequence_score += 1.0
            if "run_gap_detector" in actions and "run_writer" in actions:
                if actions.index("run_gap_detector") < actions.index("run_writer"):
                    sequence_score += 1.0
            sequence_score /= 3.0
            
        components["sequence_score"] = sequence_score

        conn.close()
    except Exception as exc:
        if verbose:
            print(f"  [reward] DB read error: {exc}")
        components.setdefault("contradiction_yield", 0.0)
        components.setdefault("critique_depth",      0.0)
        components.setdefault("gap_novelty",         0.0)
        components.setdefault("sequence_score",      0.0)

    #Review completeness
    review_text = ""
    if review_path and pathlib.Path(str(review_path)).exists():
        try:
            review_text = pathlib.Path(str(review_path)).read_text(encoding="utf-8")
        except Exception:
            pass
    found = sum(1 for sec in _EXPECTED_SECTIONS if sec in review_text)
    components["review_completeness"] = found / len(_EXPECTED_SECTIONS)

    #Efficiency
    excess = max(agent_invocations - _IDEAL_INVOCATIONS, 0)
    components["efficiency"] = max(1.0 - excess / _IDEAL_INVOCATIONS, 0.0)

    #Weighted sum
    reward = sum(_WEIGHTS[k] * components.get(k, 0.0) for k in _WEIGHTS)
    reward = float(np.clip(reward, 0.0, 1.0))

    if verbose:
        print(f"[reward] breakdown: {components}")
        print(f"[reward] total={reward:.4f}")

    return reward, components


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    """Return True if `table` exists in the connected database."""
    row = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return row is not None