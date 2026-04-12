from __future__ import annotations
import sqlite3
import numpy as np
import pathlib
from agents.planner_agent import AgentState

ACTIONS = [
    "run_comparator",    
    "run_critic",       
    "run_gap_detector",  
    "run_writer",        
]
ACTION_INDEX: dict[str, int] = {a: i for i, a in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)


STATE_LABELS = [
    "log_papers",         
    "log_claims",         
    "log_entities",        
    "log_contradictions",  
    "log_critiques",       
    "log_gaps",           
    "fired_comparator",   
    "fired_critic",       
    "fired_gap_detector",  
    "fired_writer",        
    "step_fraction",      
]
STATE_DIM = len(STATE_LABELS)  
MAX_STEPS = 6                 


def get_state_vector(
    state: "AgentState",
    agents_fired: list[str],
    db_path: pathlib.Path | str,
) -> np.ndarray:

    paper_count = claim_count = entity_count = 0
    contradiction_count = critique_count = gap_count = 0

    try:
        conn = sqlite3.connect(str(db_path))
        cur  = conn.cursor()

        def _count(table: str) -> int:
            try:
                return cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                return 0

        paper_count        = _count("papers")
        claim_count        = _count("claims")
        entity_count       = _count("entities")
        contradiction_count = _count("comparisons")
        critique_count     = _count("critiques")
        gap_count          = _count("gaps")
        conn.close()
    except Exception:
        pass 

    fired_comparator  = 1.0 if "run_comparator"   in agents_fired else 0.0
    fired_critic      = 1.0 if "run_critic"        in agents_fired else 0.0
    fired_gap         = 1.0 if "run_gap_detector"  in agents_fired else 0.0
    fired_writer      = 1.0 if "run_writer"        in agents_fired else 0.0

    step_fraction = min(state.get("step_count", 0) / MAX_STEPS, 1.0)

    raw = np.array([
        paper_count,
        claim_count,
        entity_count,
        contradiction_count,
        critique_count,
        gap_count,
        fired_comparator,
        fired_critic,
        fired_gap,
        fired_writer,
        step_fraction,
    ], dtype=np.float32)

    # log1p on count features (indices 0-5) keeps zeros as zeros and compresses heavy-tailed counts into a similar range
    raw[:6] = np.log1p(raw[:6])

    return raw