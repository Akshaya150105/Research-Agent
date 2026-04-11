from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sqlite3
import sys
import uuid
from typing import TypedDict, Annotated
import operator

# LangGraph 
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    print(
        "  langgraph not installed. Run: pip install langgraph\n"
        "   Planner will run in sequential fallback mode.",
        file=sys.stderr,
    )

_AGENTS_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

# Agent imports 
from reader_agent     import ReaderAgent
from writer_agent     import WriterAgent
from comparator_agent import ComparatorAgent
import comparator_agent as _comp_module   

try:
    from critic_agent import CriticAgent
    CRITIC_AVAILABLE = True
except ImportError:
    CRITIC_AVAILABLE = False
    print("  critic_agent not found — critic step skipped.", file=sys.stderr)

try:
    from gap_detector_agent import GapDetectorAgent
    GAP_AVAILABLE = True
except ImportError:
    GAP_AVAILABLE = False
    print("  gap_detector_agent not found — gap step skipped.", file=sys.stderr)

MEMORY_DIR    = pathlib.Path("memory")
SHARED_MEMORY = pathlib.Path("shared_memory")
DB_PATH       = SHARED_MEMORY / "research.db"


#  SHARED STATE
class AgentState(TypedDict, total=False):
    
    #Single dict that flows through every LangGraph node.
    #Every node reads from it, returns a partial update.
    #LangGraph merges updates; react_trace uses operator.add so entries
    #from all nodes are appended rather than overwritten.
    
    session_id:              str
    topic:                   str
    papers_dir:              str   
    ollama_host:             str   
    no_extraction:           bool
    memory_dir:              str
    use_llm:                 bool
    verbose:                 bool
    step_count:              int
    coverage_score:          float
    coverage_gain_threshold: float

    reader_report:     dict
    comparator_report: dict
    critic_report:     dict
    gap_report:        dict
    writer_report:     dict

    session_complete: bool

    react_trace: Annotated[list, operator.add]


#  ROUTING LOGIC
class PlannerLogic:

    @staticmethod
    def build_state_vector(state: AgentState) -> list[float]:

        rr = state.get("reader_report", {})
        cr = state.get("comparator_report", {})
        gr = state.get("gap_report") or {}

        return [
            min(float(state.get("coverage_score", 0.0)), 1.0),
            min(float(rr.get("papers_read", 0)) / 20, 1.0),
            min(float(cr.get("n_contradictions_found", 0)) / 10, 1.0),
            min(float(cr.get("n_complements_found", 0)) / 10, 1.0),
            min(float(len(gr.get("gaps", []))) / 10, 1.0),
            min(float(state.get("step_count", 0)) / 20, 1.0),
            float(bool(state.get("reader_report"))),
            float(bool(state.get("comparator_report"))),
            float(bool(state.get("critic_report"))),
            float(bool(state.get("gap_report"))),
        ]

    @staticmethod
    def route_after_reader(state: AgentState) -> str:
        
        #2 papers available → comparator.
        #<2 papers → writer directly (nothing to compare).
    
        rr    = state.get("reader_report", {})
        total = rr.get("papers_read", 0) + rr.get("papers_already_in_db", 0)
        return "comparator" if total >= 2 else "writer"

    @staticmethod
    def route_after_comparator(state: AgentState) -> str:
        return "critic" if CRITIC_AVAILABLE else "gap_detector"

    @staticmethod
    def route_after_critic(state: AgentState) -> str:
        return "gap_detector" if GAP_AVAILABLE else "writer"

    @staticmethod
    def route_after_gap(state: AgentState) -> str:
        return "writer"

    @staticmethod
    def route_after_writer(state: AgentState) -> str:
        return "end"


#  LANGGRAPH NODE FUNCTIONS
#  Each receives the full state dict, returns a PARTIAL update dict.
#  LangGraph merges the partial update into the running state.
def _vlog(state: AgentState, agent: str, msg: str) -> None:
    """Print a timestamped log line if verbose mode is on."""
    if state.get("verbose", False):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}][planner→{agent}] {msg}")

#  LLM BACKEND DETECTION 
def _detect_gemini_backend(use_llm: bool) -> str:
    return "gemini" if (use_llm and os.environ.get("GEMINI_API_KEY")) else "none"


def _detect_ollama_backend(use_llm: bool) -> str:
    if not use_llm:
        return "none"
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        import requests
        r = requests.get(f"{host}/api/tags", timeout=3)
        if r.status_code == 200:
            return "ollama"
    except Exception:
        pass
    return "none"

def node_reader(state: AgentState) -> dict:
    
    #Node 1 — Reader.
    #Scans memory/ for claims_output.json files, computes coverage gain
    #per paper, writes new papers to SQLite, returns coverage report.
    
    _vlog(state, "reader", "Starting")

    agent = ReaderAgent(
        memory_dir = pathlib.Path(state.get("memory_dir", "memory")),
        db_path    = DB_PATH,
        verbose    = state.get("verbose", False),
    )
    # full state passed
    updated = agent.run(dict(state))

    rr       = updated.get("reader_report", {})
    coverage = rr.get("coverage_score", 0.0)

    _vlog(state, "reader",
          f"Done — read={rr.get('papers_read',0)}, "
          f"skipped={rr.get('papers_skipped',0)}, "
          f"already_in_db={rr.get('papers_already_in_db',0)}, "
          f"coverage={coverage:.2f}")
    
    print("DB_PATH =", DB_PATH)

    return {
        "reader_report":  rr,
        "coverage_score": coverage,
        "step_count":     state.get("step_count", 0) + 1,
        "react_trace":    [
            f"[reader] papers_read={rr.get('papers_read',0)}, "
            f"coverage={coverage:.2f}"
        ],
    }


def node_comparator(state: AgentState) -> dict:
    
    _vlog(state, "comparator", "Starting")

    mem_path = pathlib.Path(state.get("memory_dir", "memory"))
    _comp_module.MEMORY_DIR = mem_path

    backend = "gemini" if (
        state.get("use_llm", True) and os.environ.get("GEMINI_API_KEY")
    ) else "none"

    agent  = ComparatorAgent(
        llm_backend = backend,
        verbose     = state.get("verbose", False),
        session_id  = state.get("session_id", ""),
    )

    # run_session() reads papers from memory/ and writes to SQLite + JSON
    report = agent.run_session({
        "session_id": state.get("session_id", ""),
        "memory_dir": str(mem_path),
    })

    _vlog(state, "comparator",
          f"Done — contradictions={report.get('n_contradictions_found',0)}, "
          f"complements={report.get('n_complements_found',0)}, "
          f"pairs={report.get('n_pairs_compared',0)}")

    return {
        "comparator_report": report,
        "step_count":        state.get("step_count", 0) + 1,
        "react_trace":       [
            f"[comparator] contradictions={report.get('n_contradictions_found',0)}, "
            f"complements={report.get('n_complements_found',0)}"
        ],
    }


def node_critic(state: AgentState) -> dict:

    _vlog(state, "critic", "Starting")

    if not CRITIC_AVAILABLE:
        return {"critic_report": {"skipped": True}, "react_trace": ["[critic] skipped"]}

    backend = "ollama" if (state.get("use_llm") and _detect_ollama_backend(True) == "ollama") else "none"
    
    agent = CriticAgent(
        llm_backend=backend,
        verbose=state.get("verbose", False)
    )
    
    report = agent.run_session(dict(state))

    return {
        "critic_report": report,
        "step_count":    state.get("step_count", 0) + 1,
        "react_trace":   report.get("react_trace", [f"[critic] done: {report['total_weaknesses']} weaknesses"])
    }


def node_gap_detector(state: AgentState) -> dict:

    _vlog(state, "gap_detector", "Starting")

    if not GAP_AVAILABLE:
        return {"gap_report": {"gaps": [], "skipped": True}, "react_trace": ["[gap] skipped"]}

    backend = "ollama" if (state.get("use_llm") and _detect_ollama_backend(True) == "ollama") else "none"
    
    agent = GapDetectorAgent(
        llm_backend=backend,
        verbose=state.get("verbose", False)
    )
    
    report = agent.run_session(dict(state))

    return {
        "gap_report":  report,
        "step_count":  state.get("step_count", 0) + 1,
        "react_trace": report.get("react_trace", [f"[gap] {report.get('n_gaps', 0)} gaps detected"])
    }


def node_writer(state: AgentState) -> dict:
    
    #Node 5 — Writer.
    #Reads all structured outputs from SQLite + agent JSON files.
    #Synthesises a Markdown literature review section by section.
    _vlog(state, "writer", "Starting")

    agent   = WriterAgent(verbose=state.get("verbose", False))
    updated = agent.run(dict(state))

    wr       = updated.get("writer_report", {})
    out_path = wr.get("output_path", "")

    _vlog(state, "writer", f"Done — review saved: {out_path}")

    return {
        "writer_report":    wr,
        "session_complete": True,
        "step_count":       state.get("step_count", 0) + 1,
        "react_trace":      [f"[writer] review saved: {out_path}"],
    }


#  PLANNER AGENT

class PlannerAgent:
    
    #Builds the LangGraph StateGraph and runs the full pipeline.

    #Graph topology:
    #  START → reader → comparator → critic → gap_detector → writer → END

    #Each edge is conditional — the route_after_* functions decide whether to skip an agent .

    VERSION = "1.1.0"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def _print(self, msg: str):
        if self.verbose:
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}][planner] {msg}")

    # Graph construction

    def build_graph(self):
        if not LANGGRAPH_AVAILABLE:
            return None

        g = StateGraph(AgentState)

        g.add_node("reader",       node_reader)
        g.add_node("comparator",   node_comparator)
        g.add_node("critic",       node_critic)
        g.add_node("gap_detector", node_gap_detector)
        g.add_node("writer",       node_writer)

        g.set_entry_point("reader")

        g.add_conditional_edges("reader", PlannerLogic.route_after_reader, {
            "comparator": "comparator",
            "writer":     "writer",
        })
        g.add_conditional_edges("comparator", PlannerLogic.route_after_comparator, {
            "critic":       "critic",
            "gap_detector": "gap_detector",
        })
        g.add_conditional_edges("critic", PlannerLogic.route_after_critic, {
            "gap_detector": "gap_detector",
            "writer":       "writer",
        })
        g.add_conditional_edges("gap_detector", PlannerLogic.route_after_gap, {
            "writer": "writer",
        })
        g.add_conditional_edges("writer", PlannerLogic.route_after_writer, {
            "end": END,
        })

        return g.compile()


    def run(
        self,
        topic:      str  = "",
        memory_dir: str  = "memory",
        papers_dir: str  = "data_1/papers",      
        ollama_host: str = "",          
        no_extraction: bool = False,
        use_llm:    bool = True,
    ) -> dict:

        #Returns the final AgentState dict (all agent reports included).
        
        session_id = str(uuid.uuid4())[:8]
        self._print(f"Session {session_id} | topic='{topic}' | llm={use_llm}")

        resolved_ollama = (
            ollama_host 
            or os.environ.get("OLLAMA_HOST", "") 
            or "http://localhost:11434"
        )

        initial: AgentState = {
            "session_id":              session_id,
            "topic":                   topic,
            "papers_dir":              papers_dir,      
            "memory_dir":              memory_dir,
            "ollama_host":             resolved_ollama, 
            "no_extraction":           no_extraction or (not use_llm), 
            "use_llm":                 use_llm,
            "verbose":                 self.verbose,
            "step_count":              0,
            "coverage_score":          0.0,
            "coverage_gain_threshold": 0.10,
            "session_complete":        False,
            "react_trace":             [],
        }

        if LANGGRAPH_AVAILABLE:
            self._print("LangGraph active — running graph...")
            compiled    = self.build_graph()
            final_state = compiled.invoke(initial)
        else:
            self._print("LangGraph not installed — running sequential fallback...")
            final_state = self._sequential_fallback(initial)

        self._write_session_log(final_state)
        self._print_summary(final_state)
        print("LOG DB =", DB_PATH)
        return final_state

    # Sequential fallback (no LangGraph) 

    def _sequential_fallback(self, state: AgentState) -> AgentState:
        
        #Same logic as the graph but without LangGraph overhead.
       
        state = {**state, **node_reader(state)}

        route = PlannerLogic.route_after_reader(state)

        if route == "comparator":
            state = {**state, **node_comparator(state)}
            route = PlannerLogic.route_after_comparator(state)

            if route == "critic":
                state = {**state, **node_critic(state)}
                route = PlannerLogic.route_after_critic(state)

                if route == "gap_detector":
                    state = {**state, **node_gap_detector(state)}
                # else → writer (gap not available)

            elif route == "gap_detector":
                # critic not available, jumped straight to gap
                state = {**state, **node_gap_detector(state)}

        state = {**state, **node_writer(state)}
        return state

    # Session log written to SQLite 

    def _write_session_log(self, state: AgentState) -> None:
        if not DB_PATH.exists():
            return
        try:
            conn   = sqlite3.connect(DB_PATH)
            now    = datetime.datetime.now(datetime.timezone.utc).isoformat()
            rr     = state.get("reader_report", {})
            cr     = state.get("comparator_report", {})
            wr     = state.get("writer_report", {})
            conn.execute(
                "INSERT INTO session_log"
                "(session_id, timestamp, agent, action,"
                " input_summary, output_summary, confidence, duration_ms)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    state.get("session_id", ""),
                    now,
                    "planner_agent",
                    "session_complete",
                    f"papers={rr.get('papers_read',0)}, topic={state.get('topic','')}",
                    f"contradictions={cr.get('n_contradictions_found',0)}, "
                    f"review={wr.get('output_path','')}",
                    float(state.get("coverage_score", 0.0)),
                    0,
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  [planner] session_log write failed: {e}", file=sys.stderr)


    def _print_summary(self, state: AgentState) -> None:
        rr = state.get("reader_report",     {})
        cr = state.get("comparator_report", {})
        gr = state.get("gap_report")  or    {}
        wr = state.get("writer_report",     {})

        n_papers  = rr.get("papers_read", 0)
        n_contras = cr.get("n_contradictions_found", 0)
        n_comps   = cr.get("n_complements_found", 0)
        n_gaps    = len(gr.get("gaps", []))
        out_path  = wr.get("output_path", "")

        print("\n" + "═" * 64)
        print(f"  PLANNER SESSION SUMMARY  (v{self.VERSION})")
        print("═" * 64)
        print(f"  Session ID          : {state.get('session_id','')}")
        print(f"  Topic               : {state.get('topic','(none)')}")
        print(f"  Papers processed    : {n_papers}")
        print(f"  Coverage score      : {state.get('coverage_score', 0.0):.2f}")
        print(f"  Contradictions      : {n_contras}")
        print(f"  Complements         : {n_comps}")
        print(f"  Research gaps       : {n_gaps}")
        print(f"  Steps taken         : {state.get('step_count', 0)}")
        print(f"  Session complete    : {state.get('session_complete', False)}")
        if out_path:
            print(f"  Review saved at     : {out_path}")
        print("═" * 64)
        if state.get("react_trace"):
            print("\n  Execution trace:")
            for entry in state["react_trace"]:
                print(f"    {entry}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description=f"Planner Agent v1.1.0 — full research pipeline"
    )
    parser.add_argument("--topic",      default="")
    parser.add_argument("--memory-dir", default="memory")
    parser.add_argument("--papers-dir", default="data_1/papers", help="Where raw PDFs live") 
    parser.add_argument("--ollama-host", default="", help="Ollama URL")
    parser.add_argument("--no-extraction", action="store_true", help="Skip PDF processing") 
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    planner = PlannerAgent(verbose=args.verbose)
    planner.run(
        topic         = args.topic,
        memory_dir    = args.memory_dir,
        papers_dir    = args.papers_dir,      
        ollama_host   = args.ollama_host,     
        no_extraction = args.no_extraction,   
        use_llm       = not args.no_llm,
    )


if __name__ == "__main__":
    main()