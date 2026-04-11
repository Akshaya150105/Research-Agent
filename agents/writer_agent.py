from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from dotenv import load_dotenv
from shared_schema import ensure_schema as _ensure_db_schema

load_dotenv()

_AGENTS_DIR   = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MEMORY_DIR    = pathlib.Path("memory")
SHARED_MEMORY = pathlib.Path("shared_memory")
DB_PATH       = SHARED_MEMORY / "research.db"
OUTPUT_DIR    = pathlib.Path("data_1/agent_outputs/reviews")
COMPARISONS_DIR = pathlib.Path("data_1/agent_outputs/comparisons")
CRITIQUES_DIR   = pathlib.Path("data_1/agent_outputs/critiques")
GAPS_DIR        = pathlib.Path("data_1/agent_outputs/gaps")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.5-flash:generateContent"
)

LLM_TIMEOUT     = 90
LLM_MAX_RETRIES = 3
LLM_RETRY_DELAY = 30

#  DATA MODELS

@dataclass
class WriterReport:
    agent:         str   = "writer_agent"
    agent_version: str   = "1.0.0"
    session_id:    str   = ""
    topic:         str   = ""
    output_path:   str   = ""
    n_papers:      int   = 0
    n_sections:    int   = 7
    llm_used:      bool  = False
    elapsed_s:     float = 0.0
    react_trace:   list  = field(default_factory=list)


def _call_gemini(prompt: str, max_tokens: int = 2000) -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")
    last_err = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{GEMINI_URL}?key={key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.3,
                        "maxOutputTokens": max_tokens,
                    },
                },
                timeout=LLM_TIMEOUT,
            )
            if resp.status_code == 429:
                if attempt < LLM_MAX_RETRIES:
                    time.sleep(LLM_RETRY_DELAY * attempt)
                    continue
                raise RuntimeError("Rate limited (429)")
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (requests.RequestException, KeyError, IndexError) as e:
            last_err = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_DELAY)
    raise RuntimeError(f"Gemini failed: {last_err}")


#reads ONLY structured outputs
def _load_papers_from_db(db_path: pathlib.Path) -> list[dict]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM papers ORDER BY year"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_entities_for_paper(paper_id: str, db_path: pathlib.Path) -> dict[str, list]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # join with paper_entity_relationships to find which 
    # canonical entities belong to this specific paper.
    query = """
        SELECT e.canonical_text, e.entity_type 
        FROM entities e
        JOIN paper_entity_relationships r ON e.entity_id = r.entity_id
        WHERE r.paper_id = ?
    """
    
    rows = conn.execute(query, (paper_id,)).fetchall()
    conn.close()
    
    result: dict[str, list] = {}
    for row in rows:
        etype = row["entity_type"]
        result.setdefault(etype, [])
        # Use canonical_text instead of the old 'text' column
        if row["canonical_text"] not in result[etype]:
            result[etype].append(row["canonical_text"])
    return result


def _load_claims_from_memory(paper_id: str, memory_dir: pathlib.Path) -> list[dict]:
    for folder in memory_dir.iterdir():
        if not folder.is_dir():
            continue
        path = folder / "claims_output.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("paper_id") == paper_id:
                    return data.get("claims", [])
            except Exception:
                pass
    return []


def _load_limitations_from_memory(paper_id: str, memory_dir: pathlib.Path) -> list[dict]:
    for folder in memory_dir.iterdir():
        if not folder.is_dir():
            continue
        path = folder / "claims_output.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("paper_id") == paper_id:
                    return data.get("limitations", [])
            except Exception:
                pass
    return []


def _load_comparisons(comparisons_dir: pathlib.Path) -> list[dict]:
    results = []
    abs_path = comparisons_dir.resolve()
    if not abs_path.exists():
        return results
    
    for path in abs_path.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if data: 
                    results.append(data)
        except Exception as e:
            print(f"DEBUG: Writer failed to load comparison {path.name}: {e}")
    return results


def _load_critiques(critiques_dir: pathlib.Path) -> list[dict]:
    results = []
    abs_path = critiques_dir.resolve()
    if not abs_path.exists():
        return results
        
    for path in abs_path.glob("*_critique.json"):
        try:
            with open(path, encoding="utf-8") as f:
                results.append(json.load(f))
        except Exception:
            pass
    return results


def _load_gaps(gaps_dir: pathlib.Path) -> list[dict]:
    results = []
    if not gaps_dir.exists():
        return results
    for path in sorted(gaps_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                results.extend(data.get("gaps", []))
        except Exception:
            pass
    return results

#  SECTION GENERATORS

def _make_citation(paper: dict) -> str:
    authors = json.loads(paper.get("authors", "[]"))
    first   = authors[0].split()[-1] if authors else "Unknown"
    year    = paper.get("year") or "?"
    return f"[{first} {year}]"


def _section_introduction(papers: list[dict], topic: str) -> str:
    years = [p["year"] for p in papers if p.get("year")]
    year_range = f"{min(years)}–{max(years)}" if years else "recent years"
    lines = [
        f"# Literature Review: {topic.title() if topic else 'Research Overview'}",
        "",
        "## 1. Introduction",
        "",
        f"This review synthesises {len(papers)} research papers published between "
        f"{year_range}, covering advances in {topic or 'the field'}. "
        "The papers were analysed using an automated multi-agent pipeline that "
        "extracted structured entities, compared findings across papers, "
        "identified methodological weaknesses, and surfaced unexplored research directions.",
        "",
        "**Corpus overview:**",
        "",
    ]
    for p in papers:
        cite = _make_citation(p)
        lines.append(
            f"- {cite} — {p.get('title', 'Untitled')} "
            f"({p.get('venue', 'venue unknown')})"
        )
    lines += ["", "---", ""]
    return "\n".join(lines)


def _section_methods(
    papers: list[dict],
    entities_by_paper: dict[str, dict],
    llm_fn: Optional[callable],
    topic: str,
) -> str:
    # Collect all unique methods across papers
    method_to_papers: dict[str, list[str]] = {}
    for p in papers:
        pid  = p["paper_id"]
        cite = _make_citation(p)
        ents = entities_by_paper.get(pid, {})
        for method in ents.get("method", []):
            method_to_papers.setdefault(method, [])
            if cite not in method_to_papers[method]:
                method_to_papers[method].append(cite)

    # Sort by how many papers use them (most shared first)
    sorted_methods = sorted(
        method_to_papers.items(), key=lambda x: len(x[1]), reverse=True
    )

    if llm_fn:
        top_methods = sorted_methods[:20]
        prompt = f"""Write a 200-word paragraph for a literature review section titled
"Methods and Approaches" about {topic or 'the research topic'}.

Use ONLY the following data. Do not invent anything.
Methods and which papers use them:
{json.dumps({m: paps for m, paps in top_methods}, indent=2)}

Write in academic prose. Every claim must reference the paper citation in square brackets.
Do not use bullet points. Output only the paragraph text."""
        try:
            body = llm_fn(prompt, max_tokens=600)
        except Exception:
            body = _fallback_methods_text(sorted_methods)
    else:
        body = _fallback_methods_text(sorted_methods)

    return f"## 2. Methods & Approaches\n\n{body}\n\n---\n"


def _fallback_methods_text(sorted_methods: list) -> str:
    if not sorted_methods:
        return "No method entities were extracted from the corpus."
    lines = []
    for method, cites in sorted_methods[:10]:
        lines.append(f"- **{method}** — used in {', '.join(cites)}")
    return "\n".join(lines)


def _section_results(
    papers: list[dict],
    entities_by_paper: dict[str, dict],
    claims_by_paper: dict[str, list],
    llm_fn: Optional[callable],
    topic: str,
) -> str:
    # Collect performance claims
    perf_claims = []
    for p in papers:
        pid  = p["paper_id"]
        cite = _make_citation(p)
        for c in claims_by_paper.get(pid, []):
            if c.get("claim_type") in ("performance", "comparative") and c.get("value"):
                perf_claims.append({
                    "paper": cite,
                    "claim": c.get("description", "")[:120],
                    "value": c.get("value"),
                    "entities": c.get("entities_involved", []),
                })

    if llm_fn and perf_claims:
        prompt = f"""Write a 200-word paragraph for a literature review section titled
"Results and Performance" about {topic or 'the research topic'}.

Use ONLY the following structured performance claims. Do not invent anything.
{json.dumps(perf_claims[:15], indent=2)}

Write in academic prose. Reference paper citations in square brackets.
Highlight the strongest and most surprising results.
Do not use bullet points. Output only the paragraph text."""
        try:
            body = llm_fn(prompt, max_tokens=600)
        except Exception:
            body = _fallback_results_text(perf_claims)
    else:
        body = _fallback_results_text(perf_claims)

    return f"## 3. Results & Performance\n\n{body}\n\n---\n"


def _fallback_results_text(perf_claims: list) -> str:
    if not perf_claims:
        return "No numeric performance claims were extracted from the corpus."
    lines = []
    for c in perf_claims[:8]:
        lines.append(f"- {c['paper']}: {c['claim']} (value: {c['value']})")
    return "\n".join(lines)


def _section_contradictions(
    comparisons: list[dict],
    llm_fn: Optional[callable],
    topic: str,
) -> str:
    # Collect real contradictions across all comparison files
    all_contras = []
    for comp in comparisons:
        pa    = comp.get("paper_a", "")
        pb    = comp.get("paper_b", "")
        rel   = comp.get("overall_relationship", "")
        for c in comp.get("contradictions", []):
            if c.get("llm_classification") in ("explains", "neutral"):
                continue
            all_contras.append({
                "paper_a":  pa,
                "paper_b":  pb,
                "metric":   c.get("metric", ""),
                "dataset":  c.get("dataset", ""),
                "value_a":  c.get("value_a"),
                "value_b":  c.get("value_b"),
                "severity": c.get("severity", ""),
                "rationale": c.get("llm_rationale", "")[:100],
            })

    # Collect complementary findings
    complements = []
    for comp in comparisons:
        for cf in comp.get("complementary_findings", []):
            if cf.get("llm_classification") in ("complements",) or cf.get("confidence", 0) > 0.7:
                complements.append({
                    "paper_lim": cf.get("paper_with_limitation", ""),
                    "paper_addr": cf.get("paper_addressing", ""),
                    "description": (cf.get("limitation_text") or cf.get("description", ""))[:120],
                })

    if not all_contras and not complements:
        body = "No significant contradictions or complementary findings were detected across papers."
    elif llm_fn:
        prompt = f"""Write a 200-word paragraph for a literature review section titled
"Contradictions, Debates, and Complementary Findings" about {topic or 'the research topic'}.

Use ONLY the following structured data. Do not invent anything.

Contradictions found:
{json.dumps(all_contras[:6], indent=2)}

Complementary findings:
{json.dumps(complements[:4], indent=2)}

Write in academic prose. Note severity where relevant.
Do not use bullet points. Output only the paragraph text."""
        try:
            body = llm_fn(prompt, max_tokens=600)
        except Exception:
            body = _fallback_contradictions_text(all_contras, complements)
    else:
        body = _fallback_contradictions_text(all_contras, complements)

    return f"## 4. Contradictions, Debates & Complementary Findings\n\n{body}\n\n---\n"


def _fallback_contradictions_text(all_contras: list, complements: list) -> str:
    lines = []
    if all_contras:
        lines.append("**Contradictions:**")
        for c in all_contras[:5]:
            lines.append(
                f"- [{c['severity']}] {c['metric']} on {c['dataset']}: "
                f"{c['paper_a']} reports {c['value_a']} vs "
                f"{c['paper_b']} reports {c['value_b']}"
            )
    if complements:
        lines.append("\n**Complementary findings:**")
        for c in complements[:4]:
            lines.append(f"- {c['paper_lim']} limitation addressed by {c['paper_addr']}: {c['description']}")
    return "\n".join(lines) if lines else "No contradictions or complements found."


def _section_limitations(
    papers: list[dict],
    critiques: list[dict],
    lim_by_paper: dict[str, list],
    llm_fn: Optional[callable],
    topic: str,
) -> str:
    # Collect limitation statements
    all_lims = []
    for p in papers:
        pid  = p["paper_id"]
        cite = _make_citation(p)
        for lim in lim_by_paper.get(pid, []):
            all_lims.append({
                "paper": cite,
                "text":  lim.get("text", "")[:120],
            })

    # Collect heuristic critique weaknesses (from critic agent)
    all_weaknesses = []
    for crit in critiques:
        pid  = crit.get("paper_id", "")
        for w in crit.get("weaknesses", []):
            if w.get("severity") in ("HIGH", "MEDIUM"):
                all_weaknesses.append({
                    "paper":       pid,
                    "type":        w.get("weakness_type", ""),
                    "severity":    w.get("severity", ""),
                    "description": w.get("description", "")[:100],
                })

    if not all_lims and not all_weaknesses:
        body = "No structured limitations or methodological weaknesses were found."
    elif llm_fn:
        prompt = f"""Write a 200-word paragraph for a literature review section titled
"Common Limitations and Methodological Weaknesses" about {topic or 'the research topic'}.

Use ONLY the following structured data. Do not invent anything.

Stated limitations (from papers):
{json.dumps(all_lims[:10], indent=2)}

Detected weaknesses (from automated critique):
{json.dumps(all_weaknesses[:8], indent=2)}

Write in academic prose. Group recurring patterns.
Do not use bullet points. Output only the paragraph text."""
        try:
            body = llm_fn(prompt, max_tokens=600)
        except Exception:
            body = _fallback_lim_text(all_lims, all_weaknesses)
    else:
        body = _fallback_lim_text(all_lims, all_weaknesses)

    return f"## 5. Common Limitations & Methodological Weaknesses\n\n{body}\n\n---\n"


def _fallback_lim_text(lims: list, weaknesses: list) -> str:
    lines = []
    if lims:
        lines.append("**Stated limitations:**")
        for l in lims[:6]:
            lines.append(f"- {l['paper']}: {l['text']}")
    if weaknesses:
        lines.append("\n**Detected weaknesses:**")
        for w in weaknesses[:5]:
            lines.append(f"- [{w['severity']}] {w['paper']}: {w['type']} — {w['description']}")
    return "\n".join(lines) if lines else "No limitations found."


def _section_gaps(
    gaps: list[dict],
    llm_fn: Optional[callable],
    topic: str,
) -> str:
    if not gaps:
        body = "No research gaps were detected by the Gap Detector agent."
    elif llm_fn:
        prompt = f"""Write a 200-word paragraph for a literature review section titled
"Research Gaps and Future Directions" about {topic or 'the research topic'}.

Use ONLY the following structured gap data. Do not invent anything.
{json.dumps(gaps[:8], indent=2)}

Write in academic prose. Prioritise high-confidence, unaddressed gaps.
Be specific about what combinations or directions are missing.
Do not use bullet points. Output only the paragraph text."""
        try:
            body = llm_fn(prompt, max_tokens=600)
        except Exception:
            body = _fallback_gaps_text(gaps)
    else:
        body = _fallback_gaps_text(gaps)

    return f"## 6. Research Gaps & Future Directions\n\n{body}\n\n---\n"


def _fallback_gaps_text(gaps: list) -> str:
    if not gaps:
        return "No gaps detected."
    lines = []
    for g in gaps[:6]:
        desc = g.get("description", g.get("gap_description", ""))
        conf = g.get("confidence", "?")
        status = g.get("addressed_status", "")
        lines.append(f"- {desc} (confidence={conf}, status={status})")
    return "\n".join(lines)


def _section_conclusion(
    papers: list[dict],
    topic: str,
    n_contras: int,
    n_gaps: int,
    llm_fn: Optional[callable],
) -> str:
    if llm_fn:
        paper_summaries = [
            {"title": p.get("title", ""), "year": p.get("year")}
            for p in papers
        ]
        prompt = f"""Write a 120-word conclusion paragraph for a literature review about
{topic or 'the research topic'}.

The review covered {len(papers)} papers, found {n_contras} contradictions and {n_gaps} research gaps.

Papers reviewed: {json.dumps(paper_summaries, indent=2)}

Write in academic prose. Summarise the state of the field in 2-3 sentences,
then mention 1-2 key future directions. Output only the paragraph text."""
        try:
            body = llm_fn(prompt, max_tokens=400)
        except Exception:
            body = f"This review covered {len(papers)} papers, identifying {n_contras} cross-paper contradictions and {n_gaps} unexplored research gaps. Future work should address the detected gaps to advance the field."
    else:
        body = f"This review covered {len(papers)} papers, identifying {n_contras} cross-paper contradictions and {n_gaps} unexplored research gaps. Future work should address the detected gaps to advance the field."

    return f"## 7. Conclusion\n\n{body}\n\n---\n"


def _section_references(papers: list[dict]) -> str:
    lines = ["## References", ""]
    for i, p in enumerate(papers, 1):
        authors = json.loads(p.get("authors", "[]"))
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."
        year  = p.get("year", "n.d.")
        title = p.get("title", "Untitled")
        venue = p.get("venue", "")
        lines.append(f"[{i}] {author_str} ({year}). {title}. {venue}.")
    return "\n".join(lines)


#  WRITER AGENT

class WriterAgent:

    #Synthesises all structured agent outputs into a Markdown literature review.

    #LangGraph integration:
    #  Planner calls agent.run(state) after Comparator, Critic,
    #  and Gap Detector have all completed.

    VERSION = "1.0.0"

    def __init__(
        self,
        db_path:         pathlib.Path = DB_PATH,
        memory_dir:      pathlib.Path = MEMORY_DIR,
        output_dir:      pathlib.Path = OUTPUT_DIR,
        comparisons_dir: pathlib.Path = COMPARISONS_DIR,
        critiques_dir:   pathlib.Path = CRITIQUES_DIR,
        gaps_dir:        pathlib.Path = GAPS_DIR,
        verbose:         bool         = False,
    ):
        self.db_path         = db_path
        self.memory_dir      = memory_dir
        self.output_dir      = output_dir
        self.comparisons_dir = comparisons_dir
        self.critiques_dir   = critiques_dir
        self.gaps_dir        = gaps_dir
        self.verbose         = verbose
        self.trace:          list[str] = []

    def _think(self, msg: str):
        entry = f"[THINK] {msg}"; self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _act(self, msg: str):
        entry = f"[ACT]   {msg}"; self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _observe(self, msg: str):
        entry = f"[OBS]   {msg}"; self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def run(self, state: dict) -> dict:
        #Called by the Planner node in LangGraph.

        #Args - state: The shared AgentState. Must contain: - session_id (str), optional: - topic (str)   — research topic string , - use_llm (bool)

        #Returns: Updated state dict with "writer_report" key added.
        
        session_id = state.get("session_id", str(uuid.uuid4())[:8])
        topic      = state.get("topic", "")
        use_llm    = state.get("use_llm", True)
        backend    = "gemini" if (use_llm and os.environ.get("GEMINI_API_KEY")) else "none"

        report = self._run_internal(session_id, topic, backend)
        return {**state, "writer_report": asdict(report)}
    
    def _run_internal(
        self,
        session_id: str,
        topic:      str,
        backend:    str,
    ) -> WriterReport:

        start = __import__("time").time()
        report = WriterReport(session_id=session_id, topic=topic)

        llm_fn = None
        if backend == "gemini":
            def llm_fn(prompt: str, max_tokens: int = 600) -> str:
                return _call_gemini(prompt, max_tokens)

        # Load all structured data 
        self._think("Loading papers from SQLite")
        papers = _load_papers_from_db(self.db_path)
        if not papers:
            self._observe("No papers in DB — cannot write review.")
            report.react_trace = self.trace
            return report

        report.n_papers = len(papers)
        self._observe(f"Loaded {len(papers)} papers from DB")

        self._act("Loading entities, claims, limitations per paper")
        entities_by_paper:  dict[str, dict]  = {}
        claims_by_paper:    dict[str, list]  = {}
        lim_by_paper:       dict[str, list]  = {}

        for p in papers:
            pid = p["paper_id"]
            entities_by_paper[pid] = _load_entities_for_paper(pid, self.db_path)
            claims_by_paper[pid]   = _load_claims_from_memory(pid, self.memory_dir)
            lim_by_paper[pid]      = _load_limitations_from_memory(pid, self.memory_dir)

        self._act("Loading Comparator, Critic, Gap Detector outputs")
        comparisons = _load_comparisons(self.comparisons_dir)
        critiques   = _load_critiques(self.critiques_dir)
        gaps        = _load_gaps(self.gaps_dir)

        self._observe(
            f"Comparisons: {len(comparisons)}, "
            f"Critiques: {len(critiques)}, "
            f"Gaps: {len(gaps)}"
        )

        # Count real contradictions
        n_contras = sum(
            len([c for c in comp.get("contradictions", [])
                 if c.get("llm_classification") not in ("explains", "neutral")])
            for comp in comparisons
        )

        # Generate review section by section
        self._think("Generating review sections")
        sections = []

        self._act("Section 1: Introduction")
        sections.append(_section_introduction(papers, topic))

        self._act("Section 2: Methods")
        sections.append(_section_methods(papers, entities_by_paper, llm_fn, topic))
        time.sleep(2) if llm_fn else None

        self._act("Section 3: Results")
        sections.append(_section_results(papers, entities_by_paper, claims_by_paper, llm_fn, topic))
        time.sleep(2) if llm_fn else None

        self._act("Section 4: Contradictions")
        sections.append(_section_contradictions(comparisons, llm_fn, topic))
        time.sleep(2) if llm_fn else None

        self._act("Section 5: Limitations")
        sections.append(_section_limitations(papers, critiques, lim_by_paper, llm_fn, topic))
        time.sleep(2) if llm_fn else None

        self._act("Section 6: Gaps")
        sections.append(_section_gaps(gaps, llm_fn, topic))
        time.sleep(2) if llm_fn else None

        self._act("Section 7: Conclusion")
        sections.append(_section_conclusion(papers, topic, n_contras, len(gaps), llm_fn))

        self._act("Section 8: References")
        sections.append(_section_references(papers))

        review_text = "\n\n".join(sections)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"session_{session_id}_draft.md"
        out_path.write_text(review_text, encoding="utf-8")

        self._observe(f"Review saved: {out_path}")

        report.output_path = str(out_path)
        report.llm_used    = llm_fn is not None
        report.elapsed_s   = round(__import__("time").time() - start, 1)
        report.react_trace = self.trace
        return report

    def print_report(self, report: WriterReport) -> None:
        print("\n" + "═" * 56)
        print("  WRITER AGENT REPORT  (v1.0.0)")
        print("═" * 56)
        print(f"  Topic           : {report.topic or '(none)'}")
        print(f"  Papers included : {report.n_papers}")
        print(f"  Sections        : {report.n_sections}")
        print(f"  LLM used        : {report.llm_used}")
        print(f"  Elapsed         : {report.elapsed_s}s")
        print(f"  Output          : {report.output_path}")
        print("═" * 56 + "\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Writer Agent v1.0.0")
    parser.add_argument("--topic",      default="")
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    agent = WriterAgent(verbose=args.verbose)
    state = {
        "session_id": str(uuid.uuid4())[:8],
        "topic":      args.topic,
        "use_llm":    not args.no_llm,
    }
    updated_state = agent.run(state)
    report = WriterReport(**updated_state["writer_report"])
    agent.print_report(report)


if __name__ == "__main__":
    main()