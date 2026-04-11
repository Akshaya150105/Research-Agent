from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import re
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_AGENTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from entity_resolver import EmbeddingEntityResolver, get_typed_entities


MEMORY_DIR    = pathlib.Path("memory")
SHARED_MEMORY = pathlib.Path("shared_memory")
DB_PATH       = SHARED_MEMORY / "research.db"
GEXF_PATH     = SHARED_MEMORY / "knowledge_graph.gexf"
OUTPUT_DIR    = pathlib.Path("data_1/agent_outputs/comparisons")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "qwen2.5" 

# GEMINI_URL = (
#     "https://generativelanguage.googleapis.com/v1beta"
#     "/models/gemini-2.5-flash:generateContent"
# )

MIN_OVERLAP_THRESHOLD    = 0.01
METRIC_DIFF_THRESHOLD    = 0.02
MAX_PAIRS_PER_SESSION    = 20
COMPLEMENT_TOKEN_OVERLAP = 3

LLM_TIMEOUT_SECONDS  = 60
LLM_MAX_RETRIES      = 3
LLM_RETRY_BASE_DELAY = 2

STOPWORDS = {
    "the", "a", "an", "of", "is", "are", "in", "to", "and",
    "that", "this", "we", "our", "with", "for", "on", "by",
    "from", "as", "at", "be", "was", "it", "its", "not", "but",
    "have", "has", "their", "which", "can", "also", "will", "may",
}

HYPERPARAM_PATTERNS: dict[str, list[str]] = {
    "learning_rate": [
        r"learning[\s_-]?rate[s]?\s*[=:of]*\s*([\d.e+\-]+)",
        r"\blr\s*[=:]\s*([\d.e+\-]+)",
        r"(?:set|use[sd]?)\s+(?:a\s+)?learning[\s_-]?rate\s+of\s+([\d.e+\-]+)",
    ],
    "batch_size": [
        r"batch[\s_-]?size[s]?\s*(?:of\s+)?[=:of]*\s*(\d+)",
        r"\bbs\s*[=:]\s*(\d+)",
        r"mini[\s_-]?batch(?:es)?\s+of\s+(\d+)",
    ],
    "epochs": [
        r"(\d+)\s+epoch[s]?",
        r"epoch[s]?\s*[=:]\s*(\d+)",
        r"train(?:ed|ing)?\s+for\s+(\d+)\s+epoch[s]?",
    ],
    "optimizer": [r"\b(adam(?:w)?|sgd|adagrad|rmsprop|adamax|nadam|lion)\b"],
    "weight_decay": [r"weight[\s_-]?decay\s*[=:of]*\s*([\d.e+\-]+)"],
    "dropout":      [r"dropout\s*(?:rate\s*)?[=:of]*\s*([\d.]+)"],
    "warmup_steps": [
        r"warmup[\s_-]?steps?\s*[=:of]*\s*(\d+)",
        r"linear\s+warmup\s+(?:over\s+)?(\d+)\s+step[s]?",
    ],
    "max_seq_length": [
        r"max(?:imum)?\s+seq(?:uence)?\s+length\s*[=:of]*\s*(\d+)",
        r"(?:input\s+)?(?:context\s+)?length\s+of\s+(\d+)\s+token[s]?",
    ],
}

GPU_PATTERNS: list[tuple[str, str]] = [
    (r"\bh100\b", "H100"), (r"\ba100\b", "A100"), (r"\bv100\b", "V100"),
    (r"\bt4\b",   "T4"),   (r"\bp100\b", "P100"), (r"\btpu\b",  "TPU"),
    (r"\brtx\s*[34]\d{3}", "RTX 3/4000"), (r"\bgtx\s*\d{4}", "GTX"),
    (r"\bxeon\b", "CPU-Xeon"),
]

GPU_TIER: dict[str, int] = {
    "H100": 6, "A100": 5, "V100": 4, "TPU": 4,
    "T4": 3, "P100": 3, "RTX 3/4000": 2, "GTX": 1, "CPU-Xeon": 0,
}

SETUP_SECTIONS = {
    "experiments", "experimental", "experimental setup",
    "implementation", "implementation details", "training details",
    "training setup", "setup", "methodology", "methods",
    "training", "training procedure", "hyperparameters",
}

TIME_METRICS = {
    "time", "epoch", "throughput", "latency", "speed",
    "words/sec", "tokens/sec", "training time", "inference time",
}


#  DATA MODELS
@dataclass
class ExperimentalSetup:
    paper_id:          str
    learning_rate:     Optional[float] = None
    batch_size:        Optional[int]   = None
    epochs:            Optional[int]   = None
    optimizer:         Optional[str]   = None
    weight_decay:      Optional[float] = None
    dropout:           Optional[float] = None
    warmup_steps:      Optional[int]   = None
    max_seq_length:    Optional[int]   = None
    gpu_types:         list[str]       = field(default_factory=list)
    gpu_count:         Optional[int]   = None
    is_cluster:        bool            = False
    hardware_tier:     int             = -1
    uses_quantization: bool            = False
    uses_pruning:      bool            = False
    uses_distillation: bool            = False
    precision:         Optional[str]   = None
    train_size:        Optional[int]   = None
    test_size:         Optional[int]   = None
    setup_snippets:    list[str]       = field(default_factory=list)
    extraction_confidence: float       = 0.0


@dataclass
class SetupDivergence:
    paper_a_id:            str
    paper_b_id:            str
    lr_ratio:              Optional[float] = None
    lr_divergent:          bool = False
    batch_size_ratio:      Optional[float] = None
    batch_divergent:       bool = False
    epoch_ratio:           Optional[float] = None
    epoch_divergent:       bool = False
    optimizer_mismatch:    bool = False
    different_optimizers:  tuple = ()
    hardware_divergent:    bool = False
    hardware_a:            str  = ""
    hardware_b:            str  = ""
    hardware_tier_diff:    int  = 0
    efficiency_complement: bool = False
    efficiency_details:    str  = ""
    setup_comparable:      bool = True
    incomparability_reason: str = ""
    divergence_summary:    str  = ""


@dataclass
class PairFeatures:
    """
    # TODO (Phase 8 / Person 4):
    # Replace heuristic_score with comparator_policy.predict(as_vector())
    """
    paper_a_id:           str
    paper_b_id:           str
    entity_overlap:       float
    dataset_overlap:      float
    metric_overlap:       float
    task_overlap:         float
    result_divergence:    float
    shared_dataset_count: int   = 0
    shared_method_count:  int   = 0
    citation_link:        bool  = False
    year_diff:            int   = 0
    setup_divergence:     float = 0.0
    heuristic_score:      float = 0.0

    def score(self) -> float:
        s = (
            0.30 * self.dataset_overlap +
            0.25 * self.result_divergence +
            0.15 * self.entity_overlap +
            0.10 * self.metric_overlap +
            0.05 * self.task_overlap +
            0.05 * float(self.citation_link) +
            0.10 * (1.0 - min(self.setup_divergence, 1.0))
        )
        self.heuristic_score = round(s, 4)
        return self.heuristic_score

    def as_vector(self) -> list[float]:
        return [
            self.entity_overlap, self.dataset_overlap, self.metric_overlap,
            self.task_overlap, self.result_divergence,
            float(self.shared_dataset_count), float(self.shared_method_count),
            float(self.citation_link), float(self.year_diff), self.setup_divergence,
        ]


@dataclass
class ComparisonResult:
    comparison_id:                str
    paper_a:                      str
    paper_b:                      str
    paper_a_title:                str
    paper_b_title:                str
    generated_at:                 str
    agent_version:                str   = "3.1.0"
    pair_score:                   float = 0.0
    pair_features:                dict  = field(default_factory=dict)
    setup_a:                      dict  = field(default_factory=dict)
    setup_b:                      dict  = field(default_factory=dict)
    setup_divergence:             dict  = field(default_factory=dict)
    shared_methods:               list  = field(default_factory=list)
    shared_datasets:              list  = field(default_factory=list)
    shared_tasks:                 list  = field(default_factory=list)
    contradictions:               list  = field(default_factory=list)
    complementary_findings:       list  = field(default_factory=list)
    agreements:                   list  = field(default_factory=list)
    false_contradictions_filtered: list = field(default_factory=list)
    overall_relationship:         str   = "neutral"
    overall_rationale:            str   = ""
    llm_used:                     bool  = False
    n_contradictions:             int   = 0
    n_complements:                int   = 0
    react_trace:                  list  = field(default_factory=list)

    def finalise(self):
        self.n_contradictions = len(self.contradictions)
        self.n_complements    = len(self.complementary_findings)



class ExperimentalSetupExtractor:

    def __init__(self, paper_folder: pathlib.Path, paper_id: str):
        self.paper_folder = paper_folder
        self.paper_id     = paper_id

    def extract(self) -> ExperimentalSetup:
        setup = ExperimentalSetup(paper_id=self.paper_id)
        sections_path = self.paper_folder / "sections.json"
        if not sections_path.exists():
            return setup
        try:
            with open(sections_path, encoding="utf-8") as f:
                sections = json.load(f)
        except Exception:
            return setup

        setup_text_parts: list[str] = []
        for sec in sections:
            heading  = str(sec.get("heading", "")).lower()
            sec_type = str(sec.get("section_type", "")).lower()
            text     = sec.get("text", "")
            if not text:
                continue
            if any(kw in heading or kw in sec_type for kw in SETUP_SECTIONS):
                setup_text_parts.append(text)
                sentences = re.split(r'(?<=[.!?])\s+', text)
                snippet   = " ".join(sentences[:3]).strip()
                if snippet and len(snippet) > 30:
                    setup.setup_snippets.append(snippet[:400])

        if not setup_text_parts:
            for sec in sections:
                if sec.get("text"):
                    setup_text_parts.append(sec["text"])

        full = " ".join(setup_text_parts).lower()
        if not full.strip():
            return setup

        fields_found = 0
        for param, patterns in HYPERPARAM_PATTERNS.items():
            for pattern in patterns:
                m = re.search(pattern, full, re.IGNORECASE)
                if m:
                    raw_val = m.group(1).strip()
                    try:
                        if param == "optimizer":
                            setattr(setup, param, raw_val.lower())
                        elif param in ("batch_size", "epochs", "warmup_steps", "max_seq_length"):
                            setattr(setup, param, int(float(raw_val)))
                        else:
                            setattr(setup, param, float(raw_val))
                        fields_found += 1
                    except (ValueError, TypeError):
                        pass
                    break

        gpu_found = []
        for pattern, canonical in GPU_PATTERNS:
            if re.search(pattern, full, re.IGNORECASE):
                gpu_found.append(canonical)
        if gpu_found:
            setup.gpu_types    = gpu_found
            setup.hardware_tier = max((GPU_TIER.get(g, 0) for g in gpu_found), default=-1)
            fields_found      += 1

        m = re.search(r"(\d+)\s*[×x]\s*(?:nvidia\s+)?(?:gpu[s]?|h100|a100|v100|t4|p100)", full, re.IGNORECASE)
        if m:
            setup.gpu_count = int(m.group(1))
            fields_found   += 1

        if re.search(r"\b(?:distributed|multi[\s-]?(?:gpu|node)|cluster|data[\s-]?parallel|\d+\s+node[s]?)\b", full, re.IGNORECASE):
            setup.is_cluster = True; fields_found += 1

        for prec in ("bf16", "fp16", "fp32", r"mixed[\s-]?precision"):
            if re.search(prec, full, re.IGNORECASE):
                setup.precision = prec.replace(r"[\s-]?", "-"); fields_found += 1; break

        if any(re.search(kw, full, re.IGNORECASE) for kw in ("quantiz", "4-bit", "8-bit", "int8")):
            setup.uses_quantization = True; fields_found += 1
        if re.search(r"\bprun(?:ing|ed)\b", full, re.IGNORECASE):
            setup.uses_pruning = True; fields_found += 1
        if re.search(r"\b(?:distillation|distilled|knowledge[\s-]?distill)\b", full, re.IGNORECASE):
            setup.uses_distillation = True; fields_found += 1

        setup.extraction_confidence = min(1.0, fields_found / 10)
        return setup

    @staticmethod
    def format_for_prompt(setup: ExperimentalSetup) -> str:
        if setup.extraction_confidence == 0.0:
            return "  (no experimental setup data found)"
        lines = []
        if setup.learning_rate  is not None: lines.append(f"  learning_rate  : {setup.learning_rate}")
        if setup.batch_size     is not None: lines.append(f"  batch_size     : {setup.batch_size}")
        if setup.epochs         is not None: lines.append(f"  epochs         : {setup.epochs}")
        if setup.optimizer:                  lines.append(f"  optimizer      : {setup.optimizer}")
        if setup.gpu_types:
            lines.append(f"  hardware       : {', '.join(setup.gpu_types)}"
                         f"{f' ×{setup.gpu_count}' if setup.gpu_count else ''}"
                         f"{' (distributed)' if setup.is_cluster else ''}")
        if setup.precision:         lines.append(f"  precision      : {setup.precision}")
        if setup.uses_quantization: lines.append(f"  quantization   : yes")
        if setup.uses_pruning:      lines.append(f"  pruning        : yes")
        if setup.uses_distillation: lines.append(f"  distillation   : yes")
        if setup.setup_snippets:    lines.append(f"  snippet        : \"{setup.setup_snippets[0][:200]}\"")
        return "\n".join(lines) if lines else "  (setup found but no values extracted)"


class SetupComparator:
    LR_RATIO_THRESHOLD      = 10.0
    BATCH_RATIO_THRESHOLD   = 8.0
    EPOCH_RATIO_THRESHOLD   = 5.0
    HARDWARE_TIER_THRESHOLD = 2

    def compare(self, sa: ExperimentalSetup, sb: ExperimentalSetup) -> SetupDivergence:
        div     = SetupDivergence(paper_a_id=sa.paper_id, paper_b_id=sb.paper_id)
        reasons: list[str] = []

        if sa.learning_rate and sb.learning_rate:
            ratio = max(sa.learning_rate, sb.learning_rate) / min(sa.learning_rate, sb.learning_rate)
            div.lr_ratio     = round(ratio, 2)
            div.lr_divergent = ratio >= self.LR_RATIO_THRESHOLD
            if div.lr_divergent:
                reasons.append(f"LR differs {ratio:.0f}x ({sa.learning_rate} vs {sb.learning_rate})")

        if sa.batch_size and sb.batch_size:
            ratio = max(sa.batch_size, sb.batch_size) / min(sa.batch_size, sb.batch_size)
            div.batch_size_ratio = round(ratio, 2)
            div.batch_divergent  = ratio >= self.BATCH_RATIO_THRESHOLD
            if div.batch_divergent:
                reasons.append(f"batch size differs {ratio:.0f}x")

        if sa.epochs and sb.epochs:
            ratio = max(sa.epochs, sb.epochs) / min(sa.epochs, sb.epochs)
            div.epoch_ratio     = round(ratio, 2)
            div.epoch_divergent = ratio >= self.EPOCH_RATIO_THRESHOLD
            if div.epoch_divergent:
                reasons.append(f"training duration differs {ratio:.0f}x")

        if sa.optimizer and sb.optimizer and sa.optimizer.lower() != sb.optimizer.lower():
            div.optimizer_mismatch   = True
            div.different_optimizers = (sa.optimizer, sb.optimizer)
            reasons.append(f"different optimizers: {sa.optimizer} vs {sb.optimizer}")

        if sa.hardware_tier >= 0 and sb.hardware_tier >= 0:
            tier_diff = abs(sa.hardware_tier - sb.hardware_tier)
            div.hardware_tier_diff = tier_diff
            div.hardware_a         = ", ".join(sa.gpu_types) or "unknown"
            div.hardware_b         = ", ".join(sb.gpu_types) or "unknown"
            if tier_diff >= self.HARDWARE_TIER_THRESHOLD:
                div.hardware_divergent = True
                reasons.append(f"hardware gap: {div.hardware_a} vs {div.hardware_b}")

        eff_a = sa.uses_quantization or sa.uses_pruning or sa.uses_distillation
        eff_b = sb.uses_quantization or sb.uses_pruning or sb.uses_distillation
        if eff_a != eff_b:
            div.efficiency_complement = True
            eff_setup = sa if eff_a else sb
            eff_pid   = sa.paper_id if eff_a else sb.paper_id
            base_pid  = sb.paper_id if eff_a else sa.paper_id
            techs = ([t for t in ["quantization", "pruning", "distillation"]
                      if getattr(eff_setup, f"uses_{t}", False)])
            div.efficiency_details = f"{eff_pid} uses {', '.join(techs)} while {base_pid} does not"

        if div.lr_divergent and div.hardware_divergent:
            div.setup_comparable       = False
            div.incomparability_reason = "Both LR and hardware differ significantly."
        elif div.lr_ratio and div.lr_ratio >= 100:
            div.setup_comparable       = False
            div.incomparability_reason = f"LR differs by {div.lr_ratio:.0f}x."

        div.divergence_summary = (
            "Setup differences: " + "; ".join(reasons) + "." if reasons
            else "No major setup differences detected."
        )
        return div

    @staticmethod
    def divergence_score(div: SetupDivergence) -> float:
        s = 0.0
        if div.lr_divergent:       s += 0.35
        if div.hardware_divergent: s += 0.25
        if div.batch_divergent:    s += 0.20
        if div.optimizer_mismatch: s += 0.15
        if div.epoch_divergent:    s += 0.15
        return min(1.0, s)


def load_all_papers(memory_dir: pathlib.Path = MEMORY_DIR) -> dict[str, dict]:
    papers: dict[str, dict] = {}
    for folder in sorted(memory_dir.iterdir()):
        if not folder.is_dir():
            continue
        for fname in ["claims_output.json"] + list(
            str(p) for p in folder.glob("*_enriched.json")
        ):
            path = folder / fname if "/" not in fname else pathlib.Path(fname)
            if not path.exists():
                continue
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                pid = data.get("paper_id") or folder.name
                if pid not in papers:
                    papers[pid] = data
                break
            except Exception as e:
                print(f"  [load] ⚠ {path}: {e}", file=sys.stderr)
    return papers


def find_paper_folder(paper_id: str, memory_dir: pathlib.Path = MEMORY_DIR) -> Optional[pathlib.Path]:
    direct = memory_dir / paper_id
    if direct.is_dir():
        return direct
    for folder in memory_dir.iterdir():
        if not folder.is_dir():
            continue
        p = folder / "claims_output.json"
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
                if d.get("paper_id") == paper_id:
                    return folder
            except Exception:
                pass
    return None


def load_setups(papers: dict[str, dict], memory_dir: pathlib.Path = MEMORY_DIR) -> dict[str, ExperimentalSetup]:
    result: dict[str, ExperimentalSetup] = {}
    for pid in papers:
        folder = find_paper_folder(pid, memory_dir)
        result[pid] = (ExperimentalSetupExtractor(folder, pid).extract()
                       if folder else ExperimentalSetup(paper_id=pid))
    return result


def load_precomputed_candidates(memory_dir: pathlib.Path = MEMORY_DIR) -> list[dict]:
    path = memory_dir / "contradiction_candidates.json"
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def query_sqlite_for_citations(db_path: pathlib.Path = DB_PATH) -> set[tuple[str, str]]:
    if not db_path.exists():
        return set()
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT paper_id, cited_paper_id FROM citations WHERE cited_paper_id IS NOT NULL")
        result = {(r[0], r[1]) for r in cursor.fetchall()}
        conn.close()
        return result
    except Exception:
        return set()


def jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def paper_year(paper: dict) -> Optional[int]:
    try:
        return int(paper.get("metadata", {}).get("year", 0)) or None
    except (TypeError, ValueError):
        return None


def paper_title(paper: dict) -> str:
    return paper.get("metadata", {}).get("title", paper.get("paper_id", "unknown"))


def is_valid_performance_claim(resolved: dict) -> bool:
    return (
        resolved.get("metric")  is not None and
        resolved.get("dataset") is not None
    )


def compute_result_divergence(
    paper_a: dict, paper_b: dict,
    typed_a: dict, typed_b: dict,
    resolver: EmbeddingEntityResolver,
) -> float:
    pid_a = paper_a.get("paper_id", "")
    pid_b = paper_b.get("paper_id", "")

    def perf_map(paper, pid, typed):
        m = {}
        for c in paper.get("claims", []):
            if c.get("value") is None: continue
            if c.get("claim_type") not in ("performance", "comparative"): continue
            res = resolver.resolve_claim(c, pid, typed)
            if not is_valid_performance_claim(res): continue
            key = (res["metric"], res["dataset"])
            m[key] = float(c["value"])
        return m

    map_a  = perf_map(paper_a, pid_a, typed_a)
    map_b  = perf_map(paper_b, pid_b, typed_b)
    shared = (set(map_a.keys()) & set(map_b.keys())) - {(None, None)}
    if not shared:
        return 0.0
    diffs = [abs(map_a[k] - map_b[k]) / max(abs(map_a[k]), abs(map_b[k]), 1.0) for k in shared]
    return min(1.0, sum(diffs) / len(diffs))


#  PAIR RANKING

def build_candidate_pairs(
    papers:    dict[str, dict],
    setups:    dict[str, ExperimentalSetup],
    citations: set[tuple[str, str]],
    resolver:  EmbeddingEntityResolver,
    max_pairs: int = MAX_PAIRS_PER_SESSION,
) -> list[tuple[str, str, PairFeatures]]:
    setup_cmp  = SetupComparator()
    ids        = list(papers.keys())
    candidates: list[tuple[str, str, PairFeatures]] = []

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pid_a, pid_b = ids[i], ids[j]
            pa, pb       = papers[pid_a], papers[pid_b]
            ta           = get_typed_entities(pa)
            tb           = get_typed_entities(pb)

            all_a     = ta["methods"] | ta["datasets"] | ta["tasks"]
            all_b     = tb["methods"] | tb["datasets"] | tb["tasks"]
            entity_ov = jaccard(all_a, all_b)
            if entity_ov < MIN_OVERLAP_THRESHOLD:
                continue

            dataset_ov = jaccard(ta["datasets"], tb["datasets"])
            metric_ov  = jaccard(ta["metrics"],  tb["metrics"])
            task_ov    = jaccard(ta["tasks"],     tb["tasks"])
            divergence = compute_result_divergence(pa, pb, ta, tb, resolver)
            shared_ds  = len(ta["datasets"] & tb["datasets"])
            shared_m   = len(ta["methods"]  & tb["methods"])
            ya, yb     = paper_year(pa), paper_year(pb)
            year_d     = abs((ya or 0) - (yb or 0))
            cites      = ((pid_a, pid_b) in citations or (pid_b, pid_a) in citations)

            sa   = setups.get(pid_a, ExperimentalSetup(paper_id=pid_a))
            sb   = setups.get(pid_b, ExperimentalSetup(paper_id=pid_b))
            div  = setup_cmp.compare(sa, sb)
            sdiv = SetupComparator.divergence_score(div)

            feats = PairFeatures(
                paper_a_id=pid_a, paper_b_id=pid_b,
                entity_overlap=entity_ov, dataset_overlap=dataset_ov,
                metric_overlap=metric_ov, task_overlap=task_ov,
                result_divergence=divergence,
                shared_dataset_count=shared_ds, shared_method_count=shared_m,
                citation_link=cites, year_diff=year_d, setup_divergence=sdiv,
            )
            feats.score()
            candidates.append((pid_a, pid_b, feats))

    candidates.sort(key=lambda x: x[2].heuristic_score, reverse=True)
    return candidates[:max_pairs]


#  LAYER 1 — PROGRAMMATIC SIGNAL EXTRACTION
class ProgrammaticExtractor:

    def __init__(
        self,
        paper_a:  dict,
        paper_b:  dict,
        setup_a:  ExperimentalSetup,
        setup_b:  ExperimentalSetup,
        div:      SetupDivergence,
        resolver: EmbeddingEntityResolver,
    ):
        self.pa       = paper_a
        self.pb       = paper_b
        self.sa       = setup_a
        self.sb       = setup_b
        self.div      = div
        self.resolver = resolver
        self.ta       = get_typed_entities(paper_a)
        self.tb       = get_typed_entities(paper_b)
        self.pid_a    = paper_a.get("paper_id", "paper_a")
        self.pid_b    = paper_b.get("paper_id", "paper_b")

    def shared_methods(self)  -> list[str]: return sorted(self.ta["methods"]  & self.tb["methods"])
    def shared_datasets(self) -> list[str]: return sorted(self.ta["datasets"] & self.tb["datasets"])
    def shared_tasks(self)    -> list[str]: return sorted(self.ta["tasks"]    & self.tb["tasks"])

    def find_metric_contradictions(self) -> tuple[list[dict], list[dict]]:
        real:     list[dict] = []
        filtered: list[dict] = []

        perf_a = [c for c in self.pa.get("claims", [])
                if c.get("value") is not None
                and c.get("claim_type") in ("performance", "comparative")]
        perf_b = [c for c in self.pb.get("claims", [])
                if c.get("value") is not None
                and c.get("claim_type") in ("performance", "comparative")]

        for ca in perf_a:
            ra = self.resolver.resolve_claim(ca, self.pid_a, self.ta)

            if not is_valid_performance_claim(ra):
                continue

            for cb in perf_b:
                rb = self.resolver.resolve_claim(cb, self.pid_b, self.tb)

                if not is_valid_performance_claim(rb):
                    continue

                # same canonical metric AND dataset
                if ra["metric"] != rb["metric"] or ra["dataset"] != rb["dataset"]:
                    continue

                # values actually differ
                diff = abs(float(ca["value"]) - float(cb["value"]))
                if diff <= METRIC_DIFF_THRESHOLD:
                    continue

                # Only flag as contradiction when subjects are identical or unknown.
                ra_method = ra["methods"][0] if ra.get("methods") else None
                rb_method = rb["methods"][0] if rb.get("methods") else None

                if ra_method and rb_method and ra_method != rb_method:
                    # Different subjects — reclassify as inter-system comparison
                    filtered.append({
                        "finding_id":         f"inter_{uuid.uuid4().hex[:8]}",
                        "type":               "inter_system_comparison",
                        "severity":           "LOW",
                        "metric":             ra["metric"],
                        "dataset":            ra["dataset"],
                        "value_a":            float(ca["value"]),
                        "value_b":            float(cb["value"]),
                        "diff":               round(diff, 4),
                        "subject_a":          ra_method,
                        "subject_b":          rb_method,
                        "claim_text_a":       ca.get("description", "")[:200],
                        "claim_text_b":       cb.get("description", "")[:200],
                        "filter_reason":      "different_subjects",
                        "setup_note":         (
                            f"FILTERED — '{ra_method}' vs '{rb_method}' are different "
                            f"systems evaluated on the same benchmark. "
                            f"This is a comparison finding, not a contradiction."
                        ),
                        "llm_classification": "inter_system",
                        "llm_rationale":      "",
                        "source":             "heuristic",
                        "confidence":         1.0,
                    })
                    continue

                metric    = ra["metric"]
                ratio_kws = ("f1", "accuracy", "bleu", "rouge", "precision",
                            "recall", "auc", "map", "meteor")
                severity  = ("HIGH" if diff > 0.05 else "MEDIUM") \
                            if any(kw in metric for kw in ratio_kws) \
                            else ("HIGH" if diff > 5.0 else "MEDIUM")
                is_time   = any(kw in metric.lower() for kw in TIME_METRICS)

                finding = {
                    "finding_id":          f"contra_{uuid.uuid4().hex[:8]}",
                    "type":                "metric_contradiction",
                    "severity":            severity,
                    "metric":              ra["metric"],
                    "dataset":             ra["dataset"],
                    "subject_method":      ra_method,   
                    "value_a":             float(ca["value"]),
                    "value_b":             float(cb["value"]),
                    "diff":                round(diff, 4),
                    "claim_text_a":        ca.get("description", "")[:200],
                    "claim_text_b":        cb.get("description", "")[:200],
                    "source":              "heuristic",
                    "confidence":          1.0,
                    "llm_classification":  None,
                    "llm_rationale":       "",
                    "setup_note":          "",
                }

                if not self.div.setup_comparable:
                    finding["setup_note"]         = f"FILTERED — {self.div.incomparability_reason}"
                    finding["filter_reason"]      = "incomparable_setup"
                    finding["llm_classification"] = "explains"
                    filtered.append(finding)
                    continue

                if is_time and self.div.hardware_divergent:
                    finding["setup_note"]         = (
                        f"FILTERED — hardware variance. "
                        f"{self.div.hardware_a} vs {self.div.hardware_b}"
                    )
                    finding["filter_reason"]      = "hardware_variance"
                    finding["llm_classification"] = "neutral"
                    filtered.append(finding)
                    continue

                if self.div.lr_divergent:
                    finding["setup_note"] = (
                        f"LR diverges {self.div.lr_ratio:.0f}x — "
                        f"performance gap may reflect optimization, not methodology."
                    )
                    finding["severity"] = "MEDIUM"

                if self.div.optimizer_mismatch:
                    existing = finding.get("setup_note", "")
                    finding["setup_note"] = (
                        existing + f" Optimizers differ: "
                        f"{self.div.different_optimizers[0]} vs "
                        f"{self.div.different_optimizers[1]}."
                    ).strip()

                real.append(finding)

        return real, filtered

    def find_complement_signals(self) -> list[dict]:
        # Keyword overlap between limitations and claims
        complements: list[dict] = []

        def tokens(text: str) -> set:
            return {w for w in re.findall(r"\b[a-z][a-z0-9\-]{2,}\b", text.lower())
                    if w not in STOPWORDS}

        def check(lim_text, claims, lim_pid, claim_pid, lim_setup, claim_setup):
            lim_toks  = tokens(lim_text)
            lim_lower = lim_text.lower()
            for claim in claims:
                if claim.get("claim_type") not in ("methodological", "performance"):
                    continue
                ct      = claim.get("description", "")
                overlap = lim_toks & tokens(ct)
                if len(overlap) < COMPLEMENT_TOKEN_OVERLAP:
                    continue
                base_conf = 0.65
                memory_lim = any(kw in lim_lower for kw in (
                    "memory", "edge", "mobile", "resource",
                    "parameter", "compute", "lightweight"
                ))
                if memory_lim and (claim_setup.uses_quantization or
                                   claim_setup.uses_pruning or
                                   claim_setup.uses_distillation):
                    base_conf = 0.90
                if self.div.efficiency_complement:
                    base_conf = max(base_conf, 0.85)
                comp = {
                    "finding_id":            f"comp_{uuid.uuid4().hex[:8]}",
                    "type":                  "limitation_addressed",
                    "severity":              "LOW",
                    "paper_with_limitation": lim_pid,
                    "paper_addressing":      claim_pid,
                    "limitation_text":       lim_text[:200],
                    "addressing_claim":      ct[:200],
                    "overlap_tokens":        sorted(overlap)[:6],
                    "source":                "heuristic",
                    "confidence":            base_conf,
                    "llm_classification":    None,
                    "llm_rationale":         "",
                }
                if base_conf >= 0.85:
                    comp["setup_boost"] = (
                        f"Confidence boosted — {claim_pid} uses efficiency techniques."
                    )
                complements.append(comp)

        for lim in self.pa.get("limitations", []):
            check(lim.get("text", ""), self.pb.get("claims", []),
                  self.pid_a, self.pid_b, self.sa, self.sb)
        for lim in self.pb.get("limitations", []):
            check(lim.get("text", ""), self.pa.get("claims", []),
                  self.pid_b, self.pid_a, self.sb, self.sa)

        if self.div.efficiency_complement and self.div.efficiency_details:
            complements.append({
                "finding_id":    f"comp_eff_{uuid.uuid4().hex[:6]}",
                "type":          "efficiency_complement",
                "severity":      "LOW",
                "description":   self.div.efficiency_details,
                "source":        "setup_extractor",
                "confidence":    0.85,
                "llm_classification": None,
                "llm_rationale":      "",
            })

        return complements

    def find_agreements(self) -> list[dict]:
        # Numeric agreement on same validated (metric, dataset) pairs.
        agreements: list[dict] = []

        def perf_map(paper, pid, typed):
            m = {}
            for c in paper.get("claims", []):
                if c.get("value") is None: continue
                if c.get("claim_type") not in ("performance", "comparative"): continue
                res = self.resolver.resolve_claim(c, pid, typed)
                if not is_valid_performance_claim(res): continue
                key = (res["metric"], res["dataset"])
                m[key] = c
            return m

        map_a  = perf_map(self.pa, self.pid_a, self.ta)
        map_b  = perf_map(self.pb, self.pid_b, self.tb)
        shared = (set(map_a.keys()) & set(map_b.keys())) - {(None, None)}

        for key in shared:
            ca, cb = map_a[key], map_b[key]
            diff   = abs(float(ca["value"]) - float(cb["value"]))
            if diff <= METRIC_DIFF_THRESHOLD:
                agreements.append({
                    "finding_id":   f"agree_{uuid.uuid4().hex[:8]}",
                    "type":         "metric_agreement",
                    "severity":     "LOW",
                    "metric":       key[0] or "",
                    "dataset":      key[1] or "",
                    "value_a":      float(ca["value"]),
                    "value_b":      float(cb["value"]),
                    "diff":         round(diff, 4),
                    "claim_text_a": ca.get("description", "")[:200],
                    "claim_text_b": cb.get("description", "")[:200],
                    "source":       "heuristic",
                    "confidence":   1.0,
                })

        return agreements

    def run_all(self) -> dict:
        real_contras, filtered_contras = self.find_metric_contradictions()
        return {
            "shared_methods":                self.shared_methods(),
            "shared_datasets":               self.shared_datasets(),
            "shared_tasks":                  self.shared_tasks(),
            "metric_contradictions":         real_contras,
            "false_contradictions_filtered": filtered_contras,
            "complement_signals":            self.find_complement_signals(),
            "agreements":                    self.find_agreements(),
        }


#  LAYER 2 — LLM
def _call_ollama_raw(prompt: str) -> str:
    """Call Ollama /api/generate with JSON constraint."""
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",  
        "options": {
            "temperature": 0.1,
            "num_predict": 4096,
        },
    }
    headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama comparison failed: {e}")


def _parse_llm_json(text: str) -> dict:
    text  = re.sub(r"```(?:json)?\s*", "", text)
    text  = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found. Raw: {text[:200]}")
    depth = 0; end = -1; in_str = False; esc = False
    for i, ch in enumerate(text[start:], start):
        if esc: esc = False; continue
        if ch == "\\" and in_str: esc = True; continue
        if ch == '"': in_str = not in_str; continue
        if in_str: continue
        if ch == "{": depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0: end = i; break
    if end == -1:
        raise ValueError(f"Unclosed JSON. Raw: {text[:200]}")
    return json.loads(text[start:end + 1])


def build_llm_prompt(
    paper_a, paper_b, signals,
    setup_a, setup_b, div, pid_a, pid_b,
) -> str:
    ma  = paper_a.get("metadata", {})
    mb  = paper_b.get("metadata", {})
    fmt = ExperimentalSetupExtractor.format_for_prompt

    return f"""You are comparing two research papers for a systematic literature review.
Base your analysis ONLY on the data below. Do NOT invent information.

PAPER A: {pid_a} — {ma.get('title','?')} ({ma.get('year','?')})
PAPER B: {pid_b} — {mb.get('title','?')} ({mb.get('year','?')})

SHARED: methods={signals['shared_methods']}, datasets={signals['shared_datasets']}, tasks={signals['shared_tasks']}

SETUP A:
{fmt(setup_a)}

SETUP B:
{fmt(setup_b)}

SETUP NOTE: {div.divergence_summary}
{f"NOT COMPARABLE: {div.incomparability_reason}" if not div.setup_comparable else ""}

CONTRADICTION CANDIDATES (same metric, same dataset, different value):
{json.dumps(signals['metric_contradictions'][:4], indent=2) if signals['metric_contradictions'] else "none"}

COMPLEMENT SIGNALS (limitation of one paper addressed by the other):
{json.dumps(signals['complement_signals'][:5], indent=2) if signals['complement_signals'] else "none"}

CLASSIFY each contradiction:
  contradicts — same conditions, genuinely different methodology/result
  explains    — setup difference (optimizer, LR, hardware) explains the gap
  complements — different goals, both valid
  neutral     — not enough info
  false_positive — data extraction error or outlier (e.g. comparing values of different orders of magnitude)

UNIVERSAL DATA VALIDATION RULES:
1. Examine the values in "CONTRADICTION CANDIDATES". If one value is an extreme outlier (e.g., 10x smaller or larger) compared to the other for the same metric/dataset, classify it as "false_positive".
2. Check for "Units Mismatch": If the code compares a percentage-scale value (e.g., 0.95) to a point-scale value (e.g., 95.0), classify as "false_positive".
3. Check for "Context Mismatch": If a value looks like a hyperparameter (e.g., 0.001) or loss value but is being compared against a performance metric, classify as "false_positive".
4. IGNORE all "false_positive" findings when determining the "overall_relationship".

SETUP REASONING: 
- If optimizer or LR differ significantly, prefer "explains". 
- If Paper B uses the same core architecture as Paper A but achieves better results by scaling up the hardware, dataset size, or batch size, the relationship is "extends".

overall_relationship: contradicts | complements | extends | parallel | neutral

Reply ONLY valid JSON, no markdown, rationale under 15 words:
{{
  "contradiction_classifications": [
    {{"metric":"...","dataset":"...","classification":"...","severity":"...","rationale":"..."}}
  ],
  "complement_confirmations": [
    {{"finding_id":"...","confirmed":true,"summary":"..."}}
  ],
  "additional_complement_findings": [],
  "overall_relationship": "...",
  "overall_rationale": "..."
}}"""


def apply_llm_enrichment(
    result: ComparisonResult,
    signals: dict,
    llm_resp: dict,
    verbose: bool = False,
) -> None:
    SEV = ("HIGH", "MEDIUM", "LOW")
    for clf in llm_resp.get("contradiction_classifications", []):
        for c in result.contradictions:
            if c.get("metric") == clf.get("metric") and c.get("dataset") == clf.get("dataset"):
                c["llm_classification"] = clf.get("classification")
                c["llm_rationale"]      = clf.get("rationale", "")
                proposed = clf.get("severity", "").upper()
                if proposed in SEV and SEV.index(proposed) > SEV.index(c["severity"]):
                    c["severity"] = proposed

    confs = {cc["finding_id"]: cc for cc in llm_resp.get("complement_confirmations", [])
             if cc.get("confirmed")}
    for comp in result.complementary_findings:
        fid = comp.get("finding_id", "")
        if fid in confs:
            comp["confidence"]         = 0.90 if comp.get("setup_boost") else 0.85
            comp["llm_classification"] = "complements"
            comp["llm_rationale"]      = confs[fid].get("summary", "")

    for finding in llm_resp.get("additional_complement_findings", []):
        desc = finding.get("description", "").strip()
        if not desc:
            continue
        result.complementary_findings.append({
            "finding_id":    f"comp_llm_{uuid.uuid4().hex[:6]}",
            "type":          "llm_complement", "severity": "LOW",
            "description":   desc, "source": "llm",
            "confidence":    float(finding.get("confidence", 0.75)),
            "llm_classification": "complements", "llm_rationale": desc,
        })

    result.overall_relationship = llm_resp.get("overall_relationship", "neutral")
    result.overall_rationale    = llm_resp.get("overall_rationale", "")
    result.llm_used             = True


#  MEMORY WRITE-BACK

def _ensure_db_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS comparisons (
            comparison_id TEXT PRIMARY KEY,
            paper_a       TEXT NOT NULL, paper_b TEXT NOT NULL,
            finding_type  TEXT NOT NULL, severity TEXT DEFAULT 'LOW',
            metric TEXT DEFAULT '', dataset TEXT DEFAULT '',
            value_a REAL, value_b REAL, diff REAL,
            rationale TEXT DEFAULT '', confidence REAL DEFAULT 1.0,
            source TEXT DEFAULT 'heuristic',
            llm_used INTEGER DEFAULT 0, setup_used INTEGER DEFAULT 0,
            generated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    TEXT NOT NULL, timestamp TEXT NOT NULL,
            agent TEXT NOT NULL, action TEXT NOT NULL,
            input_summary TEXT DEFAULT '', output_summary TEXT DEFAULT '',
            confidence REAL DEFAULT 1.0, duration_ms INTEGER DEFAULT 0
        );
    """)
    conn.commit()


def write_to_sqlite(result: ComparisonResult, session_id: str,
                    db_path: pathlib.Path = DB_PATH, verbose: bool = False) -> None:
    if not db_path.exists():
        if verbose: print("  [sqlite] DB not found — skipping")
        return
    try:
        conn = sqlite3.connect(db_path)
        _ensure_db_schema(conn)
        try:
            conn.execute("ALTER TABLE session_log ADD COLUMN input_summary TEXT DEFAULT ''")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE session_log ADD COLUMN output_summary TEXT DEFAULT ''")
        except Exception:
            pass
        conn.commit()

        cursor = conn.cursor()
        now    = datetime.datetime.now(datetime.timezone.utc).isoformat()

        def ins(fid, pa, pb, ft, sv, mt, ds, va, vb, di, rt, cf, sr, su):
            cursor.execute(
                "INSERT OR REPLACE INTO comparisons VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (fid, pa, pb, ft, sv, mt, ds, va, vb, di, rt, cf, sr,
                 int(result.llm_used), int(su), now)
            )

        for c in result.contradictions:
            ins(c.get("finding_id", str(uuid.uuid4())),
                result.paper_a, result.paper_b,
                c.get("llm_classification") or "contradicts",
                c.get("severity", "MEDIUM"),
                c.get("metric", ""), c.get("dataset", ""),
                c.get("value_a"), c.get("value_b"), c.get("diff"),
                c.get("llm_rationale") or c.get("claim_text_a", "")[:200],
                c.get("confidence", 1.0), c.get("source", "heuristic"), False)

        for comp in result.complementary_findings:
            ins(comp.get("finding_id", str(uuid.uuid4())),
                comp.get("paper_with_limitation", result.paper_a),
                comp.get("paper_addressing",      result.paper_b),
                "complements", "LOW", "", "", None, None, None,
                comp.get("llm_rationale") or comp.get("description", "")[:200],
                comp.get("confidence", 0.65), comp.get("source", "heuristic"), False)

        for ag in result.agreements:
            ins(ag.get("finding_id", str(uuid.uuid4())),
                result.paper_a, result.paper_b,
                "agrees_with", "LOW",
                ag.get("metric", ""), ag.get("dataset", ""),
                ag.get("value_a"), ag.get("value_b"), ag.get("diff"),
                ag.get("claim_text_a", "")[:200], 1.0, "heuristic", False)

        cursor.execute(
            "INSERT INTO session_log"
            "(session_id,timestamp,agent,action,input_summary,output_summary,confidence,duration_ms)"
            " VALUES(?,?,?,?,?,?,?,?)",
            (session_id, now, "comparator_agent", "compare_pair",
             f"{result.paper_a} ↔ {result.paper_b}",
             f"relationship={result.overall_relationship}, "
             f"contradictions={result.n_contradictions}, "
             f"complements={result.n_complements}",
             float(result.pair_score), 0)
        )
        conn.commit(); conn.close()
        if verbose:
            print(f"  [sqlite] ✅ {result.n_contradictions} contradictions, "
                  f"{result.n_complements} complements")
    except Exception as e:
        print(f"  [sqlite] ⚠ Write failed: {e}", file=sys.stderr)


def write_to_gexf(result: ComparisonResult,
                  gexf_path: pathlib.Path = GEXF_PATH,
                  verbose: bool = False) -> None:
    try:
        import networkx as nx
    except ImportError:
        print("  [gexf] networkx not installed", file=sys.stderr); return

    G = nx.read_gexf(str(gexf_path)) if gexf_path.exists() else nx.DiGraph()
    if not gexf_path.exists():
        gexf_path.parent.mkdir(parents=True, exist_ok=True)

    pid_a = result.paper_a; pid_b = result.paper_b
    now   = datetime.datetime.now(datetime.timezone.utc).isoformat()

    for pid, title in [(pid_a, result.paper_a_title), (pid_b, result.paper_b_title)]:
        if pid not in G:
            G.add_node(pid, node_type="Paper", label=title or pid)

    rel = result.overall_relationship
    if rel == "contradicts":
        for c in result.contradictions:
            if c.get("llm_classification") in ("explains", "neutral"):
                continue
            G.add_edge(pid_a, pid_b, edge_type="contradicts",
                       severity=c.get("severity", "MEDIUM"),
                       metric=c.get("metric", ""), dataset=c.get("dataset", ""),
                       value_a=str(c.get("value_a", "")), value_b=str(c.get("value_b", "")),
                       rationale=c.get("llm_rationale", "")[:200],
                       confidence=str(c.get("confidence", 1.0)), generated_at=now)
    elif rel == "complements":
        G.add_edge(pid_a, pid_b, edge_type="complements",
                   rationale=result.overall_rationale[:200],
                   confidence="0.90" if result.llm_used else "0.65", generated_at=now)
    elif rel == "extends":
        G.add_edge(pid_a, pid_b, edge_type="extends",
                   rationale=result.overall_rationale[:200], generated_at=now)
    elif rel in ("parallel", "neutral"):
        shared = result.shared_methods or result.shared_datasets
        if shared:
            G.add_edge(pid_a, pid_b, edge_type="similar_to",
                       shared=",".join(shared[:5]), generated_at=now)
    for ag in result.agreements:
        G.add_edge(pid_a, pid_b, edge_type="agrees_with",
                   metric=ag.get("metric", ""), dataset=ag.get("dataset", ""),
                   value_a=str(ag.get("value_a", "")), value_b=str(ag.get("value_b", "")),
                   generated_at=now)
    try:
        nx.write_gexf(G, str(gexf_path))
        if verbose:
            print(f"  [gexf] ✅ {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"  [gexf] ⚠ Save failed: {e}", file=sys.stderr)


def save_comparison_json(result: ComparisonResult,
                         output_dir: pathlib.Path = OUTPUT_DIR) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.paper_a}__{result.paper_b}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False, default=str)
    return out_path


#  REACT AGENT
class ComparatorAgent:
    VERSION = "3.1.0"

    def __init__(self, llm_backend: str = "none",
                 verbose: bool = False, session_id: str = ""):
        self.llm_backend = llm_backend
        self.verbose     = verbose
        self.session_id  = session_id or str(uuid.uuid4())[:8]
        self.trace:  list[str] = []
        self._n_contradictions = 0
        self._n_complements    = 0
        self._n_neutral        = 0
        self._n_compared       = 0
        self._n_false_filtered = 0
        self._methods_in_findings: set[str] = set()

    def _think(self, msg: str):
        entry = f"[THINK] {msg}"; self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _act(self, msg: str):
        entry = f"[ACT]   {msg}"; self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _observe(self, msg: str):
        entry = f"[OBS]   {msg}"; self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def run_session(self, session_state: dict = None) -> dict:
        start_time = time.time()
        if session_state:
            self.session_id = session_state.get("session_id", self.session_id)
           
            mem = session_state.get("memory_dir")
            if mem:
                global MEMORY_DIR
                MEMORY_DIR = pathlib.Path(mem)

        self._think("Loading papers"); self._act("load_all_papers()")
        papers = load_all_papers()
        self._observe(f"Loaded {len(papers)} papers: {list(papers.keys())}")

        if len(papers) < 2:
            self._observe("Need ≥ 2 papers — exiting.")
            return self._build_session_report(0.0)

        self._think("Initialising EmbeddingEntityResolver")
        resolver = EmbeddingEntityResolver(verbose=self.verbose)
        rag_ok   = resolver._init_rag()
        self._observe(
            f"Resolver: RAG={'active' if rag_ok else 'unavailable'}, "
            f"{'ChromaDB semantic matching' if rag_ok else 'string fallback'}"
        )

        self._think("Extracting experimental setups")
        setups = load_setups(papers)
        for pid, s in setups.items():
            self._observe(
                f"Setup '{pid}': confidence={s.extraction_confidence:.2f}, "
                f"lr={s.learning_rate}, hw={s.gpu_types}, opt={s.optimizer}"
            )

        precomputed = load_precomputed_candidates()
        citations   = query_sqlite_for_citations()
        self._observe(f"Pre-computed: {len(precomputed)}, citations: {len(citations)}")

        self._think("Ranking pairs")
        pairs = build_candidate_pairs(papers, setups, citations, resolver)
        self._observe(f"{len(pairs)} pairs selected")

        if not pairs:
            self._observe("No pairs meet overlap threshold.")
            return self._build_session_report(time.time() - start_time)

        results: list[ComparisonResult] = []
        for n, (pid_a, pid_b, features) in enumerate(pairs, 1):
            self._think(
                f"Pair {n}/{len(pairs)}: {pid_a} ↔ {pid_b} "
                f"(score={features.heuristic_score:.3f})"
            )
            r = self._compare_one_pair(
                papers[pid_a], papers[pid_b],
                setups[pid_a], setups[pid_b], features, resolver,
            )
            results.append(r)
            self._n_compared       += 1
            self._n_contradictions += r.n_contradictions
            self._n_complements    += r.n_complements
            self._n_false_filtered += len(r.false_contradictions_filtered)
            if r.overall_relationship == "neutral":
                self._n_neutral += 1
            for m in r.shared_methods:
                self._methods_in_findings.add(m)

            stats = resolver.cache_stats()
            self._observe(
                f"Done: relationship={r.overall_relationship}, "
                f"contradictions={r.n_contradictions}, "
                f"false-filtered={len(r.false_contradictions_filtered)}, "
                f"complements={r.n_complements} | "
                f"resolver {stats['resolved']}/{stats['total_queries']} resolved"
            )

            recent = [x.overall_relationship for x in results[-3:]]
            if len(recent) == 3 and all(x == "neutral" for x in recent):
                self._observe("3 consecutive neutral — early stop.")
                break
            if self.llm_backend == "gemini" and n < len(pairs):
                time.sleep(4)

        elapsed = time.time() - start_time
        report  = self._build_session_report(elapsed)
        self._observe(
            f"Session done in {elapsed:.1f}s. "
            f"Contradictions: {self._n_contradictions} real, "
            f"{self._n_false_filtered} false-filtered, "
            f"Complements: {self._n_complements}"
        )
        return report

    def _compare_one_pair(
        self, paper_a, paper_b,
        setup_a, setup_b, features, resolver,
    ) -> ComparisonResult:
        pid_a = paper_a.get("paper_id", "unknown_a")
        pid_b = paper_b.get("paper_id", "unknown_b")
        div   = SetupComparator().compare(setup_a, setup_b)
        self._observe(f"Setup: {div.divergence_summary[:100]}")

        result = ComparisonResult(
            comparison_id=str(uuid.uuid4()), paper_a=pid_a, paper_b=pid_b,
            paper_a_title=paper_title(paper_a), paper_b_title=paper_title(paper_b),
            generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            pair_score=features.heuristic_score, pair_features=asdict(features),
            setup_a=asdict(setup_a), setup_b=asdict(setup_b),
            setup_divergence=asdict(div), react_trace=self.trace,
        )

        self._act(f"ProgrammaticExtractor.run_all(): {pid_a} ↔ {pid_b}")
        extractor = ProgrammaticExtractor(
            paper_a, paper_b, setup_a, setup_b, div, resolver
        )
        signals = extractor.run_all()

        result.shared_methods                = signals["shared_methods"]
        result.shared_datasets               = signals["shared_datasets"]
        result.shared_tasks                  = signals["shared_tasks"]
        result.contradictions                = signals["metric_contradictions"]
        result.false_contradictions_filtered = signals["false_contradictions_filtered"]
        result.complementary_findings        = signals["complement_signals"]
        result.agreements                    = signals["agreements"]

        self._observe(
            f"Programmatic: {len(signals['metric_contradictions'])} contradictions, "
            f"{len(signals['false_contradictions_filtered'])} false-filtered, "
            f"{len(signals['complement_signals'])} complements, "
            f"{len(signals['agreements'])} agreements"
        )

        has_signals = (signals["metric_contradictions"] or
                       signals["complement_signals"]    or
                       signals["shared_datasets"]       or
                       signals["agreements"])

        if self.llm_backend == "ollama":
            self._act(f"llm_enrich(): {pid_a} ↔ {pid_b}")
            limited = {
                **signals,
                "metric_contradictions": signals["metric_contradictions"][:3],
                "complement_signals":    signals["complement_signals"][:5],
                "agreements":            signals["agreements"][:2],
            }
            try:
                prompt   = build_llm_prompt(
                    paper_a, paper_b, limited,
                    setup_a, setup_b, div, pid_a, pid_b,
                )
                text     = _call_ollama_raw(prompt) 
                llm_resp = _parse_llm_json(text)
                apply_llm_enrichment(result, limited, llm_resp, self.verbose)
                self._observe(f"LLM done. overall={result.overall_relationship}")
            except Exception as e:
                self._observe(f"LLM failed ({e}) — keeping heuristic results.")

        if not result.llm_used:
            if result.contradictions:
                result.overall_relationship = "contradicts"
            elif result.complementary_findings:
                result.overall_relationship = "complements"
            elif result.agreements:
                result.overall_relationship = "agrees_with"
            elif result.shared_methods or result.shared_datasets:
                result.overall_relationship = "parallel"

        result.finalise()

        out = save_comparison_json(result)
        self._observe(f"Saved: {out}")
        write_to_sqlite(result, self.session_id, verbose=self.verbose)
        write_to_gexf(result, verbose=self.verbose)
        return result

    def _build_session_report(self, elapsed: float) -> dict:
        all_methods = set()
        for _, paper in load_all_papers().items():
            all_methods |= get_typed_entities(paper)["methods"]
        coverage = (
            len(self._methods_in_findings) / len(all_methods)
            if all_methods else 0.0
        )
        return {
            "agent":                      "comparator_agent",
            "session_id":                 self.session_id,
            "agent_version":              self.VERSION,
            "n_pairs_compared":           self._n_compared,
            "n_contradictions_found":     self._n_contradictions,
            "n_false_positives_filtered": self._n_false_filtered,
            "n_complements_found":        self._n_complements,
            "n_neutral":                  self._n_neutral,
            "method_space_coverage":      round(coverage, 3),
            "llm_used":                   self.llm_backend == "ollama",
            "elapsed_seconds":            round(elapsed, 1),
            "react_trace":                self.trace,
            "rl_note": (
                "Heuristic pair scoring. Replace with "
                "comparator_policy.predict(features.as_vector()) in Phase 8."
            ),
        }

    def can_run(self) -> tuple[bool, str]:
        papers = load_all_papers()
        if len(papers) < 2:
            return False, f"Only {len(papers)} paper(s) — need ≥ 2"
        n = len(papers); max_possible = n * (n - 1) // 2
        existing = len(list(OUTPUT_DIR.glob("*.json")))
        if existing >= max_possible:
            return False, f"All {max_possible} pairs already compared"
        return True, f"{len(papers)} papers, {max_possible - existing} pairs remaining"


def print_session_summary(report: dict) -> None:
    ICONS = {
        "contradicts": "🔴", "complements": "🟢", "extends":    "🔵",
        "parallel":    "⚪", "neutral":     "⚫", "agrees_with": "🟡",
    }
    print("\n" + "═" * 64)
    print("  COMPARATOR SESSION REPORT  (v3.1.0)")
    print("═" * 64)
    for k, label in [
        ("session_id",                 "Session ID"),
        ("n_pairs_compared",           "Pairs compared"),
        ("n_contradictions_found",     "Real contradictions"),
        ("n_false_positives_filtered", "False positives caught"),
        ("n_complements_found",        "Complements"),
        ("method_space_coverage",      "Method coverage"),
        ("llm_used",                   "LLM used"),
        ("elapsed_seconds",            "Elapsed (s)"),
    ]:
        val = report.get(k, "")
        if k == "method_space_coverage":
            val = f"{val:.1%}"
        print(f"  {label:<28}: {val}")
    print("═" * 64)

    if OUTPUT_DIR.exists():
        print("\n  Comparison results:")
        for path in sorted(OUTPUT_DIR.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    r = json.load(f)
                rel  = r.get("overall_relationship", "neutral")
                icon = ICONS.get(rel, "•")
                print(f"\n  {icon} {r.get('paper_a')} ↔ {r.get('paper_b')}")
                print(f"     relationship  : {rel}")
                print(f"     contradictions: {r.get('n_contradictions', 0)}")
                print(f"     false-filtered: {len(r.get('false_contradictions_filtered', []))}")
                print(f"     complements   : {r.get('n_complements', 0)}")
                if r.get("overall_rationale"):
                    print(f"     rationale     : {r['overall_rationale'][:100]}")
                for c in r.get("contradictions", [])[:3]:
                    note = f" ← {c['setup_note'][:40]}" if c.get("setup_note") else ""
                    print(f"       [REAL][{c.get('severity','?')}] "
                          f"{c.get('metric')} on {c.get('dataset')}: "
                          f"{c.get('value_a')} vs {c.get('value_b')}{note}")
                for c in r.get("false_contradictions_filtered", [])[:2]:
                    print(f"       [FILTERED] {c.get('metric')} on "
                          f"{c.get('dataset')}: "
                          f"{c.get('filter_reason', 'setup difference')}")
            except Exception:
                pass

    print("\n" + "═" * 64 + "\n")


def main() -> None:
    global MEMORY_DIR, MAX_PAIRS_PER_SESSION
    parser = argparse.ArgumentParser(description="Comparator Agent v3.1.0")
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--max-pairs",  type=int, default=MAX_PAIRS_PER_SESSION)
    parser.add_argument("--memory-dir", default=str(MEMORY_DIR))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    MEMORY_DIR            = pathlib.Path(args.memory_dir)
    MAX_PAIRS_PER_SESSION = args.max_pairs

    backend = "none" if args.no_llm else "ollama"

    if args.verbose:
        print(f"\nComparator Agent v{ComparatorAgent.VERSION}")
        print(f"  Memory dir  : {MEMORY_DIR}")
        print(f"  Output dir  : {OUTPUT_DIR}")
        print(f"  SQLite DB   : {DB_PATH}")
        print(f"  GEXF path   : {GEXF_PATH}")
        print(f"  LLM backend : {backend}")
        print(f"  Max pairs   : {MAX_PAIRS_PER_SESSION}\n")

    agent = ComparatorAgent(llm_backend=backend, verbose=args.verbose)
    can, reason = agent.can_run()
    if not can:
        print(f"  ⚠ Cannot run: {reason}"); sys.exit(0)

    report = agent.run_session()
    print_session_summary(report)
    print(f"  Comparisons : {OUTPUT_DIR}")
    print(f"  SQLite       : {DB_PATH}")
    print(f"  Graph        : {GEXF_PATH}\n")


if __name__ == "__main__":
    main()