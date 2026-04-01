from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
import datetime
import pathlib
from dataclasses import dataclass, field, asdict
from enum import Enum

import requests
from dotenv import load_dotenv

load_dotenv()

import re
from pathlib import Path

def extract_year(data: dict, file_path: str) -> int | None:
    """
    Robust year extraction that works for ANY paper format.
    """

    meta = data.get("metadata", {})
    year = meta.get("year")

    # 1. Valid metadata year
    if isinstance(year, int) and 1900 <= year <= 2025:
        return year

    # 2. Try arXiv ID (folder name or paper_id)
    candidates = []
    candidates.append(Path(file_path).parent.name)

    if "paper_id" in data:
        candidates.append(data["paper_id"])

    for text in candidates:
        # arXiv pattern: 1706.03762 or 1706.03762v7
        match = re.search(r"(\d{2})(\d{2})\.\d+", text)
        if not match:
            match = re.search(r"(\d{2})(\d{2})", text)

        if match:
            yy = int(match.group(1))
            return 1900 + yy if yy >= 90 else 2000 + yy

    # 3. Try generic 4-digit year
    for text in candidates:
        match = re.search(r"(19|20)\d{2}", text)
        if match:
            return int(match.group())

    return None

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────

OUTPUT_DIR = pathlib.Path("data_1/agent_outputs/critiques")

OLLAMA_HOST      = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL     = "qwen2.5"

ABLATION_KEYWORDS = [
    "ablation", "ablate", "sensitivity analysis",
    "component analysis", "remove", "without",
    "w/o", "effect of", "contribution of",
]

VARIANT_KEYWORDS = [
    "variant", "variants", "different versions",
    "we test", "we compare", "configuration",
    "setting", "hyperparameter", "heads", "layers"
]

SIGNIFICANCE_KEYWORDS = [
    "p<", "p <", "p-value", "t-test", "wilcoxon",
    "statistical significance", "confidence interval",
    "error bar", "std dev", "standard deviation", "±",
]
REPRODUCIBILITY_KEYWORDS = [
    "github", "code", "implementation", "available at",
    "open source", "open-source", "released",
    "https://", "http://",
]
SPEED_KEYWORDS = [
    "words/sec", "words per second", "tokens/sec", "throughput",
    "training time", "faster", "speed", "latency", "step/sec",
]


# Known ML/NLP baseline publication years.
# Used by check_outdated_baselines() to flag stale comparisons.
# Covers the most common baselines seen in NLP/ML papers 2015-2024.
# Add more as needed — this dict never needs to be exhaustive,
# unknown baselines are flagged separately at LOW severity.
KNOWN_BASELINE_YEARS: dict[str, int] = {
    # RNN / LSTM family
    "lstm":             1997,   # Hochreiter & Schmidhuber 1997
    "biglstm":          2016,
    "lstmp":            2014,
    "rnn":              1990,
    "gru":              2014,
    # Transformer family
    "transformer":      2017,
    "attention is all": 2017,
    "bert":             2018,
    "roberta":          2019,
    "gpt":              2018,
    "gpt-2":            2019,
    "gpt-3":            2020,
    "t5":               2020,
    "xlnet":            2019,
    "albert":           2020,
    "distilbert":       2019,
    "electra":          2020,
    # CNN family
    "alexnet":          2012,
    "vgg":              2014,
    "resnet":           2016,
    "inception":        2015,
    "densenet":         2017,
    # Seq2seq / encoder-decoder
    "seq2seq":          2014,
    "bahdanau":         2015,
    "attention":        2015,
    # Classic NLP
    "word2vec":         2013,
    "glove":            2014,
    "fasttext":         2017,
    "elmo":             2018,
    # RL
    "dqn":              2015,
    "a3c":              2016,
    "ppo":              2017,
    "sac":              2018,
}

# How many years old a baseline can be before flagging (relative to paper year)
BASELINE_STALENESS_THRESHOLD = 4
# Year threshold: papers published before this year get softer severity
# for norms (significance testing, ablations) that weren't standard then
NORM_YEAR_THRESHOLD = 2020

# Severity ladder for upgrade validation
SEV_LADDER = ["LOW", "MEDIUM", "HIGH"]

LLM_TIMEOUT_SECONDS  = 60
LLM_MAX_RETRIES      = 3
LLM_RETRY_BASE_DELAY = 2


# ─────────────────────────────────────────────
#  DATA MODELS
# ─────────────────────────────────────────────

class Severity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


class PaperType(str, Enum):
    EXPERIMENTAL = "experimental"
    THEORETICAL  = "theoretical"
    SURVEY       = "survey"


@dataclass
class Weakness:
    weakness_id:   str
    weakness_type: str
    severity:      Severity
    description:   str
    evidence:      str
    suggestion:    str
    source:        str        # "heuristic" | "llm"
    section:       str  = ""
    confidence:    float = 1.0   # heuristic=1.0, llm=0.85, llm_filtered=0.0 (rejected)
    validation_status: str = "accepted"  # "accepted" | "rejected_hallucination" | "rejected_contradiction"
    needs_review:      bool = False       # True when confidence < 0.7 — flag for planner/writer


@dataclass
class CritiqueResult:
    paper_id:               str
    title:                  str
    authors:                list[str]
    year:                   int | None
    critique_id:            str
    generated_at:           str
    agent_version:          str = "2.0.0"
    depth:                  str = "heuristic"
    paper_type:             str = ""
    weaknesses:             list[Weakness] = field(default_factory=list)
    hallucinations_filtered: list[dict]    = field(default_factory=list)
    severity_counts:        dict           = field(default_factory=dict)
    overall_quality:        str            = ""
    react_trace:            list[str]      = field(default_factory=list)
    agent_report: dict = field(default_factory=dict)  


    def compute_summary(self):
        counts = {s.value: 0 for s in Severity}
        for w in self.weaknesses:
            counts[w.severity.value] += 1
        self.severity_counts = counts
        if counts["HIGH"] >= 2:
            self.overall_quality = "HIGH_CONCERN"
        elif counts["HIGH"] >= 1 or counts["MEDIUM"] >= 3:
            self.overall_quality = "MODERATE"
        else:
            self.overall_quality = "ACCEPTABLE"


# ─────────────────────────────────────────────
#  INPUT LOADING
# ─────────────────────────────────────────────

def load_claims_output(path: str | pathlib.Path) -> dict:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with open(p, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object, got {type(raw)}")
    if "paper_id" not in raw:
        raise ValueError("Missing required field: paper_id")

    for claim in raw.get("claims", []):
        if "text" not in claim and "description" in claim:
            claim["text"] = claim["description"]
        if "type" not in claim and "claim_type" in claim:
            claim["type"] = claim["claim_type"]

    for lim in raw.get("limitations", []):
        if "text" not in lim:
            lim["text"] = str(lim)

    if "entity_index" not in raw and "entities" in raw:
        raw["entity_index"] = raw["entities"] if isinstance(raw["entities"], dict) else {}

    raw.setdefault("entity_index", {})
    raw.setdefault("claims",       [])
    raw.setdefault("limitations",  [])
    raw.setdefault("future_work",  [])
    raw.setdefault("metadata",     {})
    return raw



# ─────────────────────────────────────────────
#  ENTITY NAME NORMALISATION
# ─────────────────────────────────────────────

def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def _stem(tok: str) -> str:
    """Minimal stemmer — strips trailing 's'/'es' from tokens longer than 3 chars."""
    return tok.rstrip("es").rstrip("s") if len(tok) > 3 else tok


def _name_similarity(a: str, b: str) -> float:
    """
    Jaccard similarity on stemmed tokens.
    0.0 = completely different, 1.0 = identical.
    """
    ta = {_stem(t) for t in _normalise_name(a).split()}
    tb = {_stem(t) for t in _normalise_name(b).split()}
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _deduplicate_entity_names(names: list[str], threshold: float = 0.60) -> list[str]:
    """
    Remove near-duplicate entity names using stemmed Jaccard similarity.
    Keeps the shorter (more canonical) form when two names are similar enough.

    Threshold 0.60 merges:
      "one billion word benchmark" ↔ "one billion words benchmark evaluation"
      "CoNLL-2003"                 ↔ "CoNLL 2003"

    But keeps separate:
      "SQuAD" ↔ "SQuAD 2.0"   (0.33 — different versions)
      "GLUE"  ↔ "SuperGLUE"   (0.00 — different benchmarks)
    """
    if len(names) <= 1:
        return names

    keep = [True] * len(names)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if not keep[i] or not keep[j]:
                continue
            sim = _name_similarity(names[i], names[j])
            if sim >= threshold:
                # Keep the shorter name (more canonical), drop the longer one
                if len(names[i]) <= len(names[j]):
                    keep[j] = False
                else:
                    keep[i] = False

    return [names[i] for i in range(len(names)) if keep[i]]


# ─────────────────────────────────────────────
#  GROUND TRUTH INDEX
#  Built once from structured data — used by the hallucination filter
# ─────────────────────────────────────────────

def build_ground_truth_index(data: dict) -> dict:
    """
    Build a flat index of what is ACTUALLY present in the structured data.
    This is the source of truth the hallucination filter checks against.
    """
    ei = data.get("entity_index", {})
    if not isinstance(ei, dict):
        ei = {}

    all_claim_text = " ".join(
        c.get("text", "") for c in data.get("claims", [])
    ).lower()

    all_lim_text = " ".join(
        l.get("text", "") for l in data.get("limitations", [])
    ).lower()

    abstract = data.get("metadata", {}).get("abstract", "").lower()
    full_text = all_claim_text + " " + abstract

    datasets  = list(ei.get("dataset", {}).keys())
    methods   = list(ei.get("method",  {}).keys())
    metrics   = list(ei.get("metric",  {}).keys())
    tasks     = list(ei.get("task",    {}).keys())

    # Deduplicate near-identical dataset/entity names before counting
    datasets = _deduplicate_entity_names(datasets)

    # Flatten all entity text for keyword matching
    all_entity_text = " ".join(datasets + methods + metrics + tasks).lower()

    paper_id = data.get("paper_id", "")
    meta = data.get("metadata", {})

    # 🔧 Fix year using robust extractor
    file_path = data.get("_file_path", "")
    year = extract_year(data, file_path)

    if year:
        meta["year"] = year

    if not year or year > 2025:
        try:
            prefix = paper_id.split('.')[0]
            year_prefix = int(prefix[:2])

            if year_prefix >= 90:
                inferred_year = 1900 + year_prefix
            else:
                inferred_year = 2000 + year_prefix

            meta["year"] = inferred_year
        except:
            pass

    return {
        "datasets":          datasets,
        "methods":           methods,
        "metrics":           metrics,
        "tasks":             tasks,
        "n_datasets":        len(datasets),
        "n_methods":         len(methods),
        "n_metrics":         len(metrics),
        "n_limitations":     len(data.get("limitations", [])),
        "n_future_work":     len(data.get("future_work", [])),
        "n_claims":          len(data.get("claims", [])),
        "has_ablation":      detect_ablation_signals(all_claim_text),
        "has_significance":  any(kw in all_claim_text for kw in SIGNIFICANCE_KEYWORDS),
        "has_reproducibility": any(kw in full_text for kw in REPRODUCIBILITY_KEYWORDS),
        "has_speed_metrics": any(kw in all_entity_text + all_claim_text for kw in SPEED_KEYWORDS),
        "paper_year":        data.get("metadata", {}).get("year"),
        "all_claim_text":    all_claim_text,
        "all_entity_text":   all_entity_text,
    }

def detect_ablation_signals(text: str) -> bool:
    """
    Improved ablation detection using multiple signals:
    - keyword match (existing)
    - variant/config comparison
    - structural signal (multiple model comparisons)
    """
    text = text.lower()

    keyword_hit = any(k in text for k in ABLATION_KEYWORDS)
    variant_hit = any(k in text for k in VARIANT_KEYWORDS)

    # Structural signal: multiple mentions of model + comparison words
    structural_hit = (
        text.count("model") >= 3 and
        ("compare" in text or "different" in text or "vary" in text)
    )

    return keyword_hit or variant_hit or structural_hit

# ─────────────────────────────────────────────
#  HALLUCINATION FILTER
# ─────────────────────────────────────────────

# Maps weakness_type keywords → what the LLM is claiming is ABSENT
# If that thing is actually PRESENT in ground_truth, the weakness is a hallucination
ABSENCE_CLAIM_MAP = {
    "speed":         "has_speed_metrics",
    "training_time": "has_speed_metrics",
    "throughput":    "has_speed_metrics",
    "fast":          "has_speed_metrics",
    "ablation":      "has_ablation",
    "significance":  "has_significance",
    "statistical":   "has_significance",
    "reproducib":    "has_reproducibility",
    "code":          "has_reproducibility",
    "github":        "has_reproducibility",
    "dataset":       None,   # needs count check
    "baseline":      None,   # needs description check — can't auto-validate
}


def validate_llm_weakness(
    weakness: Weakness,
    gt: dict,
    heuristic_findings: dict,
    paper_year: int | None,
    verbose: bool = False,
) -> tuple[bool, str]:
    """
    Validate a single LLM-generated weakness against ground truth.

    Returns (is_valid, rejection_reason).
    is_valid=True  → accept the weakness
    is_valid=False → reject with reason
    """
    wtype = weakness.weakness_type.lower()
    desc  = weakness.description.lower()
    ev    = weakness.evidence.lower()
    combined = wtype + " " + desc + " " + ev

    # ── Rule 1: Contradiction check ────────────────────────────────────────
    # If LLM claims X is missing but heuristics confirmed X is present, reject.
    for keyword, gt_key in ABSENCE_CLAIM_MAP.items():
        if keyword in combined:
            if gt_key and gt.get(gt_key, False):
                reason = (
                    f"CONTRADICTION: LLM claims '{keyword}' is absent/missing "
                    f"but ground_truth['{gt_key}'] = True. "
                    f"Heuristic confirmed this is present in structured data."
                )
                if verbose:
                    print(f"    [filter] ❌ REJECTED '{weakness.weakness_type}': {reason[:120]}")
                return False, reason

    # ── Rule 2: Dataset count contradiction ────────────────────────────────
    # LLM says "only one dataset" but there are actually 2+
    if ("single dataset" in combined or "only one dataset" in combined
            or "single-dataset" in combined):
        if gt["n_datasets"] >= 2:
            reason = (
                f"CONTRADICTION: LLM claims single-dataset evaluation but "
                f"ground_truth shows {gt['n_datasets']} datasets: {gt['datasets']}"
            )
            if verbose:
                print(f"    [filter] ❌ REJECTED '{weakness.weakness_type}': {reason[:120]}")
            return False, reason

    # ── Rule 3: Evidence grounding check ───────────────────────────────────
    # LLM evidence must reference something that actually appears in the data.
    # If evidence is completely generic (no paper-specific terms), flag as low confidence.
    # We don't reject on this alone but reduce confidence to 0.6.
    evidence_tokens = set(re.findall(r'\b\w{4,}\b', ev))
    data_tokens     = set(re.findall(r'\b\w{4,}\b', gt["all_claim_text"] + " " + gt["all_entity_text"]))
    overlap = evidence_tokens & data_tokens

    if len(evidence_tokens) > 3 and len(overlap) < 2:
        # Evidence has specific words but none match the actual paper content
        # Likely hallucinated evidence — reduce confidence but don't fully reject
        weakness.confidence = 0.60
        weakness.needs_review = True
        if verbose:
            print(f"    [filter] ⚠ LOW CONFIDENCE '{weakness.weakness_type}' (needs_review=True): "
                  f"evidence tokens don't match paper content ({len(overlap)}/5 overlap)")

    # ── Rule 4: Heuristic authority check ──────────────────────────────────
    # If heuristics already confirmed something exists, LLM can't say it's missing.
    heuristic_present = {
        "ablation":     heuristic_findings.get("has_ablation", False),
        "significance": heuristic_findings.get("has_significance", False),
        "repro":        heuristic_findings.get("has_reproducibility", False),
        "speed":        heuristic_findings.get("has_speed_metrics", False),
    }
    for key, present in heuristic_present.items():
        if present and key in combined:
            reason = (
                f"HEURISTIC_AUTHORITY: Heuristic confirmed '{key}' is present "
                f"in the paper. LLM cannot flag it as missing."
            )
            if verbose:
                print(f"    [filter] ❌ REJECTED '{weakness.weakness_type}': {reason[:120]}")
            return False, reason

    if verbose:
        print(f"    [filter] ✅ ACCEPTED '{weakness.weakness_type}' "
              f"(confidence: {weakness.confidence})")
    return True, ""


def validate_severity_adjustment(
    weakness_type: str,
    current_severity: Severity,
    proposed_severity: Severity,
    paper_year: int | None,
    justification: str,
    verbose: bool = False,
) -> tuple[Severity, str]:
    """
    Validate and possibly cap a severity adjustment proposed by the LLM.

    Rules:
    - LLM can freely DOWNGRADE severity (always safe)
    - LLM can upgrade by MAX ONE LEVEL
    - For pre-NORM_YEAR_THRESHOLD papers: statistical significance and ablation
      cannot be upgraded to HIGH (these weren't standard practice then)
    - Any upgrade requires justification text
    """
    current_idx  = SEV_LADDER.index(current_severity.value)
    proposed_idx = SEV_LADDER.index(proposed_severity.value)

    # Downgrade — always allow
    if proposed_idx <= current_idx:
        if verbose and proposed_idx < current_idx:
            print(f"    [severity] ✅ Downgrade allowed: {current_severity.value} → {proposed_severity.value}")
        return proposed_severity, "downgrade_allowed"

    # Upgrade by more than one level — cap at one level
    if proposed_idx > current_idx + 1:
        capped = Severity(SEV_LADDER[current_idx + 1])
        if verbose:
            print(f"    [severity] ⚠ Upgrade capped: {proposed_severity.value} → {capped.value} "
                  f"(max one level per adjustment)")
        return capped, f"capped_at_one_level_from_{current_severity.value}"

    # Pre-norm-year guard: significance and ablation can't reach HIGH for old papers
    OLD_PAPER_PROTECTED = ["missing_statistical_significance", "missing_ablation_study"]
    if (paper_year and int(paper_year) < NORM_YEAR_THRESHOLD
            and weakness_type in OLD_PAPER_PROTECTED
            and proposed_severity == Severity.HIGH):
        if verbose:
            print(f"    [severity] ⚠ Year-cap applied: {weakness_type} cannot be HIGH "
                  f"for paper from {paper_year} (threshold: {NORM_YEAR_THRESHOLD})")
        return Severity.MEDIUM, f"year_cap_{paper_year}_lt_{NORM_YEAR_THRESHOLD}"

    # Upgrade by one level with justification — allow
    if justification and len(justification.strip()) > 10:
        if verbose:
            print(f"    [severity] ✅ Upgrade allowed: {current_severity.value} → {proposed_severity.value}")
        return proposed_severity, "upgrade_with_justification"

    # Upgrade with no justification — reject upgrade, keep current
    if verbose:
        print(f"    [severity] ⚠ Upgrade rejected (no justification): keeping {current_severity.value}")
    return current_severity, "upgrade_rejected_no_justification"


# ─────────────────────────────────────────────
#  LLM CALL HELPER (exponential backoff)
# ─────────────────────────────────────────────

def _llm_call_raw(prompt: str, llm_backend: str) -> str:
    """
    Call Ollama /api/generate with format='json' so the model is
    constrained to produce valid JSON output.
    All prompts in this agent already end with 'Reply with ONLY valid JSON'
    instructions — the format='json' flag enforces this at the API level.
    """
    if llm_backend != "ollama":
        raise EnvironmentError(
            f"Unsupported LLM backend '{llm_backend}'. Only 'ollama' is supported."
        )

    ollama_host = OLLAMA_HOST.rstrip("/")
    last_error  = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{ollama_host}/api/generate",
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                    "ngrok-skip-browser-warning": "true",
                },
                json={
                    "model":  OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",           # Ollama-native JSON mode — enforces valid JSON
                    "options": {
                        "temperature": 0.0,     # Deterministic output
                        "num_predict": 4096,
                    },
                },
                timeout=LLM_TIMEOUT_SECONDS,
            )

            if resp.status_code == 503:
                delay = LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                if attempt < LLM_MAX_RETRIES:
                    print(f"  [llm] Ollama 503 (attempt {attempt}/{LLM_MAX_RETRIES}). Retrying in {delay}s...")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"Ollama unavailable (503) after {LLM_MAX_RETRIES} attempts")

            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

        except (requests.RequestException, KeyError) as e:
            last_error = e
            if attempt < LLM_MAX_RETRIES:
                time.sleep(LLM_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

    raise RuntimeError(f"LLM call failed after {LLM_MAX_RETRIES} attempts. Last: {last_error}")


# ─────────────────────────────────────────────
#  PAPER TYPE DETECTION
# ─────────────────────────────────────────────

def detect_paper_type(data: dict, llm_backend: str, verbose: bool = False) -> PaperType:
    abstract = data.get("metadata", {}).get("abstract", "")[:800]

    if not abstract or llm_backend == "none":
        if verbose:
            print("    [paper_type] No abstract or no LLM — defaulting to EXPERIMENTAL")
        return PaperType.EXPERIMENTAL

    prompt = (
        "Classify the following research paper abstract into exactly one category:\n"
        "  experimental  — proposes and evaluates a method with experiments\n"
        "  theoretical   — proves theorems or derives formulas, minimal experiments\n"
        "  survey        — reviews and summarises existing work\n\n"
        f"Abstract:\n{abstract}\n\n"
        "Reply with one word only: experimental, theoretical, or survey."
    )

    try:
        text = _llm_call_raw(prompt, llm_backend).strip().lower()
        if "survey"       in text: ptype = PaperType.SURVEY
        elif "theoretical" in text: ptype = PaperType.THEORETICAL
        else:                       ptype = PaperType.EXPERIMENTAL
        if verbose:
            print(f"    [paper_type] Detected: {ptype.value} (raw: '{text[:40]}')")
        return ptype
    except Exception as e:
        if verbose:
            print(f"    [paper_type] Detection failed ({e}) — defaulting to EXPERIMENTAL")
        return PaperType.EXPERIMENTAL


# ─────────────────────────────────────────────
#  HEURISTIC CHECKS  (Layer 1 — ground truth)
# ─────────────────────────────────────────────

class HeuristicChecker:

    def __init__(self, data: dict, paper_type: PaperType, gt: dict, verbose: bool = False):
        self.data       = data
        self.paper_type = paper_type
        self.gt         = gt
        self.verbose    = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"    [heuristic] {msg}")

    def _make_id(self, wtype: str) -> str:
        return f"w_{wtype}_{uuid.uuid4().hex[:6]}"

    def check_single_dataset(self) -> list[Weakness]:
        if self.paper_type in (PaperType.SURVEY, PaperType.THEORETICAL):
            self._log("skipping single_dataset — not applicable")
            return []

        n = self.gt["n_datasets"]
        self._log(f"datasets: {n} → {self.gt['datasets']}")

        if n == 0:
            return [Weakness(
                weakness_id   = self._make_id("no_dataset"),
                weakness_type = "no_dataset_reported",
                severity      = Severity.HIGH,
                description   = (
                    "No named dataset was extracted from this paper. "
                    "Reproducibility cannot be assessed without knowing what data was used."
                ),
                evidence      = "entity_index.dataset is empty.",
                suggestion    = "Verify dataset names exist in the paper and re-run the NER pipeline.",
                source        = "heuristic",
            )]
        if n == 1:
            return [Weakness(
                weakness_id   = self._make_id("single_dataset"),
                weakness_type = "single_dataset_evaluation",
                severity      = Severity.HIGH,
                description   = (
                    f"Results are reported on only one dataset ('{self.gt['datasets'][0]}'). "
                    "Single-dataset evaluation limits generalisability."
                ),
                evidence      = f"Only dataset in entity_index: '{self.gt['datasets'][0]}'",
                suggestion    = "Evaluate on at least two additional benchmarks.",
                source        = "heuristic",
            )]
        return []

    def check_ablation(self) -> list[Weakness]:
        if self.paper_type in (PaperType.SURVEY, PaperType.THEORETICAL):
            self._log("skipping ablation — not applicable")
            return []

        self._log(f"ablation keywords found: {self.gt['has_ablation']}")
        if not self.gt["has_ablation"]:
            return [Weakness(
                weakness_id   = self._make_id("ablation"),
                weakness_type = "missing_ablation_study",
                severity      = Severity.MEDIUM,
                description   = (
                    "No ablation study or component analysis was detected. "
                    "Without ablation results it is impossible to determine which "
                    "components of the proposed method drive the observed performance gains."
                ),
                evidence      = "None of the ablation keywords appear in extracted claims.",
                suggestion    = "Add an ablation table isolating each novel component.",
                source        = "heuristic",
            )]
        return []

    def check_statistical_significance(self) -> list[Weakness]:
        if self.paper_type == PaperType.THEORETICAL:
            self._log("skipping significance — theoretical paper")
            return []

        perf_claims = [
            c for c in self.data.get("claims", [])
            if c.get("type") in ("performance", "comparative")
        ]
        if not perf_claims:
            self._log("no perf/comparative claims — skipping")
            return []

        self._log(f"significance keywords: {self.gt['has_significance']}, perf claims: {len(perf_claims)}")
        if not self.gt["has_significance"]:
            snippets = [c.get("text", "")[:120] for c in perf_claims[:3]]
            return [Weakness(
                weakness_id   = self._make_id("significance"),
                weakness_type = "missing_statistical_significance",
                severity      = Severity.MEDIUM,
                description   = (
                    f"The paper makes {len(perf_claims)} performance or comparative "
                    "claims but reports no statistical significance testing."
                ),
                evidence      = "Performance claims: " + " | ".join(snippets),
                suggestion    = "Report mean ± std dev. Use a paired t-test across 3+ seeds.",
                source        = "heuristic",
            )]
        return []

    def check_reproducibility(self) -> list[Weakness]:
        if self.paper_type == PaperType.THEORETICAL:
            self._log("skipping reproducibility — theoretical paper")
            return []

        self._log(f"reproducibility indicators: {self.gt['has_reproducibility']}")
        if not self.gt["has_reproducibility"]:
            return [Weakness(
                weakness_id   = self._make_id("repro"),
                weakness_type = "missing_reproducibility",
                severity      = Severity.LOW,
                description   = "No code repository or reproducibility statement was detected.",
                evidence      = "No reproducibility indicators in extracted text.",
                suggestion    = "Release code on GitHub with hyperparameter settings and seeds.",
                source        = "heuristic",
            )]
        return []

    def check_future_work(self) -> list[Weakness]:
        fw  = self.gt["n_future_work"]
        lim = self.gt["n_limitations"]
        self._log(f"future_work: {fw}, limitations: {lim}")

        if fw == 0 and lim <= 1:
            return [Weakness(
                weakness_id   = self._make_id("future_work"),
                weakness_type = "missing_limitations_future_work",
                severity      = Severity.MEDIUM,
                description   = (
                    f"The paper reports minimal limitations ({lim}) "
                    "and no future work directions."
                ),
                evidence      = f"future_work empty. limitations: {lim} item(s).",
                suggestion    = "Add a Limitations section with 2–3 concrete future directions.",
                source        = "heuristic",
            )]
        return []

    def check_metric_diversity(self) -> list[Weakness]:
        if self.paper_type in (PaperType.SURVEY, PaperType.THEORETICAL):
            self._log("skipping metric diversity — not applicable")
            return []

        n     = self.gt["n_metrics"]
        tasks = self.gt["tasks"]
        self._log(f"metrics: {n}, tasks: {len(tasks)}")

        if n <= 1 and len(tasks) >= 2:
            return [Weakness(
                weakness_id   = self._make_id("metrics"),
                weakness_type = "limited_evaluation_metrics",
                severity      = Severity.LOW,
                description   = (
                    f"Only {n} metric(s) detected despite {len(tasks)} tasks addressed."
                ),
                evidence      = f"Metrics: {self.gt['metrics']}. Tasks: {tasks[:4]}.",
                suggestion    = "Report complementary metrics and efficiency measures.",
                source        = "heuristic",
            )]
        return []


    def check_outdated_baselines(self) -> list[Weakness]:
        """
        Check if baselines referenced in comparative claims are outdated
        relative to the paper's publication year.

        Strategy:
          1. Extract method names from comparative claims.
          2. Look them up in KNOWN_BASELINE_YEARS.
          3. Flag if median known-baseline year is > BASELINE_STALENESS_THRESHOLD years old.
          4. Flag unknown baselines at LOW severity if no year can be found.
        """
        if self.paper_type == PaperType.SURVEY:
            self._log("skipping outdated baselines — survey paper")
            return []

        paper_year = self.gt.get("paper_year")
        if not paper_year:
            self._log("no paper year — skipping outdated baselines check")
            return []

        paper_year = int(paper_year)

        # Gather all methods from comparative claims
        comparative_claims = [
            c for c in self.data.get("claims", [])
            if c.get("type") == "comparative"
        ]
        if not comparative_claims:
            self._log("no comparative claims — skipping outdated baselines check")
            return []

        # Collect all entities involved in comparative claims
        baseline_entities: list[str] = []
        for c in comparative_claims:
            for e in c.get("entities_involved", []):
                baseline_entities.append(e.lower().strip())

        # Deduplicate — only use entities from comparative claims,
        # NOT all gt methods (that would flag unrelated referenced methods)
        baseline_entities = list(set(baseline_entities))

        # Look up known years
        found_years:   list[tuple[str, int]] = []  # (name, year)
        unknown_names: list[str]             = []

        def _normalise_for_lookup(s: str) -> str:
            import re as _re
            return _re.sub(r'[^a-z0-9]', '', s.lower())

        for entity in baseline_entities:
            entity_norm = _normalise_for_lookup(entity)
            matched = False
            for key, year in KNOWN_BASELINE_YEARS.items():
                key_norm = _normalise_for_lookup(key)
                # Exact normalised match only — prevents "f-lstm" matching "lstm"
                if entity_norm == key_norm:
                    found_years.append((entity, year))
                    matched = True
                    break
            if not matched:
                unknown_names.append(entity)

        self._log(
            f"baseline check: {len(found_years)} known, "
            f"{len(unknown_names)} unknown. paper_year={paper_year}"
        )

        weaknesses: list[Weakness] = []

        # Check known baselines for staleness
        if found_years:
            stale = [
                (name, year) for name, year in found_years
                if (paper_year - year) > BASELINE_STALENESS_THRESHOLD
            ]
            if stale:
                stale_str = ", ".join(
                    f"{name} ({year}, {paper_year - year}y old)"
                    for name, year in stale[:4]
                )
                # Severity depends on how stale
                max_age = max(paper_year - yr for _, yr in stale)
                severity = Severity.HIGH if max_age > 7 else Severity.MEDIUM

                weaknesses.append(Weakness(
                    weakness_id   = self._make_id("baselines"),
                    weakness_type = "outdated_baselines",
                    severity      = severity,
                    description   = (
                        f"{len(stale)} baseline(s) appear to be more than "
                        f"{BASELINE_STALENESS_THRESHOLD} years old relative to "
                        f"this paper's publication year ({paper_year}). "
                        "Comparisons against outdated baselines weaken the paper's "
                        "claims about state-of-the-art performance."
                    ),
                    evidence      = f"Stale baselines: {stale_str}",
                    suggestion    = (
                        "Replace outdated baselines with methods published within "
                        f"{BASELINE_STALENESS_THRESHOLD} years of this paper. "
                        "Include at least one contemporary strong baseline."
                    ),
                    source        = "heuristic",
                ))

        return weaknesses

    def run_all(self) -> list[Weakness]:
        weaknesses: list[Weakness] = []
        for check_fn in [
            self.check_single_dataset,
            self.check_ablation,
            self.check_statistical_significance,
            self.check_reproducibility,
            self.check_future_work,
            self.check_metric_diversity,
            self.check_outdated_baselines,
        ]:
            try:
                weaknesses.extend(check_fn())
            except Exception as e:
                print(f"  ⚠ '{check_fn.__name__}' failed: {e}", file=sys.stderr)
        return weaknesses


# ─────────────────────────────────────────────
#  LLM ENRICHMENT  (Layer 2)
# ─────────────────────────────────────────────

def _build_enrichment_prompt(
    data: dict,
    gt: dict,
    heuristic_weaknesses: list[Weakness],
    paper_type: PaperType,
) -> str:
    meta     = data.get("metadata", {})
    title    = meta.get("title",    "Unknown")
    year     = meta.get("year",     "Unknown")
    abstract = meta.get("abstract", "")[:400]

    claims_sample = [
        c.get("text", "")[:200]
        for c in data.get("claims", [])
        if c.get("type") in ("performance", "comparative")
    ][:3]

    limitations_sample = [l.get("text", "")[:200] for l in data.get("limitations", [])][:4]

    flags_text = (
        "\n".join(
            f"  - [{w.severity.value}] {w.weakness_type}: {w.description[:150]}"
            for w in heuristic_weaknesses
        ) if heuristic_weaknesses else "  (none)"
    )

    # Tell the LLM exactly what IS present so it can't hallucinate absences
    present_facts = f"""CONFIRMED PRESENT IN THIS PAPER (do NOT flag these as missing):
- Datasets ({gt['n_datasets']}): {gt['datasets']}
- Metrics ({gt['n_metrics']}): {gt['metrics']}
- Methods: {gt['methods'][:6]}
- Speed/throughput metrics present: {gt['has_speed_metrics']}
- Reproducibility indicators present: {gt['has_reproducibility']}
- Ablation study present: {gt['has_ablation']}
- Statistical significance present: {gt['has_significance']}
- Limitations count: {gt['n_limitations']}
- Future work count: {gt['n_future_work']}"""

    return f"""You are a rigorous scientific peer reviewer specialising in ML and NLP.

Paper type: {paper_type.value}  |  Year: {year}

PAPER INFORMATION
Title: {title}
Abstract: {abstract}
Tasks: {gt['tasks']}

Performance/comparative claims:
{chr(10).join("  • " + c for c in claims_sample) or "  (none)"}

Stated limitations:
{chr(10).join("  • " + l for l in limitations_sample) or "  (none)"}

{present_facts}

ALREADY DETECTED BY HEURISTICS
{flags_text}

YOUR TASK
1. Identify ADDITIONAL weaknesses NOT already listed and NOT contradicting CONFIRMED PRESENT facts above.
2. Focus on: narrow task scope, overclaiming vs actual results, missing SOTA context,
   weak baseline justification, generalisability beyond tested domain, vague limitations.
3. You may ADJUST severity on heuristic flags if strongly justified.
4. Apply standards appropriate for a {paper_type.value} paper from {year}.
   Papers before 2020 should not have significance/ablation upgraded to HIGH.

IMPORTANT: Only flag something as missing if it is genuinely absent from the confirmed facts above.
If a fact is listed as present above, you must NOT claim it is missing.

Severity guide:
  HIGH   = threatens validity or reproducibility of core conclusions
  MEDIUM = limits generalisation or strength of claims
  LOW    = minor best-practice miss

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Return ONLY a single valid JSON object. No markdown fences, no preamble, no explanation, nothing before or after.
- Empty lists must be [] not null.
- Do not invent content not supported by the paper information above.

Reply with ONLY valid JSON — no markdown fences, no extra text:

{{
  "additional_weaknesses": [
    {{
      "weakness_type": "short_snake_case_label",
      "severity": "HIGH|MEDIUM|LOW",
      "description": "Clear 2-3 sentence explanation.",
      "evidence": "Specific quote or reference from paper info above.",
      "suggestion": "Concrete actionable fix."
    }}
  ],
  "severity_adjustments": [
    {{
      "weakness_type": "type_from_heuristic_list",
      "new_severity": "HIGH|MEDIUM|LOW",
      "justification": "Specific reason referencing paper content."
    }}
  ],
  "overall_assessment": "One sentence quality assessment."
}}"""


def _parse_llm_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object in LLM response. Raw: {text[:300]}")

    depth = 0
    end   = -1
    in_string = False
    escape    = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False; continue
        if ch == "\\" and in_string:
            escape = True; continue
        if ch == '"':
            in_string = not in_string; continue
        if in_string:
            continue
        if ch == "{":   depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i; break

    if end == -1:
        raise ValueError(f"Unclosed JSON object in LLM response. Raw: {text[:300]}")

    return json.loads(text[start:end + 1])


def llm_enrich(
    data: dict,
    gt: dict,
    heuristic_weaknesses: list[Weakness],
    paper_type: PaperType,
    llm_backend: str,
    verbose: bool = False,
) -> tuple[list[Weakness], list[dict], list[dict]]:
    """
    Returns (accepted_weaknesses, severity_adjustments, rejected_hallucinations)
    """
    prompt = _build_enrichment_prompt(data, gt, heuristic_weaknesses, paper_type)

    if verbose:
        print(f"\n  [llm] Sending enrichment prompt ({len(prompt)} chars) to {llm_backend}...")

    text   = _llm_call_raw(prompt, llm_backend)
    parsed = _parse_llm_json(text)

    if verbose:
        print("  [llm] Response parsed successfully")

    # Build heuristic findings dict for the filter
    heuristic_findings = {
        "has_ablation":      gt["has_ablation"],
        "has_significance":  gt["has_significance"],
        "has_reproducibility": gt["has_reproducibility"],
        "has_speed_metrics": gt["has_speed_metrics"],
    }

    accepted:   list[Weakness] = []
    rejected:   list[dict]     = []

    for item in parsed.get("additional_weaknesses", []):
        try:
            sev = Severity(item.get("severity", "LOW").upper())
        except ValueError:
            sev = Severity.LOW

        candidate = Weakness(
            weakness_id   = f"w_{item.get('weakness_type', 'llm')}_{uuid.uuid4().hex[:6]}",
            weakness_type = item.get("weakness_type", "llm_detected"),
            severity      = sev,
            description   = item.get("description", ""),
            evidence      = item.get("evidence", ""),
            suggestion    = item.get("suggestion", ""),
            source        = "llm",
            confidence    = 0.85,
        )

        is_valid, reason = validate_llm_weakness(
            candidate, gt, heuristic_findings,
            gt.get("paper_year"), verbose,
        )

        if is_valid:
            accepted.append(candidate)
        else:
            candidate.validation_status = (
                "rejected_contradiction" if "CONTRADICTION" in reason
                else "rejected_hallucination"
            )
            rejected.append({
                "weakness_type": candidate.weakness_type,
                "rejected_reason": reason,
                "original_severity": sev.value,
                "description": candidate.description[:200],
            })

    # Post-processing: downgrade limited_empirical_scope to LOW
    # if fewer than 3 tasks claimed (scope claim isn't strong enough to be MEDIUM)
    for w in accepted:
        if w.weakness_type == "limited_empirical_scope" and w.severity == Severity.MEDIUM:
            tasks_count = len(gt.get("tasks", []))
            if tasks_count < 3:
                w.severity = Severity.LOW
                if verbose:
                    print(f"    [post] Downgraded limited_empirical_scope to LOW "
                          f"(only {tasks_count} tasks — insufficient for MEDIUM scope claim)")

    return accepted, parsed.get("severity_adjustments", []), rejected


# ─────────────────────────────────────────────
#  REACT AGENT
# ─────────────────────────────────────────────

class CriticAgent:
    VERSION = "2.2.0"

    def __init__(self, llm_backend: str = "none", verbose: bool = False):
        self.llm_backend = llm_backend
        self.verbose     = verbose
        self.trace: list[str] = []

    def _think(self, msg: str):
        entry = f"[THINK] {msg}"
        self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _act(self, msg: str):
        entry = f"[ACT]   {msg}"
        self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def _observe(self, msg: str):
        entry = f"[OBS]   {msg}"
        self.trace.append(entry)
        if self.verbose: print(f"  {entry}")

    def run(self, claims_path: str | pathlib.Path) -> CritiqueResult:
        claims_path = pathlib.Path(claims_path)

        # Step 0: Load
        self._think(f"Loading input from {claims_path}")
        self._act("load_claims_output()")
        data = load_claims_output(claims_path)
                # 🧪 DEBUG + FORCE YEAR FIX
        from pathlib import Path
        import re

        print("DEBUG: RUN FUNCTION EXECUTED")
        print("DEBUG: file path =", claims_path)

        folder_name = Path(claims_path).parent.name
        print("DEBUG: folder_name =", folder_name)

        match = re.search(r"(\d{2})(\d{2})", folder_name)

        if match:
            yy = int(match.group(1))
            year = 1900 + yy if yy >= 90 else 2000 + yy
            print("DEBUG: extracted year =", year)

            data.setdefault("metadata", {})["year"] = year
            print("DEBUG: updated metadata year =", data["metadata"]["year"])
        data["_file_path"] = str(claims_path)
        meta = data.get("metadata", {})
        # 🔧 FORCE year correction at source
        year = extract_year(data, str(claims_path))
        if year:
            data.setdefault("metadata", {})["year"] = year

        # refresh meta AFTER fixing
        meta = data.get("metadata", {})
        self._observe(
            f"Loaded '{meta.get('title','?')}' ({meta.get('year','?')}). "
            f"Claims: {len(data['claims'])}, Limitations: {len(data['limitations'])}, "
            f"Future work: {len(data['future_work'])}"
        )

        # Step 1: Build ground truth index
        self._think("Building ground truth index from structured data")
        self._act("build_ground_truth_index()")
        gt = build_ground_truth_index(data)
        self._observe(
            f"Ground truth: {gt['n_datasets']} datasets, {gt['n_metrics']} metrics, "
            f"speed={gt['has_speed_metrics']}, ablation={gt['has_ablation']}, "
            f"significance={gt['has_significance']}, repro={gt['has_reproducibility']}"
        )

        # Step 2: Detect paper type
        self._think(f"Detecting paper type (backend: {self.llm_backend})")
        self._act("detect_paper_type()")
        paper_type = detect_paper_type(data, self.llm_backend, self.verbose)
        self._observe(f"Paper type: {paper_type.value}")

        result = CritiqueResult(
            paper_id      = data["paper_id"],
            title         = meta.get("title",   ""),
            authors       = meta.get("authors", []),
            year          = meta.get("year"),
            critique_id   = str(uuid.uuid4()),
            generated_at  = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            agent_version = self.VERSION,
            paper_type    = paper_type.value,
            react_trace   = self.trace,
        )

        # Step 3: Heuristic layer (ground truth)
        self._think("Running adaptive heuristic checks (Layer 1 — ground truth)")
        self._act("HeuristicChecker.run_all()")
        checker              = HeuristicChecker(data, paper_type, gt, verbose=self.verbose)
        heuristic_weaknesses = checker.run_all()
        result.weaknesses.extend(heuristic_weaknesses)
        result.depth = "heuristic"
        self._observe(
            f"Heuristic complete. {len(heuristic_weaknesses)} weaknesses "
            f"({sum(1 for w in heuristic_weaknesses if w.severity==Severity.HIGH)} HIGH, "
            f"{sum(1 for w in heuristic_weaknesses if w.severity==Severity.MEDIUM)} MEDIUM, "
            f"{sum(1 for w in heuristic_weaknesses if w.severity==Severity.LOW)} LOW)"
        )

        # Step 4: LLM enrichment with hallucination filtering
        self._think(f"LLM backend: '{self.llm_backend}'. Running filtered enrichment.")

        if self.llm_backend == "ollama":
            self._act(f"llm_enrich() via {self.llm_backend}")
            result.depth = "deep"
            try:
                additional, adjustments, rejected = llm_enrich(
                    data, gt, heuristic_weaknesses, paper_type,
                    self.llm_backend, self.verbose,
                )

                self._observe(
                    f"LLM enrichment complete. "
                    f"Accepted: {len(additional)}, "
                    f"Rejected (hallucinations): {len(rejected)}, "
                    f"Severity adjustments: {len(adjustments)}"
                )

                result.weaknesses.extend(additional)
                result.hallucinations_filtered = rejected

                # Log each rejection in trace
                for r in rejected:
                    self._observe(
                        f"FILTERED '{r['weakness_type']}': {r['rejected_reason'][:100]}"
                    )

                # Apply validated severity adjustments
                for adj in adjustments:
                    wtype = adj.get("weakness_type", "")
                    justification = adj.get("justification", "")
                    try:
                        proposed_sev = Severity(adj.get("new_severity", "").upper())
                        for w in result.weaknesses:
                            if w.weakness_type == wtype and w.source == "heuristic":
                                validated_sev, reason = validate_severity_adjustment(
                                    wtype, w.severity, proposed_sev,
                                    result.year, justification, self.verbose,
                                )
                                if validated_sev != w.severity:
                                    old = w.severity
                                    w.severity = validated_sev
                                    self._observe(
                                        f"Severity adjusted '{wtype}': "
                                        f"{old.value} → {validated_sev.value} ({reason})"
                                    )
                    except ValueError:
                        pass

            except EnvironmentError as e:
                self._observe(f"LLM skipped — {e}. Heuristic-only output.")
                result.depth = "heuristic"
            except RuntimeError as e:
                self._observe(f"LLM failed — {e}. Heuristic-only output.")
                result.depth = "heuristic"
        else:
            self._think("No LLM backend — heuristic-only output.")

        # Step 5: Finalise
        self._think("Computing summary and overall quality score.")
        result.compute_summary()
        self._observe(
            f"Final: {len(result.weaknesses)} weaknesses accepted, "
            f"{len(result.hallucinations_filtered)} filtered. "
            f"Quality: {result.overall_quality}. Depth: {result.depth}."
        )

        result.agent_report = {
            "paper_id": result.paper_id,
            "critique_id": result.critique_id,
            "depth": result.depth,
            "paper_type": result.paper_type,
            "weakness_count": len(result.weaknesses),
            "severity_distribution": result.severity_counts,
            "overall_quality": result.overall_quality,
            "timestamp": result.generated_at,
        }

        return result , data

    def as_langgraph_node(self):
        """
        Returns a callable for graph.add_node("critic", node_fn).
        """
        agent = self

        def node_fn(state: dict) -> dict:
            papers = state.get("papers_to_critique", [])
            critiques = dict(state.get("critiques", {}))
            reports = list(state.get("agent_reports", []))

            for paper_path in papers:
                result = agent.run(paper_path)

                result.agent_report = {
                    "agent_name": "critic_agent",
                    "agent_version": agent.VERSION,
                    "status": "completed",
                    "paper_id": result.paper_id,
                    "depth": result.depth,
                    "total_weaknesses": len(result.weaknesses),
                    "has_high_weakness": result.severity_counts.get("HIGH", 0) > 0,
                    "high_weakness_types": [
                        w.weakness_type for w in result.weaknesses
                        if getattr(w.severity, "value", w.severity) == "HIGH"
                    ],
                    "duration_seconds": 0.0,
                    "recommended_next": (
                        "comparator"
                        if result.severity_counts.get("HIGH", 0) > 0
                        else "gap_detector"
                    ),
                    "coverage_gained": 0.0,
                    "planner_signal": {
                        "recommend_more_papers": result.severity_counts.get("HIGH", 0) > 2,
                        "recommend_comparator": result.severity_counts.get("HIGH", 0) > 1,
                        "coverage_concern": len(result.weaknesses) > 5,
                    },
                }

                critiques[result.paper_id] = result
                reports.append(result.agent_report)

            high_concern_count = sum(
                1
                for r in critiques.values()
                if hasattr(r, "severity_counts") and r.severity_counts.get("HIGH", 0) > 0
            )

            return {
                "critiques": critiques,
                "agent_reports": reports,
                "critic_summary": {
                    "papers_critiqued": len(papers),
                    "high_concern_papers": high_concern_count,
                    "recommended_next": (
                        "comparator" if high_concern_count > 0 else "gap_detector"
                    ),
                },
            }

        return node_fn

# ─────────────────────────────────────────────
#  OUTPUT WRITER
# ─────────────────────────────────────────────

def save_critique(result: CritiqueResult, output_dir: pathlib.Path) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{result.paper_id}_critique.json"

    data_dict = asdict(result)
    for w in data_dict.get("weaknesses", []):
        if hasattr(w.get("severity"), "value"):
            w["severity"] = w["severity"].value

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)

    return out_path




def save_enriched_paper_json(
    result: CritiqueResult,
    original_claims_path: str | pathlib.Path,
    data: dict,
    verbose: bool = False,
) -> pathlib.Path | None:
    """
    Write-back: load the original claims_output.json, inject the critique
    results as a new "critiques" key, and save as {paper_id}_enriched.json
    in the same folder.

    This is the shared memory update step — downstream agents (gap detector,
    writer) read the enriched JSON instead of the bare claims JSON.

    Never mutates the original file. Writes to a separate _enriched.json.

    TODO (Phase 4): Also update knowledge graph with has_weakness edges
    once the comparator agent and graph are available.
    """
    original_path = pathlib.Path(original_claims_path)
    if not original_path.exists():
        if verbose:
            print(f"  [write-back] ⚠ Original claims file not found: {original_path}")
        return None

    try:
        paper_data = data

        # Build the critiques array — one entry per accepted weakness
        critiques_array = []
        for w in result.weaknesses:
            sev = w.severity.value if hasattr(w.severity, "value") else str(w.severity)
            critiques_array.append({
                "weakness_id":       w.weakness_id,
                "weakness_type":     w.weakness_type,
                "severity":          sev,
                "description":       w.description,
                "evidence":          w.evidence,
                "suggestion":        w.suggestion,
                "source":            w.source,
                "confidence":        w.confidence,
                "needs_review":      getattr(w, "needs_review", False),
            })

        # Inject critique results into the paper JSON
        paper_data["critiques"] = critiques_array
        paper_data["critique_summary"] = {
            "critique_id":            result.critique_id,
            "generated_at":           result.generated_at,
            "agent_version":          result.agent_version,
            "depth":                  result.depth,
            "paper_type":             result.paper_type,
            "overall_quality":        result.overall_quality,
            "severity_counts":        result.severity_counts,
            "total_weaknesses":       len(result.weaknesses),
            "hallucinations_filtered": len(result.hallucinations_filtered),
            # Signals for downstream agents
            "has_high_weaknesses":    result.severity_counts.get("HIGH", 0) > 0,
            "high_weakness_types":    [
                w.weakness_type for w in result.weaknesses
                if (w.severity.value if hasattr(w.severity, "value") else w.severity) == "HIGH"
            ],
            # TODO (planner): coverage_signal field will be added here
            # when planner_agent.py is built and its expected format is known.
        }

        # Save as enriched copy — never overwrite original
        out_path = original_path.parent / f"{result.paper_id}_enriched.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False, default=str)

        if verbose:
            print(f"  [write-back] ✅ Enriched JSON saved: {out_path}")

        return out_path

    except Exception as e:
        if verbose:
            print(f"  [write-back] ⚠ Write-back failed: {e}")
        return None

def print_summary(result: CritiqueResult):
    SEV_ICONS     = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
    QUALITY_ICONS = {
        "HIGH_CONCERN": "⚠️  HIGH CONCERN",
        "MODERATE":     "⚡ MODERATE",
        "ACCEPTABLE":   "✅ ACCEPTABLE",
    }
    print("\n" + "═" * 64)
    print("  CRITIQUE REPORT  (v2.2.0 — production)")
    print("═" * 64)
    print(f"  Paper    : {result.title}")
    print(f"  Authors  : {', '.join(result.authors)}")
    print(f"  Year     : {result.year}")
    print(f"  Type     : {result.paper_type}")
    print(f"  Depth    : {result.depth}")
    print(f"  Quality  : {QUALITY_ICONS.get(result.overall_quality, result.overall_quality)}")
    print(f"  Counts   : {result.severity_counts}")
    if result.hallucinations_filtered:
        print(f"  Filtered : {len(result.hallucinations_filtered)} LLM hallucination(s) rejected")
    print("─" * 64)

    if not result.weaknesses:
        print("  No weaknesses detected.")
    else:
        for i, w in enumerate(result.weaknesses, 1):
            sev  = w.severity.value if hasattr(w.severity, "value") else str(w.severity)
            icon = SEV_ICONS.get(sev, "•")
            review_flag = " ⚑ needs_review" if getattr(w, "needs_review", False) else ""
            conf = f" [conf:{w.confidence:.2f}{review_flag}]" if w.source == "llm" else ""
            print(f"\n  {i}. {icon} [{sev}] {w.weakness_type}  (src: {w.source}{conf})")
            print(f"     {w.description}")
            print(f"     Evidence : {w.evidence[:120]}")
            print(f"     Fix      : {w.suggestion[:120]}")

    if result.hallucinations_filtered:
        print("\n  ── Filtered (hallucinations) ──────────────────────────")
        for h in result.hallucinations_filtered:
            print(f"  ✗ {h['weakness_type']} [{h['original_severity']}]")
            print(f"    Reason: {h['rejected_reason'][:120]}")

    print("\n" + "═" * 64)


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def detect_llm_backend() -> str:
    if os.environ.get("OLLAMA_HOST") or _ollama_is_reachable():
        return "ollama"
    return "none"


def _ollama_is_reachable() -> bool:
    """Quick ping to check if Ollama is running at the default host."""
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5,
                         headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Critic Agent v2.2.0 — production-grade methodological weakness detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agents/critic_agent.py data_1/parsed/claims_output.json
  python agents/critic_agent.py data_1/parsed/claims_output.json --verbose
  python agents/critic_agent.py data_1/parsed/claims_output.json --no-llm
        """,
    )
    parser.add_argument("input",  help="Path to claims_output.json")
    parser.add_argument("--llm",  choices=["ollama", "auto"], default="auto")
    parser.add_argument("--ollama-host", default=None,
                        help="Ollama host URL (overrides OLLAMA_HOST env var, default: http://localhost:11434)")
    parser.add_argument("--no-llm",     action="store_true")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Allow CLI override of Ollama host
    if args.ollama_host:
        os.environ["OLLAMA_HOST"] = args.ollama_host
        import importlib, sys as _sys
        # Refresh module-level OLLAMA_HOST constant
        globals()["OLLAMA_HOST"] = args.ollama_host

    backend = "none" if args.no_llm else (
        detect_llm_backend() if args.llm == "auto" else args.llm
    )

    if args.verbose:
        print(f"\nCritic Agent v{CriticAgent.VERSION}")
        print(f"  Input      : {args.input}")
        print(f"  LLM backend: {backend}")
        print(f"  Output dir : {args.output_dir}")

    agent = CriticAgent(llm_backend=backend, verbose=args.verbose)

    try:
        result, data = agent.run(args.input)
    except FileNotFoundError as e:
        print(f"\n❌ {e}", file=sys.stderr); sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Invalid input: {e}", file=sys.stderr); sys.exit(1)

    print_summary(result)
    out_path = save_critique(result, pathlib.Path(args.output_dir))
    print(f"\n  ✅ Saved to: {out_path}")

    # Write-back: inject critique into the original paper JSON
    enriched_path = save_enriched_paper_json(result, args.input,data,verbose=args.verbose)
    if enriched_path:
        print(f"  ✅ Enriched paper JSON: {enriched_path}\n")
    else:
        print(f"  ⚠ Write-back skipped (original file not found)\n")


if __name__ == "__main__":
    main()