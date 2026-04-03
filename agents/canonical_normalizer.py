"""
canonical_normalizer.py  v1.0.0
================================
Folder: agents/

Purpose
-------
Cross-paper entity normalization so that:
  "BLEU score"  ==  "bleu"  ==  "BLEU-4"       → "bleu"
  "WMT'14 En-De" == "wmt 2014 english-german"   → "wmt14-en-de"

Two layers:
  Layer 1 — Regex pattern rules (fast, zero hardcoded values, generalizes)
  Layer 2 — LLM fallback (only called when Layer 1 fails, result cached to disk)

The LLM cache file (agents/llm_norm_cache.json) grows automatically.
After ~20 papers it rarely needs to call the LLM at all.

Usage
-----
  from canonical_normalizer import normalize

  normalize("BLEU score",       "metric")   → "bleu"
  normalize("WMT'14 En-De",     "dataset")  → "wmt14-en-de"
  normalize("val ppl",          "metric")   → "perplexity"
  normalize("SQuAD 1.1",        "dataset")  → "squad-v1"
  normalize("top-1 accuracy",   "metric")   → "top-1"
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import time
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────
#  LLM CACHE  (persists to disk so LLM is called at most once per string)
# ─────────────────────────────────────────────────────────────

_CACHE_PATH = pathlib.Path(__file__).parent / "llm_norm_cache.json"
_llm_cache: dict[str, str] = {}


def _load_cache() -> None:
    global _llm_cache
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, encoding="utf-8") as f:
                _llm_cache = json.load(f)
        except Exception:
            _llm_cache = {}


def _save_cache() -> None:
    try:
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_llm_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


_load_cache()

# ─────────────────────────────────────────────────────────────
#  LAYER 1 — REGEX PATTERN RULES
#
#  These are RULES, not hardcoded values.
#  Each rule is a transformation that generalizes across all papers.
#  e.g. the WMT rule handles wmt14, wmt15, wmt16 ... wmt23 automatically.
# ─────────────────────────────────────────────────────────────

# Language pair normalization table
# Maps various ways to write a language pair → standard code
# Add new pairs here as you add papers from other domains
_LANG_PAIRS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"english[\s\-]?(to[\s\-])?german",  re.I), "en-de"),
    (re.compile(r"german[\s\-]?(to[\s\-])?english",  re.I), "de-en"),
    (re.compile(r"english[\s\-]?(to[\s\-])?french",  re.I), "en-fr"),
    (re.compile(r"french[\s\-]?(to[\s\-])?english",  re.I), "fr-en"),
    (re.compile(r"english[\s\-]?(to[\s\-])?chinese", re.I), "en-zh"),
    (re.compile(r"english[\s\-]?(to[\s\-])?japanese",re.I), "en-ja"),
    (re.compile(r"english[\s\-]?(to[\s\-])?romanian",re.I), "en-ro"),
    (re.compile(r"\ben[\s\-]de\b",                   re.I), "en-de"),
    (re.compile(r"\ben[\s\-]fr\b",                   re.I), "en-fr"),
    (re.compile(r"\ben[\s\-]zh\b",                   re.I), "en-zh"),
    (re.compile(r"\bde[\s\-]en\b",                   re.I), "de-en"),
]


def _normalize_lang_pair(s: str) -> str:
    """Replace any language pair mention with standard xx-yy code."""
    for pattern, code in _LANG_PAIRS:
        s = pattern.sub(code, s)
    return s


def _normalize_metric(raw: str) -> Optional[str]:
    """
    Apply metric normalization rules.
    Returns normalized string, or None if no rule matched.
    All rules are patterns — nothing is a specific hardcoded value.
    """
    s = raw.lower().strip()

    # Rule: strip common suffix noise
    # "bleu score" → "bleu",  "f1 value" → "f1",  "accuracy metric" → "accuracy"
    s = re.sub(r"\s+(score|rate|value|metric|result|measure)s?$", "", s).strip()

    # Rule: perplexity family
    # "val ppl", "validation perplexity", "test ppl", "dev perplexity" → "perplexity"
    if re.match(r"(val(idation)?|test|dev|train(ing)?)[\s\-]?(ppl|perplexity)$", s):
        return "perplexity"
    if re.match(r"ppl$", s):
        return "perplexity"

    # Rule: loss family
    # "nll", "negative log likelihood", "cross entropy loss", "val loss" → "nll"
    if re.match(r"(negative[\s\-]?log[\s\-]?likelihood|cross[\s\-]?entropy|"
                r"(val(idation)?|test|train(ing)?)[\s\-]?loss|nll)$", s):
        return "nll"

    # Rule: BLEU family
    # "bleu-4", "bleu4", "tokenized bleu", "detokenized bleu",
    # "uncased bleu", "sacrebleu" → "bleu"
    if re.match(r"(tokenized|detokenized|uncased|cased|multi[\s\-]?ref)?[\s\-]?"
                r"(sacre)?bleu[\s\-]?\d*$", s):
        return "bleu"

    # Rule: ROUGE family — preserve number/letter variant
    # "rouge-l" → "rouge-l", "rouge1" → "rouge-1", "rougel" → "rouge-l"
    m = re.match(r"rouge[\s\-]?([l\d]+)$", s)
    if m:
        variant = m.group(1).replace(" ", "")
        return f"rouge-{variant}"
    if re.match(r"rouge$", s):
        return "rouge-l"  # default to rouge-l when unspecified

    # Rule: F-score family
    # "f-1", "f 1", "f1 score", "macro-f1", "micro f1"
    m = re.match(r"(macro[\s\-]?|micro[\s\-]?|weighted[\s\-]?)?f[\s\-]?(\d)$", s)
    if m:
        prefix = m.group(1)
        num    = m.group(2)
        if prefix:
            return f"f{num}-{prefix.strip().rstrip('-')}"
        return f"f{num}"

    # Rule: accuracy family
    # "top-1 accuracy", "top1 acc", "top-5 acc" → "top-1" / "top-5"
    m = re.match(r"top[\s\-]?(\d)[\s\-]?(accuracy|acc)?$", s)
    if m:
        return f"top-{m.group(1)}"
    if re.match(r"(test[\s\-]?|val[\s\-]?|train[\s\-]?)?acc(uracy)?$", s):
        return "accuracy"

    # Rule: WER/CER family
    if re.match(r"(word[\s\-]?error[\s\-]?rate|wer)$", s):
        return "wer"
    if re.match(r"(char(acter)?[\s\-]?error[\s\-]?rate|cer)$", s):
        return "cer"

    # Rule: training time / throughput — keep as-is but normalized
    if re.match(r"(training|inference)[\s\-]?time$", s):
        return s.replace(" ", "-")

    # No rule matched — return cleaned string as-is
    # (LLM fallback will handle it if needed)
    return s if s != raw.lower().strip() else None


def _normalize_dataset(raw: str) -> Optional[str]:
    """
    Apply dataset normalization rules.
    Returns normalized string, or None if no rule matched.
    All rules are patterns — nothing is a specific hardcoded value.
    """
    s = raw.lower().strip()

    # Rule: strip trailing noise
    # "wmt14 dataset" → "wmt14",  "squad corpus" → "squad"
    s = re.sub(r"\s+(dataset|corpus|data|set|benchmark|task|"
               r"collection|test set|training set)s?$", "", s).strip()

    # Rule: WMT family — handles ANY year, ANY language pair
    # "WMT'14 En-De" → "wmt14-en-de"
    # "WMT 2014 English-German" → "wmt14-en-de"
    # "WMT'16 En-Fr" → "wmt16-en-fr"
    # "WMT 2020 English-Chinese" → "wmt20-en-zh"  ← never hardcoded
    m = re.match(
        r"wmt[\s'\u2019\u0060]?"   # "wmt", "wmt'", "wmt`"
        r"(\d{2,4})"               # year: 14, 2014, etc.
        r"[\s\-]?"                 # optional separator
        r"(.*)$",                  # rest: lang pair or empty
        s, re.IGNORECASE
    )
    if m:
        year = m.group(1)
        year = year[-2:]           # "2014" → "14", "14" → "14"
        rest = m.group(2).strip().strip("-").strip()
        rest = _normalize_lang_pair(rest) if rest else ""
        if rest:
            return f"wmt{year}-{rest}"
        return f"wmt{year}"

    # Rule: bare "wmt" → keep as "wmt" (cross-year reference)
    if re.match(r"^wmt$", s):
        return "wmt"

    # Rule: newstest — standard WMT test sets
    # "newstest2014", "newstest14", "ntst14" → "newstest2014"
    m = re.match(r"(newstest|ntst)[\s\-]?(\d{2,4})$", s, re.IGNORECASE)
    if m:
        year = m.group(2)
        year = f"20{year}" if len(year) == 2 else year
        return f"newstest{year}"

    # Rule: SQuAD family
    # "squad 1.1", "squad v1.1", "squad1.1" → "squad-v1"
    # "squad 2.0", "squad v2"               → "squad-v2"
    m = re.match(r"squad[\s\-]?v?(\d)[\.\d]*$", s, re.IGNORECASE)
    if m:
        return f"squad-v{m.group(1)}"
    if re.match(r"squad$", s, re.IGNORECASE):
        return "squad-v1"  # default version

    # Rule: GLUE family — preserve task name
    # "glue-mnli", "mnli-matched" → "glue-mnli"
    # "sst-2", "sst2"             → "glue-sst2"
    glue_tasks = {"mnli", "snli", "qnli", "qqp", "cola",
                  "mrpc", "rte", "wnli", "stsb", "sst"}
    for task in glue_tasks:
        if re.match(rf"(glue[\s\-]?)?{task}[\s\-]?(matched|mismatched|\d)?$",
                    s, re.IGNORECASE):
            clean = re.sub(r"[\s\-](matched|mismatched)$", "", s).strip()
            clean = re.sub(r"^glue[\s\-]?", "", clean).strip()
            clean = re.sub(r"[\s\-]", "", clean)
            return f"glue-{clean}"

    # Rule: Penn Treebank / WSJ
    # "penn treebank", "ptb", "wsj", "wall street journal" → "penn-treebank"
    if re.match(r"(penn[\s\-]?treebank|ptb|wall[\s\-]?street[\s\-]?journal|wsj)$",
                s, re.IGNORECASE):
        return "penn-treebank"

    # Rule: ImageNet family
    # "imagenet", "ilsvrc", "imagenet-1k", "imagenet 2012" → "imagenet-1k"
    if re.match(r"(imagenet|ilsvrc)[\s\-]?(1k|2012)?$", s, re.IGNORECASE):
        return "imagenet-1k"

    # Rule: IWSLT family — handles ANY year, ANY language pair
    # "IWSLT14 De-En" → "iwslt14-de-en"
    m = re.match(r"iwslt[\s'\u2019]?(\d{2,4})[\s\-]?(.*)$", s, re.IGNORECASE)
    if m:
        year = m.group(1)[-2:]
        rest = _normalize_lang_pair(m.group(2).strip()) if m.group(2).strip() else ""
        return f"iwslt{year}-{rest}" if rest else f"iwslt{year}"

    # Rule: CIFAR family
    m = re.match(r"cifar[\s\-]?(\d+)$", s, re.IGNORECASE)
    if m:
        return f"cifar-{m.group(1)}"

    # No rule matched
    return s if s != raw.lower().strip() else None


# ─────────────────────────────────────────────────────────────
#  LAYER 2 — LLM FALLBACK
# ─────────────────────────────────────────────────────────────

_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.5-flash:generateContent"
)


def _call_llm_normalize(raw: str, entity_type: str) -> Optional[str]:
    """
    Ask Gemini to normalize a single entity string.
    Called ONLY when Layer 1 regex rules return None.
    Result cached to disk immediately so it's never called twice for the same input.
    """
    cache_key = f"{entity_type}::{raw.lower().strip()}"
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None  # no key — skip silently

    prompt = f"""You are normalizing {entity_type} names from ML research papers.
Convert the input to a short, lowercase, canonical form.

Rules:
- metrics: remove trailing "score/rate/value", normalize variants to one name
- datasets: use format like "wmt14-en-de", "squad-v1", "imagenet-1k"
- Return ONLY the canonical string, nothing else, no explanation

Examples for metric:
  "BLEU score"           → bleu
  "val perplexity"       → perplexity  
  "Top-1 Accuracy"       → top-1
  "Negative Log Likelihood" → nll

Examples for dataset:
  "WMT 2014 English-German dataset" → wmt14-en-de
  "SQuAD version 1.1"               → squad-v1
  "Penn Treebank"                   → penn-treebank

Input: "{raw}"
Canonical form:"""

    try:
        resp = requests.post(
            f"{_GEMINI_URL}?key={api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 32},
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        result = result.strip().lower().strip('"').strip("'")

        # Basic sanity check — reject if LLM returned something too long
        if result and len(result) < 80 and "\n" not in result:
            _llm_cache[cache_key] = result
            _save_cache()
            return result
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────

def normalize(raw: str, entity_type: str, use_llm: bool = True) -> str:
    """
    Normalize a raw entity string to a canonical form.

    Args:
        raw:         The raw string from the paper, e.g. "BLEU score"
        entity_type: "metric" or "dataset"
        use_llm:     Whether to call Gemini if rules don't match (default True)

    Returns:
        Canonical string, e.g. "bleu"
        Falls back to lowercased raw if nothing works.

    Examples:
        normalize("BLEU score",            "metric")  → "bleu"
        normalize("WMT'14 En-De",          "dataset") → "wmt14-en-de"
        normalize("validation perplexity", "metric")  → "perplexity"
        normalize("SQuAD 1.1",             "dataset") → "squad-v1"
        normalize("newstest2014",          "dataset") → "newstest2014"
        normalize("Top-1 Accuracy",        "metric")  → "top-1"
    """
    if not raw or not raw.strip():
        return raw

    # Layer 1: regex rules
    if entity_type == "metric":
        result = _normalize_metric(raw)
    elif entity_type == "dataset":
        result = _normalize_dataset(raw)
    else:
        result = None

    if result is not None:
        return result

    # Layer 2: LLM fallback (only for metric/dataset, only if key available)
    if use_llm and entity_type in ("metric", "dataset"):
        llm_result = _call_llm_normalize(raw, entity_type)
        if llm_result:
            return llm_result

    # Last resort: clean lowercase
    return raw.lower().strip()


def normalize_resolved(resolved: dict) -> dict:
    """
    Normalize the metric and dataset fields of an already-resolved claim dict.
    Call this at the END of resolve_claim() before returning.

    Input:  {"metric": "bleu score", "dataset": "wmt 2014 english-german dataset", ...}
    Output: {"metric": "bleu",       "dataset": "wmt14-en-de", ...}
    """
    out = dict(resolved)
    if out.get("metric"):
        out["metric"] = normalize(out["metric"], "metric")
    if out.get("dataset"):
        out["dataset"] = normalize(out["dataset"], "dataset")
    return out


# ─────────────────────────────────────────────────────────────
#  QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # (input, type, expected)
        ("BLEU score",                        "metric",  "bleu"),
        ("bleu",                              "metric",  "bleu"),
        ("BLEU-4",                            "metric",  "bleu"),
        ("tokenized BLEU",                    "metric",  "bleu"),
        ("SacreBLEU",                         "metric",  "bleu"),
        ("val perplexity",                    "metric",  "perplexity"),
        ("validation perplexity",             "metric",  "perplexity"),
        ("ppl",                               "metric",  "perplexity"),
        ("NLL",                               "metric",  "nll"),
        ("negative log likelihood",           "metric",  "nll"),
        ("cross entropy",                     "metric",  "nll"),
        ("Top-1 Accuracy",                    "metric",  "top-1"),
        ("top-5 acc",                         "metric",  "top-5"),
        ("F1 score",                          "metric",  "f1"),
        ("macro-F1",                          "metric",  "f1-macro"),
        ("ROUGE-L",                           "metric",  "rouge-l"),
        ("rouge1",                            "metric",  "rouge-1"),
        ("WER",                               "metric",  "wer"),

        ("WMT'14 En-De",                      "dataset", "wmt14-en-de"),
        ("WMT 2014 English-German dataset",   "dataset", "wmt14-en-de"),
        ("wmt 2014 english-german dataset",   "dataset", "wmt14-en-de"),
        ("WMT'14 En-Fr",                      "dataset", "wmt14-en-fr"),
        ("WMT 2016 English-German",           "dataset", "wmt16-en-de"),
        ("WMT'20 English-Chinese",            "dataset", "wmt20-en-zh"),  # never hardcoded!
        ("newstest2014",                      "dataset", "newstest2014"),
        ("ntst14",                            "dataset", "newstest2014"),
        ("SQuAD 1.1",                         "dataset", "squad-v1"),
        ("squad v2.0",                        "dataset", "squad-v2"),
        ("MNLI",                              "dataset", "glue-mnli"),
        ("Penn Treebank",                     "dataset", "penn-treebank"),
        ("WSJ",                               "dataset", "penn-treebank"),
        ("ImageNet",                          "dataset", "imagenet-1k"),
        ("IWSLT14 De-En",                     "dataset", "iwslt14-de-en"),
        ("CIFAR-10",                          "dataset", "cifar-10"),
    ]

    print("Running self-test (no LLM)...\n")
    passed = 0
    for raw, etype, expected in tests:
        got = normalize(raw, etype, use_llm=False)
        ok  = "✅" if got == expected else "❌"
        if got != expected:
            print(f"  {ok} normalize({raw!r}, {etype!r})")
            print(f"       expected: {expected!r}")
            print(f"       got:      {got!r}")
        else:
            passed += 1

    print(f"\n{passed}/{len(tests)} passed")