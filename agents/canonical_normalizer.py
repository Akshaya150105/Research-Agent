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

#  LLM CACHE 
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


#  LAYER 1 — REGEX PATTERN RULES
# Language pair normalization table - Maps various ways to write a language pair → standard code
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
    for pattern, code in _LANG_PAIRS:
        s = pattern.sub(code, s)
    return s


def _normalize_metric(raw: str) -> Optional[str]:
    
    s = raw.lower().strip()

    # strip common suffix noise
    s = re.sub(r"\s+(score|rate|value|metric|result|measure)s?$", "", s).strip()

    # perplexity family
    if re.match(r"(val(idation)?|test|dev|train(ing)?)[\s\-]?(ppl|perplexity)$", s):
        return "perplexity"
    if re.match(r"ppl$", s):
        return "perplexity"

    # loss family
    if re.match(r"(negative[\s\-]?log[\s\-]?likelihood|cross[\s\-]?entropy|"
                r"(val(idation)?|test|train(ing)?)[\s\-]?loss|nll)$", s):
        return "nll"

    # BLEU family
    if re.match(r"(tokenized|detokenized|uncased|cased|multi[\s\-]?ref)?[\s\-]?"
                r"(sacre)?bleu[\s\-]?\d*$", s):
        return "bleu"

    # ROUGE family 
    m = re.match(r"rouge[\s\-]?([l\d]+)$", s)
    if m:
        variant = m.group(1).replace(" ", "")
        return f"rouge-{variant}"
    if re.match(r"rouge$", s):
        return "rouge-l"  

    # F-score family
    m = re.match(r"(macro[\s\-]?|micro[\s\-]?|weighted[\s\-]?)?f[\s\-]?(\d)$", s)
    if m:
        prefix = m.group(1)
        num    = m.group(2)
        if prefix:
            return f"f{num}-{prefix.strip().rstrip('-')}"
        return f"f{num}"

    # accuracy family
    m = re.match(r"top[\s\-]?(\d)[\s\-]?(accuracy|acc)?$", s)
    if m:
        return f"top-{m.group(1)}"
    if re.match(r"(test[\s\-]?|val[\s\-]?|train[\s\-]?)?acc(uracy)?$", s):
        return "accuracy"

    # WER/CER family
    if re.match(r"(word[\s\-]?error[\s\-]?rate|wer)$", s):
        return "wer"
    if re.match(r"(char(acter)?[\s\-]?error[\s\-]?rate|cer)$", s):
        return "cer"

    # training time / throughput
    if re.match(r"(training|inference)[\s\-]?time$", s):
        return s.replace(" ", "-")

    return s if s != raw.lower().strip() else None


def _normalize_dataset(raw: str) -> Optional[str]:
    
    s = raw.lower().strip()

    # strip trailing noise
    s = re.sub(r"\s+(dataset|corpus|data|set|benchmark|task|"
               r"collection|test set|training set)s?$", "", s).strip()

    # WMT family 
    m = re.match(
        r"wmt[\s'\u2019\u0060]?"   
        r"(\d{2,4})"               
        r"[\s\-]?"                 
        r"(.*)$",                  
        s, re.IGNORECASE
    )
    if m:
        year = m.group(1)
        year = year[-2:]        
        rest = m.group(2).strip().strip("-").strip()
        rest = _normalize_lang_pair(rest) if rest else ""
        if rest:
            return f"wmt{year}-{rest}"
        return f"wmt{year}"

    if re.match(r"^wmt$", s):
        return "wmt"

    # newstest 
    m = re.match(r"(newstest|ntst)[\s\-]?(\d{2,4})$", s, re.IGNORECASE)
    if m:
        year = m.group(2)
        year = f"20{year}" if len(year) == 2 else year
        return f"newstest{year}"

    # SQuAD family
    m = re.match(r"squad[\s\-]?v?(\d)[\.\d]*$", s, re.IGNORECASE)
    if m:
        return f"squad-v{m.group(1)}"
    if re.match(r"squad$", s, re.IGNORECASE):
        return "squad-v1"  

    # GLUE family 
    glue_tasks = {"mnli", "snli", "qnli", "qqp", "cola",
                  "mrpc", "rte", "wnli", "stsb", "sst"}
    for task in glue_tasks:
        if re.match(rf"(glue[\s\-]?)?{task}[\s\-]?(matched|mismatched|\d)?$",
                    s, re.IGNORECASE):
            clean = re.sub(r"[\s\-](matched|mismatched)$", "", s).strip()
            clean = re.sub(r"^glue[\s\-]?", "", clean).strip()
            clean = re.sub(r"[\s\-]", "", clean)
            return f"glue-{clean}"

    # Penn Treebank / WSJ
    if re.match(r"(penn[\s\-]?treebank|ptb|wall[\s\-]?street[\s\-]?journal|wsj)$",
                s, re.IGNORECASE):
        return "penn-treebank"

    # ImageNet 
    if re.match(r"(imagenet|ilsvrc)[\s\-]?(1k|2012)?$", s, re.IGNORECASE):
        return "imagenet-1k"

    # IWSLT
    m = re.match(r"iwslt[\s'\u2019]?(\d{2,4})[\s\-]?(.*)$", s, re.IGNORECASE)
    if m:
        year = m.group(1)[-2:]
        rest = _normalize_lang_pair(m.group(2).strip()) if m.group(2).strip() else ""
        return f"iwslt{year}-{rest}" if rest else f"iwslt{year}"

    # CIFAR
    m = re.match(r"cifar[\s\-]?(\d+)$", s, re.IGNORECASE)
    if m:
        return f"cifar-{m.group(1)}"

    return s if s != raw.lower().strip() else None



#  LAYER 2 — LLM FALLBACK
_GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta"
    "/models/gemini-2.5-flash:generateContent"
)


def _call_llm_normalize(raw: str, entity_type: str) -> Optional[str]:

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


def normalize(raw: str, entity_type: str, use_llm: bool = True) -> str:
   
    if not raw or not raw.strip():
        return raw

    # regex rules
    if entity_type == "metric":
        result = _normalize_metric(raw)
    elif entity_type == "dataset":
        result = _normalize_dataset(raw)
    else:
        result = None

    if result is not None:
        return result

    # LLM fallback 
    if use_llm and entity_type in ("metric", "dataset"):
        llm_result = _call_llm_normalize(raw, entity_type)
        if llm_result:
            return llm_result

    return raw.lower().strip()


def normalize_resolved(resolved: dict) -> dict:
  
    out = dict(resolved)
    if out.get("metric"):
        out["metric"] = normalize(out["metric"], "metric")
    if out.get("dataset"):
        out["dataset"] = normalize(out["dataset"], "dataset")
    return out

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
        ("WMT'20 English-Chinese",            "dataset", "wmt20-en-zh"),  
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