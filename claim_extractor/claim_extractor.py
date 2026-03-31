import json
import time
import re
import os
from typing import List, Dict, Any

import requests


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a scientific information extraction system for academic research papers.
Your job is to extract structured information from any research paper section, regardless of domain.
You ALWAYS respond with valid JSON only. No preamble, no explanation, no markdown fences.
"""

SECTION_EXTRACTION_PROMPT = """You are analyzing a section from a research paper.

SECTION TYPE: {section_type}
SECTION HEADING: {section_heading}

SECTION TEXT:
{section_text}

CANDIDATE ENTITIES (from a preliminary NER pass — may contain noise and tokenization errors):
{ner_hints}

YOUR TASK:
Extract the following from this section as a JSON object with exactly these keys:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. "entities"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clean, normalized named entities ACTUALLY mentioned in the text.

NORMALIZATION RULES:
  - Merge tokenization fragments: "Ran" + "##dom" + "##Forest" → "RandomForest"
  - Normalize spacing/punctuation: "SVM - RBF" → "SVM-RBF"
  - Use the most specific name when a specific variant is mentioned:
    prefer "BERT-large" over "BERT", "ResNet-50" over "ResNet"
  - Deduplicate abbreviations: if "CNN" and "Convolutional Neural Network" refer
    to the same thing in the text, keep only the spelled-out form

ENTITY TYPES — exactly one of these four:

  "method"  : A named model, algorithm, architecture, technique, or procedure.
              Domain examples:
                ML/CS   → "Transformer", "BERT", "Random Forest", "Adam optimizer"
                Bio/Med → "CRISPR-Cas9", "Western blot", "PCR", "Kaplan-Meier"
                Physics → "Monte Carlo simulation", "Density Functional Theory"
                General → any named procedure, system, or algorithmic approach
              NOT a method:
                - Generic infrastructure or hardware ("GPU cluster", "96-well plate")
                - Phenomena or challenges ("overfitting", "vanishing gradient",
                  "batch effects", "selection bias")
                - Vague descriptors ("large model", "deep network", "novel approach")

  "dataset" : A named corpus, benchmark, database, or collection used for evaluation or training.
              Domain examples:
                ML/CS   → "ImageNet", "SQuAD", "Penn Treebank"
                Bio/Med → "TCGA", "UK Biobank", "UniProt"
                Physics → "Sloan Digital Sky Survey", "LHC Run 2 dataset"
              NOT a dataset:
                - Generic references ("the training set", "our collected data",
                  "publicly available data")

  "metric"  : A named quantitative measure used to evaluate or report results.
              Domain examples:
                ML/CS   → "perplexity", "F1 score", "BLEU", "accuracy", "AUC-ROC"
                Bio/Med → "sensitivity", "specificity", "p-value", "hazard ratio"
                Physics → "signal-to-noise ratio", "chi-squared", "luminosity"
              NOT a metric:
                - Vague terms ("performance", "results", "improvement", "score")
                - Raw measured quantities that are not evaluation metrics
                  ("temperature", "sample size" unless used as an eval criterion)

  "task"    : A concrete, named problem or application the method is applied to.
              Domain examples:
                ML/CS   → "machine translation", "object detection", "link prediction"
                Bio/Med → "protein structure prediction", "tumour classification",
                           "drug response prediction"
                Physics → "gravitational wave detection", "galaxy classification"
              NOT a task:
                - A property of a method ("model parallelism", "generalization")
                - A challenge or difficulty ("handling class imbalance",
                  "learning long-range dependencies")
                - A generic goal ("improving performance", "reducing complexity")

Each entity object:
  {{
    "text": "<canonical name>",
    "entity_type": "method" | "dataset" | "metric" | "task",
    "confidence": <float 0.0–1.0>
  }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. "claims"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Factual assertions made in this section that involve at least one entity from
your entities list above. Do not include claims with no entities_involved.

CLAIM TYPES:

  "performance"  : A concrete result or measurement achieved by a specific method.
                   Always capture the numeric value when one is stated.
                   Examples across domains:
                     "BERT-large achieves 93.2 F1 on SQuAD 2.0."         → value: 93.2
                     "The proposed classifier reaches 97% sensitivity."   → value: 97
                     "Monte Carlo simulation converges in 1,200 steps."   → value: 1200
                     "Drug X reduces tumour size by 43% in treated mice." → value: 43

  "comparative"  : A direct comparison between two or more methods, models,
                   or approaches on the same criterion. has terms like "outperforms", "improves over", "is better than", "is faster than", "differs from", etc.
                   Examples:
                     "Method A outperforms baseline B by 3.2 points on metric C."
                     "Our approach uses 4× fewer parameters than the prior state of the art."
                     "Model X differs from Model Y in that it does not require labelled data."

  "methodological": How a method works, its design rationale, or its theoretical properties.
                   Examples:
                     "The encoder uses self-attention over variable-length sequences."
                     "CRISPR-Cas9 introduces a double-strand break at the target locus."
                     "The Kalman filter assumes linear dynamics and Gaussian noise."

TABLE EXTRACTION RULE:
  If the section contains a results table, extract ONE "performance" claim per data row
  using the exact numeric values from the table. Do not summarize multiple rows into one claim.
  Extract every quantitative column (accuracy, speed, parameter count, p-value, etc.)
  as a separate claim with its own numeric value.

Each claim object:
  {{
    "claim_type": "performance" | "comparative" | "methodological",
    "description": "<one clear declarative sentence>",
    "entities_involved": ["<entity text>", ...],
    "value": <number | null>,
    "confidence": <float 0.0–1.0>
  }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. "limitations"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY statements where the AUTHORS explicitly acknowledge that their OWN proposed
method, experiment, or finding has a weakness, failed to show a benefit, or is
constrained in scope.

THESE ARE limitations ✓
  - "Our method did not improve over the baseline on dataset X."
  - "We observed degraded performance when [condition] was applied."
  - "This approach is limited to [specific constraint or assumption]."
  - "We did not find statistically significant results for [outcome]."
  - "The proposed model underperforms on [task] compared to [baseline]."
  - "[Our method] showed no benefit in our initial experiments."

THESE ARE NOT limitations ✗
  - Background challenges of the field ("class imbalance is a known problem")
  - Motivation for the work ("existing methods are slow")
  - Limitations of prior or baseline work
  - General statements about difficulty ("this task is inherently hard")
  - Scope statements that are not admissions of failure
    ("we focus only on English" is a scope choice, not a failure)

Each limitation object:
  {{
    "text": "<one sentence, as close to the original wording as possible>",
    "entities_involved": ["<entity text>", ...],
    "confidence": <float 0.0–1.0>
  }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. "future_work"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Statements where the AUTHORS explicitly propose, suggest, or leave something
for their own future investigation. A statement belongs in "future_work" if it proposes exploration.
It belongs in "limitations" only if it admits a failure or constraint.
Never place the same sentence in both lists.

Signal phrases to look for:
  "future work", "we plan to", "we intend to", "remains to be explored",
  "could be extended to", "an interesting direction", "left for future research",
  "it might be possible to", "we will investigate", "warrants further study"

THESE ARE future work ✓
  - "We plan to extend this approach to multilingual settings."
  - "Applying this method to [domain] remains a topic of future research."
  - "It would be interesting to explore [X] in future work."

THESE ARE NOT future work ✗
  - General open problems ("the field still lacks X")
  - What other researchers might do ("others could explore")
  - Background motivation statements

Each future_work object:
  {{
    "text": "<one sentence, as close to the original wording as possible>",
    "entities_involved": ["<entity text>", ...],
    "confidence": <float 0.0–1.0>
  }}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Return ONLY a single valid JSON object. No markdown, no explanation, nothing before or after.
- Empty lists must be [] not null.
- Do not invent content not supported by the text.
- entity text values in claims/limitations/future_work must exactly match
  the "text" field of an entity in your entities list.
- Confidence reflects how certain you are the item is correctly extracted
  and grounded in the section text.

OUTPUT FORMAT:
{{
  "entities": [...],
  "claims": [...],
  "limitations": [...],
  "future_work": [...]
}}"""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, requests_per_minute: int = 12):
        self.min_interval = 60.0 / requests_per_minute
        self.last_call = 0.0

    def wait(self):
        elapsed = time.time() - self.last_call
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_call = time.time()


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class LLMClaimExtractor:

    MODEL_NAME = "qwen2.5"
    MAX_RETRIES = 6
    RETRY_BACKOFF = 15   # base seconds; multiplied by attempt number

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        requests_per_minute: int = 12,
    ):
        self.ollama_host = ollama_host.rstrip("/")
        self.rate_limiter = RateLimiter(requests_per_minute)
        self._verify_connection()
        print(f"[ClaimExtractor] Model : {self.MODEL_NAME}")
        print(f"[ClaimExtractor] Host  : {self.ollama_host}")

    def _verify_connection(self):
        """Quick ping to confirm Ollama is reachable."""
        try:
            r = requests.get(f"{self.ollama_host}/api/tags", timeout=10,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.ollama_host}. "
                f"Check OLLAMA_HOST or --ollama-host. Error: {e}"
            )

    def _call_ollama(self, prompt: str) -> str:
        """
        Call Ollama /api/generate with format='json' so the model is
        constrained to produce valid JSON output.
        """
        payload = {
            "model": self.MODEL_NAME,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "format": "json",          # Ollama-native JSON mode
            "options": {
                "temperature": 0.0,    # Deterministic output
                "num_predict": 4096,
            },
        }
        response = requests.post(
            f"{self.ollama_host}/api/generate",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0",
                "ngrok-skip-browser-warning": "true", 
            },
            json=payload,
            timeout=300,               # Large sections may take time
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    def extract_from_section(
        self,
        section: Dict[str, Any],
        ner_hints: List[Dict[str, Any]],
    ) -> Dict[str, Any]:

        section_type    = section.get("section_type", "Unknown")
        section_heading = section.get("heading", "")
        section_text    = section.get("text", "").strip()

        if not section_text:
            return self._empty_result(section_type, section_heading)

        hints_str = self._format_ner_hints(ner_hints)
        prompt = SECTION_EXTRACTION_PROMPT.format(
            section_type=section_type,
            section_heading=section_heading,
            section_text=section_text,
            ner_hints=hints_str,
        )

        self.rate_limiter.wait()
        print(f"[ClaimExtractor] Processing: {section_type} ({len(section_text)} chars)")

        for attempt in range(self.MAX_RETRIES):
            try:
                raw = self._call_ollama(prompt)

                # Strip any accidental markdown fences
                raw = re.sub(r"^```json\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                parsed = json.loads(raw)
                result = self._validate_and_clean(parsed, section_type, section_heading)
                print(
                    f"[ClaimExtractor]   -> {len(result['entities'])} entities, "
                    f"{len(result['claims'])} claims, "
                    f"{len(result['limitations'])} limitations, "
                    f"{len(result['future_work'])} future_work"
                )
                return result

            except json.JSONDecodeError as e:
                print(f"[ClaimExtractor] JSON parse error in {section_type}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (attempt + 1)
                    print(f"[ClaimExtractor] Retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                return self._empty_result(section_type, section_heading, error=str(e))

            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else "?"
                # 503 = tunnel temporarily unavailable; 429 = rate limit — both are transient
                if status in (429, 503) and attempt < self.MAX_RETRIES - 1:
                    wait = self.RETRY_BACKOFF * (attempt + 1)
                    reason = "rate limit" if status == 429 else "tunnel unavailable (503)"
                    print(f"[ClaimExtractor] {status} {reason} on '{section_type}' "
                          f"(attempt {attempt+1}/{self.MAX_RETRIES}). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"[ClaimExtractor] HTTP error in {section_type}: {e}")
                return self._empty_result(section_type, section_heading, error=str(e))

            except Exception as e:
                print(f"[ClaimExtractor] Error in {section_type}: {str(e)[:200]}")
                return self._empty_result(section_type, section_heading, error=str(e)[:200])

        return self._empty_result(section_type, section_heading, error="max retries exceeded")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_ner_hints(self, ner_entities: List[Dict]) -> str:
        if not ner_entities:
            return "None"
        lines = []
        for e in ner_entities:
            lines.append(
                f"  - [{e.get('entity_type','?')}] {e.get('text','')} "
                f"(conf={e.get('confidence',0):.2f})"
            )
        return "\n".join(lines)

    def _validate_and_clean(self, parsed, section_type, section_heading):
        entities = []
        for e in parsed.get("entities", []):
            if not isinstance(e, dict):
                continue
            text  = str(e.get("text", "")).strip()
            etype = str(e.get("entity_type", "")).lower()
            conf  = float(e.get("confidence", 0.0))
            if text and etype in ("method", "dataset", "metric", "task") and conf >= 0.5:
                entities.append({
                    "text": text,
                    "entity_type": etype,
                    "confidence": round(conf, 4),
                    "section_type": section_type,
                    "section_heading": section_heading,
                    "source": "llm",
                })

        claims = []
        for c in parsed.get("claims", []):
            if not isinstance(c, dict):
                continue
            desc     = str(c.get("description", "")).strip()
            ctype    = str(c.get("claim_type", "")).lower()
            involved = [str(x) for x in c.get("entities_involved", []) if str(x).strip()]
            conf     = float(c.get("confidence", 0.0))
            value    = c.get("value")
            if desc and ctype in ("performance", "comparative", "methodological") and involved:
                claims.append({
                    "claim_type": ctype,
                    "description": desc,
                    "entities_involved": involved,
                    "value": value,
                    "confidence": round(conf, 4),
                    "section_type": section_type,
                    "section_heading": section_heading,
                    "source": "llm",
                })

        limitations = []
        for lim in parsed.get("limitations", []):
            if not isinstance(lim, dict):
                continue
            text     = str(lim.get("text", "")).strip()
            involved = [str(x) for x in lim.get("entities_involved", []) if str(x).strip()]
            conf     = float(lim.get("confidence", 0.0))
            if text:
                limitations.append({
                    "text": text,
                    "entities_involved": involved,
                    "confidence": round(conf, 4),
                    "section_type": section_type,
                    "section_heading": section_heading,
                    "source": "llm",
                })

        future_work = []
        for fw in parsed.get("future_work", []):
            if not isinstance(fw, dict):
                continue
            text     = str(fw.get("text", "")).strip()
            involved = [str(x) for x in fw.get("entities_involved", []) if str(x).strip()]
            conf     = float(fw.get("confidence", 0.0))
            if text:
                future_work.append({
                    "text": text,
                    "entities_involved": involved,
                    "confidence": round(conf, 4),
                    "section_type": section_type,
                    "section_heading": section_heading,
                    "source": "llm",
                })

        return {
            "section_type": section_type,
            "section_heading": section_heading,
            "entities": entities,
            "claims": claims,
            "limitations": limitations,
            "future_work": future_work,
            "error": None,
        }

    def _empty_result(self, section_type, section_heading, error=None):
        return {
            "section_type": section_type,
            "section_heading": section_heading,
            "entities": [],
            "claims": [],
            "limitations": [],
            "future_work": [],
            "error": error,
        }