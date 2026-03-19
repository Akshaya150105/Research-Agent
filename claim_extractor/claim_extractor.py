import json
import time
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a scientific information extraction system.
Your job is to extract structured information from research paper sections.
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

1. "entities": Clean, normalized list of real named entities actually mentioned in the text.
   Fix NER errors: merge fragments (e.g. "Rec" + "##urrent Neural Networks" -> "Recurrent Neural Networks"),
   remove noise (e.g. "large", "training of", "models" are NOT entities),
   normalize names (e.g. "F - LSTM" -> "F-LSTM", "G - LSTM" -> "G-LSTM").
   Each entity must have:
     - "text": normalized name (e.g. "F-LSTM", "BIGLSTM", "One Billion Word Benchmark")
     - "entity_type": one of "method", "dataset", "metric", "task"
     - "confidence": float 0.0-1.0

2. "claims": Structured claims made in this section. Only include claims that involve
   at least one entity from your cleaned entities list. Each claim must have:
     - "claim_type": one of "performance", "comparative", "methodological"
     - "description": one clear sentence describing the claim
     - "entities_involved": list of entity texts from your entities list
     - "value": numeric value if mentioned (e.g. 35.1 for perplexity), else null
     - "confidence": float 0.0-1.0

3. "limitations": Atomic limitation statements found in this section.
   Only sentences that explicitly admit a weakness or constraint of the work.
   Each must have:
     - "text": the limitation statement (one sentence)
     - "entities_involved": list of entity texts
     - "confidence": float 0.0-1.0

4. "future_work": Atomic future work statements found in this section.
   Only sentences that propose something to explore in future.
   Each must have:
     - "text": the future work statement (one sentence)
     - "entities_involved": list of entity texts
     - "confidence": float 0.0-1.0

STRICT RULES:
- Return ONLY valid JSON. No markdown, no explanation before or after.
- If a list is empty, return [] not null.
- Do not invent claims not supported by the text.
- Do not include claims with no entities_involved.
- Keep entity names consistent across all four lists.

JSON SCHEMA:
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

    MODEL_NAME = "gemini-2.5-flash"
    MAX_RETRIES = 4
    RETRY_BACKOFF = 10

    def __init__(self, api_key: str, requests_per_minute: int = 12):
        self.client = genai.Client(api_key=api_key)
        self.rate_limiter = RateLimiter(requests_per_minute)
        print(f"[ClaimExtractor] Loaded model: {self.MODEL_NAME}")

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
                response = self.client.models.generate_content(
                    model=self.MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                    ),
                )
                raw = response.text.strip()
                raw = re.sub(r"^```json\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)

                parsed = json.loads(raw)
                result = self._validate_and_clean(parsed, section_type, section_heading)
                print(f"[ClaimExtractor]   -> {len(result['entities'])} entities, "
                      f"{len(result['claims'])} claims, "
                      f"{len(result['limitations'])} limitations, "
                      f"{len(result['future_work'])} future_work")
                return result

            except json.JSONDecodeError as e:
                print(f"[ClaimExtractor] JSON parse error in {section_type}: {e}")
                return self._empty_result(section_type, section_heading, error=str(e))

            except Exception as e:
                err_str = str(e)
                # Extract retry delay from 429 response if present
                retry_match = re.search(r"retryDelay.*?'(\d+)s'", err_str)
                wait = int(retry_match.group(1)) + self.RETRY_BACKOFF if retry_match else 60
                if "429" in err_str and attempt < self.MAX_RETRIES - 1:
                    print(f"[ClaimExtractor] 429 rate limit on {section_type} "
                          f"(attempt {attempt+1}/{self.MAX_RETRIES}). "
                          f"Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"[ClaimExtractor] API error in {section_type}: {err_str[:200]}")
                return self._empty_result(section_type, section_heading, error=err_str[:200])

        return self._empty_result(section_type, section_heading, error="max retries exceeded")

    def _format_ner_hints(self, ner_entities: List[Dict]) -> str:
        if not ner_entities:
            return "None"
        lines = []
        for e in ner_entities:
            lines.append(f"  - [{e.get('entity_type','?')}] {e.get('text','')} "
                         f"(conf={e.get('confidence',0):.2f})")
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
                    "text": text, "entity_type": etype,
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
                    "claim_type": ctype, "description": desc,
                    "entities_involved": involved, "value": value,
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
                    "text": text, "entities_involved": involved,
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
                    "text": text, "entities_involved": involved,
                    "confidence": round(conf, 4),
                    "section_type": section_type,
                    "section_heading": section_heading,
                    "source": "llm",
                })

        return {
            "section_type": section_type, "section_heading": section_heading,
            "entities": entities, "claims": claims,
            "limitations": limitations, "future_work": future_work,
            "error": None,
        }

    def _empty_result(self, section_type, section_heading, error=None):
        return {
            "section_type": section_type, "section_heading": section_heading,
            "entities": [], "claims": [], "limitations": [], "future_work": [],
            "error": error,
        }