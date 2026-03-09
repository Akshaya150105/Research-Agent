import json
import re
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

SCIIE_LABEL_MAP = {
    "Task":     "task",
    "Method":   "method",
    "Metric":   "metric",
    "Material": "dataset",   
    "B-Task":     "task",
    "I-Task":     "task",
    "B-Method":   "method",
    "I-Method":   "method",
    "B-Metric":   "metric",
    "I-Metric":   "metric",
    "B-Material": "dataset",
    "I-Material": "dataset",
}

HIGH_PRIORITY_SECTIONS = {
    "Methods", "Experiments", "Results", "Evaluation",
    "Abstract", "Introduction", "Discussion", "Conclusion",
    "Related Work", "Limitations", "Future Work"
}


def normalize_entity_text(raw_text: str) -> str:
    text = raw_text.strip()
    # Remove possessives
    text = re.sub(r"'s$", "", text)
    # Remove trailing noise words that are never part of entity names
    text = re.sub(r"\s+(-based|-style|-like|-specific|-level|-aware)$", "", text, flags=re.IGNORECASE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Strip leading/trailing punctuation (but keep hyphens inside like GPT-4)
    text = text.strip(".,;:()[]{}\"'")
    return text


# ---------------------------------------------------------------------------
# Token → span reconstruction (BIO scheme)
# ---------------------------------------------------------------------------

def bio_tokens_to_spans(
    tokens: List[str],
    labels: List[str],
    offsets: Optional[List[Tuple[int, int]]] = None
) -> List[Dict[str, Any]]:
    """
    Convert BIO-tagged token list into entity spans.
    Handles both B-/I- prefixed labels and plain labels.

    Returns list of dicts:
        {text, label, start_char, end_char, tokens}
    """
    spans = []
    current_tokens = []
    current_label = None
    current_start = None
    current_end = None

    for i, (token, label) in enumerate(zip(tokens, labels)):
        # Normalise label
        clean_label = label.replace("B-", "").replace("I-", "").replace("S-", "")
        is_begin = label.startswith("B-") or label.startswith("S-")
        is_inside = label.startswith("I-")
        is_other = label in ("O", "outside", "Other")

        if is_other:
            if current_tokens:
                spans.append(_build_span(current_tokens, current_label, current_start, current_end))
                current_tokens, current_label, current_start, current_end = [], None, None, None
            continue

        entity_type = SCIIE_LABEL_MAP.get(label) or SCIIE_LABEL_MAP.get(f"B-{clean_label}")
        if entity_type is None:
            # Label not in our map — skip
            if current_tokens:
                spans.append(_build_span(current_tokens, current_label, current_start, current_end))
                current_tokens, current_label, current_start, current_end = [], None, None, None
            continue

        if is_begin or (not is_inside and current_label != entity_type):
            # Flush previous
            if current_tokens:
                spans.append(_build_span(current_tokens, current_label, current_start, current_end))
            current_tokens = [token]
            current_label = entity_type
            current_start = offsets[i][0] if offsets else None
            current_end = offsets[i][1] if offsets else None
        else:
            # Continuation
            current_tokens.append(token)
            if offsets:
                current_end = offsets[i][1]

    if current_tokens:
        spans.append(_build_span(current_tokens, current_label, current_start, current_end))

    return spans


def _build_span(tokens, label, start, end):
    raw_text = " ".join(tokens)
    return {
        "raw_text":   raw_text,
        "text":       normalize_entity_text(raw_text),
        "label":      label,
        "start_char": start,
        "end_char":   end,
        "tokens":     tokens,
    }


# ---------------------------------------------------------------------------
# Main NER runner — wraps HuggingFace pipeline
# ---------------------------------------------------------------------------

class SciBERTNERExtractor:
    """
    Loads the SciBERT NER model once and runs it section-by-section.

    Model: RJuro/SciNERTopic
      - SciBERT fine-tuned on SciERC dataset (publicly available, no auth needed)
      - Recognises: Task, Method, Metric, Material (we map Material → dataset)
      - ~440MB download, cached locally after first run
    """

    DEFAULT_MODEL = "RJuro/SciNERTopic"
    # Max tokens per chunk — SciBERT has 512 token limit
    MAX_TOKENS = 400
    # Stride for sliding window (overlap to not miss cross-boundary entities)
    STRIDE = 50

    def __init__(self, model_name: Optional[str] = None, device: int = -1):
        """
        Args:
            model_name: HuggingFace model ID or local path.
            device: -1 for CPU, 0 for first GPU.
        """
        from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

        self.model_name = model_name or self.DEFAULT_MODEL
        print(f"[NER] Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        # aggregation_strategy="simple" reconstructs word-level spans from subword tokens
        self.pipe = pipeline(
            "ner",
            model=model,
            tokenizer=self.tokenizer,
            aggregation_strategy="simple",
            device=device,
        )
        print(f"[NER] Model loaded on {'CPU' if device == -1 else f'GPU {device}'}")

    def extract_from_section(
        self,
        section_text: str,
        section_type: str,
        section_heading: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Run NER on a single section. Handles long sections via sliding window.

        Returns list of entity dicts with provenance fields attached.
        """
        if not section_text or not section_text.strip():
            return []

        raw_entities = self._run_ner_windowed(section_text)
        entities = []

        seen_normalized = set()  # deduplicate within section

        for ent in raw_entities:
            label = ent.get("entity_group", ent.get("entity", ""))
            entity_type = SCIIE_LABEL_MAP.get(label)
            if entity_type is None:
                continue

            raw_text = ent["word"].strip()
            norm_text = normalize_entity_text(raw_text)
            
            if not norm_text or len(norm_text) < 2:
                continue

            dedup_key = (entity_type, norm_text.lower())
            if dedup_key in seen_normalized:
                continue
            seen_normalized.add(dedup_key)

            entities.append({
                "raw_text":      raw_text,
                "text":          norm_text,
                "entity_type":   entity_type,
                "confidence":    round(float(ent.get("score", 0.0)), 4),
                "start_char":    ent.get("start"),
                "end_char":      ent.get("end"),
                "section_type":  section_type,
                "section_heading": section_heading,
            })

        return entities

    def _run_ner_windowed(self, text: str) -> List[Dict]:
        """
        Sliding window over long texts to stay within the 512-token limit.
        Uses character-level splitting on sentence boundaries.
        """
        # Split into sentences first (simple heuristic — good enough for scientific text)
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_tokens = len(self.tokenizer.tokenize(sent))
            if current_len + sent_tokens > self.MAX_TOKENS and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Stride: keep last N tokens worth of sentences
                current_chunk = current_chunk[-2:]  # Keep last 2 sentences as overlap
                current_len = sum(len(self.tokenizer.tokenize(s)) for s in current_chunk)
            current_chunk.append(sent)
            current_len += sent_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Run NER on each chunk
        all_results = []
        char_offset = 0
        for chunk in chunks:
            try:
                results = self.pipe(chunk)
                # Adjust char offsets relative to original text
                chunk_start = text.find(chunk[:50])  # find chunk start in original
                offset_adjust = chunk_start if chunk_start >= 0 else char_offset
                for r in results:
                    r = dict(r)
                    if "start" in r and r["start"] is not None:
                        r["start"] += offset_adjust
                        r["end"] += offset_adjust
                    all_results.append(r)
                char_offset += len(chunk)
            except Exception as e:
                print(f"[NER] Warning: chunk NER failed — {e}")
                continue

        return all_results


# ---------------------------------------------------------------------------
# Section-level orchestrator
# ---------------------------------------------------------------------------

def extract_entities_from_sections(
    sections: List[Dict[str, Any]],
    extractor: SciBERTNERExtractor,
    priority_only: bool = False,
) -> Dict[str, Any]:
    """
    Run NER over all sections and return a structured entity report.

    Args:
        sections:      List of section dicts from sections.json
                       Expected keys: section_type, heading, text
        extractor:     Loaded SciBERTNERExtractor instance
        priority_only: If True, only process HIGH_PRIORITY_SECTIONS

    Returns:
        {
          "entities_by_section": { section_type: [entity, ...] },
          "entities_flat":       [ entity, ... ],   # all entities, deduplicated globally
          "entity_index": {
              "method":  { normalized_text: [entity, ...] },
              "dataset": { ... },
              "metric":  { ... },
              "task":    { ... },
          },
          "section_coverage": { section_type: entity_count }
        }
    """
    entities_by_section: Dict[str, List] = {}
    all_entities: List[Dict] = []

    # Global deduplication across sections: track (type, normalized_text)
    global_seen: Dict[Tuple, Dict] = {}

    for section in sections:
        sec_type = section.get("section_type", "Unknown")
        heading = section.get("heading", "")
        text = section.get("text", "")

        if priority_only and sec_type not in HIGH_PRIORITY_SECTIONS:
            continue

        if not text.strip():
            continue

        print(f"[NER] Processing section: {sec_type} ({len(text)} chars)")
        section_entities = extractor.extract_from_section(text, sec_type, heading)

        entities_by_section[sec_type] = section_entities

        for ent in section_entities:
            key = (ent["entity_type"], ent["text"].lower())
            if key not in global_seen:
                global_seen[key] = ent
                all_entities.append(ent)
            else:
                # Update provenance — entity appears in multiple sections
                existing = global_seen[key]
                if "also_in_sections" not in existing:
                    existing["also_in_sections"] = []
                if sec_type not in existing["also_in_sections"]:
                    existing["also_in_sections"].append(sec_type)
                # Keep highest confidence score
                if ent["confidence"] > existing["confidence"]:
                    existing["confidence"] = ent["confidence"]

    # Build index by entity type
    entity_index: Dict[str, Dict[str, List]] = {
        "method": {}, "dataset": {}, "metric": {}, "task": {}
    }
    for ent in all_entities:
        etype = ent["entity_type"]
        if etype in entity_index:
            key = ent["text"].lower()
            if key not in entity_index[etype]:
                entity_index[etype][key] = []
            entity_index[etype][key].append(ent)

    section_coverage = {
        sec: len(ents) for sec, ents in entities_by_section.items()
    }

    return {
        "entities_by_section": entities_by_section,
        "entities_flat":       all_entities,
        "entity_index":        entity_index,
        "section_coverage":    section_coverage,
        "total_entities":      len(all_entities),
        "entity_type_counts": {
            etype: len(index)
            for etype, index in entity_index.items()
        }
    }