from pydantic import BaseModel, Field
from typing import List, Optional

# --- Stage 1 Outputs (NER) ---
class EntityEncounter(BaseModel):
    """A grounded entity from the SciBERT NER stage."""
    text: str = Field(description="The exact text of the extracted entity.")
    label: str = Field(description="The category: METHOD, DATASET, METRIC, or TASK.")
    section: str = Field(description="The section of the paper where this entity was found.")
    start_char: int
    end_char: int

# --- Stage 2 Outputs (LLM Claims) ---
class Claim(BaseModel):
    """A sentence-level claim extracted by the LLM, grounded to a specific entity."""
    entity: str = Field(description="The grounded entity this claim is about (from Stage 1).")
    entity_type: str = Field(description="METHOD, DATASET, METRIC, or TASK")
    claim_type: str = Field(description="e.g., PERFORMANCE_ASSERTION, METHODOLOGICAL_FLAW, FUTURE_WORK, COMPARATIVE_FINDING")
    statement: str = Field(description="The actual claim or limitation stated in the text.")
    sentence_context: str = Field(description="The verbatim sentence from the text containing the claim.")

class SectionClaims(BaseModel):
    """All claims extracted from a single section of the paper."""
    section_name: str
    claims: List[Claim]

class PaperExtraction(BaseModel):
    """The final structured extraction for the entire paper."""
    title: Optional[str] = None
    structured_claims: List[SectionClaims]
