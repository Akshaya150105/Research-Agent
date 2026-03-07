import os
from typing import List
from pydantic import BaseModel, ConfigDict
from google import genai
from google.genai import types

from extraction_agent.schema import EntityEncounter, Claim, SectionClaims

class LLMClaimExtractor:
    """
    Stage 2: LLM for Claim Extraction.
    Anchors the LLM to only extract claims about explicitly provided entities, preventing hallucinations.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is missing. Please set it before running Stage 2.")
            
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def extract_claims(self, section_name: str, section_text: str, entities: List[EntityEncounter]) -> SectionClaims:
        """
        Takes raw text and a list of entities from Stage 1 (SciBERT),
        and asks the LLM to extract claims explicitly linking back to those entities.
        """
        # Format the entities into a prompt string
        entity_str = ""
        for i, ent in enumerate(entities):
            entity_str += f"[{i+1}] '{ent.text}' (Type: {ent.label})\n"
            
        system_instruction = """
        You are an expert scientific researcher. Your task is to extract claims, limitations, performance assertions, 
        and comparative findings from a research paper section. 
        
        CRITICAL INSTRUCTION: You MUST ONLY extract claims that directly correspond to the explicitly provided list of PRE-IDENTIFIED ENTITIES. 
        Do not hallucinate new methdologies, datasets, or metrics. Limit your extraction to only the entities listed below.
        
        If an entity is provided but has no corresponding claim or limitation in the text snippet, ignore it.
        """
        
        prompt = f"""
        # Section Name: {section_name}

        # Pre-Identified Entities (Stage 1 NER)
        These are the only entities you are allowed to formulate claims about:
        {entity_str}
        
        # Raw Text
        {section_text[:50000]} # Limit to 50k chars for safety, though Gemini can handle 1M+
        
        Please extract any claims or limitations that directly involve the above entities. Return a valid JSON matching the SectionClaims schema.
        """

        print(f"Calling Gemini API for section '{section_name}' with {len(entities)} entities...")
        
        # Call the new Google GenAI SDK using ResponseSchema to enforce the structure
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=SectionClaims,
                temperature=0.0, # Zero temperature for maximum determinism
            ),
        )
        
        # The response.parsed field will contain the strongly typed Pydantic object
        print(f"Successfully extracted {len(response.parsed.claims)} claims for section '{section_name}'.")
        return response.parsed
