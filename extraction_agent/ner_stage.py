import torch
from transformers import pipeline
from typing import List, Dict, Any
from extraction_agent.schema import EntityEncounter

class SciBERTNER:
    """
    Stage 1: NER for Entity Grounding.
    Uses a fine-tuned SciBERT model to extract Methods, Datasets, Metrics, and Tasks.
    """
    def __init__(self, model_name: str = "dslim/bert-base-NER"):
        # Note: "dslim/bert-base-NER" is a generic placeholder. 
        # For true SciIE, a model like "allenai/scibert_scivocab_uncased" fine-tuned on SciIE should be used.
        # The user can substitute the exact HF Hub model name here.
        print(f"Loading NER Model: {model_name}...")
        self.device = 0 if torch.cuda.is_available() else -1
        self.nlp = pipeline("ner", model=model_name, aggregation_strategy="simple", device=self.device)
        print("Model loaded successfully.")

    def map_label_to_sci_domain(self, label: str) -> str:
        """
        Maps generic NER labels to the SciIE domain if using a generic model for testing.
        If the model is already fine-tuned on SciIE, it will output METHOD, DATASET natively.
        """
        label = label.upper()
        if "MISC" in label or "ORG" in label:
            return "METHOD" # Rough approximation for testing
        if "LOC" in label:
            return "DATASET"
        return "TASK" # Default

    def extract_entities(self, text: str, section_name: str) -> List[EntityEncounter]:
        """
        Runs the NER pipeline on a chunk of text and returns grounded EntityEncounters.
        """
        # Chunk text if it's too long for BERT (512 tokens)
        max_chars = 2000 # Rough chunking
        chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
        
        all_entities = []
        offset = 0
        
        for chunk in chunks:
            if not chunk.strip():
                offset += len(chunk)
                continue
                
            predictions = self.nlp(chunk)
            
            for pred in predictions:
                # Map raw model labels to our target taxonomy (Method, Dataset, Metric, Task)
                mapped_label = self.map_label_to_sci_domain(pred.get("entity_group", pred.get("entity", "UNK")))
                
                entity = EntityEncounter(
                    text=pred["word"],
                    label=mapped_label,
                    section=section_name,
                    start_char=offset + pred["start"],
                    end_char=offset + pred["end"]
                )
                all_entities.append(entity)
                
            offset += len(chunk)
            
        return all_entities
