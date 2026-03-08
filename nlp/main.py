import json
from pathlib import Path

from extraction_agent.schema import EntityEncounter, Claim, SectionClaims, PaperExtraction
from extraction_agent.ner_stage import SciBERTNER
from extraction_agent.llm_stage import LLMClaimExtractor

def run_hybrid_extraction(parsed_json_path: str, output_path: str):
    """
    Executes the Hybrid NER + LLM Pipeline.
    """
    print(f"Reading structured parsed JSON from: {parsed_json_path}")
    with open(parsed_json_path, 'r', encoding='utf-8') as f:
        doc_data = json.load(f)
        
    metadata_path = Path(parsed_json_path).parent / 'metadata.json'
    title = "Unknown Title"
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as mf:
            meta = json.load(mf)
            title = meta.get("title", "Unknown Title")
        
    print("\n--- Initializing Stage 1: NER (SciBERT) ---")
    ner_model = SciBERTNER()
    
    print("\n--- Initializing Stage 2: Claim Extraction (Gemini) ---")
    llm_extractor = LLMClaimExtractor()
    
    # We will treat each 'page' as a section for this prototype, 
    # but ideally the parser would group elements by headers.
    final_paper_extraction = PaperExtraction(
        title=title,
        structured_claims=[]
    )
    
    # Iterate over pages 
    pages_list = doc_data if isinstance(doc_data, list) else doc_data.get("pages", [])
    for page in pages_list:
        page_num = page.get("page_number", "Unknown")
        
        # Reconstruct text for the page
        page_text = ""
        for element in page.get("elements", []):
            if element.get("type") == "text":
                page_text += element.get("content", "") + "\n\n"
                
        if not page_text.strip():
            continue
            
        section_name = f"Page {page_num}"
        
        # Stage 1: Grounding
        print(f"\n[Stage 1] Extracting entities for {section_name}...")
        grounded_entities = ner_model.extract_entities(page_text, section_name)
        
        print(f"          -> Found {len(grounded_entities)} entities (Methods, Tasks, etc.).")
        
        if not grounded_entities:
            print("          -> Skipping Stage 2 (no entities to ground claims to).")
            continue
            
        # Stage 2: Claim LLM Extraction
        print(f"[Stage 2] Extracting claims grounded on entities for {section_name}...")
        try:
            section_claims_obj = llm_extractor.extract_claims(section_name, page_text, grounded_entities)
            final_paper_extraction.structured_claims.append(section_claims_obj)
        except Exception as e:
            print(f"          -> Error combining entities via LLM: {e}")
            break # Failsafe mostly for testing limits

    # Save final pipeline output
    print(f"\n--- Writing final Knowledge Graph / Extraction JSON to {output_path} ---")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_paper_extraction.model_dump(), f, indent=2)
        
    print(f"Pipeline complete. Check {output_path} for the results!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Entity-Claim Extractor")
    parser.add_argument("input_json", help="Path to parsed PDF structured_content.json file")
    parser.add_argument("-o", "--output", default="extraction_results.json", help="Output JSON path")
    args = parser.parse_args()
    
    run_hybrid_extraction(args.input_json, args.output)
