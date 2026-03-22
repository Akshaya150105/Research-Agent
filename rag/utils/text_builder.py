"""
text_builder.py
---------------
Builds the embed_text string for each chunk type.
 
This is separated from chunker.py deliberately.
The embed_text is what actually gets encoded into a vector.
Small changes here have large effects on retrieval quality,
so keeping it isolated makes it easy to iterate on independently.
 
Design principle:
  embed_text = type context + location context + content + entity hints
  display_text = just the content (what gets shown to the user)
 
The type and location prefixes help the embedding model produce
better-separated vectors for different claim types — a performance
claim and a limitation about the same method should NOT cluster
together even if the words overlap.
"""
 
 
def build_claim_text(claim: dict, paper_title: str) -> str:
    """
    Builds embed text for a claim, limitation, or future_work chunk.
 
    Example output:
        "comparative claim in Results | Training Efficiency:
         STGCN achieves 14x acceleration over GCGRU.
         entities: STGCN, GCGRU, training_speed"
    """
    claim_type  = claim.get("claim_type", "")
    chunk_type  = claim.get("_chunk_type", "claim")   # injected by chunker
    section     = claim.get("section_type", "")
    heading     = claim.get("section_heading", "")
    description = claim.get("description") or claim.get("text", "")
    entities    = claim.get("entities_involved", [])
 
    # header line
    if chunk_type == "limitation":
        type_label = "limitation"
    elif chunk_type == "future_work":
        type_label = "future work"
    else:
        type_label = f"{claim_type} claim" if claim_type else "claim"
 
    location = f"{section}"
    if heading and heading != section:
        location += f" | {heading}"
 
    # entity hint — helps retrieval when user mentions method/dataset names
    entity_hint = ""
    if entities:
        entity_hint = f"\nentities: {', '.join(entities)}"
 
    return f"{type_label} in {location}:\n{description}{entity_hint}"
 
 
def build_entity_text(entity: dict, paper_title: str) -> str:
    """
    Builds embed text for an entity chunk.
 
    Example output:
        "method: STGCN
         appears in: Abstract, Experiments, Results, Future Work
         paper: Spatio-Temporal Graph Convolutional Networks..."
    """
    entity_type = entity.get("entity_type", "")
    entity_text = entity.get("text", "")
    section     = entity.get("section_type", "")
    also_in     = entity.get("also_in_sections", [])
 
    sections_list = [section]
    if also_in:
        sections_list += also_in
    sections_str = ", ".join(dict.fromkeys(sections_list))   # deduplicate, preserve order
 
    return (
        f"{entity_type}: {entity_text}\n"
        f"appears in: {sections_str}\n"
        f"paper: {paper_title}"
    )
 
 
def build_section_text(section: dict, paper_title: str) -> str:
    """
    Builds embed text for a section chunk.
    Sections get the full raw text — no truncation here.
    The embedder handles batching; long texts get mean-pooled by the model.
 
    Example output:
        "Methods section — Network Architecture | paper: STGCN...\n[full text]"
    """
    section_type    = section.get("section_type", "")
    heading         = section.get("heading", "")
    text            = section.get("text", "")
 
    location = f"{section_type} section"
    if heading and heading != section_type:
        location += f" — {heading}"
 
    return f"{location} | paper: {paper_title}\n\n{text}"
 
 
def build_figure_text(figure: dict, paper_title: str) -> str:
    """
    Builds embed text for a figure chunk.
 
    Example output:
        "figure 2 caption | paper: STGCN...:
         Architecture of spatio-temporal graph convolutional networks..."
    """
    label   = figure.get("label", "")
    caption = figure.get("caption", "")
 
    label_str = f"figure {label}" if label else "figure"
 
    return f"{label_str} caption | paper: {paper_title}:\n{caption}"