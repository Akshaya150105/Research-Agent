"""
test_real_stgcn.py
------------------
Run this from your Research-Agent root folder:

    python test_real_stgcn.py

It will test chunker.py against your actual stgcn_yu_2018 output folder
and print a full diagnostic report so you can verify everything is correct.

Make sure your folder structure looks like:
    memory/
    └── stgcn_yu_2018/
        ├── claims_output.json
        ├── sections.json
        └── figures.json   (optional)
"""

import json
import sys
import os
import logging
from pathlib import Path
from collections import Counter

# ── make sure imports work from project root ──────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

logging.basicConfig(level=logging.WARNING)   # suppress INFO during test

# ── config ────────────────────────────────────────────────────────────────
FOLDER   = "memory/stgcn_yu_2018"   # change this if your folder name differs
PAPER_ID = "stgcn_yu_2018"

# expected approximate ranges based on your claims_output.json summary
EXPECTED = {
    "claim":       (130, 160),    # summary says 141
    "limitation":  (15,  25),     # summary says 18
    "future_work": (2,   8),      # summary says 4
    "entity":      (80,  160),    # deduplicated from ~202 raw
    "section":     (5,   20),     # depends on how many sections
    "figure":      (0,   10),     # optional file
}


# ─────────────────────────────────────────────────────────────────────────
def sep(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)


def ok(msg):   print(f"  ✓  {msg}")
def fail(msg): print(f"  ✗  {msg}"); sys.exit(1)
def warn(msg): print(f"  ⚠  {msg}")
def info(msg): print(f"     {msg}")


# ─────────────────────────────────────────────────────────────────────────
def check_folder():
    sep("1. FOLDER CHECK")
    folder = Path(FOLDER)

    if not folder.exists():
        fail(f"Folder not found: {FOLDER}\n"
             f"     Make sure you run this from Research-Agent root\n"
             f"     and that the folder name is '{PAPER_ID}'")

    ok(f"Folder found: {folder.resolve()}")

    for fname, required in [
        ("claims_output.json", True),
        ("sections.json",      True),
        ("figures.json",       False),
    ]:
        path = folder / fname
        if path.exists():
            size_kb = path.stat().st_size / 1024
            ok(f"{fname} exists ({size_kb:.1f} KB)")
        elif required:
            fail(f"{fname} is missing — Phase 1 may not have completed")
        else:
            warn(f"{fname} not found — figure chunks will be skipped")

    return folder


# ─────────────────────────────────────────────────────────────────────────
def check_json_structure(folder):
    sep("2. JSON STRUCTURE CHECK")

    # claims_output.json
    claims_path = folder / "claims_output.json"
    with open(claims_path, encoding="utf-8") as f:
        data = json.load(f)

    top_keys = list(data.keys())
    info(f"Top-level keys in claims_output.json: {top_keys}")

    required_keys = ["metadata", "claims", "limitations", "future_work"]
    for k in required_keys:
        if k in data:
            ok(f"'{k}' key present")
        else:
            fail(f"'{k}' key missing from claims_output.json")

    # counts from summary
    n_claims  = len(data.get("claims", []))
    n_lims    = len(data.get("limitations", []))
    n_fw      = len(data.get("future_work", []))
    n_ents    = len(data.get("llm_entities", data.get("entities", [])))
    title     = data.get("metadata", {}).get("title", "NOT FOUND")

    info(f"Paper title : {title[:70]}...")
    info(f"Raw claims  : {n_claims}")
    info(f"Limitations : {n_lims}")
    info(f"Future work : {n_fw}")
    info(f"Raw entities: {n_ents}")

    if n_claims == 0:
        fail("No claims found — check claims_output.json structure")
    if n_ents == 0:
        warn("No entities found — entity chunks will be empty")

    # sections.json
    sec_path = folder / "sections.json"
    with open(sec_path, encoding="utf-8") as f:
        secs = json.load(f)

    info(f"Sections    : {len(secs)}")
    if len(secs) == 0:
        fail("sections.json is empty")

    for s in secs:
        if not s.get("text"):
            warn(f"Empty text in section: {s.get('section_type', '?')}")

    ok("JSON structure looks correct")
    return data, secs


# ─────────────────────────────────────────────────────────────────────────
def run_chunker():
    sep("3. RUNNING CHUNKER")
    from rag.chunker import chunk_paper

    print(f"  chunking '{PAPER_ID}' from '{FOLDER}' ...")
    chunks = chunk_paper(FOLDER, PAPER_ID)
    ok(f"Chunker completed — {len(chunks)} total chunks")
    return chunks


# ─────────────────────────────────────────────────────────────────────────
def check_counts(chunks):
    sep("4. CHUNK COUNT VERIFICATION")
    counts = Counter(c["chunk_type"] for c in chunks)

    print(f"  {'type':<16} {'count':>6}   {'expected range'}")
    print(f"  {'─'*16} {'─'*6}   {'─'*20}")
    all_pass = True
    for ctype, (lo, hi) in EXPECTED.items():
        count = counts.get(ctype, 0)
        in_range = lo <= count <= hi
        status = "✓" if in_range else ("⚠" if count == 0 else "?")
        print(f"  {status}  {ctype:<14} {count:>6}   {lo}–{hi}")
        if not in_range and ctype not in ("figure",):
            all_pass = False

    if all_pass:
        ok("All chunk counts within expected ranges")
    else:
        warn("Some counts outside expected range — review above")

    return counts


# ─────────────────────────────────────────────────────────────────────────
def check_uniqueness(chunks):
    sep("5. CHUNK ID UNIQUENESS")
    ids = [c["chunk_id"] for c in chunks]
    duplicates = [id for id, count in Counter(ids).items() if count > 1]

    if duplicates:
        fail(f"Duplicate chunk_ids found: {duplicates[:5]}")
    else:
        ok(f"All {len(ids)} chunk_ids are unique")


# ─────────────────────────────────────────────────────────────────────────
def check_metadata_schema(chunks):
    sep("6. METADATA SCHEMA (no None values)")
    required_fields = [
        "chunk_id", "paper_id", "chunk_type", "embed_text", "display_text",
        "claim_type", "section_type", "confidence", "has_numeric_value",
        "numeric_value", "entities_mentioned", "also_in_papers",
        "embed_model", "embed_model_version"
    ]

    none_violations = []
    missing_fields  = []

    for c in chunks:
        for f in required_fields:
            if f not in c:
                missing_fields.append((c["chunk_type"], f))
            elif c[f] is None:
                none_violations.append((c["chunk_type"], f, c.get("chunk_id")))

    if missing_fields:
        for ctype, f in missing_fields[:5]:
            warn(f"Missing field '{f}' in chunk_type='{ctype}'")
        fail("Missing required fields found")
    else:
        ok("All required fields present")

    if none_violations:
        for ctype, f, cid in none_violations[:5]:
            warn(f"None value in '{f}' for chunk_type='{ctype}' id={cid}")
        fail("None values found — ChromaDB will reject these")
    else:
        ok("No None values in metadata (ChromaDB safe)")


# ─────────────────────────────────────────────────────────────────────────
def check_entity_dedup(chunks):
    sep("7. ENTITY DEDUPLICATION")
    entities = [c for c in chunks if c["chunk_type"] == "entity"]
    names    = [c["entity_text"] for c in entities]

    # check for exact duplicates
    dup_names = [n for n, cnt in Counter(names).items() if cnt > 1]
    if dup_names:
        warn(f"Duplicate entity names: {dup_names[:10]}")
    else:
        ok(f"No duplicate entity names ({len(entities)} unique entities)")

    # show entity type breakdown
    type_counts = Counter(c["entity_type"] for c in entities)
    info(f"Entity type breakdown: {dict(type_counts)}")

    # show a sample
    info("Sample entities (first 10):")
    for e in entities[:10]:
        info(f"  [{e['entity_type']:8}] {e['entity_text']}")


# ─────────────────────────────────────────────────────────────────────────
def check_numeric_values(chunks):
    sep("8. NUMERIC VALUE EXTRACTION")
    numeric = [c for c in chunks if c.get("has_numeric_value")]

    ok(f"Found {len(numeric)} claims with numeric values")
    info("(These are used for contradiction detection across papers)")
    info("")
    for c in numeric:
        info(f"  value={c['numeric_value']:>10.2f}  |  "
             f"{c['display_text'][:60]}...")


# ─────────────────────────────────────────────────────────────────────────
def check_embed_text_quality(chunks):
    sep("9. EMBED TEXT QUALITY SPOT CHECK")

    # pick one of each type and show embed vs display
    seen = set()
    for c in chunks:
        ct = c["chunk_type"]
        if ct in seen:
            continue
        seen.add(ct)

        embed   = c["embed_text"]
        display = c["display_text"]

        # checks
        if not embed:
            fail(f"Empty embed_text in chunk_type='{ct}'")
        if embed == display:
            warn(f"embed_text == display_text for '{ct}' — type prefix missing?")

        print(f"\n  [{ct}]")
        print(f"  embed  : {embed[:110]}")
        print(f"  display: {display[:80]}")

    ok("Embed text spot check complete — review output above")


# ─────────────────────────────────────────────────────────────────────────
def check_paper_id(chunks):
    sep("10. PAPER ID CONSISTENCY")
    wrong = [c for c in chunks if c["paper_id"] != PAPER_ID]
    if wrong:
        fail(f"{len(wrong)} chunks have wrong paper_id")
    else:
        ok(f"All chunks have paper_id='{PAPER_ID}'")


# ─────────────────────────────────────────────────────────────────────────
def print_full_sample_chunk(chunks, chunk_type):
    sep(f"FULL SAMPLE: {chunk_type.upper()} CHUNK")
    matches = [c for c in chunks if c["chunk_type"] == chunk_type]
    if not matches:
        warn(f"No chunks of type '{chunk_type}'")
        return
    # pick an interesting one
    if chunk_type == "claim":
        # find one with a numeric value if possible
        target = next((c for c in matches if c["has_numeric_value"]), matches[0])
    else:
        target = matches[0]

    print(json.dumps(target, indent=4))


# ─────────────────────────────────────────────────────────────────────────
def final_summary(chunks, counts):
    sep("FINAL SUMMARY")
    print(f"\n  paper_id   : {PAPER_ID}")
    print(f"  folder     : {Path(FOLDER).resolve()}")
    print(f"  total chunks: {len(chunks)}")
    print()
    for k, v in sorted(counts.items()):
        print(f"  {k:<16}: {v}")
    print()
    print("  Next step: build embedder.py (Step 2)")
    print("  Run:  from rag.embedder import embed_chunks")
    sep()


# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  REAL DATA TEST — STGCN PAPER")
    print("="*60)

    folder             = check_folder()
    claims_data, secs  = check_json_structure(folder)
    chunks             = run_chunker()
    counts             = check_counts(chunks)
    check_uniqueness(chunks)
    check_metadata_schema(chunks)
    check_entity_dedup(chunks)
    check_numeric_values(chunks)
    check_embed_text_quality(chunks)
    check_paper_id(chunks)

    # print one full claim chunk so you can inspect every field
    print_full_sample_chunk(chunks, "claim")

    final_summary(chunks, counts)