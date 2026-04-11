
import json
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import streamlit as st
from arxiv_utils import (ARXIV_MAX_RESULTS, arxiv_id_to_filename,arxiv_search_with_expansion,check_duplicate_arxiv, download_arxiv_pdf,   load_registry, register_paper, save_registry, sha256_of)

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from grobid_parser.client import GROBIDClient
from grobid_parser.tei_parser import parse_tei
from grobid_parser.utils import save_grobid_output
if "arxiv_expansion" not in st.session_state:
    st.session_state.arxiv_expansion = None

@st.cache_resource(show_spinner="Loading SciBERT NER model (one-time)…")
def load_ner_extractor():
    from ner_pipeline.ner_extractor import SciBERTNERExtractor
    return SciBERTNERExtractor()


st.set_page_config(
    page_title="Research Paper Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem; border-radius: 16px; margin-bottom: 2rem; color: white;
}
.main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.main-header p  { margin: 0.4rem 0 0; opacity: 0.9; font-size: 1.05rem; }

.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 10px; padding: 1rem 1.4rem; min-width: 140px; text-align: center;
}
.metric-box .val {
    font-size: 1.8rem; font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.metric-box .label { font-size: 0.82rem; color: #9ca3af; margin-top: 0.2rem; }

.upload-zone {
    border: 2px dashed rgba(102,126,234,0.4);
    border-radius: 14px; padding: 2rem; text-align: center;
    background: rgba(102,126,234,0.04); transition: border-color 0.3s;
}
.upload-zone:hover { border-color: rgba(102,126,234,0.7); }

/* ArXiv card */
.arxiv-card {
    background: rgba(30,30,50,0.6);
    border: 1px solid rgba(102,126,234,0.2);
    border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 0.6rem;
}
.arxiv-card .paper-title { font-weight: 600; font-size: 0.97rem; color: #c4b5fd; }
.arxiv-card .paper-meta  { font-size: 0.78rem; color: #9ca3af; margin-top: 0.3rem; }
.arxiv-card .paper-abs   {
    font-size: 0.8rem; color: #d1d5db; margin-top: 0.5rem;
    display: -webkit-box; -webkit-line-clamp: 3;
    -webkit-box-orient: vertical; overflow: hidden;
}

/* Badges */
.dup-badge {
    display: inline-block;
    background: rgba(234,179,8,0.15); border: 1px solid rgba(234,179,8,0.4);
    border-radius: 6px; padding: 0.15rem 0.5rem;
    font-size: 0.75rem; color: #fbbf24; font-weight: 600;
}
.new-badge {
    display: inline-block;
    background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.35);
    border-radius: 6px; padding: 0.15rem 0.5rem;
    font-size: 0.75rem; color: #4ade80; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 Research Paper Analyzer</h1>
    <p>ArXiv Search &nbsp;·&nbsp; PDF Upload &nbsp;·&nbsp; GROBID &nbsp;·&nbsp;
       SciBERT NER &nbsp;·&nbsp; LLM Claim Extraction</p>
</div>
""", unsafe_allow_html=True)

#sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
 
    grobid_url = st.text_input(
        "GROBID URL", value="http://localhost:8070",
        help="URL of the running GROBID service",
    )
 
    st.divider()
    st.markdown("#### 🔍 ArXiv Query Expansion")
    st.caption("Groq understands your natural language query and generates ArXiv search terms.")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help=(
            "Groq API key for query expansion. "
            "Get one free at https://console.groq.com/keys"
        ),
    )
    groq_model = st.selectbox(
        "Groq model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        index=0,
        help="llama-3.3-70b-versatile is fast and highly capable.",
    )
 
    st.divider()
    st.markdown("#### 🧠 Claim Extraction (Ollama)")
    st.caption("Ollama handles GROBID → NER → claim extraction. Separate from search.")
    ollama_host = st.text_input(
        "Ollama Host",
        value=os.environ.get("OLLAMA_HOST", "https://468e-136-109-132-83.ngrok-free.app"),
        help="URL of your Ollama server for Stage 3 (Claim Extraction).",
    )
    rpm = st.slider("Requests per minute (Ollama)", 1, 60, 12)
 
    output_root = st.text_input(
        "Output folder",
        value=str(PROJECT_ROOT / "memory"),
        help="Root folder. Each paper gets a subfolder named by its paper_id.",
    )
 
    st.divider()
    if st.button("🩺 Check GROBID"):
        from grobid_parser.client import GROBIDClient
        client = GROBIDClient(base_url=grobid_url)
        if client.is_alive():
            st.success(f"GROBID is running at {grobid_url}")
        else:
            st.error(f"GROBID NOT reachable at {grobid_url}")
 
    st.divider()
    st.markdown("### 📚 Paper Registry")
    _reg = load_registry(output_root)
    st.metric("Unique papers indexed", len(_reg["hashes"]))
    if _reg["hashes"] and st.button("🔍 Show registry"):
        for h, pid in _reg["hashes"].items():
            st.code(f"{h[:12]}…  →  {pid}", language=None)


#session state
if "arxiv_results"   not in st.session_state: st.session_state.arxiv_results   = []
if "arxiv_selected"  not in st.session_state: st.session_state.arxiv_selected  = {}
if "pipeline_queue"  not in st.session_state: st.session_state.pipeline_queue  = []
if "last_topic"      not in st.session_state: st.session_state.last_topic      = ""
if "writer_review"   not in st.session_state: st.session_state.writer_review   = ""
if "writer_out_path" not in st.session_state: st.session_state.writer_out_path = ""


# TABS

tab_arxiv, tab_upload = st.tabs(["🌐 ArXiv Search", "📂 Upload PDFs"])

# TAB 1 — ArXiv Search
with tab_arxiv:
    st.markdown("### 🔍 Search ArXiv in natural language")
    st.caption(
        "Describe what you're looking for in plain English. "
        "The LLM will understand your intent and generate the best ArXiv search terms — "
        "then the top matching papers are shown below."
    )
 
    col_q, col_n, col_btn = st.columns([5, 1, 1])
    with col_q:
        arxiv_query = st.text_input(
            "Query",
            placeholder='e.g. "I want papers on how transformers handle long sequences"',
            label_visibility="collapsed",
        )
    with col_n:
        n_results = st.number_input(
            "Max", min_value=1, max_value=50, value=ARXIV_MAX_RESULTS
        )
    with col_btn:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
 
    # ── Run search 
    if search_clicked and arxiv_query.strip():
        with st.spinner("🤖 Understanding your query…"):
            try:
                papers, expansion = arxiv_search_with_expansion(
                    natural_language_query=arxiv_query.strip(),
                    groq_api_key=groq_api_key,
                    max_results=int(n_results),
                    model=groq_model,
                )
                st.session_state.arxiv_results   = papers
                st.session_state.arxiv_selected  = {}
                st.session_state.arxiv_expansion = expansion
                st.session_state.last_topic = (
                    expansion.get("primary_query") or arxiv_query.strip()
                )
            except Exception as exc:
                st.error(f"Search failed: {exc}")
 
    expansion = st.session_state.get("arxiv_expansion")
    if expansion:
        status_icon = "✅" if expansion["expansion_ok"] else "⚠️"
        tag_html = " ".join(
            f'<span style="background:rgba(102,126,234,0.18);border:1px solid '
            f'rgba(102,126,234,0.35);border-radius:6px;padding:2px 8px;'
            f'font-size:0.78rem;color:#c4b5fd;">{t}</span>'
            for t in expansion["topic_tags"]
        )
        related_html = " ".join(
            f'<span style="background:rgba(52,211,153,0.1);border:1px solid '
            f'rgba(52,211,153,0.3);border-radius:6px;padding:2px 8px;'
            f'font-size:0.78rem;color:#6ee7b7;">{r}</span>'
            for r in expansion["related_terms"]
        )
        st.markdown(f"""
<div style="background:rgba(20,20,40,0.7);border:1px solid rgba(102,126,234,0.25);
            border-radius:12px;padding:1rem 1.4rem;margin:0.8rem 0 1.2rem 0;">
  <div style="font-size:0.8rem;color:#9ca3af;margin-bottom:0.4rem;">
    {status_icon} <strong style="color:#e2e8f0;">LLM understood your query as:</strong>
  </div>
  <div style="font-size:0.96rem;color:#f0f0f0;margin-bottom:0.5rem;">
    {expansion['intent_summary']}
  </div>
  <div style="font-size:0.8rem;color:#9ca3af;margin-bottom:0.3rem;">
    🔎 <strong>ArXiv query sent:</strong>
    <code style="color:#a5b4fc;background:rgba(102,126,234,0.1);
                 padding:1px 6px;border-radius:4px;">
      {expansion['primary_query']}
    </code>
  </div>
  {"<div style='margin-top:0.5rem;'>🏷 " + tag_html + "</div>" if tag_html else ""}
  {"<div style='margin-top:0.4rem;'>🔗 <span style='font-size:0.78rem;color:#9ca3af;'>Related:</span> " + related_html + "</div>" if related_html else ""}
</div>
""", unsafe_allow_html=True)
 
    # ── Results ───────────────────────────────────────────────────────
    results = st.session_state.arxiv_results
    if results:
        registry = load_registry(output_root)
        st.markdown(f"**{len(results)} result(s)** — tick papers to add to the pipeline queue:")
 
        if st.button("☑ Select all new papers"):
            for p in results:
                if not check_duplicate_arxiv(p["arxiv_id"], registry):
                    st.session_state.arxiv_selected[p["arxiv_id"]] = True
 
        for paper in results:
            aid = paper["arxiv_id"]
            existing_pid = check_duplicate_arxiv(aid, registry)
 
            col_cb, col_card = st.columns([0.05, 0.95])
            with col_cb:
                checked = st.checkbox(
                    "", key=f"cb_{aid}",
                    value=st.session_state.arxiv_selected.get(aid, False),
                    disabled=bool(existing_pid),
                )
                st.session_state.arxiv_selected[aid] = checked
 
            with col_card:
                badge = (
                    f'<span class="dup-badge">⚠ Already indexed as <code>{existing_pid}</code></span>'
                    if existing_pid
                    else '<span class="new-badge">✦ New</span>'
                )
                authors_str = ", ".join(paper["authors"][:3])
                if len(paper["authors"]) > 3:
                    authors_str += f" +{len(paper['authors'])-3} more"
 
                st.markdown(f"""
<div class="arxiv-card">
  <div class="paper-title">{paper['title']}</div>
  <div class="paper-meta">
    {badge}&nbsp;&nbsp;
    📅 {paper['published']} &nbsp;·&nbsp;
    👤 {authors_str} &nbsp;·&nbsp;
    🆔 <code>{aid}</code>
  </div>
  <div class="paper-abs">{paper['abstract']}</div>
</div>
""", unsafe_allow_html=True)
 
        selected_ids = [
            aid for aid, sel in st.session_state.arxiv_selected.items() if sel
        ]
        if selected_ids:
            if st.button(
                f"⬇️ Download & queue {len(selected_ids)} selected paper(s)",
                type="primary",
            ):
                already_in_queue = {item["name"] for item in st.session_state.pipeline_queue}
                prog = st.progress(0)
                downloaded = skipped = 0
 
                for i, aid in enumerate(selected_ids):
                    fname = arxiv_id_to_filename(aid)
                    if fname in already_in_queue:
                        skipped += 1
                        prog.progress((i + 1) / len(selected_ids))
                        continue
                    meta = next(p for p in results if p["arxiv_id"] == aid)
                    with st.spinner(f"Downloading {aid}…"):
                        try:
                            pdf_bytes = download_arxiv_pdf(meta["pdf_url"])
                            st.session_state.pipeline_queue.append({
                                "name":      fname,
                                "bytes":     pdf_bytes,
                                "source":    "arxiv",
                                "arxiv_id":  aid,
                                "title":     meta["title"],
                            })
                            downloaded += 1
                        except Exception as exc:
                            st.warning(f"Could not download {aid}: {exc}")
                    prog.progress((i + 1) / len(selected_ids))
 
                prog.empty()
                st.success(f"✅ {downloaded} paper(s) queued; {skipped} already in queue.")

# TAB 2 — Upload PDFs
with tab_upload:
    st.markdown("### Upload PDF files")
    

    st.text_input(
        "Research Topic (for Planner Agent)",
        key="last_topic",
        placeholder="e.g. Graph Neural Networks for Small Datasets",
        help="Specify the topic of these papers so the Planner Agent can generate a focused literature review."
    )

    st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload research paper PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_files:
        already_in_queue = {item["name"] for item in st.session_state.pipeline_queue}
        added = 0
        for uf in uploaded_files:
            if uf.name not in already_in_queue:
                st.session_state.pipeline_queue.append({
                    "name":     uf.name,
                    "bytes":    uf.getvalue(),
                    "source":   "upload",
                    "arxiv_id": None,
                    "title":    uf.name,
                })
                added += 1
        if added:
            st.success(f"➕ {added} file(s) added to the pipeline queue.")


# Pipeline Queue — visible below tabs
st.divider()
st.markdown("## 🗂 Pipeline Queue")

queue = st.session_state.pipeline_queue
if not queue:
    st.info("Queue is empty. Search ArXiv or upload PDFs above.")
    st.stop()

registry           = load_registry(output_root)
seen_hashes: dict  = {}
item_statuses      = []

for item in queue:
    h = sha256_of(item["bytes"])
    item["_hash"] = h

    if h in registry["hashes"]:
        item_statuses.append(("registry_dup", registry["hashes"][h]))
    elif h in seen_hashes:
        item_statuses.append(("batch_dup", seen_hashes[h]))
    else:
        seen_hashes[h] = item["name"]
        item_statuses.append(("new", None))

# ── Summary 
c1, c2, c3 = st.columns(3)
c1.metric("In queue",           len(queue))
c2.metric("Already indexed",    sum(1 for s, _ in item_statuses if s == "registry_dup"))
c3.metric("Ready to process",   sum(1 for s, _ in item_statuses if s == "new"))

# ── Per-item rows 
for item, (status, ref) in zip(queue, item_statuses):
    icon = "🌐" if item["source"] == "arxiv" else "📂"
    if status == "registry_dup":
        st.markdown(
            f"{icon} ~~{item['name']}~~ &nbsp;"
            f'<span class="dup-badge">⚠ Already processed — stored as <code>{ref}</code></span>',
            unsafe_allow_html=True,
        )
    elif status == "batch_dup":
        st.markdown(
            f"{icon} ~~{item['name']}~~ &nbsp;"
            f'<span class="dup-badge">⚠ Identical content as <code>{ref}</code> in this batch</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"{icon} **{item['name']}** &nbsp;"
            f'<span class="new-badge">✦ New</span>',
            unsafe_allow_html=True,
        )

col_clear, _ = st.columns([1, 4])
with col_clear:
    if st.button("🗑 Clear queue"):
        st.session_state.pipeline_queue = []
        st.rerun()

new_items = [item for item, (s, _) in zip(queue, item_statuses) if s == "new"]

if not new_items:
    st.warning("All queued papers have already been processed. Nothing to run.")
    st.stop()


# GROBID worker (thread-pool target)
def _grobid_parse_one(pdf_bytes: bytes, pdf_name: str, grobid_url: str):
    tmp_dir        = tempfile.mkdtemp(prefix="research_pipeline_")
    pdf_path       = Path(tmp_dir) / pdf_name
    pdf_path.write_bytes(pdf_bytes)
    grobid_out_dir = Path(tmp_dir) / "grobid_output"

    try:
        client = GROBIDClient(base_url=grobid_url)
        if not client.is_alive():
            return pdf_name, tmp_dir, None, None, None, "GROBID not reachable"

        tei_xml = client.process_fulltext(str(pdf_path))
        if not tei_xml:
            return pdf_name, tmp_dir, None, None, None, "GROBID returned no output"

        result = parse_tei(tei_xml, pdf_path=str(pdf_path))
        if not result.success:
            return pdf_name, tmp_dir, None, None, None, f"TEI parse error: {result.error}"

        save_grobid_output(result, tei_xml, str(grobid_out_dir))
        return pdf_name, tmp_dir, grobid_out_dir, result, tei_xml, None

    except Exception as e:
        return pdf_name, tmp_dir, None, None, None, str(e)


# Run Pipeline
st.divider()

if st.button(
    f"🚀 Run Pipeline on {len(new_items)} new paper(s)",
    type="primary",
    use_container_width=True,
):
    total   = len(new_items)
    overall = st.progress(0, text="Starting pipeline…")
    registry = load_registry(output_root)

    # ── STAGE 1: GROBID 
    grobid_results: dict[str, tuple] = {}

    with st.status(
        f"🔍 Stage 1 — GROBID Parsing ({total} papers concurrently)…", expanded=True
    ) as s1:
        futures: dict = {}
        with ThreadPoolExecutor(max_workers=min(total, 4)) as pool:
            for item in new_items:
                fut = pool.submit(_grobid_parse_one, item["bytes"], item["name"], grobid_url)
                futures[fut] = item["name"]

            done = 0
            for fut in as_completed(futures):
                done += 1
                pdf_name, tmp_dir, grobid_out_dir, result, tei_xml, error = fut.result()
                grobid_results[pdf_name] = (tmp_dir, grobid_out_dir, result, tei_xml, error)
                if error:
                    st.write(f"❌ **{pdf_name}**: {error}")
                else:
                    title = result.metadata.title or "N/A"
                    st.write(f"✅ **{pdf_name}** — {title[:60]} ({len(result.sections)} sections)")
                overall.progress(done / (total * 3), text=f"GROBID: {done}/{total}")

        s1.update(label=f"✅ Stage 1 — GROBID complete ({done}/{total})", state="complete")

    # ── STAGE 2: NER 
    ner_extractor = load_ner_extractor()
    from ner_pipeline.ner_extractor import extract_entities_from_sections
    from ner_pipeline.pipeline import load_sections, _make_paper_id

    ner_results: dict[str, tuple] = {}

    with st.status(f"🧬 Stage 2 — SciBERT NER ({total} papers)…", expanded=True) as s2:
        done = 0
        for item in new_items:
            pdf_name = item["name"]
            done += 1
            tmp_dir, grobid_out_dir, _, _, grobid_error = grobid_results[pdf_name]

            if grobid_error:
                ner_results[pdf_name] = (None, None, "Skipped (GROBID failed)")
                st.write(f"⏭️ **{pdf_name}**: skipped (GROBID failed)")
                continue

            try:
                ner_out_dir = Path(tmp_dir) / "ner_results"
                ner_out_dir.mkdir(parents=True, exist_ok=True)

                sections, metadata = load_sections(grobid_out_dir)
                result   = extract_entities_from_sections(sections, ner_extractor)
                paper_id = _make_paper_id(metadata)

                enriched = {
                    "paper_id": paper_id,
                    "metadata": metadata,
                    "ner_summary": {
                        "total_entities":     result["total_entities"],
                        "entity_type_counts": result["entity_type_counts"],
                        "section_coverage":   result["section_coverage"],
                        "model_used":         ner_extractor.model_name,
                        "priority_only":      False,
                    },
                    "entities":            result["entities_flat"],
                    "entity_index":        result["entity_index"],
                    "entities_by_section": result["entities_by_section"],
                }
                (ner_out_dir / "enriched_entities.json").write_text(
                    json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                ner_results[pdf_name] = (ner_out_dir, enriched, None)
                st.write(f"✅ **{pdf_name}** — {result['total_entities']} entities")

            except Exception as e:
                ner_results[pdf_name] = (None, None, str(e))
                st.write(f"❌ **{pdf_name}**: {e}")

            overall.progress((total + done) / (total * 3), text=f"NER: {done}/{total}")
        s2.update(label="✅ Stage 2 — NER complete", state="complete")

    # Claims + save + register 
    with st.status("🧠 Stage 3 — LLM Claim Extraction…", expanded=True) as s3:
        done = 0
        for item in new_items:
            pdf_name  = item["name"]
            pdf_bytes = item["bytes"]
            arxiv_id  = item.get("arxiv_id")
            done += 1

            tmp_dir, grobid_out_dir, _, _, grobid_error = grobid_results[pdf_name]
            ner_out_dir, enriched_data, ner_error = ner_results.get(
                pdf_name, (None, None, "No NER")
            )

            if grobid_error or ner_error:
                st.write(f"⏭️ **{pdf_name}**: skipped")
                overall.progress((2*total+done)/(total*3), text=f"Claims: {done}/{total}")
                continue

            paper_id = (enriched_data or {}).get("paper_id") or re.sub(
                r"[^\w\-]", "_", Path(pdf_name).stem
            )
            claims_data = None

            try:
                from claim_extractor.pipeline import run_pipeline as run_claims
                claims_path = run_claims(
                    grobid_output_dir=str(grobid_out_dir),
                    ner_results_dir=str(ner_out_dir),
                    ollama_host=ollama_host,
                    requests_per_minute=rpm,
                )
                with open(claims_path, encoding="utf-8") as f:
                    claims_data = json.load(f)
                paper_id = claims_data.get("paper_id", paper_id)
                n_claims = claims_data.get("summary", {}).get("total_claims", 0)
                st.write(f"✅ **{pdf_name}** — {n_claims} claims extracted")
            except Exception as e:
                st.write(f"⚠️ **{pdf_name}**: claim extraction error — {e}")

            paper_folder = Path(output_root) / paper_id
            paper_folder.mkdir(parents=True, exist_ok=True)

            for src_dir in (grobid_out_dir, ner_out_dir):
                if src_dir and Path(src_dir).exists():
                    for f in Path(src_dir).glob("*.json"):
                        shutil.copy2(f, paper_folder / f.name)

            
            pdf_out_folder = PROJECT_ROOT / "data_1" / "papers"
            pdf_out_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(tmp_dir) / pdf_name, pdf_out_folder / pdf_name)
            
            shutil.rmtree(tmp_dir, ignore_errors=True)

            register_paper(pdf_bytes, paper_id, registry, arxiv_id=arxiv_id)
            save_registry(output_root, registry)

            st.write(f"📂 Saved to `{paper_folder}`  |  hash registered ✓")
            overall.progress((2*total+done)/(total*3), text=f"Claims: {done}/{total}")

        s3.update(label="✅ Stage 3 — Claims complete", state="complete")

    _planner_topic = st.session_state.get("last_topic", "")

    with st.status(
        "🤖 Stage 4 — Planner Agent (Literature Review)…", expanded=True
    ) as s4:
        if _planner_topic:
            st.write(f"🎯 Research topic: **{_planner_topic}**")
        else:
            st.write("ℹ️ No ArXiv topic found — running Planner without topic filter.")

        try:
            _agents_dir = str(PROJECT_ROOT / "agents")
            if _agents_dir not in sys.path:
                sys.path.insert(0, _agents_dir)
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from planner_agent import PlannerAgent  

            st.write("⚙️ Building LangGraph pipeline…")
            planner     = PlannerAgent(verbose=True)
            final_state = planner.run(
                topic      = _planner_topic,
                memory_dir = output_root,           
                use_llm    = True,
            )

            rr = final_state.get("reader_report",     {})
            cr = final_state.get("comparator_report", {})
            gr = final_state.get("gap_report")  or    {}
            wr = final_state.get("writer_report",     {})

            st.write(
                f"📚 Papers read: **{rr.get('papers_read', 0)}** &nbsp;·&nbsp; "
                f"🔁 Contradictions: **{cr.get('n_contradictions_found', 0)}** &nbsp;·&nbsp; "
                f"🔍 Gaps: **{len(gr.get('gaps', []))}**"
            )

            out_path = wr.get("output_path", "")
            if out_path and Path(out_path).exists():
                review_md = Path(out_path).read_text(encoding="utf-8")
                st.session_state.writer_review   = review_md
                st.session_state.writer_out_path = out_path
                st.write(f"✅ Review saved → `{out_path}`")
            else:
                st.write("⚠️ Writer completed but no review file found.")

        except Exception as _e:
            import traceback
            st.error(f"Planner Agent error: {_e}")
            st.code(traceback.format_exc(), language="python")

        s4.update(label="✅ Stage 4 — Literature Review complete", state="complete")

    overall.progress(1.0, text="All done!")
    st.balloons()
    st.success(f"🎉 {total} paper(s) processed! Results in `{output_root}`")

    processed = {item["name"] for item in new_items}
    st.session_state.pipeline_queue = [
        i for i in st.session_state.pipeline_queue if i["name"] not in processed
    ]

    for item in new_items:
        pdf_name = item["name"]
        _, enriched_data, ner_error = ner_results.get(pdf_name, (None, None, None))
        if ner_error or not enriched_data:
            continue

        paper_id     = enriched_data.get("paper_id", pdf_name)
        paper_folder = Path(output_root) / paper_id
        claims_file  = paper_folder / "claims_output.json"

        st.markdown(f"---\n### 📄 {enriched_data.get('metadata', {}).get('title', pdf_name)}")
        st.caption(f"`paper_id`: {paper_id}")

        if claims_file.exists():
            with open(claims_file, encoding="utf-8") as f:
                claims_data = json.load(f)

            s = claims_data.get("summary", {})
            st.markdown(f"""
<div class="metric-row">
  <div class="metric-box"><div class="val">{s.get('total_llm_entities',0)}</div><div class="label">Entities</div></div>
  <div class="metric-box"><div class="val">{s.get('total_claims',0)}</div><div class="label">Claims</div></div>
  <div class="metric-box"><div class="val">{s.get('total_limitations',0)}</div><div class="label">Limitations</div></div>
  <div class="metric-box"><div class="val">{s.get('total_future_work',0)}</div><div class="label">Future Work</div></div>
</div>
""", unsafe_allow_html=True)

            claims_list = claims_data.get("claims", [])
            if claims_list:
                with st.expander(f"📋 Claims ({len(claims_list)})", expanded=False):
                    for c in claims_list:
                        st.markdown(
                            f"**[{c.get('claim_type','')}]** {c.get('description','')}\n\n"
                            f"- Entities: `{c.get('entities_involved',[])}`\n"
                            f"- Value: `{c.get('value','')}` | Confidence: `{c.get('confidence','')}`"
                        )
                        st.divider()

            st.download_button(
                label=f"⬇️ Download {paper_id} claims",
                data=json.dumps(claims_data, indent=2, ensure_ascii=False),
                file_name=f"{paper_id}_claims_output.json",
                mime="application/json",
                key=f"dl_{paper_id}",
            )


if st.session_state.get("writer_review"):
    st.divider()
    st.markdown("## 📝 Literature Review")
    _col_info, _col_dl = st.columns([3, 1])
    with _col_info:
        if st.session_state.get("last_topic"):
            st.caption(f"Topic: **{st.session_state.last_topic}**")
        if st.session_state.get("writer_out_path"):
            st.caption(f"Saved at: `{st.session_state.writer_out_path}`")
    with _col_dl:
        st.download_button(
            label="⬇️ Download .md",
            data=st.session_state.writer_review,
            file_name="literature_review.md",
            mime="text/markdown",
        )

    _review_text = st.session_state.writer_review
    import re as _re
    _sections = _re.split(r"(?=^## )", _review_text, flags=_re.MULTILINE)
    if len(_sections) > 1:
        for _sec in _sections:
            if not _sec.strip():
                continue
            _title = _sec.splitlines()[0].lstrip("#").strip() or "Section"
            with st.expander(_title, expanded=(_title.lower().startswith("intro"))):
                st.markdown(_sec)
    else:
        st.markdown(_review_text)