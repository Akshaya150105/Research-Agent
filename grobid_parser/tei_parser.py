from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Optional

_NS = "http://www.tei-c.org/ns/1.0"

def _t(tag: str) -> str:
    return f"{{{_NS}}}{tag}"

_XML_ID = "{http://www.w3.org/XML/1998/namespace}id"


# Output dataclasses

@dataclass
class ParsedFigure:
    figure_id: str        
    label: str            
    caption: str          


@dataclass
class ParsedTable:
    table_id: str         
    label: str           
    caption: str          
    rows: list[list[str]] 


@dataclass
class ParsedFormula:
    formula_id: str       
    label: str            
    content: str          


@dataclass
class ParsedSection:
    section_type: str
    heading: str
    text: str                             


@dataclass
class ParsedMetadata:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    doi: str = ""
    abstract: str = ""


@dataclass
class TEIParseResult:
    sections: list[ParsedSection]
    metadata: ParsedMetadata
    success: bool
    error: str = ""
    figures: list[ParsedFigure] = field(default_factory=list)
    tables: list[ParsedTable] = field(default_factory=list)


# Heading classification

_HEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"abstract",                                              re.I), "Abstract"),
    (re.compile(r"introduction",                                          re.I), "Introduction"),
    (re.compile(r"related\s+work|prior\s+work|literature\s+review",       re.I), "Related Work"),
    (re.compile(r"^background$|^preliminar",                              re.I), "Background"),
    (re.compile(r"future\s+(research|work|direction)",                    re.I), "Future Work"),
    (re.compile(r"^training$|training\s+(data|detail|regime|procedure|setup|batching)"
                r"|hardware|schedule|optimizer|regularization",           re.I), "Experiments"),
    (re.compile(r"experiment|empirical|evaluation\s+setup|implementation\s+detail", re.I), "Experiments"),
    (re.compile(r"result|performance|benchmark|english\s+constituency|machine\s+translation", re.I), "Results"),
    (re.compile(r"variation|ablation|benefit\s+of|training\s+efficienc",  re.I), "Results"),
    (re.compile(r"method|approach|architecture|framework|proposed|models?", re.I), "Methods"),
    (re.compile(r"discussion|why\s+|analysis|comparison|motivation",      re.I), "Discussion"),
    (re.compile(r"conclusion|summary",                                    re.I), "Conclusion"),
    (re.compile(r"limitation",                                            re.I), "Limitations"),
    (re.compile(r"acknowledg",                                            re.I), "Acknowledgements"),
    (re.compile(r"reference|bibliograph|appendix",                        re.I), "References"),
]

_SKIP_TYPES: set[str] = set()

_GARBAGE_HEADING_RE = re.compile(
    r"^(input[-\s]input|figure\s+\d|table\s+\d|attention\s+visualization|"
    r"layer\d|encoder[-\s]decoder)",
    re.I,
)

# Formula elements whose text is actually a figure/diagram description misplaced by GROBID
_FIGURE_FORMULA_RE = re.compile(
    r"(ST-Conv|Conv\s+Block|Output\s+Layer|Gated.Conv|Graph.Conv|GLU|"
    r"vt-M\+|Temporal\s+Gated|Spatial\s+Graph)",
    re.I,
)

# figDesc text that is clearly garbled — either body-text prose or raw numeric table data
_GARBLED_FIGDESC_RE = re.compile(
    r"^(Full-Connected|We execute|locate the best|data points|"
    r"in the next|Evaluation Metric"
    r"|[\d\.]+/[\d\.]+)"  # starts with numbers like ".23/ 6.12" (raw table data)
    r"|^\.",              # starts with a bare period (mid-sentence fragment)
    re.I,
)

# Table rows that contain garbled body-text fragments (single long cell, sentence-like)
_BODY_TEXT_ROW_RE = re.compile(r"\b(we|the|to|in|and|of|a)\b", re.I)


def _classify_heading(heading: str) -> str:
    clean = re.sub(r"^\d+(\.\d+)*\.?\s*", "", heading).strip()
    for pattern, label in _HEADING_PATTERNS:
        if pattern.search(clean):
            return label
    return "Unknown"


_SKIP_EXTRACT_TAGS = {_t("ref"), _t("note")}


def _element_text(el: ET.Element) -> str:
    parts: list[str] = []

    def _walk(node: ET.Element):
        if node.tag in _SKIP_EXTRACT_TAGS:
            if node.tail and node.tail.strip():
                parts.append(node.tail.strip())
            return
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        for child in node:
            _walk(child)
            if child.tag not in _SKIP_EXTRACT_TAGS:
                if child.tail and child.tail.strip():
                    parts.append(child.tail.strip())

    _walk(el)
    return " ".join(parts)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_title(raw: str) -> str:
    raw = _clean(raw)
    sentence_parts = re.split(r'\.\s+', raw)
    if len(sentence_parts) > 1:
        candidate = sentence_parts[-1].strip()
        if len(candidate.split()) >= 3:
            return candidate
    words = raw.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'^[-\s]+', '', word)
        if (clean_word.isupper()
                and len(clean_word) >= 5
                and not re.match(r'^\d{4}$', clean_word)):
            if i > 0 and any(not w.lstrip('-').isupper() for w in words[:i]):
                return " ".join(words[i:])
            break
    return raw



# Author filtering

_AFFILIATION_WORDS = re.compile(
    r"\b(university|institute|laboratory|lab|department|dept|"
    r"brain|research|google|facebook|meta|microsoft|amazon|apple|"
    r"deepmind|openai|nvidia|intel|ibm|mit|stanford|cmu|oxford|"
    r"college|school|center|centre)\b",
    re.I,
)


def _is_valid_author(name: str) -> bool:
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    if _AFFILIATION_WORDS.search(name):
        return False
    if len(parts) == 1 and parts[0].isupper():
        return False
    return True



def _get_n(head_el: Optional[ET.Element]) -> str:
    if head_el is None:
        return ""
    return head_el.get("n", "").strip()


def _is_subsection(n: str) -> bool:
    return "." in n if n else False


def _is_unnumbered(n: str) -> bool:
    return n == ""


def _top_level_number(n: str) -> str:
    if not n:
        return ""
    return n.split(".")[0]


# Global figure / table / formula extraction

def _is_garbled_caption(text: str) -> bool:
    #True if figDesc looks like body-text accidentally included as a caption.
    return bool(_GARBLED_FIGDESC_RE.search(text)) if text else True


def _is_garbled_table(rows: list[list[str]]) -> bool:
   
    #True if a table's rows look like flowing body-text rather than tabular data.
    #Heuristic: all rows have exactly one cell and that cell reads like a sentence.
   
    if not rows:
        return True
    single_cell_rows = [r for r in rows if len(r) == 1]
    if len(single_cell_rows) < len(rows) * 0.8:
        return False  
    
    prose_count = sum(
        1 for r in single_cell_rows
        if len(r[0].split()) > 5 and _BODY_TEXT_ROW_RE.search(r[0])
    )
    return prose_count >= len(single_cell_rows) * 0.6


def _parse_all_figures(root: ET.Element) -> dict[str, ParsedFigure]:
    
    #Parse every real <figure> (not table) in the document.
    #Returns a dict keyed by xml:id.
    #Filters out figures with no label (fragments) or garbled captions.
    
    figures: dict[str, ParsedFigure] = {}
    for fig in root.iter(_t("figure")):
        if fig.get("type") == "table":
            continue
        fig_id = fig.get(_XML_ID, "")
        label_el = fig.find(_t("label"))
        figdesc_el = fig.find(_t("figDesc"))

        label = _clean(label_el.text or "") if label_el is not None else ""
        caption = _clean(_element_text(figdesc_el)) if figdesc_el is not None else ""

        # Skip fragments: no label AND caption is garbled or very short
        if not label and (not caption or _is_garbled_caption(caption) or len(caption) < 20):
            continue

        figures[fig_id] = ParsedFigure(
            figure_id=fig_id,
            label=label,
            caption=caption,
        )
    return figures


def _rescue_caption_from_garbled_figdesc(raw: str) -> str:

    sentences = re.split(r'\.\s+', raw)
    for candidate in reversed(sentences):
        candidate = candidate.strip()
        words = candidate.split()
        if len(words) >= 5 and candidate[0].isupper():
            # Ensure it ends with a period
            return candidate if candidate.endswith('.') else candidate + '.'
    return ""


def _parse_all_tables(root: ET.Element) -> dict[str, ParsedTable]:
    
    #Parse every <figure type="table"> in the document.
    #Garbled captions are replaced with a rescued sentence from their tail.
    #Garbled rows (body-text fragments) are discarded.
    
    tables: dict[str, ParsedTable] = {}
    for fig in root.iter(_t("figure")):
        if fig.get("type") != "table":
            continue
        tab_id = fig.get(_XML_ID, "")
        label_el = fig.find(_t("label"))
        figdesc_el = fig.find(_t("figDesc"))
        table_el = fig.find(_t("table"))

        label = _clean(label_el.text or "") if label_el is not None else ""
        raw_caption = _clean(_element_text(figdesc_el)) if figdesc_el is not None else ""

        if _is_garbled_caption(raw_caption):
            caption = _rescue_caption_from_garbled_figdesc(raw_caption)
        else:
            caption = raw_caption

        rows: list[list[str]] = []
        if table_el is not None:
            for row in table_el.iter(_t("row")):
                cells = [_clean(_element_text(c)) for c in row.findall(_t("cell"))]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(cells)

        # Discard rows that are flowing body-text fragments
        if _is_garbled_table(rows):
            rows = []

        # Skip entirely if no label and no usable content at all
        if not label and not caption and not rows:
            continue

        tables[tab_id] = ParsedTable(
            table_id=tab_id,
            label=label,
            caption=caption,
            rows=rows,
        )
    return tables


def _normalise_formula_label(label: str) -> str:
    label = label.strip()
    if re.fullmatch(r'\d+', label):
        return f"({label})"
    return label


def _merge_formula_fragments(formulas: list[tuple[str, str, str]]) -> list[ParsedFormula]:
    
    #Merge consecutive <formula> fragments that GROBID split from one equation.

    #Two passes:
    #Pass 1 (forward): merge tiny/unlabelled fragments into their PRECEDING entry
    #  when the preceding entry is also unlabelled.
    #Pass 2 (forward): merge any remaining unlabelled fragment into the FOLLOWING
    #  labelled entry (it is a preamble fragment of that equation).
    
    if not formulas:
        return []

    # merge trailing tiny fragments backward
    p1: list[tuple[str, str, str]] = []
    for fid, label, content in formulas:
        if not content:
            continue
        is_tiny = len(content) <= 6
        prev_unlabelled = bool(p1 and not p1[-1][1])
        if not label and p1 and (is_tiny or prev_unlabelled):
            prev_fid, prev_label, prev_content = p1[-1]
            p1[-1] = (prev_fid, prev_label, (prev_content + " " + content).strip())
        else:
            p1.append((fid, label, content))

    # merge unlabelled preamble fragments forward into the next labelled entry
    p2: list[tuple[str, str, str]] = []
    i = 0
    while i < len(p1):
        fid, label, content = p1[i]
        if (not label
                and i + 1 < len(p1)
                and p1[i + 1][1]):          
            # absorb this fragment into the next entry
            next_fid, next_label, next_content = p1[i + 1]
            p1[i + 1] = (next_fid, next_label, (content + " " + next_content).strip())
            i += 1
            continue
        p2.append((fid, label, content))
        i += 1

    return [ParsedFormula(formula_id=fid, label=label, content=content)
            for fid, label, content in p2]


def _extract_div_formulas(div: ET.Element) -> list[ParsedFormula]:
    """Extract and clean all <formula> elements directly inside a div."""
    raw: list[tuple[str, str, str]] = []
    for f in div.findall(_t("formula")):
        fid = f.get(_XML_ID, "")
        content = _clean(_element_text(f))
        if _FIGURE_FORMULA_RE.search(content):
            continue 

        label_el = f.find(_t("label"))
        raw_label = _clean(label_el.text or "") if label_el is not None else ""
        label = _normalise_formula_label(raw_label)

        # Strip label from content 
        if label:
            bare = label.strip("()")
            for pat in (re.escape(label), re.escape(bare)):
                content = re.sub(r'\s*' + pat + r'\s*$', '', content).strip()
                content = re.sub(r'^\s*' + pat + r'\s*', '', content).strip()

        if content:
            raw.append((fid, label, content))

    return _merge_formula_fragments(raw)


def _collect_figure_refs(div: ET.Element) -> list[str]:
    ids: list[str] = []
    for ref in div.iter(_t("ref")):
        if ref.get("type") == "figure":
            tgt = ref.get("target", "").lstrip("#")
            if tgt:
                ids.append(tgt)

    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# Per-div text (paragraphs only — no figure/table/formula dumps)

def _div_text(div: ET.Element) -> str:
    """Extract paragraph prose and formulas from a div."""
    parts: list[str] = []
    for child in div:
        if child.tag in (_t("p"), _t("formula")):
            t = _clean(_element_text(child))
            if t:
                parts.append(t)
    return " ".join(parts)


# Section assembly

def _parse_sections(root: ET.Element, all_tables: Optional[dict[str, ParsedTable]] = None) -> list[ParsedSection]:
    
    #Extract sections with per-section figures, tables, and formulas.

 
    # Pre-build global figure and table registries
    all_figures = _parse_all_figures(root)
    if all_tables is None:
        all_tables = _parse_all_tables(root)

    sections: list[ParsedSection] = []

    # Abstract
    header = root.find(f".//{_t('teiHeader')}")
    if header is not None:
        abstract_el = header.find(f".//{_t('abstract')}")
        if abstract_el is not None:
            text = _clean(_element_text(abstract_el))
            if text:
                sections.append(ParsedSection("Abstract", "Abstract", text))

    body = root.find(f".//{_t('body')}")
    if body is None:
        return sections

    last_top_level_n = ""
    last_top_level_type = "Unknown"

    def _make_section(section_type, heading, div) -> ParsedSection:
        return ParsedSection(
            section_type=section_type,
            heading=heading,
            text=_div_text(div),
        )

    def _append_div_to_prev(prev: ParsedSection, div: ET.Element) -> ParsedSection:
        """Merge a div's content into an existing section."""
        extra_text = _div_text(div)
        return ParsedSection(
            section_type=prev.section_type,
            heading=prev.heading,
            text=(prev.text + " " + extra_text).strip() if extra_text else prev.text,
        )

    for div in body.findall(_t("div")):
        head_el = div.find(_t("head"))
        heading = _clean(_element_text(head_el)) if head_el is not None else ""
        n = _get_n(head_el)

        if heading and _GARBAGE_HEADING_RE.match(heading):
            if sections:
                sections[-1] = _append_div_to_prev(sections[-1], div)
            continue

        text_check = _div_text(div)
        has_content = bool(text_check)

        if not heading and not has_content:
            continue

        section_type = _classify_heading(heading) if heading else "Unknown"
        if section_type in _SKIP_TYPES:
            continue

        is_sub = _is_subsection(n)
        is_unnumbered = _is_unnumbered(n)
        top_n = _top_level_number(n)

        if not is_sub and not is_unnumbered:
            last_top_level_n = n
            last_top_level_type = section_type
            if has_content:
                sections.append(_make_section(section_type, heading, div))

        elif is_sub:
            if section_type != "Unknown":
                sections.append(_make_section(section_type, heading, div))
                if top_n != last_top_level_n:
                    last_top_level_n = top_n
                    last_top_level_type = section_type
            else:
                inferred_type = last_top_level_type if last_top_level_type != "Unknown" else "Unknown"
                if has_content:
                    if sections and sections[-1].section_type == inferred_type:
                        sections[-1] = _append_div_to_prev(sections[-1], div)
                    else:
                        sections.append(_make_section(inferred_type, heading, div))
                if top_n != last_top_level_n:
                    last_top_level_n = top_n

        else:  # unnumbered
            if not has_content:
                continue
            if section_type != "Unknown":
                sections.append(_make_section(section_type, heading, div))
            else:
                if sections:
                    sections[-1] = _append_div_to_prev(sections[-1], div)

    back = root.find(f".//{_t('back')}")
    if back is not None:
        for div in back.iter(_t("div")):
            head_el = div.find(_t("head"))
            heading = _clean(_element_text(head_el)) if head_el is not None else ""
            section_type = _classify_heading(heading) if heading else "Unknown"
            if section_type in _SKIP_TYPES or section_type in ("Unknown", "References"):
                continue
            text = _div_text(div)
            if text:
                sections.append(_make_section(section_type, heading, div))

    return _merge_same_type(sections)


def _merge_same_type(sections: list[ParsedSection]) -> list[ParsedSection]:
    #Merge consecutive sections of the same type; drop sections with no content.
    if not sections:
        return sections
    merged = [sections[0]]
    for s in sections[1:]:
        if s.section_type == merged[-1].section_type:
            prev = merged[-1]
            # Merge text
            new_text = (prev.text + " " + s.text).strip() if s.text else prev.text
            merged[-1] = ParsedSection(
                prev.section_type, prev.heading, new_text
            )
        else:
            merged.append(s)
    return [
        s for s in merged
        if s.text.strip()
    ]


# Metadata extraction

def _parse_metadata(root: ET.Element) -> ParsedMetadata:
    meta = ParsedMetadata()

    header = root.find(f".//{_t('teiHeader')}")
    if header is None:
        return meta

    for title_el in header.findall(f".//{_t('title')}"):
        if title_el.get("type") == "main" and title_el.text:
            meta.title = _clean_title(title_el.text)
            break
    if not meta.title:
        title_el = header.find(f".//{_t('title')}")
        if title_el is not None and title_el.text:
            meta.title = _clean_title(title_el.text)

    for author in header.findall(f".//{_t('analytic')}//{_t('author')}"):
        persName = author.find(_t("persName"))
        if persName is None:
            continue
        parts = []
        for fn in persName.findall(_t("forename")):
            if fn.text:
                parts.append(fn.text.strip())
        sn = persName.find(_t("surname"))
        if sn is not None and sn.text:
            parts.append(sn.text.strip())
        name = " ".join(parts).strip()
        if name and _is_valid_author(name):
            meta.authors.append(name)

    for date_el in header.findall(f".//{_t('date')}"):
        when = date_el.get("when", "")
        m = re.search(r"\b(19|20)\d{2}\b", when)
        if m:
            meta.year = int(m.group())
            break

    venue_el = header.find(f".//{_t('monogr')}/{_t('title')}")
    if venue_el is not None and venue_el.text:
        meta.venue = venue_el.text.strip()

    for idno in header.findall(f".//{_t('idno')}"):
        if idno.get("type", "").upper() == "DOI" and idno.text:
            meta.doi = idno.text.strip()
            break

    abstract_el = header.find(f".//{_t('abstract')}")
    if abstract_el is not None:
        meta.abstract = _clean(_element_text(abstract_el))

    return meta


# Public API

def _extract_docling_tables(pdf_path: str) -> list[ParsedTable]:
    import os
    import shutil
    _orig_symlink = getattr(os, "symlink", None)
    def _fallback_symlink(src, dst, *args, **kwargs):
        if os.path.exists(dst):
            return
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    os.symlink = _fallback_symlink
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    from docling.document_converter import DocumentConverter
    converter = DocumentConverter()
    res = converter.convert(pdf_path)
    doc = res.document
    tables = []
    t_idx = 1
    for item, level in doc.iterate_items():
        label_name = getattr(item.label, "name", str(item.label)).upper()
        if label_name == "TABLE":
            cap = ""
            if hasattr(item, "captions") and item.captions:
                cap = " ".join(c.text for c in item.captions if hasattr(c, "text"))
            rows = []
            if hasattr(item, "export_to_dataframe"):
                try:
                    df = item.export_to_dataframe()
                    headers = [str(c) for c in df.columns]
                    if any(headers): rows.append(headers)
                    for _, r in df.iterrows():
                        rows.append([str(v) for v in r.values])
                except Exception:
                    pass
            tid = f"docling_tab_{t_idx}"
            tables.append(ParsedTable(table_id=tid, label=str(t_idx), caption=cap, rows=rows))
            t_idx += 1
    return tables


def parse_tei(tei_xml: str, pdf_path: Optional[str] = None) -> TEIParseResult:

    if not tei_xml or not tei_xml.strip():
        return TEIParseResult([], ParsedMetadata(), False, "Empty TEI XML.")

    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as e:
        return TEIParseResult([], ParsedMetadata(), False, f"XML parse error: {e}")

    metadata = _parse_metadata(root)
    
    all_tables = _parse_all_tables(root)
    if pdf_path:
        try:
            print("[Hybrid] Extracting high-accuracy tables using Docling...")
            doc_tables = _extract_docling_tables(pdf_path)
            g_keys = list(all_tables.keys())
            for i, dt in enumerate(doc_tables):
                if i < len(g_keys):
                    all_tables[g_keys[i]].rows = dt.rows
                else:
                    all_tables[dt.table_id] = dt
        except Exception as e:
            print(f"[Hybrid] Warning: Failed to extract Docling tables: {e}")

    sections = _parse_sections(root, all_tables)

    all_figures_dict = _parse_all_figures(root)

    if not sections:
        return TEIParseResult([], metadata, False, "No sections extracted from TEI.", list(all_figures_dict.values()), list(all_tables.values()))

    return TEIParseResult(
        sections=sections,
        metadata=metadata,
        success=True,
        figures=list(all_figures_dict.values()),
        tables=list(all_tables.values())
    )