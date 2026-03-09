from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Optional

_NS = "http://www.tei-c.org/ns/1.0"

def _t(tag: str) -> str:
    return f"{{{_NS}}}{tag}"


# Output dataclasses

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



_HEADING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"abstract",                                        re.I), "Abstract"),
    (re.compile(r"introduction",                                    re.I), "Introduction"),
    (re.compile(r"related\s+work|prior\s+work|literature\s+review", re.I), "Related Work"),
    (re.compile(r"background|preliminar",                           re.I), "Related Work"),
    (re.compile(r"future\s+research|future\s+work|future\s+direction", re.I), "Future Work"),
    (re.compile(r"^training$|training\s+(data|detail|regime|procedure|setup)", re.I), "Experiments"),
    (re.compile(r"experiment|empirical|evaluation\s+setup|implementation\s+detail", re.I), "Experiments"),
    (re.compile(r"result|performance|benchmark|english\s+constituency|machine\s+translation", re.I), "Results"),
    (re.compile(r"variation|ablation",                              re.I), "Results"),
    (re.compile(r"method|approach|architecture|framework|proposed|models?", re.I), "Methods"),
    (re.compile(r"discussion|why\s+|analysis|comparison|motivation", re.I), "Discussion"),
    (re.compile(r"conclusion|summary",                              re.I), "Conclusion"),
    (re.compile(r"limitation",                                      re.I), "Limitations"),
    (re.compile(r"acknowledg",                                      re.I), "Acknowledgements"),
    (re.compile(r"reference|bibliograph|appendix",                  re.I), "References"),
]

_SKIP_TYPES = {"References", "Acknowledgements"}

_GARBAGE_HEADING_RE = re.compile(
    r"^(input[-\s]input|figure\s+\d|table\s+\d|attention\s+visualization|"
    r"layer\d|encoder[-\s]decoder)",
    re.I,
)


def _classify_heading(heading: str) -> str:
    """
    Map a heading string to a section type.
    Strips leading section numbers first: '3.2.1 Scaled Dot-Product' -> 'Scaled Dot-Product'
    """
    clean = re.sub(r"^\d+(\.\d+)*\.?\s*", "", heading).strip()
    for pattern, label in _HEADING_PATTERNS:
        if pattern.search(clean):
            return label
    return "Unknown"

def _element_text(el: ET.Element) -> str:
    """Recursively extract readable text, skipping math formulas and figures."""
    skip_tags = {_t("formula"), _t("figure")}
    parts: list[str] = []

    def _walk(node: ET.Element):
        if node.tag in skip_tags:
            return
        if node.text and node.text.strip():
            parts.append(node.text.strip())
        for child in node:
            _walk(child)
            if child.tail and child.tail.strip():
                parts.append(child.tail.strip())

    _walk(el)
    return " ".join(parts)


def _clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _clean_title(raw: str) -> str:
    """
    Strip prefixes that GROBID includes in the title field:
    """
    raw = _clean(raw)

    # Case 1: sentence ending with '. ' before actual title
    # Take the last sentence if there are multiple
    sentence_parts = re.split(r'\.\s+', raw)
    if len(sentence_parts) > 1:
        candidate = sentence_parts[-1].strip()
        # Only accept if candidate is a plausible title (>= 3 words, not a sentence fragment)
        if len(candidate.split()) >= 3:
            return candidate

    # Case 2: ALL-CAPS word (>=5 chars, not a year) preceded by mixed-case prefix
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


def _div_text(div: ET.Element) -> str:
    """Collect direct <p> text from a div, skip <head>."""
    parts = []
    for child in div:
        if child.tag == _t("head"):
            continue
        if child.tag == _t("p"):
            t = _clean(_element_text(child))
            if t:
                parts.append(t)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Author filtering
# ---------------------------------------------------------------------------

# Common affiliation/institution words that appear in author slots in GROBID output
_AFFILIATION_WORDS = re.compile(
    r"\b(university|institute|laboratory|lab|department|dept|"
    r"brain|research|google|facebook|meta|microsoft|amazon|apple|"
    r"deepmind|openai|nvidia|intel|ibm|mit|stanford|cmu|oxford|"
    r"college|school|center|centre)\b",
    re.I,
)


def _is_valid_author(name: str) -> bool:
    """
    Return True if the name looks like a person, not an affiliation string.
    Heuristics:
      - Must have at least 2 parts (first + last name)
      - Must not match known affiliation keywords
      - Must not be all-caps single word (likely an acronym like 'NLP')
    """
    parts = name.strip().split()
    if len(parts) < 2:
        return False
    if _AFFILIATION_WORDS.search(name):
        return False
    if len(parts) == 1 and parts[0].isupper():
        return False
    return True


# ---------------------------------------------------------------------------
# Sub-section detection
# ---------------------------------------------------------------------------

def _get_n(head_el: Optional[ET.Element]) -> str:
    """Return the 'n' attribute value from a <head> element, stripped."""
    if head_el is None:
        return ""
    return head_el.get("n", "").strip()


def _is_subsection(n: str) -> bool:
    """True if n has a dot (e.g. '3.2', '3.2.1') — indicating a sub-section."""
    return "." in n if n else False


def _is_unnumbered(n: str) -> bool:
    """True if n is empty — GROBID couldn't assign a section number."""
    return n == ""


def _top_level_number(n: str) -> str:
    """Extract top-level number: '3.2.1' -> '3', '5' -> '5', '' -> ''."""
    if not n:
        return ""
    return n.split(".")[0]


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------

def _parse_sections(root: ET.Element) -> list[ParsedSection]:
    """
    Extract sections from GROBID TEI body.

    Key behaviours:
    1. Top-level sections (n='1', n='2') always become their own entry.
    2. Sub-sections (n='1.1', n='3.2.1'):
       - Classifies as known type -> own entry
       - Unknown -> fold text into preceding section
    3. Orphaned sub-sections (e.g. n='6.1' with no preceding n='6'):
       - Create a virtual parent section from the sub-section's classified type
    4. Unnumbered divs (n=''):
       - Garbage headings (figure captions, 'Input-Input Layer5') -> skip
       - Others -> treat as sub-sections (fold or own entry)
    """
    sections: list[ParsedSection] = []

    # Abstract from header
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

    # Track the last top-level section number seen (e.g. '3' for n='3')
    last_top_level_n = ""

    for div in body.findall(_t("div")):
        head_el = div.find(_t("head"))
        heading = _clean(_element_text(head_el)) if head_el is not None else ""
        n = _get_n(head_el)
        text = _div_text(div)

        # Skip garbage headings (figure artifacts, etc.)
        if heading and _GARBAGE_HEADING_RE.match(heading):
            continue

        # Skip empty headings with no text
        if not heading and not text:
            continue

        section_type = _classify_heading(heading) if heading else "Unknown"

        if section_type in _SKIP_TYPES:
            continue

        is_sub = _is_subsection(n)
        is_unnumbered = _is_unnumbered(n)
        top_n = _top_level_number(n)

        if not is_sub and not is_unnumbered:
            # ── Top-level section ──────────────────────────────────────────
            last_top_level_n = n
            sections.append(ParsedSection(section_type, heading, text))

        elif is_sub:
            # ── Numbered sub-section (e.g. n='3.2') ───────────────────────
            parent_n = top_n

            # Check if the parent top-level section exists in sections list
            # An orphaned sub-section has no parent (e.g. n='6.1' with no n='6')
            parent_exists = (parent_n == last_top_level_n)

            if section_type != "Unknown":
                # Classifiable sub-section -> own entry (e.g. 'Related Work', 'Future Work')
                sections.append(ParsedSection(section_type, heading, text))
                # Update last_top_level_n so further siblings know this sub was added
            else:
                # Unknown sub-section -> fold into preceding section
                if sections and text:
                    prev = sections[-1]
                    sections[-1] = ParsedSection(
                        prev.section_type,
                        prev.heading,
                        (prev.text + " " + text).strip(),
                    )
                elif text:
                    # No parent yet (orphaned unknown sub) -> add as-is
                    sections.append(ParsedSection("Unknown", heading, text))

            if not parent_exists and section_type != "Unknown" and text:
                # Orphaned classifiable sub-section (no n='6' but we have n='6.1')
                # The section was already added above. Update tracking.
                last_top_level_n = parent_n

        else:
            # ── Unnumbered div (n='') ──────────────────────────────────────
            # Skip if heading looks like a garbage artifact
            if not heading or not text:
                continue
            if section_type != "Unknown":
                sections.append(ParsedSection(section_type, heading, text))
            else:
                # Fold into preceding section
                if sections and text:
                    prev = sections[-1]
                    sections[-1] = ParsedSection(
                        prev.section_type,
                        prev.heading,
                        (prev.text + " " + text).strip(),
                    )

    # Back matter (limitations, future work in some papers)
    back = root.find(f".//{_t('back')}")
    if back is not None:
        for div in back.findall(f".//{_t('div')}"):
            head_el = div.find(_t("head"))
            heading = _clean(_element_text(head_el)) if head_el is not None else ""
            section_type = _classify_heading(heading) if heading else "Unknown"
            if section_type in _SKIP_TYPES or section_type == "Unknown":
                continue
            text = _div_text(div)
            if text:
                sections.append(ParsedSection(section_type, heading, text))

    return _merge_same_type(sections)


def _merge_same_type(sections: list[ParsedSection]) -> list[ParsedSection]:
    """
    Merge consecutive sections of the same type.
    Drop sections with no text content.
    """
    if not sections:
        return sections
    merged = [sections[0]]
    for s in sections[1:]:
        if s.section_type == merged[-1].section_type:
            merged[-1] = ParsedSection(
                merged[-1].section_type,
                merged[-1].heading,
                (merged[-1].text + " " + s.text).strip(),
            )
        else:
            merged.append(s)
    return [s for s in merged if s.text.strip()]


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def _parse_metadata(root: ET.Element) -> ParsedMetadata:
    meta = ParsedMetadata()

    header = root.find(f".//{_t('teiHeader')}")
    if header is None:
        return meta

    # Title
    for title_el in header.findall(f".//{_t('title')}"):
        if title_el.get("type") == "main" and title_el.text:
            meta.title = _clean_title(title_el.text)
            break
    if not meta.title:
        title_el = header.find(f".//{_t('title')}")
        if title_el is not None and title_el.text:
            meta.title = _clean_title(title_el.text)

    # Authors — filter out affiliation strings
    for author in header.findall(f".//{_t('analytic')}//{_t('author')}"):
        parts = []
        fn = author.find(f".//{_t('forename')}")
        sn = author.find(f".//{_t('surname')}")
        if fn is not None and fn.text:
            parts.append(fn.text.strip())
        if sn is not None and sn.text:
            parts.append(sn.text.strip())
        name = " ".join(parts).strip()
        if name and _is_valid_author(name):
            meta.authors.append(name)

    # Year
    for date_el in header.findall(f".//{_t('date')}"):
        when = date_el.get("when", "")
        m = re.search(r"\b(19|20)\d{2}\b", when)
        if m:
            meta.year = int(m.group())
            break

    # Venue
    venue_el = header.find(f".//{_t('monogr')}/{_t('title')}")
    if venue_el is not None and venue_el.text:
        meta.venue = venue_el.text.strip()

    # DOI
    for idno in header.findall(f".//{_t('idno')}"):
        if idno.get("type", "").upper() == "DOI" and idno.text:
            meta.doi = idno.text.strip()
            break

    # Abstract
    abstract_el = header.find(f".//{_t('abstract')}")
    if abstract_el is not None:
        meta.abstract = _clean(_element_text(abstract_el))

    return meta


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_tei(tei_xml: str) -> TEIParseResult:
    """
    Parse a TEI XML string returned by GROBID.

    Returns TEIParseResult with sections, metadata, and success flag.
    On failure, success=False and error contains the reason.
    """
    if not tei_xml or not tei_xml.strip():
        return TEIParseResult([], ParsedMetadata(), False, "Empty TEI XML.")

    try:
        root = ET.fromstring(tei_xml)
    except ET.ParseError as e:
        return TEIParseResult([], ParsedMetadata(), False, f"XML parse error: {e}")

    metadata = _parse_metadata(root)
    sections = _parse_sections(root)

    if not sections:
        return TEIParseResult([], metadata, False, "No sections extracted from TEI.")

    return TEIParseResult(sections=sections, metadata=metadata, success=True)