## Part 2: Hybrid GROBID + Docling Scientific Structure Parser

The `grobid_parser` module extracts **semantic scientific structure** from research PDFs. It uses a hybrid approach:
* **GROBID (GeneRation Of BIbliographic Data)** handles the overarching scientific document structure, text extraction, inline formulas, and metadata formatting.
* **Docling** handles high-precision table extraction, seamlessly injecting perfectly aligned DataFrames into the data payload to override GROBID's layout logic.

It converts PDFs into strictly mapped models including:
* Abstract
* Introduction
* Related Work
* Methods
* Experiments
* Results
* Discussion
* Conclusion

along with complete **paper metadata**.

---

## Requirements

GROBID must be running locally. The easiest way is using Docker:

```bash
docker run --rm --init -p 8070:8070 lfoppiano/grobid:0.8.0
```

Verify that the service is running:

```bash
curl http://localhost:8070/api/isalive
```

Expected response `true`.

---

## Usage

Run the generic GROBID parser using the CLI:

```bash
python -m grobid_parser.cli "..\Data\your_paper.pdf" -o "output_folder"
```

## CLI Options

* `-o "output_folder"` : Directory where parsed results will be saved
* `--grobid-url` : Custom GROBID server URL (default: `http://localhost:8070`)
* `--tei-cache` : Use a cached TEI XML file instead of calling GROBID
* `--no-consolidate` : Disable CrossRef metadata enrichment
* `--check` : Verify that the GROBID service is running
* `-v` : Print preview of extracted section text

---

## Output Files

The parser produces fully detached, structured JSON payloads ensuring media objects are hoisted perfectly for downstream inference models:

```
output_folder/
├── metadata.json
├── sections.json
├── tables.json
├── figures.json
├── tei_raw.xml
└── summary.txt
```

### metadata.json
Contains extracted paper metadata (title, authors, year, venue, doi).

### sections.json
Contains structured paragraph text sections where mathematical `<formulas>` are uniquely preserved inline exactly where they appear in the original paragraph text.

### tables.json and figures.json
Completely isolated files representing high-accuracy DataFrame structures extracted via the Docling hybrid-hijack algorithms and standard figures.
