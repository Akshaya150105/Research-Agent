## Part 1: Modular PDF Parser

The `modular_parser` is a completely local, python-based parser that reads academic PDFs and outputs a strictly structured JSON file while preserving natural reading order. It intelligently extracts text, images, auto-detects synthetic figures (vector graphics), and native tables.

### Setup
1. Install the requirements (Requires PyMuPDF for advanced features, and pdfplumber as a table fallback):
   ```bash
   pip install pymupdf pdfplumber PyPDF2
   ```

### Usage
Run the parser using the modular CLI:

```bash
python -m modular_parser.cli "..\Data\your_paper.pdf" -o "output_folder" -a -i
```

**Flags:**
* `-o "output_folder"` : Where to save the output files (JSON, extracted images, text).
* `-a` : Auto-detect synthetic figures (diagrams) and save them as PNGs.
* `-i` : Extract standard rasterized images from the PDF.
* `--no-structured`: Disables the generation of the `structured_content.json` reading-order file.

**Output:**
The most important output is `output_folder/structured_content.json`, which contains the reading-order extracted text, table matrices, and image bounding boxes.
