import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


class PDFExtractor:
    """Universal PDF content extractor"""
    
    def __init__(self, pdf_path: str):
        """Initialize with PDF file path"""
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = str(self.pdf_path)
        self.extractor = None
        self.doc = None
        self.pdf = None
        self.secondary_pdfplumber = None
        
        self._initialize_extractor()
    
    def _initialize_extractor(self):
        """Initialize the best available PDF library"""
        if HAS_PYMUPDF:
            self.extractor = 'pymupdf'
            self.doc = fitz.open(self.pdf_path)
        elif HAS_PDFPLUMBER:
            self.extractor = 'pdfplumber'
            self.pdf = pdfplumber.open(self.pdf_path)
        elif HAS_PYPDF2:
            self.extractor = 'pypdf2'
            self.pdf = PyPDF2.PdfReader(self.pdf_path)
        else:
            raise ImportError("No PDF library found. Install one of: pymupdf, pdfplumber, or pypdf2")
            
        # Initialize secondary library (pdfplumber) for robust table extraction fallback
        if HAS_PDFPLUMBER and self.extractor != 'pdfplumber':
            self.secondary_pdfplumber = pdfplumber.open(self.pdf_path)
    
    def extract_text(self, page_numbers: Optional[List[int]] = None, figure_bboxes: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None) -> str:
        """Extract all text from PDF, optionally filtering out regions in figure_bboxes"""
        if self.extractor == 'pymupdf':
            return self._extract_text_pymupdf(page_numbers, figure_bboxes)
        elif self.extractor == 'pdfplumber':
            return self._extract_text_pdfplumber(page_numbers, figure_bboxes)
        elif self.extractor == 'pypdf2':
            return self._extract_text_pypdf2(page_numbers)
        return ""
    
    def _extract_text_pymupdf(self, page_numbers: Optional[List[int]] = None, figure_bboxes: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None) -> str:
        """Extract text using PyMuPDF"""
        text = ""
        pages = range(len(self.doc)) if page_numbers is None else page_numbers
        
        for page_num in pages:
            if page_num < len(self.doc):
                page = self.doc[page_num]
                
                if figure_bboxes and page_num in figure_bboxes:
                    bboxes = [fitz.Rect(bbox) for bbox in figure_bboxes[page_num]]
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        block_rect = fitz.Rect(block[:4])
                        if not any(block_rect.intersects(b) for b in bboxes):
                            text += block[4] + "\n"
                else:
                    text += page.get_text() + "\n"
        
        return text
    
    def _extract_text_pdfplumber(self, page_numbers: Optional[List[int]] = None, figure_bboxes: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None) -> str:
        """Extract text using pdfplumber"""
        text = ""
        pages = range(len(self.pdf.pages)) if page_numbers is None else page_numbers
        
        for page_num in pages:
            if page_num < len(self.pdf.pages):
                page = self.pdf.pages[page_num]
                
                if figure_bboxes and page_num in figure_bboxes:
                    bboxes = figure_bboxes[page_num]
                    def is_outside_figures(obj):
                        if not isinstance(obj, dict) or 'x0' not in obj or 'top' not in obj or 'x1' not in obj or 'bottom' not in obj:
                            return True
                        x0, top, x1, bottom = obj['x0'], obj['top'], obj['x1'], obj['bottom']
                        for (fx0, fy0, fx1, fy1) in bboxes:
                            if (x0 < fx1 and x1 > fx0 and top < fy1 and bottom > fy0):
                                return False
                        return True
                    filtered_page = page.filter(is_outside_figures)
                    text += (filtered_page.extract_text() or "") + "\n"
                else:
                    text += (page.extract_text() or "") + "\n"
        
        return text
    
    def _extract_text_pypdf2(self, page_numbers: Optional[List[int]] = None) -> str:
        """Extract text using PyPDF2"""
        text = ""
        pages = range(len(self.pdf.pages)) if page_numbers is None else page_numbers
        
        for page_num in pages:
            if page_num < len(self.pdf.pages):
                page = self.pdf.pages[page_num]
                text += page.extract_text() or "" + "\n"
        
        return text
    
    def extract_metadata(self) -> Dict:
        """Extract PDF metadata"""
        metadata = {}
        
        if self.extractor == 'pymupdf':
            metadata = self.doc.metadata
        elif self.extractor == 'pdfplumber':
            metadata = self.pdf.metadata
        elif self.extractor == 'pypdf2':
            metadata = self.pdf.metadata
        
        return metadata
    
    def get_page_count(self) -> int:
        """Get total number of pages"""
        if self.extractor == 'pymupdf':
            return len(self.doc)
        elif self.extractor == 'pdfplumber':
            return len(self.pdf.pages)
        elif self.extractor == 'pypdf2':
            return len(self.pdf.pages)
        return 0
    
    def extract_page_info(self, page_number: int = 0) -> Dict:
        """Extract information about a specific page"""
        info = {
            'page_number': page_number,
            'width': 0,
            'height': 0,
            'text_length': 0
        }
        
        if self.extractor == 'pymupdf':
            page = self.doc[page_number]
            info['width'] = page.rect.width
            info['height'] = page.rect.height
            info['text_length'] = len(page.get_text())
        elif self.extractor == 'pdfplumber':
            page = self.pdf.pages[page_number]
            info['width'] = page.width
            info['height'] = page.height
            info['text_length'] = len(page.extract_text() or "")
        elif self.extractor == 'pypdf2':
            page = self.pdf.pages[page_number]
            info['text_length'] = len(page.extract_text() or "")
        
        return info
    
    def extract_images(self, output_dir: Optional[str] = None) -> List[str]:
        """Extract images from PDF (requires PyMuPDF)"""
        if self.extractor != 'pymupdf':
            print("Warning: Image extraction requires PyMuPDF")
            return []
        
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        image_paths = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                if output_dir:
                    image_path = Path(output_dir) / f"page_{page_num}_img_{img_index}.{image_ext}"
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_paths.append(str(image_path))
                else:
                    image_paths.append(f"Image {page_num}_{img_index} ({len(image_bytes)} bytes)")
        
        return image_paths
    
    def extract_synthetic_figures(self, figure_bboxes: Dict[int, List[Tuple[float, float, float, float]]], output_dir: str) -> List[str]:
        """
        Extract specified bounding boxes as rendered images (synthetic figures).
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        image_paths = []
        
        if self.extractor == 'pymupdf':
            zoom_matrix = fitz.Matrix(300 / 72, 300 / 72)
            for page_num, bboxes in figure_bboxes.items():
                if page_num < len(self.doc):
                    page = self.doc[page_num]
                    for idx, bbox in enumerate(bboxes):
                        rect = fitz.Rect(bbox)
                        pix = page.get_pixmap(matrix=zoom_matrix, clip=rect)
                        img_path = str(Path(output_dir) / f"synthetic_fig_page_{page_num}_{idx}.png")
                        pix.save(img_path)
                        image_paths.append(img_path)
                        
        elif self.extractor == 'pdfplumber':
            for page_num, bboxes in figure_bboxes.items():
                if page_num < len(self.pdf.pages):
                    page = self.pdf.pages[page_num]
                    for idx, bbox in enumerate(bboxes):
                        cropped_page = page.crop(bbox)
                        im = cropped_page.to_image(resolution=300)
                        img_path = str(Path(output_dir) / f"synthetic_fig_page_{page_num}_{idx}.png")
                        im.save(img_path, format="PNG")
                        image_paths.append(img_path)
        else:
            print("Warning: Synthetic figure rendering requires PyMuPDF or pdfplumber")
            
        return image_paths
    
    def extract_links(self) -> List[Dict]:
        """Extract hyperlinks from PDF"""
        links = []
        
        if self.extractor == 'pymupdf':
            for page_num in range(len(self.doc)):
                page = self.doc[page_num]
                page_links = page.get_links()
                for link in page_links:
                    links.append({
                        'page': page_num,
                        'type': link.get('kind', 'unknown'),
                        'uri': link.get('uri', ''),
                        'from': link.get('from', {}),
                        'to': link.get('to', {})
                    })
        
        return links
    
    def extract_tables(self) -> List[List[List[str]]]:
        """Extract tables from PDF using PyMuPDF or pdfplumber"""
        tables = []
        
        # Try PyMuPDF native table extraction first
        if self.extractor == 'pymupdf':
            try:
                for page in self.doc:
                    if hasattr(page, "find_tables"):
                        tabs = page.find_tables()
                        if tabs and tabs.tables:
                            for tab in tabs.tables:
                                tables.append(tab.extract())
            except Exception as e:
                pass
                
        # Fallback to pdfplumber
        if not tables:
            plumber_doc = None
            if self.extractor == 'pdfplumber':
                plumber_doc = self.pdf
            elif self.secondary_pdfplumber:
                plumber_doc = self.secondary_pdfplumber
                
            if plumber_doc:
                for page in plumber_doc.pages:
                    page_tables = page.extract_tables()
                    tables.extend(page_tables)
            elif self.extractor != 'pdfplumber' and not tables:
                print("Warning: Table extraction requires pdfplumber or PyMuPDF 1.23.0+")
        
        return tables

    def extract_structured(self, figure_bboxes: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None, extract_images_to: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extracts elements in natural reading order."""
        if self.extractor != 'pymupdf':
            print("Warning: Structured extraction requires PyMuPDF")
            return []
            
        pages_data = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            elements = []
            
            # 1. Tables
            table_rects = []
            if hasattr(page, "find_tables"):
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    for tab in tabs.tables:
                        bbox = list(tab.bbox)
                        table_rects.append(fitz.Rect(bbox))
                        elements.append({
                            "type": "table",
                            "bbox": bbox,
                            "content": tab.extract()
                        })
            
            # Table fallback
            if not table_rects and self.secondary_pdfplumber:
                try:
                    p_page = self.secondary_pdfplumber.pages[page_num]
                    for tab in p_page.find_tables():
                        bbox = [tab.bbox[0], tab.bbox[1], tab.bbox[2], tab.bbox[3]]
                        table_rects.append(fitz.Rect(bbox))
                        elements.append({
                            "type": "table",
                            "bbox": bbox,
                            "content": tab.extract()
                        })
                except Exception:
                    pass

            # 2. Synthetic Figures
            synthetic_rects = []
            if figure_bboxes and page_num in figure_bboxes:
                for idx, bbox in enumerate(figure_bboxes[page_num]):
                    synthetic_rects.append(fitz.Rect(bbox))
                    img_dict = {
                        "type": "synthetic_figure",
                        "bbox": bbox
                    }
                    if extract_images_to:
                        img_path = str(Path(extract_images_to) / f"synthetic_fig_page_{page_num}_{idx}.png")
                        if os.path.exists(img_path):
                            img_dict["image_path"] = img_path
                    elements.append(img_dict)

            # 3. Text & Standard Images
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                bbox = block.get("bbox")
                target_rect = fitz.Rect(bbox)
                
                # Filter out elements that heavily overlap with tables or synthetic figures
                overlaps = False
                for r in table_rects + synthetic_rects:
                    if target_rect.intersects(r):
                        intersect_area = target_rect.intersect(r).get_area()
                        if target_rect.get_area() > 0 and (intersect_area / target_rect.get_area()) > 0.5:
                            overlaps = True
                            break
                            
                if overlaps:
                    continue
                    
                if block.get("type") == 1: # Image
                    img_element = {
                        "type": "image",
                        "bbox": bbox,
                        "ext": block.get("ext", "png")
                    }
                    image_data = block.get("image")
                    if image_data:
                        if extract_images_to:
                            img_filename = f"page_{page_num}_img_{len(elements)}.png"
                            img_path = Path(extract_images_to) / img_filename
                            with open(img_path, "wb") as f:
                                f.write(image_data)
                            img_element["image_path"] = str(img_path)
                        else:
                            import base64
                            img_element["base64"] = base64.b64encode(image_data).decode('utf-8')
                    elements.append(img_element)
                    
                elif block.get("type") == 0: # Text
                    text_content = ""
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_content += span.get("text", "") + " "
                        text_content += "\n"
                    text_content = text_content.strip()
                    if text_content:
                        elements.append({
                            "type": "text",
                            "bbox": bbox,
                            "content": text_content
                        })

            # Sort elements by reading order (y0, then x0)
            def sort_key(elem):
                y0 = round(elem["bbox"][1] / 10) * 10
                x0 = elem["bbox"][0]
                return (y0, x0)
                
            elements.sort(key=sort_key)
            pages_data.append({
                "page_number": page_num + 1,
                "width": page.rect.width,
                "height": page.rect.height,
                "elements": elements
            })
            
        return pages_data
    
    def detect_figures(self) -> Dict[int, List[Tuple[float, float, float, float]]]:
        """
        Auto-detect synthetic figures by finding clusters of vector graphics/paths.
        Requires PyMuPDF.
        """
        if self.extractor != 'pymupdf':
            print("Warning: Auto figure detection requires PyMuPDF")
            return {}
            
        figure_bboxes = {}
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            
            # Get drawings (paths, lines, curves)
            drawings = page.get_drawings()
            
            if not drawings:
                continue
                
            rects = [fitz.Rect(d["rect"]) for d in drawings]
            if not rects:
                continue
                
            page_area = page.rect.width * page.rect.height
            valid_rects = []
            for r in rects:
                area = r.width * r.height
                if area > 100 and area < (page_area * 0.8): 
                    valid_rects.append(r)
            
            if not valid_rects:
                continue
                
            merged = True
            while merged:
                merged = False
                new_rects = []
                while valid_rects:
                    current = valid_rects.pop(0)
                    to_remove = []
                    for i, other in enumerate(valid_rects):
                        inflated = current + (-10, -10, 10, 10)
                        if inflated.intersects(other):
                            current |= other 
                            to_remove.append(i)
                            merged = True
                            
                    for i in reversed(to_remove):
                        valid_rects.pop(i)
                        
                    new_rects.append(current)
                valid_rects = new_rects
                
            final_rects = []
            for r in valid_rects:
                 if r.width > 50 and r.height > 50:
                     r += (-5, -5, 5, 5)
                     final_rects.append((r.x0, r.y0, r.x1, r.y1))
                     
            if final_rects:
                figure_bboxes[page_num] = final_rects
                
        return figure_bboxes
    
    def close(self):
        """Close PDF resources"""
        if self.extractor == 'pymupdf':
            self.doc.close()
        elif self.extractor == 'pdfplumber':
            self.pdf.close()
        elif self.extractor == 'pypdf2':
            self.pdf = None
            
        if self.secondary_pdfplumber:
            self.secondary_pdfplumber.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def extract_all(pdf_path: str, output_dir: Optional[str] = None, figure_bboxes: Optional[Dict[int, List[Tuple[float, float, float, float]]]] = None, auto_detect_figures: bool = False, extract_structured: bool = True) -> Dict:
    """Wrapper to run extractor over everything."""
    results = {
        'metadata': {},
        'text': '',
        'page_count': 0,
        'images': [],
        'synthetic_figures': [],
        'links': [],
        'tables': [],
        'structured_pages': [],
        'errors': []
    }
    
    try:
        with PDFExtractor(pdf_path) as extractor:
            if auto_detect_figures and extractor.extractor == 'pymupdf':
                detected_bboxes = extractor.detect_figures()
                if figure_bboxes: # Merge if both provided
                    for p_num, bboxes in detected_bboxes.items():
                        if p_num in figure_bboxes:
                            figure_bboxes[p_num].extend(bboxes)
                        else:
                            figure_bboxes[p_num] = bboxes
                else:
                    figure_bboxes = detected_bboxes
                    
            results['metadata'] = extractor.extract_metadata()
            results['page_count'] = extractor.get_page_count()
            results['images'] = extractor.extract_images(output_dir)
            
            if figure_bboxes and output_dir:
                results['synthetic_figures'] = extractor.extract_synthetic_figures(figure_bboxes, output_dir)
                
            results['text'] = extractor.extract_text(figure_bboxes=figure_bboxes)
            results['links'] = extractor.extract_links()
            results['tables'] = extractor.extract_tables()
            
            if extract_structured:
                results['structured_pages'] = extractor.extract_structured(figure_bboxes, output_dir)
                
    except Exception as e:
        results['errors'].append(str(e))
    
    return results
