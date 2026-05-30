"""
pdf_parser.py — Parse PDFs into structured text chunks with metadata.
Uses PyMuPDF (fitz) as primary parser, pdfplumber for tables.
"""
import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
import re


class PDFParser:
    """
    Extracts text from PDFs with page-level metadata.
    Handles: text blocks, tables, multi-column layouts.
    """

    def __init__(self, extract_tables: bool = True):
        self.extract_tables = extract_tables

    def parse(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Parse a PDF file and return list of page-level text blocks.

        Returns:
            List of dicts: {text, page, source, section, type}
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {path.name}")
        pages = []

        # Primary: PyMuPDF for text extraction
        doc = fitz.open(str(path))
        toc = self._extract_toc(doc)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = self._extract_page_text(page)

            if not text.strip():
                continue

            section = self._find_section(toc, page_num + 1)

            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": path.name,
                "source_path": str(path),
                "section": section,
                "type": "pdf",
                "url": None,
            })

        doc.close()

        # Optional: Extract tables with pdfplumber
        if self.extract_tables:
            table_blocks = self._extract_tables(str(path))
            pages.extend(table_blocks)

        logger.info(f"Parsed {len(pages)} blocks from {path.name}")
        return pages

    def _extract_page_text(self, page: fitz.Page) -> str:
        """Extract clean text from a PDF page."""
        blocks = page.get_text("blocks")
        texts = []

        for block in blocks:
            if block[6] == 0:  # Text block (not image)
                text = block[4].strip()
                if text and len(text) > 20:  # Filter noise
                    texts.append(text)

        return "\n\n".join(texts)

    def _extract_toc(self, doc: fitz.Document) -> List[Dict]:
        """Extract table of contents for section detection."""
        toc = []
        try:
            raw_toc = doc.get_toc()
            for level, title, page in raw_toc:
                toc.append({"level": level, "title": title, "page": page})
        except Exception:
            pass
        return toc

    def _find_section(self, toc: List[Dict], page_num: int) -> Optional[str]:
        """Find the section title for a given page number."""
        section = None
        for item in toc:
            if item["page"] <= page_num:
                section = item["title"]
            else:
                break
        return section

    def _extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber."""
        table_blocks = []
        path = Path(pdf_path)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    tables = page.extract_tables()
                    for table in tables:
                        if table and len(table) > 1:
                            # Convert table to readable text
                            headers = [str(h or "").strip() for h in table[0]]
                            rows = []
                            for row in table[1:]:
                                cells = [str(c or "").strip() for c in row]
                                row_text = " | ".join(
                                    f"{h}: {c}" for h, c in zip(headers, cells) if c
                                )
                                if row_text:
                                    rows.append(row_text)

                            if rows:
                                table_text = f"Table on page {page_num}:\n" + "\n".join(rows)
                                table_blocks.append({
                                    "text": table_text,
                                    "page": page_num,
                                    "source": path.name,
                                    "source_path": str(path),
                                    "section": "Table",
                                    "type": "pdf_table",
                                    "url": None,
                                })
        except Exception as e:
            logger.warning(f"Table extraction failed: {e}")

        return table_blocks