from pathlib import Path
from typing import Protocol

import pymupdf


class PageParser(Protocol):
    """One adapter per worker batch; expensive initialization is reused."""

    def parse_page(self, pdf_path: str, page_index: int) -> str: ...


class NeedsRichParser(ValueError):
    """The adapter detected structure it cannot reliably preserve."""


class PlainPyMuPDFParser:
    def parse_page(self, pdf_path: str, page_index: int) -> str:
        with pymupdf.open(pdf_path) as document:
            return document[page_index].get_text(sort=True).strip()


class PyMuPDF4LLMParser:
    def parse_page(self, pdf_path: str, page_index: int) -> str:
        import pymupdf4llm

        pymupdf4llm.use_layout(True)
        chunks = pymupdf4llm.to_markdown(
            pdf_path, pages=[page_index], page_chunks=True,
            show_progress=False, use_ocr=False, write_images=False,
            embed_images=False, header=True, footer=True,
        )
        if not isinstance(chunks, list) or len(chunks) != 1:
            raise ValueError("Expected one layout page")
        chunk = chunks[0]
        if chunk["metadata"]["page_number"] != page_index + 1:
            raise ValueError("Layout returned the wrong source page")
        # The supplied tables expose merged columns/rows in layout 1.28.2.
        # Do not accept plausible-looking but structurally incorrect Markdown.
        if any(box.get("class") == "table" for box in chunk.get("page_boxes", [])):
            raise NeedsRichParser("Tables require Docling")
        return chunk["text"]


class DoclingParser:
    def __init__(self, force_ocr: bool = False):
        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMode, PdfPipelineOptions, RapidOcrOptions
        from docling.datamodel.settings import settings
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.do_ocr = True
        options.do_table_structure = True
        options.document_timeout = 120
        options.enable_remote_services = False
        options.allow_external_plugins = False
        options.artifacts_path = settings.artifacts_path or settings.cache_dir / "models"
        if not Path(options.artifacts_path).is_dir():
            raise FileNotFoundError("Prefetch Docling models before scraping PDFs")
        options.accelerator_options = AcceleratorOptions(num_threads=1, device="cpu")
        options.ocr_options = RapidOcrOptions(
            lang=["en"], backend="onnxruntime",
            mode=OcrMode.FULL_PAGE if force_ocr else OcrMode.DEFAULT,
        )
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)},
        )
        self.converter.initialize_pipeline(InputFormat.PDF)

    def parse_page(self, pdf_path: str, page_index: int) -> str:
        from docling.datamodel.base_models import ConversionStatus

        result = self.converter.convert(
            pdf_path, page_range=(page_index + 1, page_index + 1), raises_on_error=False,
        )
        if result.status != ConversionStatus.SUCCESS or result.errors:
            raise ValueError("Docling conversion incomplete")
        return result.document.export_to_markdown(image_placeholder="")
