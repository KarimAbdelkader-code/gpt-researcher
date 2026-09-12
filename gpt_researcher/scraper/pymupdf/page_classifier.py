from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import pymupdf

Route = Literal["plain", "layout", "docling", "blank"]


@dataclass(frozen=True)
class PageFeatures:
    alnum_count: int = 0
    garbled_ratio: float = 0
    image_coverage: float = 0
    image_text_chars: int = 0
    has_text: bool = False
    has_paint: bool = False
    has_table: bool = False
    has_two_columns: bool = False
    has_heading: bool = False
    geometry_unknown: bool = False


def garbled_ratio(text: str) -> float:
    visible = [char for char in text if char not in " \t\r\n"]
    bad = sum(char == "\ufffd" or not char.isprintable() for char in visible)
    return bad / len(visible) if visible else 0


def is_usable(markdown) -> bool:
    return isinstance(markdown, str) and bool(markdown.strip()) and garbled_ratio(markdown) < 0.20


def classify(features: PageFeatures) -> Route:
    if features.garbled_ratio >= 0.20:
        return "docling"
    if not features.has_text:
        return "docling" if features.has_paint else "blank"
    if features.image_coverage >= 0.70 and features.image_text_chars < 40:
        return "docling"
    if features.has_table:
        return "docling"
    if (features.has_two_columns or features.has_heading
            or features.geometry_unknown):
        return "layout"
    return "plain"


def weighted_median_font_size(spans: Sequence[tuple[float, int]]) -> float:
    midpoint = sum(weight for _, weight in spans) / 2
    running = 0
    for size, weight in sorted(spans):
        running += weight
        if running >= midpoint:
            return size
    return 0


def union_area(rectangles) -> float:
    """Sweep x edges and merge y intervals to count overlapping scans once."""
    edges = sorted({x for r in rectangles for x in (r.x0, r.x1)})
    area = 0.0
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((r.y0, r.y1) for r in rectangles if r.x0 < right and r.x1 > left)
        end = float("-inf")
        height = 0.0
        for low, high in intervals:
            height += max(0, high - max(low, end))
            end = max(end, high)
        area += (right - left) * height
    return area


def has_two_columns(page: pymupdf.Page, total_chars: int) -> bool:
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]
    midpoint = page.rect.x0 + page.rect.width / 2
    left = [b for b in blocks if b[2] <= midpoint]
    right = [b for b in blocks if b[0] >= midpoint]
    if not left or not right:
        return False
    counts = [sum(sum(c.isalnum() for c in b[4]) for b in group) for group in (left, right)]
    if min(counts) < max(20, total_chars * 0.20):
        return False
    if min(b[0] for b in right) - max(b[2] for b in left) < page.rect.width * 0.02:
        return False
    tops = [min(b[1] for b in group) for group in (left, right)]
    bottoms = [max(b[3] for b in group) for group in (left, right)]
    overlap = min(bottoms) - max(tops)
    height = min(bottom - top for top, bottom in zip(tops, bottoms))
    return height > 0 and overlap / height >= 0.25


def has_usable_table(page: pymupdf.Page) -> bool:
    for strategy in ("lines_strict", "text"):
        for table in page.find_tables(strategy=strategy).tables:
            rows = table.extract()
            columns = max((len(row) for row in rows), default=0)
            populated = sum(bool(cell and cell.strip()) for row in rows for cell in row)
            if len(rows) >= 2 and columns >= 2 and populated >= len(rows) * columns / 2:
                return True
    return False


def inspect_page(
    page: pymupdf.Page,
    table_detector: Callable[[pymupdf.Page], bool] = has_usable_table,
) -> PageFeatures:
    # Otherwise MuPDF replaces broken Unicode mappings with plausible glyph IDs,
    # hiding precisely the damaged text layer that needs full-page OCR.
    text = page.get_text(flags=pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_USE_CID_FOR_UNKNOWN_UNICODE)
    alnum = sum(c.isalnum() for c in text)
    rectangles = [pymupdf.Rect(info["bbox"]) & page.rect for info in page.get_image_info()]
    rectangles = [r for r in rectangles if not r.is_empty]
    image_text = sum(
        sum(c.isalnum() for c in word[4]) for word in page.get_text("words")
        if any(pymupdf.Point((word[0] + word[2]) / 2, (word[1] + word[3]) / 2) in r for r in rectangles)
    )
    spans, short_spans = [], []
    for block in page.get_text("dict", flags=pymupdf.TEXTFLAGS_TEXT)["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                weight = sum(c.isalnum() for c in span["text"])
                if weight:
                    item = (float(span["size"]), weight)
                    spans.append(item)
                    if len(span["text"].strip()) <= 120:
                        short_spans.append(item)
    body = weighted_median_font_size(spans)
    heading = body > 0 and any(size >= body * 1.35 for size, _ in short_spans)
    columns = has_two_columns(page, alnum)
    table, unknown = False, False
    if text.strip() and not heading and not columns:
        try:
            table = table_detector(page)
        except Exception:
            unknown = True
    paint = bool(rectangles)
    if not text.strip() and not paint:
        paint = bool(page.get_drawings() or list(page.annots() or ())
                     or any(widget.field_value for widget in page.widgets() or ()))
    return PageFeatures(
        alnum_count=alnum, garbled_ratio=garbled_ratio(text),
        image_coverage=union_area(rectangles) / max(1, page.rect.get_area()),
        image_text_chars=image_text, has_text=bool(text.strip()), has_paint=paint,
        has_table=table, has_two_columns=columns, has_heading=heading,
        geometry_unknown=unknown,
    )
