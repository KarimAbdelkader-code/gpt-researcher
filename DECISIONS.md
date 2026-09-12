# PDF PageRouter: engineering decisions

## Integration and interface design

The scraper boundary stays unchanged. PyMuPDFScraper still downloads or opens the PDF, handles session reuse, SSL retry, and (5, 30) HTTP timeouts. Once bytes are available it calls PageRouter(pdf_bytes, source_url).parse(), then returns the existing (content, [], title) tuple. A blank but valid PDF keeps its metadata title; total extraction failure returns the existing empty result.

The implementation is split by responsibility: page_router.py owns orchestration and result ordering, page_classifier.py owns inspection and routing signals, page_parsers.py contains the backend adapters, and pdf_workers.py supervises child processes. Each parser implements PageParser.parse_page(path, zero_based_index) -> str. Adding a backend therefore requires an adapter and a routing/fallback decision, not changes to downstream research code. Worker processes receive independent document handles so PyMuPDF and model state are never shared across caller threads.

## How pages are classified, and alternatives considered

Classification uses cheap, deterministic evidence already present in PyMuPDF: native text, replacement/non-printable characters, image coverage, text-block geometry, font sizes, drawings, and table geometry. The capability ladder is:

1. Plain PyMuPDF for healthy, simple digital pages.
2. PyMuPDF4LLM for columns or heading structure, with OCR disabled.
3. Docling with local RapidOCR for scans, corrupt text, and tables.

The checks are deliberately ordered. Corrupt text (at least 20% replacement or non-printable characters) is detected before blankness. A page with no text but painted content, or at least 70% raster coverage and fewer than 40 native alphanumeric characters in its image regions, is treated as a scan. Image rectangles are unioned so tiled or overlapping images are not counted twice. Column detection requires separated populated regions with overlapping vertical ranges; heading detection compares short spans with the character-weighted median body font using a 1.35× threshold.

I chose observable geometry over a learned classifier because there is no representative labelled corpus and adding another model would increase startup cost and failure modes. I also rejected “always Docling”: it gives stronger structure, but it is much slower and can use substantial memory. The supplied fixtures exposed an important table case: PyMuPDF4LLM merged the RoBERTa table columns, while Docling recovered the aligned cells. Detected tables therefore escalate to Docling. If table inspection fails, the geometry is treated as unknown and the page gets the safer layout route rather than silently taking plain text.

## Backends implemented and the gap each closes

- **Plain PyMuPDF** is the fast path for ordinary digital pages; it avoids model startup and preserves the existing extraction behavior.
- **PyMuPDF4LLM** improves reading order and heading structure for digital columns without paying for full document conversion or OCR.
- **Docling + RapidOCR** handles scans, damaged character maps, and tables where plausible native text would be incomplete or misleading.

OCR and layout models are local-only in this implementation. Remote services and external plugins are disabled. The richer dependencies make installation heavier, and model artifacts should be prefetched rather than downloaded during a request. RapidOCR/model redistribution and existing PyMuPDF Artifex licensing still need deployment-level review.

## How work is parallelised, and what constrains it

Inspection and extraction run in spawned child processes, each on its own main thread and document handle. Extraction batches run concurrently. A process-wide semaphore caps PDF workers at min(4, CPU count) and a second semaphore caps Docling/OCR work at two workers. Recovery pages are grouped where possible to reuse converter initialization, and original page indexes restore document order.

I chose disposable workers rather than a permanent shared pool. Startup is more expensive, but a crashed or hung native/model process cannot poison later requests. The 240-second document deadline includes queue admission, inspection, extraction, and retries. The limits are per application process, not deployment-wide; a multi-replica deployment still needs container-level CPU and memory quotas. I did not add a hard input-size, page-count, or RSS quota in this exercise.

## Backend failures, exceptions, and hangs

Worker messages are validated for page identity, batch membership, duplicates, missing results, and malformed output. Failures escalate once along the capability ladder: plain → layout → Docling. A page/backend pair is not retried indefinitely. Successful pages survive failures, and a page that never started may be reassigned once.

Inspection receives a 30-second allowance; light extraction receives 30 seconds and Docling 120 seconds, all capped by the 240-second document deadline. Only a valid, in-budget start for the next expected page renews the allowance. Duplicate, late, overlapping, or out-of-batch messages do not extend it. A hung child is terminated, given one second to exit, then killed and reaped before its capacity slot is released.

Safe native text can survive a formatting-only failure. For scans, corruption, tables, or uncertain column order, I prefer an explicit omission to plausible but wrong text. Final omissions log the sanitized source, page, backend, reason, and outcome; error text and URL query data do not enter research content. Private temporary PDFs are cleaned up even after write, parse, or worker failures.

## Fixture evidence

I compared the unchanged local-file baseline (PyMuPDFLoader.load() plus newline joining) with the router on all four supplied, hash-pinned fixtures. Default tests use fakes and remain offline; opt-in real-model tests run after model prefetch.

| Fixture | Baseline | Routed result |
| --- | --- | --- |
| BERT two-column paper | Fast, but reading order is flattened | Stable anchors from all 16 pages, heading/body structure, and paragraph continuity |
| RoBERTa borderless tables | Table cells are flattened | Stable anchors from all 13 pages and aligned Markdown cells |
| Scanned letter | No text returned | OCR recovers the expected letter phrases |
| Mixed five-page document | Structured pages degrade and scan is empty | All five page anchors survive in order with mixed routing |

Five targeted real-model fixture tests passed. Focused Python 3.11 validation passed 52 tests, covering routing, ordering, escalation, malformed messages, initialization and later-page hangs, timeout renewal, process reaping, scraper compatibility, retries, and temporary-file cleanup. Earlier timing runs showed the richer path is much slower than plain extraction, but they overlapped and were not controlled benchmarks; I do not present them as precise throughput claims.

## Trade-offs and work deliberately excluded

- **Quality versus cost:** simple pages stay on native extraction; only risky pages pay for layout or OCR.
- **Isolation versus startup time:** disposable processes improve failure containment but cost startup and model initialization.
- **Partial output versus corruption:** safe completed pages are retained, while unresolved semantic risk is omitted.
- **Scope:** I did not add equation reconstruction, chart understanding, multilingual OCR, password handling, custom rendering, or a vision-language backend. Each needs separate validation, cost, and privacy decisions.

## What I would do with another week

I would build a larger labelled page corpus and measure routing precision and recall by failure type. I would run controlled warm and cold throughput/RSS tests under concurrent scraper load, test the Python 3.12–3.14 matrix, add explicit input/resource budgets, and pin model artifacts and licences in the deployment image. I would tune thresholds only after those measurements show where the current heuristics fail.

## Known weaknesses and uncertainties

The main weakness is classification generality: the thresholds were tuned against four fixtures, so unusual layouts can still produce false positives or false negatives. Structural validation cannot prove that readable OCR is correct or that every paragraph is in the ideal order. OCR is currently English-only, model startup is expensive, and the per-process resource caps do not coordinate across replicas. These are explicit limits, not claims that the router solves every kind of PDF.
