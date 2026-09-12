from __future__ import annotations

from collections import deque
from contextlib import closing
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit, urlunsplit

import pymupdf

from .page_classifier import (
    PageFeatures as _PageFeatures,
    Route,
    classify as _classify,
    garbled_ratio as _garbled_ratio,
    has_two_columns as _has_two_columns,
    has_usable_table as _has_usable_table,
    inspect_page as _inspect_page_with,
    is_usable as _is_usable,
    union_area as _union_area,
    weighted_median_font_size as _weighted_median_font_size,
)
from .page_parsers import (
    DoclingParser,
    NeedsRichParser,
    PageParser,
    PlainPyMuPDFParser,
    PyMuPDF4LLMParser,
)
from .pdf_workers import Batch, MAX_WORKERS, run_batches

logger = logging.getLogger(__name__)
_ROUTER_TIMEOUT_SECONDS = 240


@dataclass(frozen=True)
class PageOutcome:
    index: int
    markdown: str = ""
    error: str = ""


def _inspect_page(page: pymupdf.Page) -> _PageFeatures:
    return _inspect_page_with(page, _has_usable_table)


def _worker(send, pdf_path: str, batch: Batch):
    try:
        if batch.route == "inspect":
            with pymupdf.open(pdf_path) as document:
                if document.needs_pass:
                    raise ValueError("encrypted PDF requires a password")
                send.send(("metadata", (document.metadata or {}).get("title") or "", len(document)))
                for index, page in enumerate(document):
                    try:
                        features = _inspect_page(page)
                        route = _classify(features)
                        forced = features.garbled_ratio >= 0.20
                    except Exception:
                        route, forced = "docling", True
                    send.send(("profile", index, route, forced))
            return
        parser: PageParser
        parser = (DoclingParser(batch.force_ocr) if batch.route == "docling" else
                  PyMuPDF4LLMParser() if batch.route == "layout" else PlainPyMuPDFParser())
        for index in batch.indices:
            send.send(("started", index))
            try:
                outcome = PageOutcome(index, parser.parse_page(pdf_path, index))
            except Exception as error:
                outcome = PageOutcome(index, error=type(error).__name__)
            send.send(("page", outcome))
    except Exception as error:
        send.send(("error", type(error).__name__))
    finally:
        send.close()


def _chunks(indices, count):
    size = max(1, math.ceil(len(indices) / count))
    return [tuple(indices[i:i + size]) for i in range(0, len(indices), size)]


class PageRouter:
    def __init__(self, pdf_bytes: bytes, source_url: str):
        if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
            raise ValueError("pdf_bytes must be non-empty bytes")
        if not isinstance(source_url, str) or not source_url:
            raise ValueError("source_url must be a non-empty string")
        self._pdf_bytes = pdf_bytes
        source = urlsplit(source_url)
        self._source_url = (urlunsplit((source.scheme, source.hostname or "", source.path, "", ""))
                            if source.scheme else source_url)
        self._title = ""

    @property
    def title(self) -> str:
        return self._title

    def parse(self) -> str:
        deadline = time.monotonic() + _ROUTER_TIMEOUT_SECONDS
        self._title = ""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf:
                temp_path = pdf.name
                pdf.write(self._pdf_bytes)
            return self._parse(temp_path, deadline)
        finally:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary PDF %s", temp_path)

    def _parse(self, pdf_path, deadline):
        profiles, page_count = {}, None
        with closing(run_batches(pdf_path, deque([Batch("inspect")]), deadline, _worker)) as events:
            for batch, message in events:
                if message[0] == "metadata":
                    self._title, page_count = message[1:]
                elif message[0] == "profile":
                    profiles[message[1]] = (message[2], message[3])
                elif message[0] == "error":
                    logger.warning("PDF inspection failed for %s: %s", self._source_url, message[1])
        if page_count is None:
            raise ValueError("PDF inspection failed or timed out")
        for index in range(page_count):
            profiles.setdefault(index, ("docling", True))
        logger.info("PDF initial routes for %s: %s", self._source_url,
                    {route: sum(value[0] == route for value in profiles.values())
                     for route in ("plain", "layout", "docling", "blank")})
        pending = deque()
        for route in ("plain", "layout", "docling"):
            for forced in (False, True):
                indices = [i for i, profile in profiles.items() if profile == (route, forced)]
                for chunk in _chunks(indices, min(2, MAX_WORKERS) if route == "docling" else MAX_WORKERS):
                    pending.append(Batch(route, chunk, forced))
        parsed, successful, completed, started = {}, {}, {}, {}
        last_route = {index: route for index, (route, _) in profiles.items()}

        def fail(batch, index, reason):
            logger.warning("%s failed for page %d from %s: %s", batch.route, index + 1,
                           self._source_url, reason)
            next_route = {"plain": "layout", "layout": "docling"}.get(batch.route)
            if next_route:
                forced = next_route == "docling" and reason != "NeedsRichParser"
                # Coalesce queued recovery pages, reusing a converter per batch.
                for queued in list(pending):
                    if queued.route == next_route and queued.force_ocr == forced and not queued.retry:
                        pending.remove(queued)
                        pending.append(Batch(next_route, queued.indices + (index,), forced))
                        break
                else:
                    pending.append(Batch(next_route, (index,), forced))

        with closing(run_batches(pdf_path, pending, deadline, _worker)) as events:
            for batch, message in events:
                for index in batch.indices:
                    last_route[index] = batch.route
                done = completed.setdefault(batch, set())
                begun = started.setdefault(batch, set())
                kind = message[0] if isinstance(message, tuple) and message else None
                if kind == "started" and len(message) > 1 and message[1] in batch.indices:
                    begun.add(message[1])
                elif kind == "page":
                    outcome = message[1]
                    if not isinstance(outcome, PageOutcome) or outcome.index not in batch.indices:
                        continue
                    if outcome.index in done:
                        continue
                    done.add(outcome.index)
                    if not outcome.error and _is_usable(outcome.markdown):
                        parsed[outcome.index] = outcome.markdown.strip()
                        successful[outcome.index] = batch.route
                    else:
                        fail(batch, outcome.index, outcome.error or "unusable output")
                elif kind == "finished":
                    untouched = []
                    for index in set(batch.indices) - done:
                        if index not in begun and not batch.retry:
                            untouched.append(index)
                        else:
                            fail(batch, index, "worker stopped without a result")
                    if untouched:
                        pending.append(Batch(batch.route, tuple(sorted(untouched)), batch.force_ocr, 1))
                elif kind == "error":
                    for index in set(batch.indices) - done:
                        done.add(index)
                        fail(batch, index, message[1])
        omitted = [i for i, (route, _) in profiles.items() if route != "blank" and i not in parsed]
        if omitted:
            for index in omitted:
                logger.warning("%s omitted page %d from %s", last_route[index], index + 1, self._source_url)
        if not parsed and omitted:
            self._title = ""
        logger.info("PDF successful backends for %s: %s", self._source_url,
                    {route: sum(value == route for value in successful.values())
                     for route in ("plain", "layout", "docling")})
        return "\n\n".join(parsed[i] for i in sorted(parsed))
