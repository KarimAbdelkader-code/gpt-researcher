from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import multiprocessing
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pymupdf
import pytest

from gpt_researcher.scraper.pymupdf import page_router as router_module
from gpt_researcher.scraper.pymupdf import pdf_workers
from gpt_researcher.scraper.pymupdf.page_router import (
    Batch, PageOutcome, PageRouter, _classify, _inspect_page, _is_usable,
)


def _pdf_bytes(texts=("Appendix A",), title=""):
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        doc.set_metadata({"title": title})
        return doc.tobytes()


def _no_network(*args, **kwargs):
    raise AssertionError("A PDF unit-test worker attempted network access")


def _checked_worker(send, pdf_path, batch):
    # Spawn does not inherit the pytest network monkeypatch.
    socket.socket.connect = _no_network
    assert threading.current_thread() is threading.main_thread()
    router_module._worker(send, pdf_path, batch)


def _fake_worker(send, pdf_path, batch):
    """Real subprocess/IPC, deterministic parsers, no models or network."""
    socket.socket.connect = _no_network
    try:
        if batch.route == "inspect":
            with pymupdf.open(pdf_path) as doc:
                send.send(("metadata", "Metadata", len(doc)))
                for index in range(len(doc)):
                    send.send(("profile", index, "plain", False))
        else:
            for index in batch.indices:
                send.send(("started", index))
                if index == 0 and batch.route != "docling":
                    send.send(("page", PageOutcome(index, error="synthetic failure")))
                else:
                    time.sleep(0.15 if index == 1 else 0.01)
                    send.send(("page", PageOutcome(index, f"page {index} via {batch.route}")))
    finally:
        send.close()


def _hang_worker(send, path, batch):
    socket.socket.connect = _no_network
    send.send(("pid", os.getpid()))
    send.send(("started", 0))
    send.send(("page", PageOutcome(0, "survives")))
    if batch.route == "crash":
        os._exit(6)
    while True:
        time.sleep(1)


def _echo_worker(send, path, batch):
    socket.socket.connect = _no_network
    send.send(("pid", os.getpid(), time.monotonic()))
    time.sleep(1)
    send.send(("end", time.monotonic()))
    send.close()


def _partial_hang_worker(send, path, batch):
    if batch.route == "inspect":
        _checked_worker(send, path, batch)
    else:
        _hang_worker(send, path, batch)


def _renewing_worker(send, path, batch):
    socket.socket.connect = _no_network
    time.sleep(0.12)
    for index in batch.indices:
        send.send(("started", index))
        time.sleep(0.12)
        send.send(("page", PageOutcome(index, str(index))))
    send.close()


def _initialization_hang_worker(send, path, batch):
    socket.socket.connect = _no_network
    send.send(("pid", os.getpid()))
    while True:
        time.sleep(1)


def _later_page_hang_worker(send, path, batch):
    socket.socket.connect = _no_network
    send.send(("started", batch.indices[0]))
    send.send(("page", PageOutcome(batch.indices[0], "complete")))
    send.send(("started", batch.indices[1]))
    while True:
        time.sleep(1)


def _invalid_start_worker(send, path, batch):
    socket.socket.connect = _no_network
    send.send(("clock", time.monotonic()))
    send.send(("started", batch.indices[0]))
    for message in (("started", batch.indices[0]), ("started",),
                    ("started", 99), ("started", batch.indices[1])):
        time.sleep(0.08)
        send.send(message)
    while True:
        time.sleep(1)


def _global_deadline_worker(send, path, batch):
    socket.socket.connect = _no_network
    send.send(("clock", time.monotonic()))
    send.send(("started", batch.indices[0]))
    send.send(("page", PageOutcome(batch.indices[0], "complete")))
    send.send(("started", batch.indices[1]))
    while True:
        time.sleep(1)


def _use_fast_process_start(monkeypatch):
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("subsecond worker timing requires fork")
    monkeypatch.setattr(pdf_workers.multiprocessing, "get_context", lambda _: context)


@pytest.mark.parametrize("text,route", [("", "blank"), ("A", "plain"), ("123", "plain"),
                                        ("...", "plain"), ("An ordinary short paragraph.", "plain")])
def test_real_sparse_pdf_geometry(text, route):
    with pymupdf.open(stream=_pdf_bytes((text,))) as doc:
        assert _classify(_inspect_page(doc[0])) == route
    assert _is_usable(text) == bool(text)


@pytest.mark.parametrize("value", [None, 123, "", "\ufffd" * 40, "a" * 40 + "\ufffd" * 10])
def test_invalid_output_is_not_research_content(value):
    assert not _is_usable(value)


def test_corruption_precedes_blank():
    assert _classify(router_module._PageFeatures(garbled_ratio=1)) == "docling"


def test_corrupt_native_character_map_is_not_blank():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "X" * 30)
        font = page.get_fonts()[0][0]
        cmap = doc.get_new_xref()
        doc.update_object(cmap, "<<>>")
        doc.update_stream(cmap, b"""/CIDInit /ProcSet findresource begin
12 dict begin begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def
/CMapName /Broken def /CMapType 2 def
1 begincodespacerange <00> <FF> endcodespacerange
1 beginbfchar <58> <FFFD> endbfchar
endcmap CMapName currentdict /CMap defineresource pop end end""")
        doc.xref_set_key(font, "ToUnicode", f"{cmap} 0 R")
        data = doc.tobytes()
    with pymupdf.open(stream=data) as reopened:
        features = _inspect_page(reopened[0])
        assert features.garbled_ratio == 1
        assert _classify(features) == "docling"


def test_tiled_images_and_header_route_to_ocr():
    with pymupdf.open() as doc:
        page = doc.new_page(width=400, height=600)
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(200)
        for rect in [(0, 70, 200, 600), (200, 70, 400, 600)]:
            page.insert_image(rect, pixmap=pix, keep_proportion=False)
        page.insert_text((10, 30), "A digital header outside the scanned body " * 2, fontsize=7)
        features = _inspect_page(page)
        assert features.alnum_count >= 40
        assert features.image_coverage > .85
        assert features.image_text_chars == 0
        assert _classify(features) == "docling"
        assert router_module._union_area([page.rect, page.rect]) == page.rect.get_area()


def test_logo_does_not_force_ocr():
    with pymupdf.open(stream=_pdf_bytes(("A readable paragraph with ordinary text.",))) as doc:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
        pix.clear_with(200)
        doc[0].insert_image((10, 10, 30, 30), pixmap=pix)
        assert _classify(_inspect_page(doc[0])) == "plain"


def test_heading_and_one_block_per_column():
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 72), "Heading", fontsize=24)
        page.insert_textbox((72, 120, 260, 600), "left column ordinary text " * 20)
        page.insert_textbox((330, 120, 530, 600), "right column ordinary text " * 20)
        features = _inspect_page(page)
        assert features.has_heading
        assert features.has_two_columns
        assert _classify(features) == "layout"


def test_real_table_and_unknown_geometry(monkeypatch):
    with pymupdf.open() as doc:
        page = doc.new_page()
        for x in (70, 170, 270):
            page.draw_line((x, 70), (x, 170))
        for y in (70, 120, 170):
            page.draw_line((70, y), (270, y))
        for x, y, text in [(80, 100, "Name"), (180, 100, "Value"), (80, 150, "Row"), (180, 150, "42")]:
            page.insert_text((x, y), text)
        assert _inspect_page(page).has_table
        assert _classify(_inspect_page(page)) == "docling"
    with pymupdf.open(stream=_pdf_bytes()) as doc:
        monkeypatch.setattr(router_module, "_has_usable_table", Mock(side_effect=RuntimeError()))
        assert _classify(_inspect_page(doc[0])) == "layout"


def test_real_router_preserves_short_pages_and_title(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(router_module, "_worker", _checked_worker)
    router = PageRouter(_pdf_bytes(("First", "", "123"), "Title"), "local.pdf")
    assert router.parse() == "First\n\n123"
    assert router.title == "Title"
    assert not list(tmp_path.glob("*.pdf"))
    with pytest.raises(AttributeError):
        router.title = "new"


@pytest.mark.parametrize("encrypted", [False, True])
def test_invalid_or_password_protected_pdf_is_rejected_and_removed(monkeypatch, encrypted):
    data = b"not a PDF"
    if encrypted:
        with pymupdf.open(stream=_pdf_bytes()) as doc:
            data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="secret")
    paths = []
    original = tempfile.NamedTemporaryFile

    def tracked(*args, **kwargs):
        result = original(*args, **kwargs)
        paths.append(result.name)
        return result

    monkeypatch.setattr(router_module.tempfile, "NamedTemporaryFile", tracked)
    monkeypatch.setattr(router_module, "_worker", _checked_worker)
    with pytest.raises(ValueError, match="inspection"):
        PageRouter(data, "local.pdf").parse()
    assert paths and all(not Path(p).exists() for p in paths)


def test_empty_password_encryption_is_readable(monkeypatch):
    with pymupdf.open(stream=_pdf_bytes()) as doc:
        data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="")
    monkeypatch.setattr(router_module, "_worker", _checked_worker)
    assert PageRouter(data, "local.pdf").parse() == "Appendix A"


def test_real_workers_order_and_capability_escalation(monkeypatch, caplog):
    monkeypatch.setattr(router_module, "_worker", _fake_worker)
    with caplog.at_level("INFO"):
        assert PageRouter(_pdf_bytes(("one", "two", "three")), "local.pdf").parse() == (
            "page 0 via docling\n\npage 1 via plain\n\npage 2 via plain"
        )
    assert "PDF initial routes for local.pdf" in caplog.text
    assert "PDF successful backends for local.pdf: {'plain': 2, 'layout': 0, 'docling': 1}" in caplog.text


@pytest.mark.parametrize("route", ["plain", "crash"])
def test_hanging_worker_is_reaped_and_capacity_reusable(monkeypatch, route):
    _use_fast_process_start(monkeypatch)
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 0.15)
    monkeypatch.setattr(pdf_workers, "_slots", threading.BoundedSemaphore(1))
    deadline = time.monotonic() + 10
    messages = list(pdf_workers.run_batches("unused", deque([Batch(route, (0,))]), deadline, _hang_worker))
    assert any(event[0] == "page" for _, event in messages)
    pids = [event[1] for _, event in messages if event[0] == "pid"]
    assert pids
    assert not set(pids) & {p.pid for p in multiprocessing.active_children()}
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 2)
    followup = list(pdf_workers.run_batches("unused", deque([Batch("plain", (0,))]), time.monotonic() + 10, _echo_worker))
    assert any(event[0] == "end" for _, event in followup)


def test_each_started_page_renews_the_worker_budget(monkeypatch):
    _use_fast_process_start(monkeypatch)
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 0.2)
    events = list(pdf_workers.run_batches(
        "unused", deque([Batch("plain", (0, 1))]),
        time.monotonic() + 2, _renewing_worker,
    ))
    assert [event[1].index for _, event in events if event[0] == "page"] == [0, 1]


def test_initialization_hang_expires_without_a_page_start(monkeypatch):
    _use_fast_process_start(monkeypatch)
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 0.15)
    events = list(pdf_workers.run_batches(
        "unused", deque([Batch("plain", (0,))]),
        time.monotonic() + 2, _initialization_hang_worker,
    ))
    pids = [event[1] for _, event in events if event[0] == "pid"]
    assert pids and not set(pids) & {p.pid for p in multiprocessing.active_children()}
    assert not any(event[0] in {"started", "page"} for _, event in events)


def test_later_page_hang_keeps_completed_page(monkeypatch):
    _use_fast_process_start(monkeypatch)
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 0.15)
    events = list(pdf_workers.run_batches(
        "unused", deque([Batch("plain", (0, 1))]),
        time.monotonic() + 2, _later_page_hang_worker,
    ))
    assert [event[1].index for _, event in events if event[0] == "page"] == [0]


def test_invalid_starts_do_not_renew_the_worker_budget(monkeypatch):
    _use_fast_process_start(monkeypatch)
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 0.4)
    events = list(pdf_workers.run_batches(
        "unused", deque([Batch("plain", (0, 1))]),
        time.monotonic() + 2, _invalid_start_worker,
    ))
    started = next(event[1] for _, event in events if event[0] == "clock")
    assert sum(event[0] == "started" for _, event in events) == 5
    assert time.monotonic() - started < 0.6


def test_page_renewal_never_extends_the_document_deadline(monkeypatch):
    _use_fast_process_start(monkeypatch)
    monkeypatch.setattr(pdf_workers, "_LIGHT_TIMEOUT_SECONDS", 1)
    deadline = time.monotonic() + 0.3
    events = list(pdf_workers.run_batches(
        "unused", deque([Batch("plain", (0, 1))]), deadline,
        _global_deadline_worker,
    ))
    started = next(event[1] for _, event in events if event[0] == "clock")
    assert time.monotonic() - started < 0.4
    assert any(event[0] == "page" for _, event in events)


def test_timeout_removes_pdf_only_after_children_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(router_module, "_worker", _partial_hang_worker)
    original_run = router_module.run_batches
    monotonic = time.monotonic

    def expire_after_success(path, pending, deadline, target):
        with closing(original_run(path, pending, deadline, target)) as events:
            for batch, event in events:
                yield batch, event
                if event[0] == "page":
                    # Start the short timeout only once the child is ready.
                    # Slow interpreter imports under --forked are not a failure.
                    offset = deadline - monotonic() - 0.1
                    monkeypatch.setattr(time, "monotonic", lambda: monotonic() + offset)

    monkeypatch.setattr(router_module, "run_batches", expire_after_success)
    stop = pdf_workers._stop
    observed = []

    def checked_stop(process):
        assert list(tmp_path.glob("*.pdf")), "input removed while a child still owns it"
        observed.append(process.pid)
        stop(process)

    monkeypatch.setattr(pdf_workers, "_stop", checked_stop)
    assert PageRouter(_pdf_bytes(), "local.pdf").parse() == "survives"
    assert observed and not list(tmp_path.glob("*.pdf"))
    assert not set(observed) & {p.pid for p in multiprocessing.active_children()}


def test_heavy_batches_really_overlap():
    if pdf_workers.MAX_WORKERS < 2:
        pytest.skip("two CPU slots required to demonstrate parallelism")
    jobs = deque([Batch("docling", (0,)), Batch("docling", (1,))])
    events = list(pdf_workers.run_batches("unused", jobs, time.monotonic() + 20, _echo_worker))
    starts = [event[2] for _, event in events if event[0] == "pid"]
    ends = [event[1] for _, event in events if event[0] == "end"]
    assert len(starts) == len(ends) == 2
    assert max(starts) < min(ends)


def test_concurrent_callers_never_inspect_in_parent(monkeypatch):
    data = _pdf_bytes()
    monkeypatch.setattr(router_module, "_worker", _checked_worker)
    monkeypatch.setattr(router_module.pymupdf, "open", Mock(side_effect=AssertionError("parent PDF access")))
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: PageRouter(data, "local.pdf").parse(), range(2)))
    assert results == ["Appendix A"] * 2


def test_adapter_verifies_layout_page_identity(monkeypatch):
    fake = SimpleNamespace(use_layout=Mock(), to_markdown=Mock(return_value=[{
        "metadata": {"page_number": 3}, "text": "third page",
    }]))
    monkeypatch.setitem(sys.modules, "pymupdf4llm", fake)
    parser = router_module.PyMuPDF4LLMParser()
    assert parser.parse_page("local.pdf", 2) == "third page"
    with pytest.raises(ValueError, match="wrong source page"):
        parser.parse_page("local.pdf", 0)
    assert fake.to_markdown.call_args.kwargs["use_ocr"] is False
    fake.to_markdown.return_value[0]["page_boxes"] = [{"class": "table"}]
    with pytest.raises(router_module.NeedsRichParser):
        parser.parse_page("local.pdf", 2)


def test_partial_failure_malformed_and_duplicate_outcomes(monkeypatch):
    def events(path, pending, deadline, target):
        if pending[0].route == "inspect":
            batch = pending.popleft()
            yield batch, ("metadata", "Title", 3)
            for index in range(3):
                yield batch, ("profile", index, "plain", False)
            return
        while pending:
            batch = pending.popleft()
            for index in reversed(batch.indices):
                yield batch, ("started",)
                yield batch, ("started", index)
                yield batch, ("page", PageOutcome(99, "wrong page"))
                yield batch, ("page", PageOutcome(index, None if index == 1 else str(index)))
                yield batch, ("page", PageOutcome(index, "duplicate"))
            yield batch, ("finished",)
    monkeypatch.setattr(router_module, "run_batches", events)
    assert PageRouter(_pdf_bytes(), "local.pdf").parse() == "0\n\n2"


def test_cleanup_on_write_failure(monkeypatch):
    paths = []
    original = tempfile.NamedTemporaryFile

    def broken(*args, **kwargs):
        file = original(*args, **kwargs)
        paths.append(file.name)
        file.write = Mock(side_effect=OSError("disk full"))
        return file

    monkeypatch.setattr(router_module.tempfile, "NamedTemporaryFile", broken)
    with pytest.raises(OSError, match="disk full"):
        PageRouter(_pdf_bytes(), "local.pdf").parse()
    assert paths and all(not Path(p).exists() for p in paths)


def test_cleanup_os_error_does_not_discard_success(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(PageRouter, "_parse", lambda *args: "extracted")
    monkeypatch.setattr(Path, "unlink", Mock(side_effect=PermissionError()))
    assert PageRouter(_pdf_bytes(), "local.pdf").parse() == "extracted"
    assert "Could not remove temporary PDF" in caplog.text


def test_total_failure_clears_title_without_error_markdown(monkeypatch):
    def events(path, pending, deadline, target):
        while pending:
            batch = pending.popleft()
            if batch.route == "inspect":
                yield batch, ("metadata", "Title", 1)
                yield batch, ("profile", 0, "docling", True)
            else:
                yield batch, ("error", "MissingModels")
            yield batch, ("finished",)
    monkeypatch.setattr(router_module, "run_batches", events)
    router = PageRouter(_pdf_bytes(), "https://user:secret@example.com/a.pdf?token=private")
    assert router.parse() == ""
    assert router.title == ""
    assert router._source_url == "https://example.com/a.pdf"


def test_docling_configuration_and_page_range(monkeypatch, tmp_path):
    from docling.datamodel.base_models import ConversionStatus, InputFormat
    from docling.datamodel.pipeline_options import OcrMode
    from docling.datamodel.settings import settings
    import docling.document_converter as conversion

    converter = Mock()
    factory = Mock(return_value=converter)
    monkeypatch.setattr(conversion, "DocumentConverter", factory)
    monkeypatch.setattr(settings, "artifacts_path", tmp_path)
    parser = router_module.DoclingParser(force_ocr=True)
    options = factory.call_args.kwargs["format_options"][InputFormat.PDF].pipeline_options
    assert options.artifacts_path == tmp_path
    assert options.ocr_options.mode == OcrMode.FULL_PAGE
    assert options.accelerator_options.num_threads == 1
    assert not options.enable_remote_services and not options.allow_external_plugins
    converter.initialize_pipeline.assert_called_once_with(InputFormat.PDF)
    converter.convert.return_value.status = ConversionStatus.SUCCESS
    converter.convert.return_value.errors = []
    converter.convert.return_value.document.export_to_markdown.return_value = "third"
    assert parser.parse_page("local.pdf", 2) == "third"
    assert converter.convert.call_args.kwargs["page_range"] == (3, 3)
    converter.convert.return_value.status = ConversionStatus.PARTIAL_SUCCESS
    with pytest.raises(ValueError, match="incomplete"):
        parser.parse_page("local.pdf", 2)
