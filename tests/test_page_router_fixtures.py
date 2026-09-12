import os
import json
import re
import socket
import time
from functools import lru_cache
from pathlib import Path

import pymupdf
import pytest

from gpt_researcher.scraper.pymupdf import page_router

from gpt_researcher.scraper.pymupdf.page_router import (
    PageRouter,
    _classify,
    _inspect_page,
)


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_PDF_ROUTER_FIXTURES") != "1",
    reason="set RUN_PDF_ROUTER_FIXTURES=1 after prefetching Docling models",
)

FIXTURES = Path(__file__).parents[1] / "pdf-router-fixtures"
PAGE_ANCHORS = {
    "two-column-paper.pdf": [
        "Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova",
        "word based only on its context. Unlike left-toright language model",
        "Masked Sentence A Masked Sentence B",
        "Input/Output Representations To make BERT handle a variety",
        "Figure 2: BERT input representation",
        "Table 1: GLUE Test results",
        "Table 2: SQuAD 1.1 results",
        "Table 5: Ablation over the pre-training tasks",
        "mixed results on the downstream task impact of increasing the",
        "Guo, and Llion Jones. 2018. Character-level language modeling with deeper",
        "Triviaqa: A large scale distantly supervised challenge dataset",
        "Proceedings of the 2018 EMNLP Workshop BlackboxNLP",
        "Figure 3: Differences in pre-training model architectures. BERT uses a",
        "We also observed that large data sets (e.g., 100k+ labeled",
        "Figure 4: Illustrations of Fine-tuning BERT",
        "C.1 Effect of Number of Training Steps",
    ],
    "borderless-table.pdf": [
        "Paul G. Allen School of Computer Science & Engineering",
        "alternatives that lead to better downstream task performance; (2) We",
        "optimization hyperparameters, given in Section 2, except for the peak",
        "some questions are not answered in the provided context",
        "Table 2: Development set results for base models pretrained",
        "development set accuracy for base models trained over BOOKCORPUS",
        "Table 4: Development set results for RoBERTa as we pretrain",
        "Table 5: Results on GLUE. All results are based on",
        "Table 6: Results on SQuAD",
        "We carefully evaluate a number of design decisions when pretraining",
        "Mandar Joshi, Danqi Chen, Yinhan Liu, Daniel S.",
        "Ashish Vaswani, Noam Shazeer, Niki Parmar",
        "MNLI QNLI QQP RTE SST MRPC CoLA STS",
    ],
    "scanned-page.pdf": ["SLEREXE COMPANY LIMITED"],
    "mixed-document.pdf": [
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "BERT advances the state of the art for eleven",
        "SLEREXE COMPANY LIMITED",
        "Our reimplementation (with NSP loss)",
        "Early experiments revealed only slight differences",
    ],
}


def _offline_worker(send, path, batch):
    def blocked(*args, **kwargs):
        raise AssertionError("PDF fixture attempted network access after prefetch")
    socket.socket.connect = blocked
    page_router._worker(send, path, batch)


@pytest.fixture(autouse=True)
def offline_children(monkeypatch):
    monkeypatch.setattr(page_router, "_worker", _offline_worker)


@lru_cache(maxsize=4)
def _parse(name: str) -> str:
    from langchain_community.document_loaders import PyMuPDFLoader

    path = FIXTURES / name
    start = time.monotonic()
    # Same local-file extraction and recombination as the unmodified scraper.
    baseline = "\n".join(page.page_content for page in PyMuPDFLoader(str(path)).load())
    baseline_time = time.monotonic() - start
    start = time.monotonic()
    markdown = PageRouter(path.read_bytes(), str(path)).parse()
    print(json.dumps({"fixture": name, "baseline_seconds": baseline_time,
                      "baseline_chars": len(baseline), "router_seconds": time.monotonic() - start,
                      "router_chars": len(markdown)}))
    return markdown


def _normalized(markdown):
    text = re.sub(r"-\s*\n\s*", "", markdown)
    return " ".join(text.replace("*", "").replace("#", "").split())


def _assert_page_order(markdown, fixture):
    normalized = _normalized(markdown).casefold()
    missing = [anchor for anchor in PAGE_ANCHORS[fixture]
               if anchor.casefold() not in normalized]
    assert not missing, f"missing page anchors: {missing}"
    positions = [normalized.index(anchor.casefold()) for anchor in PAGE_ANCHORS[fixture]]
    assert positions == sorted(positions), list(zip(PAGE_ANCHORS[fixture], positions))


def test_bert_reading_order():
    markdown = _parse("two-column-paper.pdf")
    _assert_page_order(markdown, "two-column-paper.pdf")
    normalized = _normalized(markdown)
    assert re.search(r"(?m)^#+\s+\**BERT: Pre-training", markdown)
    assert "There are two existing strategies" in normalized
    assert "unlabeled text by jointly conditioning on both left and right context in all layers" in normalized
    assert "BERT is conceptually simple and empirically powerful" in normalized


def test_roberta_tables_are_markdown_cells():
    markdown = _parse("borderless-table.pdf")
    _assert_page_order(markdown, "borderless-table.pdf")
    table_lines = [line for line in markdown.splitlines() if line.count("|") >= 3]
    assert any(
        all(header in line for header in ("Masking", "SQuAD", "MNLI", "SST"))
        for line in table_lines
    )
    assert any("dynamic" in line and "78.7" in line for line in table_lines)
    header_index = next(i for i, line in enumerate(table_lines) if "Masking" in line)
    cells = lambda line: [cell.strip() for cell in line.strip().strip("|").split("|")]
    headers = cells(table_lines[header_index])
    separator = cells(table_lines[header_index + 1])
    assert len(headers) == len(separator)
    assert all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator)
    dynamic = next(cells(line) for line in table_lines if "dynamic" in line and "78.7" in line)
    assert len(dynamic) == len(headers)
    assert "dynamic" in dynamic[0].lower()
    assert "78.7" in dynamic[1]


def test_scanned_letter_has_stable_ocr_phrases():
    markdown = _parse("scanned-page.pdf")
    _assert_page_order(markdown, "scanned-page.pdf")
    assert "SLEREXE COMPANY LIMITED" in markdown.upper()
    assert "facility of facsimile transmission" in markdown.lower()


def test_mixed_document_uses_multiple_routes_and_keeps_page_order():
    path = FIXTURES / "mixed-document.pdf"
    with pymupdf.open(path) as document:
        routes = [_classify(_inspect_page(page)) for page in document]

    assert routes[2] == "docling"
    assert len(set(routes)) >= 2

    markdown = _parse("mixed-document.pdf")
    _assert_page_order(markdown, "mixed-document.pdf")


def test_all_scanned_document_keeps_both_pages():
    with pymupdf.open(FIXTURES / "scanned-page.pdf") as source, pymupdf.open() as combined:
        combined.insert_pdf(source)
        combined.insert_pdf(source)
        data = combined.tobytes()
    markdown = PageRouter(data, "two-scanned-pages.pdf").parse()
    assert markdown.upper().count("SLEREXE COMPANY LIMITED") == 2
    assert markdown.lower().count("facility of facsimile transmission") == 2
