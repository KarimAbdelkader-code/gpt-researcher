from unittest.mock import Mock, call, patch

import requests

from gpt_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper


class _FakeResponse:
    def __init__(self, chunks=(b"%PDF", b" body")):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size=8192):
        assert chunk_size == 8192
        yield from self._chunks


def _mock_router(content="content", title="title"):
    router = Mock()
    router.parse.return_value = content
    router.title = title
    return router


def test_remote_pdf_uses_session_and_preserves_tuple():
    session = Mock()
    session.get.return_value = _FakeResponse()
    router = _mock_router()

    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PageRouter", return_value=router
    ) as router_class:
        result = PyMuPDFScraper("https://example.com/file.pdf", session).scrape()

    assert result == ("content", [], "title")
    session.get.assert_called_once_with(
        "https://example.com/file.pdf", timeout=(5, 30), stream=True
    )
    router_class.assert_called_once_with(b"%PDF body", "https://example.com/file.pdf")


def test_ssl_failure_retries_without_verification():
    session = Mock()
    session.get.side_effect = [requests.exceptions.SSLError(), _FakeResponse()]

    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PageRouter",
        return_value=_mock_router(),
    ):
        result = PyMuPDFScraper("https://example.com/file.pdf", session).scrape()

    assert result == ("content", [], "title")
    assert session.get.call_args_list == [
        call("https://example.com/file.pdf", timeout=(5, 30), stream=True),
        call(
            "https://example.com/file.pdf",
            timeout=(5, 30),
            stream=True,
            verify=False,
        ),
    ]


def test_download_timeout_preserves_failure_tuple():
    session = Mock()
    session.get.side_effect = requests.exceptions.Timeout()

    assert PyMuPDFScraper("https://example.com/file.pdf", session).scrape() == (
        "",
        [],
        "",
    )


def test_local_pdf_becomes_bytes(tmp_path):
    pdf_path = tmp_path / "file.pdf"
    pdf_path.write_bytes(b"local PDF")

    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PageRouter",
        return_value=_mock_router(),
    ) as router_class:
        result = PyMuPDFScraper(str(pdf_path)).scrape()

    assert result == ("content", [], "title")
    router_class.assert_called_once_with(b"local PDF", str(pdf_path))


def test_empty_router_result_preserves_total_failure_tuple(tmp_path):
    path = tmp_path / "local.pdf"
    path.write_bytes(b"PDF")
    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PageRouter",
        return_value=_mock_router(content="", title=""),
    ) as router_class:
        result = PyMuPDFScraper(str(path)).scrape()

    assert result == ("", [], "")
    router_class.assert_called_once_with(b"PDF", str(path))


def test_blank_document_preserves_metadata(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(b"PDF")
    with patch(
        "gpt_researcher.scraper.pymupdf.pymupdf.PageRouter",
        return_value=_mock_router(content="", title="Blank title"),
    ) as router_class:
        assert PyMuPDFScraper(str(path)).scrape() == ("", [], "Blank title")
    router_class.assert_called_once()
