import requests
from urllib.parse import urlparse

from .page_router import PageRouter


class PyMuPDFScraper:

    def __init__(self, link, session=None):
        """
        Initialize the scraper with a link and an optional session.

        Args:
          link (str): The URL or local file path of the PDF document.
          session (requests.Session, optional): An optional session for making HTTP requests.
        """
        self.link = link
        self.session = session

    def is_url(self) -> bool:
        """
        Check if the provided `link` is a valid URL.

        Returns:
          bool: True if the link is a valid URL, False otherwise.
        """
        try:
            result = urlparse(self.link)
            return all([result.scheme, result.netloc])  # Check for valid scheme and network location
        except Exception:
            return False

    def scrape(self) -> tuple[str, list[str], str]:
        """
        Load a PDF from the provided link or local path and route its pages to
        the lightest parser that can preserve their content.

        Returns:
          str: A string representation of the loaded document.
        """
        try:
            if self.is_url():
                http = self.session or requests
                try:
                    response = http.get(self.link, timeout=(5, 30), stream=True)
                    response.raise_for_status()
                except requests.exceptions.SSLError:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"SSL verification failed for {self.link}, retrying without verification"
                    )
                    response = http.get(self.link, timeout=(5, 30), stream=True, verify=False)
                    response.raise_for_status()

                pdf_bytes = b"".join(response.iter_content(chunk_size=8192))
            else:
                with open(self.link, "rb") as pdf_file:
                    pdf_bytes = pdf_file.read()

            router = PageRouter(pdf_bytes, str(self.link))
            content = router.parse()
            return content, [], router.title

        except requests.exceptions.Timeout:
            print(f"Download timed out. Please check the link : {self.link}")
            return "", [], ""
        except Exception as e:
            print(f"Error loading PDF : {self.link} {e}")
            return "", [], ""
