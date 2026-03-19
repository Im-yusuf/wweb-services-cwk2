"""
crawler.py — Web crawler for quotes.toscrape.com

Crawls all pages of the quotes website, extracting quote text, author, and tags.
Implements polite crawling with configurable delay and retry logic with
exponential backoff.  Designed for testability with separated HTTP request logic.

Design choices
--------------
* **Politeness** — A configurable delay (default 6 s) is enforced between
  consecutive requests using wall-clock comparison, ensuring we never hit the
  server faster than the allowed rate regardless of processing time.

* **Resilience** — Transient HTTP errors are retried up to *max_retries*
  times with exponential back-off (2^attempt seconds), preventing a single
  flaky response from aborting the entire crawl.

* **Testability** — The HTTP layer (:func:`fetch_page`) is injected as a
  callable, allowing tests to substitute a mock without monkey-patching.

Complexity
----------
``crawl()`` visits *P* pages sequentially.  Each page yields *Q* quotes
parsed in *O(H)* where *H* is the HTML size.  Overall: **O(P · (H + D))**
where *D* is the inter-request delay.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """Represents a single scraped quote."""

    text: str
    author: str
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise quote to a plain dictionary."""
        return {
            "text": self.text,
            "author": self.author,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Quote":
        """Deserialise a quote from a dictionary."""
        return cls(
            text=data["text"],
            author=data["author"],
            tags=data["tags"],
        )


@dataclass
class Page:
    """Represents a single crawled web page containing quotes.

    Each page of the website is treated as one document in the index.
    """

    page_id: int
    url: str
    quotes: List[Quote] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Combine all quote text, authors, and tags for indexing."""
        parts = []
        for q in self.quotes:
            parts.append(f"{q.text} {q.author} {' '.join(q.tags)}")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise page to a plain dictionary."""
        return {
            "page_id": self.page_id,
            "url": self.url,
            "quotes": [q.to_dict() for q in self.quotes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Page":
        """Deserialise a page from a dictionary."""
        return cls(
            page_id=data["page_id"],
            url=data["url"],
            quotes=[Quote.from_dict(q) for q in data["quotes"]],
        )


# ---------------------------------------------------------------------------
# HTTP fetching (separated for easy mocking)
# ---------------------------------------------------------------------------

def fetch_page(url: str, timeout: int = 15) -> requests.Response:
    """Perform a single HTTP GET request.

    Separated from crawling logic so tests can mock this function
    without touching the crawler's control flow.

    Args:
        url: The URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        A ``requests.Response`` object.

    Raises:
        requests.RequestException: On any network-level failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; UniversitySearchBot/1.0; "
            "+https://example.com/bot)"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def parse_quotes(html: str) -> List[Dict[str, Any]]:
    """Extract quotes from an HTML page.

    Args:
        html: Raw HTML content of a quotes page.

    Returns:
        List of dicts with keys ``text``, ``author``, ``tags``.
    """
    soup = BeautifulSoup(html, "lxml")
    results: List[Dict[str, Any]] = []

    for div in soup.select("div.quote"):
        text_span = div.select_one("span.text")
        author_span = div.select_one("small.author")
        tag_anchors = div.select("a.tag")

        if text_span is None or author_span is None:
            logger.warning("Skipping malformed quote block — missing text or author.")
            continue

        # Strip surrounding Unicode quotes (\u201c, \u201d)
        raw_text = text_span.get_text(strip=True)
        cleaned_text = raw_text.strip("\u201c\u201d\"'")

        results.append({
            "text": cleaned_text,
            "author": author_span.get_text(strip=True),
            "tags": [a.get_text(strip=True) for a in tag_anchors],
        })

    return results


def has_next_page(html: str) -> Optional[str]:
    """Return the relative URL of the next page, or ``None``.

    Args:
        html: Raw HTML of the current page.

    Returns:
        Relative path string (e.g. ``/page/2/``) or ``None``.
    """
    soup = BeautifulSoup(html, "lxml")
    next_btn = soup.select_one("li.next > a")
    if next_btn and next_btn.get("href"):
        return next_btn["href"]
    return None


# ---------------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------------

class Crawler:
    """Polite web crawler for quotes.toscrape.com.

    Attributes:
        base_url: Root URL of the target site.
        delay: Seconds to wait between consecutive requests.
        max_retries: Maximum retry attempts per request.
        fetcher: Callable used to fetch a URL (defaults to :func:`fetch_page`).
    """

    def __init__(
        self,
        base_url: str = "https://quotes.toscrape.com",
        delay: float = 6.0,
        max_retries: int = 3,
        fetcher: Optional[Callable[..., requests.Response]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.max_retries = max_retries
        self.fetcher = fetcher or fetch_page
        self._last_request_time: float = 0.0

    # ----- politeness -----

    def _wait(self) -> None:
        """Block until the politeness window has elapsed since the last request."""
        elapsed = time.time() - self._last_request_time
        remaining = self.delay - elapsed
        if remaining > 0:
            logger.debug("Sleeping %.2f s for politeness", remaining)
            time.sleep(remaining)

    # ----- fetching with retry -----

    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """Fetch a URL with exponential-backoff retry.

        Args:
            url: Absolute URL to retrieve.

        Returns:
            HTML string on success, ``None`` if all attempts fail.
        """
        for attempt in range(1, self.max_retries + 1):
            self._wait()
            try:
                self._last_request_time = time.time()
                response = self.fetcher(url)
                logger.info("Fetched %s (attempt %d)", url, attempt)
                return response.text
            except requests.RequestException as exc:
                backoff = 2 ** attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt,
                    self.max_retries,
                    url,
                    exc,
                    backoff,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)

        logger.error("All %d attempts failed for %s", self.max_retries, url)
        return None

    # ----- main entry point -----

    def crawl(self) -> List[Page]:
        """Crawl all pages of the quotes site.

        Iterates through paginated listing pages, extracts quotes, and
        groups them by page. Each page becomes one indexed document.

        Returns:
            Ordered list of :class:`Page` objects.
        """
        pages: List[Page] = []
        page_id_counter = 0
        path = "/page/1/"

        while path is not None:
            url = f"{self.base_url}{path}"
            logger.info("Crawling %s …", url)
            html = self._fetch_with_retry(url)

            if html is None:
                logger.error("Skipping page %s — could not fetch.", url)
                break

            raw_quotes = parse_quotes(html)
            quotes = [
                Quote(text=raw["text"], author=raw["author"], tags=raw["tags"])
                for raw in raw_quotes
            ]

            if quotes:
                pages.append(Page(
                    page_id=page_id_counter,
                    url=url,
                    quotes=quotes,
                ))
                page_id_counter += 1

            path = has_next_page(html)
            logger.info(
                "Crawled %d pages so far. Next page: %s",
                len(pages),
                path,
            )

        logger.info("Crawling complete — %d pages collected.", len(pages))
        return pages
