"""
test_crawler.py — Comprehensive tests for the web crawler module.

Uses mocking to avoid real network calls while thoroughly testing
crawling logic, parsing, pagination, retry behaviour, and error handling.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.crawler import Crawler, Page, Quote, fetch_page, has_next_page, parse_quotes

# ---------------------------------------------------------------------------
# Sample HTML fixtures
# ---------------------------------------------------------------------------

SAMPLE_PAGE_1 = """
<!DOCTYPE html>
<html>
<head><title>Quotes to Scrape</title></head>
<body>
<div class="container">
  <div class="quote" itemscope itemtype="http://schema.org/CreativeWork">
    <span class="text" itemprop="text">\u201cThe world as we have created it is a process of our thinking.\u201d</span>
    <span>by <small class="author" itemprop="author">Albert Einstein</small></span>
    <div class="tags">
      <a class="tag" href="/tag/change/page/1/">change</a>
      <a class="tag" href="/tag/thinking/page/1/">thinking</a>
    </div>
  </div>
  <div class="quote" itemscope itemtype="http://schema.org/CreativeWork">
    <span class="text" itemprop="text">\u201cA day without sunshine is like, you know, night.\u201d</span>
    <span>by <small class="author" itemprop="author">Steve Martin</small></span>
    <div class="tags">
      <a class="tag" href="/tag/humor/page/1/">humor</a>
    </div>
  </div>
</div>
<nav>
  <ul class="pager">
    <li class="next"><a href="/page/2/">Next <span aria-hidden="true">&rarr;</span></a></li>
  </ul>
</nav>
</body>
</html>
"""

SAMPLE_PAGE_2 = """
<!DOCTYPE html>
<html>
<head><title>Quotes to Scrape - Page 2</title></head>
<body>
<div class="container">
  <div class="quote" itemscope itemtype="http://schema.org/CreativeWork">
    <span class="text" itemprop="text">\u201cLife is what happens when you're busy making other plans.\u201d</span>
    <span>by <small class="author" itemprop="author">John Lennon</small></span>
    <div class="tags">
      <a class="tag" href="/tag/life/page/1/">life</a>
      <a class="tag" href="/tag/plans/page/1/">plans</a>
    </div>
  </div>
</div>
<nav>
  <ul class="pager">
    <li class="previous"><a href="/page/1/">Previous</a></li>
  </ul>
</nav>
</body>
</html>
"""

EMPTY_PAGE = """
<!DOCTYPE html>
<html><head><title>Empty</title></head>
<body><div class="container"></div></body>
</html>
"""

MALFORMED_HTML = """
<!DOCTYPE html>
<html><head><title>Bad</title></head>
<body>
<div class="quote">
  <span class="text">\u201cNo author here.\u201d</span>
  <!-- Missing author element -->
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Tests: parse_quotes
# ---------------------------------------------------------------------------


class TestParseQuotes:
    """Tests for the HTML parsing function."""

    def test_parse_two_quotes(self) -> None:
        """Should extract both quotes from a page with two entries."""
        result = parse_quotes(SAMPLE_PAGE_1)
        assert len(result) == 2

    def test_first_quote_text(self) -> None:
        """First quote text should be cleaned of Unicode quotes."""
        result = parse_quotes(SAMPLE_PAGE_1)
        assert "The world as we have created it" in result[0]["text"]

    def test_first_quote_author(self) -> None:
        """First quote author should be Albert Einstein."""
        result = parse_quotes(SAMPLE_PAGE_1)
        assert result[0]["author"] == "Albert Einstein"

    def test_first_quote_tags(self) -> None:
        """First quote should have two tags."""
        result = parse_quotes(SAMPLE_PAGE_1)
        assert result[0]["tags"] == ["change", "thinking"]

    def test_second_quote_author(self) -> None:
        """Second quote author should be Steve Martin."""
        result = parse_quotes(SAMPLE_PAGE_1)
        assert result[1]["author"] == "Steve Martin"

    def test_second_quote_tags(self) -> None:
        """Second quote should have one tag."""
        result = parse_quotes(SAMPLE_PAGE_1)
        assert result[1]["tags"] == ["humor"]

    def test_empty_page(self) -> None:
        """An empty page should return zero quotes."""
        result = parse_quotes(EMPTY_PAGE)
        assert result == []

    def test_malformed_html_skips_bad_entries(self) -> None:
        """Quotes missing text or author should be skipped gracefully."""
        result = parse_quotes(MALFORMED_HTML)
        assert result == []

    def test_empty_string_input(self) -> None:
        """Passing an empty string should return an empty list."""
        result = parse_quotes("")
        assert result == []

    def test_page_2_single_quote(self) -> None:
        """Page 2 contains a single quote."""
        result = parse_quotes(SAMPLE_PAGE_2)
        assert len(result) == 1
        assert result[0]["author"] == "John Lennon"


# ---------------------------------------------------------------------------
# Tests: has_next_page
# ---------------------------------------------------------------------------


class TestHasNextPage:
    """Tests for next-page detection."""

    def test_page_with_next(self) -> None:
        """Should return the next page URL when present."""
        result = has_next_page(SAMPLE_PAGE_1)
        assert result == "/page/2/"

    def test_last_page(self) -> None:
        """Should return None on the last page."""
        result = has_next_page(SAMPLE_PAGE_2)
        assert result is None

    def test_empty_html(self) -> None:
        """Should return None for empty HTML."""
        result = has_next_page("")
        assert result is None

    def test_no_nav(self) -> None:
        """Should return None when no navigation element exists."""
        result = has_next_page(EMPTY_PAGE)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Quote dataclass
# ---------------------------------------------------------------------------


class TestQuote:
    """Tests for Quote serialisation/deserialisation."""

    def test_to_dict(self) -> None:
        """Quote should serialise to a dict correctly."""
        q = Quote(text="Hello", author="Author", tags=["tag1"])
        d = q.to_dict()
        assert d == {
            "text": "Hello",
            "author": "Author",
            "tags": ["tag1"],
        }

    def test_from_dict(self) -> None:
        """Quote should deserialise from a dict correctly."""
        d = {"text": "Test", "author": "Auth", "tags": []}
        q = Quote.from_dict(d)
        assert q.text == "Test"
        assert q.author == "Auth"
        assert q.tags == []

    def test_round_trip(self) -> None:
        """to_dict → from_dict should be idempotent."""
        q = Quote(text="Round", author="Trip", tags=["a", "b"])
        q2 = Quote.from_dict(q.to_dict())
        assert q2.text == q.text
        assert q2.author == q.author
        assert q2.tags == q.tags


# ---------------------------------------------------------------------------
# Tests: Page dataclass
# ---------------------------------------------------------------------------


class TestPage:
    """Tests for Page serialisation/deserialisation."""

    def test_to_dict(self) -> None:
        """Page should serialise to a dict correctly."""
        p = Page(
            page_id=0,
            url="https://example.com/page/1/",
            quotes=[Quote(text="Hi", author="A", tags=["t"])],
        )
        d = p.to_dict()
        assert d["page_id"] == 0
        assert d["url"] == "https://example.com/page/1/"
        assert len(d["quotes"]) == 1
        assert d["quotes"][0]["text"] == "Hi"

    def test_from_dict(self) -> None:
        """Page should deserialise from a dict correctly."""
        d = {
            "page_id": 1,
            "url": "https://example.com/page/2/",
            "quotes": [{"text": "Bye", "author": "B", "tags": []}],
        }
        p = Page.from_dict(d)
        assert p.page_id == 1
        assert p.url == "https://example.com/page/2/"
        assert len(p.quotes) == 1
        assert p.quotes[0].text == "Bye"

    def test_full_text(self) -> None:
        """full_text should combine all quote text, authors and tags."""
        p = Page(
            page_id=0,
            url="https://example.com/page/1/",
            quotes=[
                Quote(text="Hello world", author="Alice", tags=["greeting"]),
                Quote(text="Goodbye", author="Bob", tags=["farewell"]),
            ],
        )
        text = p.full_text
        assert "Hello world" in text
        assert "Alice" in text
        assert "greeting" in text
        assert "Goodbye" in text
        assert "Bob" in text

    def test_round_trip(self) -> None:
        """to_dict → from_dict should be idempotent."""
        p = Page(
            page_id=5,
            url="https://example.com/page/6/",
            quotes=[Quote(text="X", author="Y", tags=["z"])],
        )
        p2 = Page.from_dict(p.to_dict())
        assert p2.page_id == p.page_id
        assert p2.url == p.url
        assert len(p2.quotes) == len(p.quotes)
        assert p2.quotes[0].text == p.quotes[0].text


# ---------------------------------------------------------------------------
# Tests: Crawler
# ---------------------------------------------------------------------------


class TestCrawler:
    """Tests for the Crawler class with mocked HTTP."""

    def _make_mock_fetcher(self, pages):
        """Create a mock fetcher that returns pages in sequence.

        Args:
            pages: dict mapping URL suffixes to (html, status_code).
        """
        def mock_fetcher(url, timeout=15):
            resp = MagicMock(spec=requests.Response)
            for suffix, html in pages.items():
                if url.endswith(suffix):
                    resp.text = html
                    resp.status_code = 200
                    return resp
            raise requests.RequestException(f"Not found: {url}")
        return mock_fetcher

    def test_crawl_two_pages(self) -> None:
        """Crawler should traverse pagination and collect all pages."""
        pages_html = {
            "/page/1/": SAMPLE_PAGE_1,
            "/page/2/": SAMPLE_PAGE_2,
        }
        crawler = Crawler(
            base_url="https://quotes.toscrape.com",
            delay=0,  # no delay for tests
            fetcher=self._make_mock_fetcher(pages_html),
        )
        pages = crawler.crawl()
        assert len(pages) == 2
        assert pages[0].page_id == 0
        assert pages[1].page_id == 1
        # Page 1 has 2 quotes, page 2 has 1 quote
        assert len(pages[0].quotes) == 2
        assert len(pages[1].quotes) == 1

    def test_crawl_assigns_sequential_page_ids(self) -> None:
        """Page IDs should be sequential starting from 0."""
        pages_html = {"/page/1/": SAMPLE_PAGE_1, "/page/2/": SAMPLE_PAGE_2}
        crawler = Crawler(delay=0, fetcher=self._make_mock_fetcher(pages_html))
        pages = crawler.crawl()
        ids = [p.page_id for p in pages]
        assert ids == [0, 1]

    def test_crawl_stores_urls(self) -> None:
        """Each page should store its full URL."""
        pages_html = {"/page/1/": SAMPLE_PAGE_1, "/page/2/": SAMPLE_PAGE_2}
        crawler = Crawler(delay=0, fetcher=self._make_mock_fetcher(pages_html))
        pages = crawler.crawl()
        assert pages[0].url == "https://quotes.toscrape.com/page/1/"
        assert pages[1].url == "https://quotes.toscrape.com/page/2/"

    def test_crawl_empty_site(self) -> None:
        """Crawler should handle a site with no quotes gracefully."""
        pages_html = {"/page/1/": EMPTY_PAGE}
        crawler = Crawler(delay=0, fetcher=self._make_mock_fetcher(pages_html))
        pages = crawler.crawl()
        assert pages == []

    def test_crawl_network_failure_aborts(self) -> None:
        """If all retries fail the crawler should stop and return what it has."""
        def failing_fetcher(url, timeout=15):
            raise requests.RequestException("Connection refused")

        crawler = Crawler(delay=0, max_retries=2, fetcher=failing_fetcher)
        pages = crawler.crawl()
        assert pages == []

    def test_retry_logic(self) -> None:
        """Crawler should retry on failure and succeed when a later attempt works."""
        call_count = 0

        def flaky_fetcher(url, timeout=15):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise requests.RequestException("Transient error")
            resp = MagicMock(spec=requests.Response)
            resp.text = SAMPLE_PAGE_2  # single-page (no next)
            resp.status_code = 200
            return resp

        crawler = Crawler(delay=0, max_retries=3, fetcher=flaky_fetcher)
        pages = crawler.crawl()
        assert len(pages) == 1
        assert call_count >= 2

    def test_politeness_delay(self) -> None:
        """Crawler should wait the configured delay between requests."""
        pages = {"/page/1/": SAMPLE_PAGE_2}
        crawler = Crawler(delay=0.1, fetcher=self._make_mock_fetcher(pages))
        start = time.time()
        crawler._last_request_time = time.time()  # simulate previous request
        crawler._wait()
        elapsed = time.time() - start
        assert elapsed >= 0.05  # allow margin

    def test_base_url_trailing_slash(self) -> None:
        """Trailing slashes in base_url should be stripped."""
        crawler = Crawler(base_url="https://example.com/")
        assert crawler.base_url == "https://example.com"

    def test_crawl_extracts_correct_data(self) -> None:
        """Verify extracted text, author, and tags for known input."""
        pages_html = {"/page/1/": SAMPLE_PAGE_2}
        crawler = Crawler(delay=0, fetcher=self._make_mock_fetcher(pages_html))
        pages = crawler.crawl()
        assert pages[0].quotes[0].author == "John Lennon"
        assert "life" in pages[0].quotes[0].tags


# ---------------------------------------------------------------------------
# Tests: fetch_page (with responses mock)
# ---------------------------------------------------------------------------


class TestFetchPage:
    """Test the standalone fetch_page function."""

    @patch("src.crawler.requests.get")
    def test_successful_fetch(self, mock_get: MagicMock) -> None:
        """fetch_page should return a response on success."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        result = fetch_page("https://example.com")
        assert result.status_code == 200
        mock_get.assert_called_once()

    @patch("src.crawler.requests.get")
    def test_fetch_raises_on_error(self, mock_get: MagicMock) -> None:
        """fetch_page should propagate HTTP errors."""
        mock_resp = MagicMock(spec=requests.Response)
        mock_resp.raise_for_status.side_effect = requests.HTTPError("404")
        mock_get.return_value = mock_resp

        with pytest.raises(requests.HTTPError):
            fetch_page("https://example.com/404")

    @patch("src.crawler.requests.get")
    def test_fetch_timeout(self, mock_get: MagicMock) -> None:
        """fetch_page should propagate timeout errors."""
        mock_get.side_effect = requests.Timeout("Timed out")
        with pytest.raises(requests.Timeout):
            fetch_page("https://example.com", timeout=1)

    @patch("src.crawler.requests.get")
    def test_fetch_connection_error(self, mock_get: MagicMock) -> None:
        """fetch_page should propagate connection errors."""
        mock_get.side_effect = requests.ConnectionError("Refused")
        with pytest.raises(requests.ConnectionError):
            fetch_page("https://example.com")
