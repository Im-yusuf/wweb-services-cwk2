"""
Shared fixtures for the test suite.

Provides reusable page collections, pre-built indexers, and search engines
to avoid duplicating setup logic across test modules.
"""

import os
import tempfile
from typing import Generator, List

import pytest

from src.crawler import Page, Quote
from src.indexer import Indexer
from src.search import SearchEngine


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_quotes() -> List[Quote]:
    """A handful of quotes for quick unit tests."""
    return [
        Quote(
            text="The world as we have created it is a process of our thinking",
            author="Albert Einstein",
            tags=["change", "thinking", "world"],
        ),
        Quote(
            text="Love all trust a few do wrong to none",
            author="William Shakespeare",
            tags=["love", "trust"],
        ),
    ]


@pytest.fixture
def sample_pages(sample_quotes: List[Quote]) -> List[Page]:
    """Three pages with varied content for integration tests."""
    return [
        Page(
            page_id=0,
            url="https://quotes.toscrape.com/page/1/",
            quotes=[sample_quotes[0], sample_quotes[1]],
        ),
        Page(
            page_id=1,
            url="https://quotes.toscrape.com/page/2/",
            quotes=[
                Quote(
                    text="Life is what happens when you are busy making other plans",
                    author="John Lennon",
                    tags=["life", "plans"],
                ),
                Quote(
                    text="The only way to do great work is to love what you do",
                    author="Steve Jobs",
                    tags=["work", "love", "inspirational"],
                ),
            ],
        ),
        Page(
            page_id=2,
            url="https://quotes.toscrape.com/page/3/",
            quotes=[
                Quote(
                    text="In three words I can sum up everything I learned about life it goes on",
                    author="Robert Frost",
                    tags=["life"],
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Pre-built components
# ---------------------------------------------------------------------------


@pytest.fixture
def built_indexer(sample_pages: List[Page]) -> Indexer:
    """An indexer already populated from :func:`sample_pages`."""
    indexer = Indexer()
    indexer.build_index(sample_pages)
    return indexer


@pytest.fixture
def search_engine(built_indexer: Indexer) -> SearchEngine:
    """A search engine wrapping :func:`built_indexer`."""
    return SearchEngine(built_indexer)


# ---------------------------------------------------------------------------
# Temporary directory for persistence tests
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_index_path() -> Generator[str, None, None]:
    """Yield a temporary file path for index save/load tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_index.json")
