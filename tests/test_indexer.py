"""
test_indexer.py — Comprehensive tests for the inverted index module.

Covers tokenization, stop-word removal, index building, TF-IDF computation,
persistence (save/load), and edge cases.
"""

import json
import math
import os
import tempfile
from typing import List

import pytest

from src.crawler import Page, Quote
from src.indexer import (
    STOP_WORDS,
    Indexer,
    remove_stop_words,
    tokenize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pages() -> List[Page]:
    """Create a small set of test pages."""
    return [
        Page(
            page_id=0,
            url="https://quotes.toscrape.com/page/1/",
            quotes=[
                Quote(
                    text="The world as we have created it is a process of our thinking",
                    author="Albert Einstein",
                    tags=["change", "thinking"],
                ),
                Quote(
                    text="A day without sunshine is like you know night",
                    author="Steve Martin",
                    tags=["humor"],
                ),
            ],
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
            ],
        ),
    ]


@pytest.fixture
def sample_pages() -> List[Page]:
    """Pytest fixture returning sample pages."""
    return _make_pages()


@pytest.fixture
def built_indexer(sample_pages: List[Page]) -> Indexer:
    """Pytest fixture returning an indexer with a built index."""
    indexer = Indexer()
    indexer.build_index(sample_pages)
    return indexer


# ---------------------------------------------------------------------------
# Tests: tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the tokenize function."""

    def test_basic_tokenization(self) -> None:
        """Simple sentence should be split into lowercase tokens."""
        tokens = tokenize("Hello World")
        assert tokens == ["hello", "world"]

    def test_punctuation_removal(self) -> None:
        """Punctuation should be stripped from tokens."""
        tokens = tokenize("Hello, World! How's it going?")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in "".join(tokens)

    def test_unicode_quotes_removed(self) -> None:
        """Unicode curly quotes should be stripped."""
        tokens = tokenize("\u201cHello\u201d")
        assert tokens == ["hello"]

    def test_numbers_excluded(self) -> None:
        """Pure digit tokens should be excluded."""
        tokens = tokenize("Room 101 is here")
        assert "101" not in tokens
        assert "room" in tokens

    def test_empty_string(self) -> None:
        """Empty string should return empty list."""
        assert tokenize("") == []

    def test_whitespace_only(self) -> None:
        """Whitespace-only string should return empty list."""
        assert tokenize("   \t\n  ") == []

    def test_mixed_case(self) -> None:
        """All tokens should be lowercased."""
        tokens = tokenize("UPPER lower MiXeD")
        assert tokens == ["upper", "lower", "mixed"]

    def test_multiple_spaces(self) -> None:
        """Multiple spaces should not produce empty tokens."""
        tokens = tokenize("hello   world")
        assert tokens == ["hello", "world"]


# ---------------------------------------------------------------------------
# Tests: remove_stop_words
# ---------------------------------------------------------------------------


class TestRemoveStopWords:
    """Tests for stop-word removal."""

    def test_removes_common_words(self) -> None:
        """Common stop words should be filtered out."""
        tokens = ["the", "world", "is", "a", "process"]
        result = remove_stop_words(tokens)
        assert "the" not in result
        assert "is" not in result
        assert "a" not in result
        assert "world" in result
        assert "process" in result

    def test_preserves_order(self) -> None:
        """Remaining tokens should maintain original order."""
        tokens = ["the", "quick", "brown", "fox"]
        result = remove_stop_words(tokens)
        assert result == ["quick", "brown", "fox"]

    def test_empty_list(self) -> None:
        """Empty list should return empty list."""
        assert remove_stop_words([]) == []

    def test_all_stop_words(self) -> None:
        """A list of only stop words should return empty."""
        tokens = ["the", "is", "a", "and", "or"]
        assert remove_stop_words(tokens) == []

    def test_no_stop_words(self) -> None:
        """A list with no stop words should be unchanged."""
        tokens = ["python", "javascript", "ruby"]
        assert remove_stop_words(tokens) == tokens


# ---------------------------------------------------------------------------
# Tests: Indexer.build_index
# ---------------------------------------------------------------------------


class TestIndexerBuild:
    """Tests for index building."""

    def test_index_has_terms(self, built_indexer: Indexer) -> None:
        """Built index should contain expected terms."""
        assert "world" in built_indexer.index
        assert "einstein" in built_indexer.index

    def test_total_docs(self, built_indexer: Indexer) -> None:
        """total_docs should match the number of indexed pages."""
        assert built_indexer.total_docs == 2

    def test_vocab_set(self, built_indexer: Indexer) -> None:
        """vocab should be a set of all unique terms."""
        assert isinstance(built_indexer.vocab, set)
        assert len(built_indexer.vocab) == len(built_indexer.index)

    def test_documents_stored(self, built_indexer: Indexer) -> None:
        """All pages should be stored with their metadata."""
        assert len(built_indexer.documents) == 2
        assert built_indexer.documents[0]["url"] == "https://quotes.toscrape.com/page/1/"
        assert len(built_indexer.documents[0]["quotes"]) == 2
        assert built_indexer.documents[1]["url"] == "https://quotes.toscrape.com/page/2/"

    def test_frequency_correct(self, built_indexer: Indexer) -> None:
        """Term frequency count should be accurate."""
        # "thinking" appears in doc 0 text AND in tags
        postings = built_indexer.get_postings("thinking")
        assert postings is not None
        assert 0 in postings
        assert postings[0]["frequency"] == 2  # once in text, once in tags

    def test_positions_recorded(self, built_indexer: Indexer) -> None:
        """Positional data should be recorded for each term."""
        postings = built_indexer.get_postings("world")
        assert postings is not None
        assert 0 in postings
        assert isinstance(postings[0]["positions"], list)
        assert len(postings[0]["positions"]) >= 1

    def test_tf_computed(self, built_indexer: Indexer) -> None:
        """TF values should be between 0 and 1."""
        for term, entry in built_indexer.index.items():
            for doc_id, posting in entry["postings"].items():
                assert 0 < posting["tf"] <= 1.0

    def test_idf_computed(self, built_indexer: Indexer) -> None:
        """IDF values should be positive."""
        for term, entry in built_indexer.index.items():
            assert entry["idf"] > 0

    def test_empty_quotes_list(self) -> None:
        """Building from an empty list should produce an empty index."""
        indexer = Indexer()
        indexer.build_index([])
        assert indexer.index == {}
        assert indexer.total_docs == 0
        assert indexer.vocab == set()

    def test_single_page(self) -> None:
        """Index should work correctly with a single page."""
        indexer = Indexer()
        indexer.build_index([
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote(text="Hello world", author="Test", tags=["greeting"])],
            ),
        ])
        assert indexer.total_docs == 1
        assert "hello" in indexer.index
        assert "world" in indexer.index
        assert "greeting" in indexer.index


# ---------------------------------------------------------------------------
# Tests: Indexer.get_* helpers
# ---------------------------------------------------------------------------


class TestIndexerLookups:
    """Tests for index lookup methods."""

    def test_get_postings_existing(self, built_indexer: Indexer) -> None:
        """get_postings should return postings for existing terms."""
        postings = built_indexer.get_postings("world")
        assert postings is not None
        assert isinstance(postings, dict)

    def test_get_postings_missing(self, built_indexer: Indexer) -> None:
        """get_postings should return None for unknown terms."""
        assert built_indexer.get_postings("xyznonexistent") is None

    def test_get_idf_existing(self, built_indexer: Indexer) -> None:
        """get_idf should return a positive float for existing terms."""
        idf = built_indexer.get_idf("world")
        assert idf > 0

    def test_get_idf_missing(self, built_indexer: Indexer) -> None:
        """get_idf should return 0.0 for unknown terms."""
        assert built_indexer.get_idf("xyznonexistent") == 0.0

    def test_get_document_existing(self, built_indexer: Indexer) -> None:
        """get_document should return metadata for valid page_id."""
        doc = built_indexer.get_document(0)
        assert doc is not None
        assert doc["url"] == "https://quotes.toscrape.com/page/1/"

    def test_get_document_missing(self, built_indexer: Indexer) -> None:
        """get_document should return None for invalid doc_id."""
        assert built_indexer.get_document(999) is None

    def test_get_term_entry(self, built_indexer: Indexer) -> None:
        """get_term_entry should return full entry with postings and IDF."""
        entry = built_indexer.get_term_entry("world")
        assert entry is not None
        assert "postings" in entry
        assert "idf" in entry

    def test_get_term_entry_missing(self, built_indexer: Indexer) -> None:
        """get_term_entry should return None for unknown terms."""
        assert built_indexer.get_term_entry("xyznonexistent") is None


# ---------------------------------------------------------------------------
# Tests: IDF calculation
# ---------------------------------------------------------------------------


class TestIDF:
    """Tests for IDF computation correctness."""

    def test_idf_rare_term_higher(self, built_indexer: Indexer) -> None:
        """A term appearing in fewer documents should have higher IDF."""
        # "einstein" appears only in doc 0
        # "world" may also appear in just one doc
        idf_einstein = built_indexer.get_idf("einstein")
        # A term common to all docs would have lower IDF
        # We'll compare against a hypothetical; at minimum IDF > 1 for rare
        assert idf_einstein > 1.0

    def test_idf_formula_correctness(self, built_indexer: Indexer) -> None:
        """Verify the smoothed IDF formula: log((1+N)/(1+df)) + 1."""
        for term, entry in built_indexer.index.items():
            df = len(entry["postings"])
            expected_idf = math.log(
                (1 + built_indexer.total_docs) / (1 + df)
            ) + 1
            assert abs(entry["idf"] - expected_idf) < 1e-9


# ---------------------------------------------------------------------------
# Tests: Persistence (save/load)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Tests for saving and loading the index."""

    def test_save_and_load_round_trip(self, built_indexer: Indexer) -> None:
        """Saving then loading should reproduce the same index state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_index.json")
            built_indexer.save_to_disk(filepath)

            loaded = Indexer()
            loaded.load_from_disk(filepath)

            assert loaded.total_docs == built_indexer.total_docs
            assert loaded.vocab == built_indexer.vocab
            assert len(loaded.documents) == len(built_indexer.documents)

            # Verify a specific term
            orig_entry = built_indexer.get_term_entry("world")
            load_entry = loaded.get_term_entry("world")
            assert orig_entry is not None
            assert load_entry is not None
            assert abs(orig_entry["idf"] - load_entry["idf"]) < 1e-9

    def test_save_creates_directory(self) -> None:
        """save_to_disk should create parent directories if missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "nested", "index.json")
            indexer = Indexer()
            indexer.build_index([
                Page(
                    page_id=0,
                    url="https://example.com/page/1/",
                    quotes=[Quote(text="Hello", author="Test", tags=[])],
                ),
            ])
            indexer.save_to_disk(filepath)
            assert os.path.exists(filepath)

    def test_load_nonexistent_file(self) -> None:
        """Loading from a nonexistent file should raise FileNotFoundError."""
        indexer = Indexer()
        with pytest.raises(FileNotFoundError):
            indexer.load_from_disk("/tmp/nonexistent_file_12345.json")

    def test_load_invalid_json(self) -> None:
        """Loading invalid JSON should raise an error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("NOT VALID JSON {{{")
            f.flush()
            try:
                indexer = Indexer()
                with pytest.raises(json.JSONDecodeError):
                    indexer.load_from_disk(f.name)
            finally:
                os.unlink(f.name)

    def test_load_missing_keys(self) -> None:
        """Loading JSON with missing required keys should raise KeyError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"wrong_key": 1}, f)
            f.flush()
            try:
                indexer = Indexer()
                with pytest.raises(KeyError):
                    indexer.load_from_disk(f.name)
            finally:
                os.unlink(f.name)

    def test_saved_file_valid_json(self, built_indexer: Indexer) -> None:
        """The saved file should be valid, loadable JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "index.json")
            built_indexer.save_to_disk(filepath)
            with open(filepath, "r") as fh:
                data = json.load(fh)
            assert "total_docs" in data
            assert "index" in data
            assert "documents" in data

    def test_doc_id_keys_are_integers_after_load(
        self, built_indexer: Indexer
    ) -> None:
        """After loading, document ID keys should be integers, not strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "index.json")
            built_indexer.save_to_disk(filepath)

            loaded = Indexer()
            loaded.load_from_disk(filepath)

            for doc_id in loaded.documents:
                assert isinstance(doc_id, int)

            for term, entry in loaded.index.items():
                for doc_id in entry["postings"]:
                    assert isinstance(doc_id, int)


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for the indexer."""

    def test_quote_with_only_stop_words(self) -> None:
        """A page with only stop words should still be indexed (tokens exist)."""
        indexer = Indexer()
        indexer.build_index([
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote(text="the is a an", author="X", tags=[])],
            ),
        ])
        # Stop words ARE indexed (we index all tokens; stop-word removal is
        # applied at search time if desired)
        assert indexer.total_docs == 1

    def test_duplicate_words_in_quote(self) -> None:
        """Duplicate words should have correct frequency and multiple positions."""
        indexer = Indexer()
        indexer.build_index([
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote(text="love love love", author="X", tags=[])],
            ),
        ])
        postings = indexer.get_postings("love")
        assert postings is not None
        assert postings[0]["frequency"] == 3
        assert len(postings[0]["positions"]) == 3

    def test_special_characters(self) -> None:
        """Special characters should be handled gracefully."""
        indexer = Indexer()
        indexer.build_index([
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote(
                    text="Hello! @world #python $money",
                    author="Test",
                    tags=[],
                )],
            ),
        ])
        assert "hello" in indexer.index
        assert "world" in indexer.index

    def test_rebuild_overwrites(self) -> None:
        """Building the index again should fully replace the previous one."""
        indexer = Indexer()
        indexer.build_index([
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote(text="first build", author="A", tags=[])],
            ),
        ])
        assert "first" in indexer.vocab

        indexer.build_index([
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote(text="second build", author="B", tags=[])],
            ),
        ])
        assert "second" in indexer.vocab
        assert "first" not in indexer.vocab
