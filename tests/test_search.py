"""
test_search.py — Comprehensive tests for the search engine module.

Covers TF-IDF ranked search, Boolean query parsing and evaluation,
phrase search (positional index), query suggestions (edit distance),
edge cases, performance benchmarks, and integration tests.
"""

import time
from typing import List

import pytest

from src.crawler import Page, Quote
from src.indexer import Indexer
from src.search import (
    QueryParser,
    SearchEngine,
    SearchResult,
    Token,
    TokenType,
    _edit_distance,
    tokenize_query,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_test_indexer() -> Indexer:
    """Build an indexer with a diverse set of pages for testing."""
    pages = [
        Page(
            page_id=0,
            url="https://quotes.toscrape.com/page/1/",
            quotes=[
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
    indexer = Indexer()
    indexer.build_index(pages)
    return indexer


@pytest.fixture
def indexer() -> Indexer:
    """Fixture: pre-built indexer."""
    return _build_test_indexer()


@pytest.fixture
def engine(indexer: Indexer) -> SearchEngine:
    """Fixture: search engine wrapping the test indexer."""
    return SearchEngine(indexer)


# ---------------------------------------------------------------------------
# Tests: tokenize_query (lexer)
# ---------------------------------------------------------------------------


class TestTokenizeQuery:
    """Tests for the Boolean query lexer."""

    def test_simple_word(self) -> None:
        """Single word should produce WORD + EOF."""
        tokens = tokenize_query("love")
        assert tokens[0].type == TokenType.WORD
        assert tokens[0].value == "love"
        assert tokens[-1].type == TokenType.EOF

    def test_and_operator(self) -> None:
        """AND should be recognized as an operator."""
        tokens = tokenize_query("love AND life")
        types = [t.type for t in tokens]
        assert TokenType.AND in types

    def test_or_operator(self) -> None:
        """OR should be recognized as an operator."""
        tokens = tokenize_query("love OR life")
        types = [t.type for t in tokens]
        assert TokenType.OR in types

    def test_not_operator(self) -> None:
        """NOT should be recognized as an operator."""
        tokens = tokenize_query("NOT hate")
        assert tokens[0].type == TokenType.NOT

    def test_parentheses(self) -> None:
        """Parentheses should be recognized."""
        tokens = tokenize_query("(love OR life) AND work")
        types = [t.type for t in tokens]
        assert TokenType.LPAREN in types
        assert TokenType.RPAREN in types

    def test_case_insensitive_operators(self) -> None:
        """Operators should be recognized regardless of case."""
        tokens = tokenize_query("love and life or work")
        types = [t.type for t in tokens]
        assert types.count(TokenType.AND) == 1
        assert types.count(TokenType.OR) == 1

    def test_empty_query(self) -> None:
        """Empty query should produce only EOF."""
        tokens = tokenize_query("")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_whitespace_query(self) -> None:
        """Whitespace-only query should produce only EOF."""
        tokens = tokenize_query("   ")
        assert len(tokens) == 1
        assert tokens[0].type == TokenType.EOF

    def test_complex_query(self) -> None:
        """Complex nested query should be lexed correctly."""
        tokens = tokenize_query("(love AND life) OR (NOT work)")
        word_tokens = [t for t in tokens if t.type == TokenType.WORD]
        assert len(word_tokens) == 3


# ---------------------------------------------------------------------------
# Tests: QueryParser
# ---------------------------------------------------------------------------


class TestQueryParser:
    """Tests for the Boolean query parser."""

    def test_single_word(self, indexer: Indexer) -> None:
        """Single word query should return docs containing that word."""
        tokens = tokenize_query("love")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        assert isinstance(result, set)
        assert len(result) > 0

    def test_and_query(self, indexer: Indexer) -> None:
        """AND should return intersection of document sets."""
        tokens = tokenize_query("love AND work")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        # Page 1 has both "love" (Jobs quote) and "work" (Jobs quote)
        assert 1 in result

    def test_or_query(self, indexer: Indexer) -> None:
        """OR should return union of document sets."""
        tokens = tokenize_query("einstein OR frost")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        assert 0 in result  # Einstein on page 0
        assert 2 in result  # Frost on page 2

    def test_not_query(self, indexer: Indexer) -> None:
        """NOT should exclude documents containing the term."""
        tokens = tokenize_query("NOT love")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        # Pages with "love" (0 and 1) should be excluded
        assert 0 not in result
        assert 1 not in result

    def test_complex_boolean(self, indexer: Indexer) -> None:
        """Complex nested Boolean should evaluate correctly."""
        tokens = tokenize_query("(love OR world) AND NOT frost")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        # Should include pages with love or world, but not Frost (page 2)
        assert 2 not in result

    def test_nonexistent_word(self, indexer: Indexer) -> None:
        """A word not in the index should return empty set."""
        tokens = tokenize_query("xyznonexistent")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        assert result == set()

    def test_parenthesised_expression(self, indexer: Indexer) -> None:
        """Parentheses should control evaluation order."""
        tokens = tokenize_query("(love)")
        parser = QueryParser(tokens, indexer)
        result = parser.parse()
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Tests: _edit_distance
# ---------------------------------------------------------------------------


class TestEditDistance:
    """Tests for the Levenshtein edit distance function."""

    @pytest.mark.parametrize("a, b, expected", [
        ("hello", "hello", 0),
        ("cat", "cats", 1),
        ("cats", "cat", 1),
        ("cat", "bat", 1),
        ("", "", 0),
        ("hello", "", 5),
        ("", "world", 5),
        ("abc", "xyz", 3),
        ("kitten", "sitting", 3),
    ])
    def test_edit_distance_cases(self, a: str, b: str, expected: int) -> None:
        """Parametrized edit distance verification."""
        assert _edit_distance(a, b) == expected

    def test_symmetry(self) -> None:
        """Distance should be symmetric."""
        assert _edit_distance("kitten", "sitting") == _edit_distance(
            "sitting", "kitten"
        )

    def test_symmetry(self) -> None:
        """Distance should be symmetric."""
        assert _edit_distance("kitten", "sitting") == _edit_distance(
            "sitting", "kitten"
        )


# ---------------------------------------------------------------------------
# Tests: SearchEngine.search (ranked TF-IDF)
# ---------------------------------------------------------------------------


class TestRankedSearch:
    """Tests for TF-IDF ranked search."""

    def test_single_word_search(self, engine: SearchEngine) -> None:
        """Single word should return relevant results."""
        results = engine.search("love")
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_sorted_by_score(self, engine: SearchEngine) -> None:
        """Results should be sorted by descending score."""
        results = engine.search("love")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multi_word_search_and_semantics(self, engine: SearchEngine) -> None:
        """Multi-word query should return only pages matching ALL terms."""
        results = engine.search("love life")
        # Only pages containing BOTH "love" and "life" should match
        for r in results:
            doc = r.document
            full_text = " ".join(
                f"{q['text']} {q['author']} {' '.join(q['tags'])}"
                for q in doc["quotes"]
            ).lower()
            assert "love" in full_text
            assert "life" in full_text

    def test_empty_query(self, engine: SearchEngine) -> None:
        """Empty query should return no results."""
        assert engine.search("") == []

    def test_whitespace_query(self, engine: SearchEngine) -> None:
        """Whitespace-only query should return no results."""
        assert engine.search("   ") == []

    def test_unknown_word(self, engine: SearchEngine) -> None:
        """Query with no matching terms should return empty."""
        assert engine.search("xyznonexistent") == []

    def test_top_k_limit(self, engine: SearchEngine) -> None:
        """Results should be limited by top_k parameter."""
        results = engine.search("love", top_k=1)
        assert len(results) <= 1

    def test_result_has_document(self, engine: SearchEngine) -> None:
        """Each result should contain page metadata."""
        results = engine.search("love")
        for r in results:
            assert "url" in r.document
            assert "quotes" in r.document

    def test_scores_positive(self, engine: SearchEngine) -> None:
        """All scores should be positive."""
        results = engine.search("love")
        for r in results:
            assert r.score > 0


# ---------------------------------------------------------------------------
# Tests: SearchEngine.boolean_search
# ---------------------------------------------------------------------------


class TestBooleanSearch:
    """Tests for Boolean query search."""

    def test_and_search(self, engine: SearchEngine) -> None:
        """AND search should return intersection."""
        results = engine.boolean_search("love AND work")
        doc_ids = {r.doc_id for r in results}
        assert 1 in doc_ids  # Page 1 has both via Jobs quote

    def test_or_search(self, engine: SearchEngine) -> None:
        """OR search should return union."""
        results = engine.boolean_search("einstein OR frost")
        doc_ids = {r.doc_id for r in results}
        assert 0 in doc_ids  # Einstein on page 0
        assert 2 in doc_ids  # Frost on page 2

    def test_not_search(self, engine: SearchEngine) -> None:
        """NOT search should exclude matching documents."""
        results = engine.boolean_search("life AND NOT frost")
        doc_ids = {r.doc_id for r in results}
        assert 2 not in doc_ids  # Frost is on page 2

    def test_empty_boolean_query(self, engine: SearchEngine) -> None:
        """Empty Boolean query should return no results."""
        assert engine.boolean_search("") == []

    def test_invalid_boolean_syntax(self, engine: SearchEngine) -> None:
        """Malformed Boolean syntax should return empty gracefully."""
        # Unmatched parenthesis
        results = engine.boolean_search("(love AND")
        assert isinstance(results, list)

    def test_boolean_results_ranked(self, engine: SearchEngine) -> None:
        """Boolean results should be ranked by TF-IDF."""
        results = engine.boolean_search("love OR life")
        if len(results) >= 2:
            assert results[0].score >= results[1].score


# ---------------------------------------------------------------------------
# Tests: SearchEngine.suggest
# ---------------------------------------------------------------------------


class TestQuerySuggestions:
    """Tests for spelling suggestion functionality."""

    def test_misspelled_word(self, engine: SearchEngine) -> None:
        """A misspelled word should return suggestions."""
        # "lov" is close to "love"
        suggestions = engine.suggest("lov")
        assert len(suggestions) > 0

    def test_correct_word_no_suggestions(self, engine: SearchEngine) -> None:
        """A correctly spelled word in vocab should return no suggestions."""
        suggestions = engine.suggest("love")
        assert suggestions == []

    def test_completely_wrong(self, engine: SearchEngine) -> None:
        """A word very far from anything should return no suggestions."""
        suggestions = engine.suggest("zzzzzzzzzzz")
        assert suggestions == []

    def test_empty_query(self, engine: SearchEngine) -> None:
        """Empty query should return no suggestions."""
        assert engine.suggest("") == []

    def test_max_suggestions_limit(self, engine: SearchEngine) -> None:
        """Should not return more than max_suggestions."""
        engine_limited = SearchEngine(engine.indexer, max_suggestions=2)
        suggestions = engine_limited.suggest("lov")
        assert len(suggestions) <= 2

    def test_suggestion_is_in_vocab(self, engine: SearchEngine) -> None:
        """All suggestions should be actual vocabulary terms."""
        suggestions = engine.suggest("worl")
        for s in suggestions:
            assert s in engine.indexer.vocab


# ---------------------------------------------------------------------------
# Tests: SearchEngine.find (auto-detect)
# ---------------------------------------------------------------------------


class TestFindAutoDetect:
    """Tests for the smart find method that detects query type."""

    def test_plain_query_uses_ranked(self, engine: SearchEngine) -> None:
        """A plain query without operators should use ranked search."""
        results = engine.find("love")
        assert len(results) > 0

    def test_boolean_query_detected(self, engine: SearchEngine) -> None:
        """A query with AND/OR/NOT should use Boolean search."""
        results = engine.find("love AND life")
        assert isinstance(results, list)

    def test_parentheses_detected(self, engine: SearchEngine) -> None:
        """A query with parentheses should use Boolean search."""
        results = engine.find("(love OR life)")
        assert isinstance(results, list)

    def test_empty_find(self, engine: SearchEngine) -> None:
        """Empty query should return empty list."""
        assert engine.find("") == []

    def test_find_top_k(self, engine: SearchEngine) -> None:
        """top_k should limit results in ranked mode."""
        results = engine.find("love", top_k=1)
        assert len(results) <= 1


# ---------------------------------------------------------------------------
# Tests: SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Tests for the SearchResult class."""

    def test_repr(self) -> None:
        """repr should not crash and include key info."""
        sr = SearchResult(doc_id=0, score=1.5, document={"text": "Hello world"})
        r = repr(sr)
        assert "doc_id=0" in r
        assert "1.5" in r

    def test_attributes(self) -> None:
        """All attributes should be accessible."""
        doc = {"text": "Test", "author": "A", "tags": []}
        sr = SearchResult(doc_id=5, score=2.0, document=doc)
        assert sr.doc_id == 5
        assert sr.score == 2.0
        assert sr.document == doc


# ---------------------------------------------------------------------------
# Tests: Phrase search (positional index)
# ---------------------------------------------------------------------------


class TestPhraseSearch:
    """Tests for exact phrase search using positional information."""

    def test_exact_phrase_found(self, engine: SearchEngine) -> None:
        """An exact adjacent phrase should be found."""
        # "love all" appears consecutively in Shakespeare quote on page 0
        results = engine.phrase_search("love all")
        doc_ids = {r.doc_id for r in results}
        assert 0 in doc_ids

    def test_non_adjacent_words_excluded(self, engine: SearchEngine) -> None:
        """Words that appear but not adjacently should NOT match."""
        # "trust wrong" — both on page 0 but not adjacent
        results = engine.phrase_search("trust wrong")
        assert len(results) == 0

    def test_single_word_falls_back(self, engine: SearchEngine) -> None:
        """Single-word phrase should degrade to ranked search."""
        results = engine.phrase_search("love")
        assert len(results) > 0

    def test_empty_phrase(self, engine: SearchEngine) -> None:
        """Empty phrase returns no results."""
        assert engine.phrase_search("") == []

    def test_unknown_phrase(self, engine: SearchEngine) -> None:
        """Phrase with unknown words returns nothing."""
        assert engine.phrase_search("xyzfoo xyzbar") == []

    def test_phrase_results_ranked(self, engine: SearchEngine) -> None:
        """Phrase results should be ranked by TF-IDF score."""
        results = engine.phrase_search("love all")
        if len(results) >= 2:
            assert results[0].score >= results[1].score

    def test_find_detects_quoted_phrase(self, engine: SearchEngine) -> None:
        """find() should route quoted queries to phrase_search."""
        results = engine.find('"love all"')
        doc_ids = {r.doc_id for r in results}
        assert 0 in doc_ids


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    """End-to-end integration tests for the full pipeline."""

    def test_full_pipeline(self) -> None:
        """Test the complete flow: pages → index → search."""
        pages = [
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[
                    Quote("The only limit is your imagination", "Unknown", ["inspiration"]),
                    Quote("Knowledge is power", "Francis Bacon", ["knowledge", "power"]),
                ],
            ),
            Page(
                page_id=1,
                url="https://example.com/page/2/",
                quotes=[
                    Quote("Imagination is more important than knowledge", "Einstein", ["imagination", "knowledge"]),
                ],
            ),
        ]

        indexer = Indexer()
        indexer.build_index(pages)
        engine = SearchEngine(indexer)

        # Ranked search — "knowledge" is on both pages
        results = engine.search("knowledge")
        assert len(results) >= 2
        doc_ids = {r.doc_id for r in results}
        assert 0 in doc_ids
        assert 1 in doc_ids

        # Boolean search
        results = engine.boolean_search("imagination AND knowledge")
        doc_ids = {r.doc_id for r in results}
        assert 1 in doc_ids  # Einstein page has both

        # Suggestions
        suggestions = engine.suggest("knowledg")
        assert "knowledge" in suggestions

    def test_save_load_search(self) -> None:
        """Test index persistence: build → save → load → search."""
        import os
        import tempfile

        pages = [
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[
                    Quote("Love all trust a few", "Shakespeare", ["love"]),
                    Quote("To be or not to be", "Shakespeare", ["philosophy"]),
                ],
            ),
        ]

        indexer = Indexer()
        indexer.build_index(pages)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            indexer.save_to_disk(filepath)

            new_indexer = Indexer()
            new_indexer.load_from_disk(filepath)
            engine = SearchEngine(new_indexer)

            results = engine.search("love")
            assert len(results) > 0
            assert results[0].doc_id == 0

    def test_boolean_not_excludes_correctly(self) -> None:
        """Integration test for NOT operator in Boolean queries."""
        pages = [
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote("I love Python programming", "Dev", ["python"])],
            ),
            Page(
                page_id=1,
                url="https://example.com/page/2/",
                quotes=[Quote("Java is also great", "Dev", ["java"])],
            ),
            Page(
                page_id=2,
                url="https://example.com/page/3/",
                quotes=[Quote("I love Java too", "Dev", ["java", "love"])],
            ),
        ]
        indexer = Indexer()
        indexer.build_index(pages)
        engine = SearchEngine(indexer)

        results = engine.boolean_search("love AND NOT java")
        doc_ids = {r.doc_id for r in results}
        assert 0 in doc_ids
        assert 2 not in doc_ids


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance tests to ensure search remains efficient."""

    def test_search_speed(self) -> None:
        """Search across a moderately sized index should complete quickly."""
        # Build a larger index
        pages = [
            Page(
                page_id=i,
                url=f"https://example.com/page/{i + 1}/",
                quotes=[
                    Quote(
                        text=f"Quote number {i} about life love world peace happiness joy "
                             f"wisdom courage strength grace beauty truth freedom hope",
                        author=f"Author {i}",
                        tags=["tag1", "tag2"],
                    ),
                ],
            )
            for i in range(500)
        ]
        indexer = Indexer()
        indexer.build_index(pages)
        engine = SearchEngine(indexer)

        start = time.time()
        for _ in range(100):
            engine.search("life love")
        elapsed = time.time() - start

        # 100 searches over 500 pages should complete in well under 2 seconds
        assert elapsed < 2.0, f"Search too slow: {elapsed:.2f}s for 100 queries"

    def test_boolean_search_speed(self) -> None:
        """Boolean search should also be efficient."""
        pages = [
            Page(
                page_id=i,
                url=f"https://example.com/page/{i + 1}/",
                quotes=[
                    Quote(
                        text=f"Quote {i} love life world peace",
                        author=f"Author {i}",
                        tags=["tag"],
                    ),
                ],
            )
            for i in range(500)
        ]
        indexer = Indexer()
        indexer.build_index(pages)
        engine = SearchEngine(indexer)

        start = time.time()
        for _ in range(100):
            engine.boolean_search("love AND life AND NOT peace")
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Boolean search too slow: {elapsed:.2f}s"

    def test_index_build_speed(self) -> None:
        """Index building should be efficient."""
        pages = [
            Page(
                page_id=i,
                url=f"https://example.com/page/{i + 1}/",
                quotes=[
                    Quote(
                        text=f"This is quote number {i} with several words to index properly",
                        author=f"Author {i}",
                        tags=["tag1", "tag2", "tag3"],
                    ),
                ],
            )
            for i in range(1000)
        ]

        start = time.time()
        indexer = Indexer()
        indexer.build_index(pages)
        elapsed = time.time() - start

        assert elapsed < 5.0, f"Index build too slow: {elapsed:.2f}s for 1000 pages"
