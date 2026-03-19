"""
search.py — Search engine with TF-IDF ranking, Boolean queries, and suggestions.

Provides a query parser that supports AND, OR, NOT operators with proper
precedence, ranked retrieval using precomputed TF-IDF scores, and simple
edit-distance-based query suggestions for misspelled terms.
"""

import logging
import re
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple

from src.indexer import Indexer, tokenize, remove_stop_words

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Search result data structure
# ---------------------------------------------------------------------------


class SearchResult:
    """A single ranked search result.

    Attributes:
        doc_id: Document identifier.
        score: Combined TF-IDF relevance score.
        document: Full document metadata dict.
    """

    def __init__(self, doc_id: int, score: float, document: Dict[str, Any]) -> None:
        self.doc_id = doc_id
        self.score = score
        self.document = document

    def __repr__(self) -> str:
        return (
            f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f}, "
            f"text={self.document.get('text', '')[:50]!r})"
        )


# ---------------------------------------------------------------------------
# Boolean query token types
# ---------------------------------------------------------------------------


class TokenType(Enum):
    """Token types for the Boolean query lexer."""

    WORD = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    LPAREN = auto()
    RPAREN = auto()
    EOF = auto()


class Token:
    """A lexer token."""

    def __init__(self, type_: TokenType, value: str = "") -> None:
        self.type = type_
        self.value = value

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value!r})"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------


def tokenize_query(query: str) -> List[Token]:
    """Tokenize a Boolean query string into a stream of :class:`Token`.

    Recognised operators: ``AND``, ``OR``, ``NOT``, ``(``, ``)``.
    All other whitespace-separated words become ``WORD`` tokens.

    Args:
        query: Raw user query string.

    Returns:
        Ordered list of tokens ending with an ``EOF`` token.
    """
    tokens: List[Token] = []
    i = 0
    query = query.strip()

    while i < len(query):
        ch = query[i]

        # Skip whitespace
        if ch.isspace():
            i += 1
            continue

        # Parentheses
        if ch == "(":
            tokens.append(Token(TokenType.LPAREN, "("))
            i += 1
            continue
        if ch == ")":
            tokens.append(Token(TokenType.RPAREN, ")"))
            i += 1
            continue

        # Read a word
        start = i
        while i < len(query) and not query[i].isspace() and query[i] not in "()":
            i += 1
        word = query[start:i]

        upper = word.upper()
        if upper == "AND":
            tokens.append(Token(TokenType.AND, "AND"))
        elif upper == "OR":
            tokens.append(Token(TokenType.OR, "OR"))
        elif upper == "NOT":
            tokens.append(Token(TokenType.NOT, "NOT"))
        else:
            tokens.append(Token(TokenType.WORD, word.lower()))

    tokens.append(Token(TokenType.EOF, ""))
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent Boolean query parser
# ---------------------------------------------------------------------------


class QueryParser:
    """Recursive-descent parser for Boolean queries.

    Grammar::

        expression := and_expr ( 'OR' and_expr )*
        and_expr   := not_expr ( 'AND' not_expr )*
        not_expr   := 'NOT' not_expr | primary
        primary    := '(' expression ')' | WORD

    The parser evaluates against the inverted index and returns a set of
    matching ``doc_id`` values.
    """

    def __init__(self, tokens: List[Token], indexer: Indexer) -> None:
        self.tokens = tokens
        self.pos = 0
        self.indexer = indexer

    def _current(self) -> Token:
        """Return the current token without advancing."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return Token(TokenType.EOF)

    def _consume(self, expected: Optional[TokenType] = None) -> Token:
        """Advance and return the current token.

        Args:
            expected: If given, raises ``ValueError`` on mismatch.
        """
        tok = self._current()
        if expected and tok.type != expected:
            raise ValueError(
                f"Expected {expected}, got {tok.type} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    # ----- grammar rules -----

    def parse(self) -> Set[int]:
        """Parse the full query and return matching document IDs."""
        result = self._expression()
        return result

    def _expression(self) -> Set[int]:
        """expression := and_expr ( 'OR' and_expr )*"""
        left = self._and_expr()
        while self._current().type == TokenType.OR:
            self._consume(TokenType.OR)
            right = self._and_expr()
            left = left | right
        return left

    def _and_expr(self) -> Set[int]:
        """and_expr := not_expr ( 'AND' not_expr )*"""
        left = self._not_expr()
        while self._current().type == TokenType.AND:
            self._consume(TokenType.AND)
            right = self._not_expr()
            left = left & right
        return left

    def _not_expr(self) -> Set[int]:
        """not_expr := 'NOT' not_expr | primary"""
        if self._current().type == TokenType.NOT:
            self._consume(TokenType.NOT)
            operand = self._not_expr()
            all_docs = set(self.indexer.documents.keys())
            return all_docs - operand
        return self._primary()

    def _primary(self) -> Set[int]:
        """primary := '(' expression ')' | WORD"""
        tok = self._current()

        if tok.type == TokenType.LPAREN:
            self._consume(TokenType.LPAREN)
            result = self._expression()
            self._consume(TokenType.RPAREN)
            return result

        if tok.type == TokenType.WORD:
            self._consume(TokenType.WORD)
            postings = self.indexer.get_postings(tok.value)
            if postings is None:
                return set()
            return set(postings.keys())

        # Unexpected token — treat as empty result
        return set()


# ---------------------------------------------------------------------------
# Edit distance for query suggestions
# ---------------------------------------------------------------------------


def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Uses the standard dynamic-programming approach.
    Time complexity: *O(len(a) * len(b))*.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Minimum number of single-character edits.
    """
    m, n = len(a), len(b)
    # Optimise: single-row DP
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost, # substitution
            )
        prev, curr = curr, prev

    return prev[n]


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------


class SearchEngine:
    """Search engine supporting ranked retrieval and Boolean queries.

    Wraps an :class:`Indexer` to provide high-level search functionality
    including TF-IDF ranking, Boolean query evaluation, and query suggestions
    based on edit distance.

    Attributes:
        indexer: The indexer instance containing the inverted index.
        max_suggestions: Maximum number of spelling suggestions to return.
        max_edit_distance: Words within this distance are considered suggestions.
    """

    def __init__(
        self,
        indexer: Indexer,
        max_suggestions: int = 5,
        max_edit_distance: int = 2,
    ) -> None:
        self.indexer = indexer
        self.max_suggestions = max_suggestions
        self.max_edit_distance = max_edit_distance

    # ----- ranked TF-IDF search -----

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Perform a ranked search using TF-IDF scoring.

        For multi-word queries, only pages containing ALL query terms are
        returned (AND semantics), ranked by the sum of TF-IDF scores.

        Args:
            query: Raw user query (one or more words).
            top_k: Maximum number of results to return.

        Returns:
            List of :class:`SearchResult` objects sorted by descending score.
        """
        if not query or not query.strip():
            return []

        terms = tokenize(query)
        if not terms:
            return []

        # Find pages containing ALL query terms (AND semantics)
        matching_docs: Optional[Set[int]] = None
        for term in terms:
            postings = self.indexer.get_postings(term)
            if postings is None:
                return []  # Term not in index → no pages match
            term_docs = set(postings.keys())
            if matching_docs is None:
                matching_docs = term_docs
            else:
                matching_docs = matching_docs & term_docs

        if not matching_docs:
            return []

        # Rank matching pages by TF-IDF sum
        scores: Dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self.indexer.get_postings(term)
            if postings is None:
                continue
            idf = self.indexer.get_idf(term)
            for doc_id in matching_docs:
                if doc_id in postings:
                    scores[doc_id] += postings[doc_id]["tf"] * idf

        # Sort by score descending, then by doc_id ascending for stability
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        results: List[SearchResult] = []
        for doc_id, score in ranked[:top_k]:
            doc = self.indexer.get_document(doc_id)
            if doc is not None:
                results.append(SearchResult(doc_id=doc_id, score=score, document=doc))

        return results

    # ----- Boolean search -----

    def boolean_search(self, query: str) -> List[SearchResult]:
        """Evaluate a Boolean query and return matching documents.

        Supports ``AND``, ``OR``, ``NOT`` operators and parenthesised
        sub-expressions.  Results are ranked by TF-IDF when possible.

        Args:
            query: Boolean query string (e.g. ``"love AND NOT hate"``).

        Returns:
            List of :class:`SearchResult`, ranked by relevance.
        """
        if not query or not query.strip():
            return []

        try:
            tokens = tokenize_query(query)
            parser = QueryParser(tokens, self.indexer)
            matching_ids = parser.parse()
        except (ValueError, IndexError) as exc:
            logger.warning("Invalid Boolean query %r: %s", query, exc)
            return []

        if not matching_ids:
            return []

        # Score matched documents by summing TF-IDF for all word tokens
        word_terms = [t.value for t in tokens if t.type == TokenType.WORD]
        scores: Dict[int, float] = defaultdict(float)

        for term in word_terms:
            postings = self.indexer.get_postings(term)
            if postings is None:
                continue
            idf = self.indexer.get_idf(term)
            for doc_id in matching_ids:
                if doc_id in postings:
                    scores[doc_id] += postings[doc_id]["tf"] * idf

        # Documents matched by NOT without any positive term get score 0
        for doc_id in matching_ids:
            if doc_id not in scores:
                scores[doc_id] = 0.0

        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))

        results: List[SearchResult] = []
        for doc_id, score in ranked:
            doc = self.indexer.get_document(doc_id)
            if doc is not None:
                results.append(SearchResult(doc_id=doc_id, score=score, document=doc))

        return results

    # ----- query suggestions -----

    def suggest(self, query: str) -> List[str]:
        """Suggest corrections for potentially misspelled query terms.

        Uses Levenshtein edit distance against the index vocabulary.

        Args:
            query: Raw user query string.

        Returns:
            List of suggested replacement words (may be empty).
        """
        if not query or not query.strip():
            return []

        terms = tokenize(query)
        if not terms:
            return []

        suggestions: List[str] = []

        for term in terms:
            # Skip if the term exists in the vocabulary
            if term in self.indexer.vocab:
                continue

            # Find close matches
            candidates: List[Tuple[int, str]] = []
            for vocab_word in self.indexer.vocab:
                # Quick length pre-filter to avoid unnecessary computation
                if abs(len(vocab_word) - len(term)) > self.max_edit_distance:
                    continue
                dist = _edit_distance(term, vocab_word)
                if dist <= self.max_edit_distance:
                    candidates.append((dist, vocab_word))

            # Sort by distance then alphabetically
            candidates.sort(key=lambda x: (x[0], x[1]))
            for _, word in candidates[: self.max_suggestions]:
                if word not in suggestions:
                    suggestions.append(word)

        return suggestions[: self.max_suggestions]

    # ----- convenience: auto-detect query type -----

    def find(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Smart search that detects Boolean operators or falls back to ranked.

        If the query contains ``AND``, ``OR``, ``NOT`` (case-insensitive) or
        parentheses, it is evaluated as a Boolean query.  Otherwise, a
        standard TF-IDF ranked search is performed.

        Args:
            query: Raw query string.
            top_k: Maximum results for ranked mode.

        Returns:
            List of :class:`SearchResult`.
        """
        if not query or not query.strip():
            return []

        # Detect Boolean operators (uppercase only, must be whole words)
        boolean_pattern = r"\b(AND|OR|NOT)\b|[()]"
        if re.search(boolean_pattern, query):
            return self.boolean_search(query)

        return self.search(query, top_k=top_k)
