"""
indexer.py — Inverted index builder with TF-IDF scoring.

Builds a positional inverted index from crawled documents, computes TF-IDF
weights, and provides efficient save/load to disk in JSON format.

Index structure
---------------
The inverted index maps each term to a dictionary::

    { term: { "idf": float,
              "postings": { doc_id: { "frequency": int,
                                      "positions": [int],
                                      "tf": float } } } }

* **Positional information** enables phrase search (see :mod:`search`).
* **TF** is *augmented frequency*: ``freq / doc_length``, preventing bias
  toward longer documents.
* **IDF** uses the smoothed formula ``log((1 + N) / (1 + df)) + 1`` to
  avoid zero-division and extreme values.

Complexity
----------
=====================  ========================================
Operation              Time complexity
=====================  ========================================
``build_index()``      *O(N · L)* — *N* pages, *L* avg tokens
``_compute_idf()``     *O(V)* — *V* unique terms
``get_postings()``     *O(1)* dict lookup
``save_to_disk()``     *O(N · V)* serialisation
``load_from_disk()``   *O(N · V)* deserialisation
=====================  ========================================
"""

import json
import logging
import math
import os
import string
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from src.crawler import Page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stop-words (common English words that add little search value)
# ---------------------------------------------------------------------------

STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "about", "up", "it", "its", "i", "me", "my",
    "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
    "hers", "herself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "if", "while", "s", "t", "d", "ll", "ve", "re", "don",
}

# ---------------------------------------------------------------------------
# Punctuation translation table (built once at import time)
# ---------------------------------------------------------------------------

# str.maketrans("", "", chars) builds a lookup table that maps every
# character in `chars` to None (i.e. delete it).  Building this once at
# module level means every tokenize() call reuses the same object instead
# of re-allocating it on every invocation. The extra Unicode chars are the
# ‘curly’ and “double” quotation marks used by the target website.
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "\u2018\u2019\u201c\u201d")

# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Convert raw text into a list of clean lowercase tokens.

    Processing steps:
        1. Convert to lowercase.
        2. Remove punctuation (ASCII + common Unicode quotes).
        3. Split on whitespace.
        4. Discard empty tokens and pure-digit tokens.

    Args:
        text: Raw text string.

    Returns:
        Ordered list of tokens preserving position for positional indexing.
    """
    text = text.lower()
    # Remove all punctuation characters in one O(n) pass using the
    # pre-built translation table rather than running a regex per token.
    text = text.translate(_PUNCT_TABLE)
    tokens = text.split()
    # Discard pure-digit tokens (page numbers, years, counters) since
    # they appear in many documents and rarely carry search intent.
    return [t for t in tokens if t and not t.isdigit()]


def remove_stop_words(tokens: List[str]) -> List[str]:
    """Filter stop-words while preserving token order.

    Args:
        tokens: List of lowercase tokens.

    Returns:
        Filtered list (original positions are **not** preserved — the caller
        should use the unfiltered list for positional indexing).
    """
    return [t for t in tokens if t not in STOP_WORDS]


# ---------------------------------------------------------------------------
# Index entry types
# ---------------------------------------------------------------------------

# Posting = per-document data for one term
# { doc_id: { "frequency": int, "positions": [int], "tf": float } }
PostingDict = Dict[int, Dict[str, Any]]

# Full inverted index
# { term: { "postings": PostingDict, "idf": float } }
IndexDict = Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------------------
# Indexer class
# ---------------------------------------------------------------------------


class Indexer:
    """Builds and manages a positional inverted index with TF-IDF scores.

    The index maps each unique term to a dictionary of document postings.
    Each posting stores term frequency, positional information, and a
    precomputed TF weight. IDF is stored at the term level and combined
    with TF during search-time scoring.

    Attributes:
        index: The inverted index dictionary.
        documents: Mapping of ``doc_id`` to serialised quote data.
        total_docs: Number of indexed documents.
        vocab: Set of all unique terms in the index.
    """

    def __init__(self) -> None:
        self.index: IndexDict = {}
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.total_docs: int = 0
        self.vocab: Set[str] = set()

    # ----- core building -----

    def build_index(self, pages: List[Page]) -> None:
        """Build the inverted index from a list of :class:`Page` objects.

        For each page the combined text of all quotes (text + author + tags)
        is tokenized and each token is recorded with its position.  After all
        pages are processed, IDF values are computed.

        Time complexity: *O(N · L)* where *N* = number of pages and
        *L* = average page text length.

        Args:
            pages: Ordered list of ``Page`` objects from the crawler.
        """
        self.index = {}
        self.documents = {}
        self.total_docs = len(pages)

        if self.total_docs == 0:
            logger.warning("build_index called with empty page list.")
            return

        for page in pages:
            self._index_page(page)

        self._compute_idf()
        self.vocab = set(self.index.keys())

        logger.info(
            "Index built: %d pages, %d unique terms.",
            self.total_docs,
            len(self.vocab),
        )

    def _index_page(self, page: Page) -> None:
        """Add a single page to the index.

        Args:
            page: The page to index.
        """
        # Store page metadata
        self.documents[page.page_id] = page.to_dict()

        # Combine all textual content from all quotes on the page
        combined_text = page.full_text
        tokens = tokenize(combined_text)

        if not tokens:
            return

        # Use enumerate so each token gets a monotonically-increasing
        # position number within this document.  These positions are stored
        # in the index to support phrase-search adjacency checks later.
        # defaultdict(list) auto-creates an empty list on first access so we
        # never need an explicit "if term not in term_positions" guard.
        term_positions: Dict[str, List[int]] = defaultdict(list)
        for position, token in enumerate(tokens):
            term_positions[token].append(position)

        doc_length = len(tokens)

        for term, positions in term_positions.items():
            if term not in self.index:
                self.index[term] = {"postings": {}, "idf": 0.0}

            frequency = len(positions)
            # Augmented term frequency to prevent bias toward longer docs
            tf = frequency / doc_length

            self.index[term]["postings"][page.page_id] = {
                "frequency": frequency,
                "positions": positions,
                "tf": tf,
            }

    def _compute_idf(self) -> None:
        """Compute IDF for every term in the index.

        Uses the smoothed formula: ``log((1 + N) / (1 + df)) + 1``
        to avoid zero-division and extreme values.
        """
        for term, entry in self.index.items():
            # df = document frequency: how many documents contain this term.
            df = len(entry["postings"])
            # Smoothed IDF formula breakdown:
            #   (1 + N)  — add 1 to total docs so N=0 never divides by zero
            #   (1 + df) — add 1 to df so rare terms don’t explode toward ∞
            #   + 1      — additive smoothing keeps IDF >= 1 even for terms
            #              that appear in every document (df == N)
            entry["idf"] = math.log((1 + self.total_docs) / (1 + df)) + 1

    # ----- lookup helpers -----

    def get_postings(self, term: str) -> Optional[PostingDict]:
        """Return postings for a term, or ``None`` if absent.

        Args:
            term: Lowercased search term.

        Returns:
            Posting dict mapping doc_id → info, or ``None``.
        """
        entry = self.index.get(term)
        if entry is None:
            return None
        return entry["postings"]

    def get_idf(self, term: str) -> float:
        """Return the IDF score for a term.

        Args:
            term: Lowercased search term.

        Returns:
            IDF value, or ``0.0`` if the term is absent.
        """
        entry = self.index.get(term)
        return entry["idf"] if entry else 0.0

    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Return metadata for a document by ID.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Document dict or ``None``.
        """
        return self.documents.get(doc_id)

    def get_term_entry(self, term: str) -> Optional[Dict[str, Any]]:
        """Return the full index entry (postings + IDF) for a term.

        Args:
            term: Lowercased search term.

        Returns:
            Entry dict or ``None``.
        """
        return self.index.get(term)

    # ----- persistence -----

    def save_to_disk(self, filepath: str = "data/index.json") -> None:
        """Serialise the index and document store to a JSON file.

        Creates parent directories if they do not exist.

        Args:
            filepath: Target file path.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)

        # JSON only supports string keys.  We store doc_ids as integers in
        # memory for O(1) dict access, but must convert them to strings when
        # serialising so json.dump doesn’t raise a TypeError.
        serialisable_index: Dict[str, Any] = {}
        for term, entry in self.index.items():
            serialisable_index[term] = {
                "idf": entry["idf"],
                "postings": {
                    str(doc_id): posting
                    for doc_id, posting in entry["postings"].items()
                },
            }

        serialisable_docs = {str(k): v for k, v in self.documents.items()}

        data = {
            "total_docs": self.total_docs,
            "index": serialisable_index,
            "documents": serialisable_docs,
        }

        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        logger.info("Index saved to %s", filepath)

    def load_from_disk(self, filepath: str = "data/index.json") -> None:
        """Load a previously saved index from JSON.

        Args:
            filepath: Path to the JSON index file.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
            KeyError: If expected keys are missing.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self.total_docs = data["total_docs"]

        # JSON stores all mapping keys as strings, so we must explicitly
        # cast each doc_id string back to int to match the in-memory schema.
        self.index = {}
        for term, entry in data["index"].items():
            self.index[term] = {
                "idf": entry["idf"],
                "postings": {
                    int(doc_id): posting
                    for doc_id, posting in entry["postings"].items()
                },
            }

        self.documents = {int(k): v for k, v in data["documents"].items()}
        self.vocab = set(self.index.keys())

        logger.info(
            "Index loaded from %s: %d documents, %d terms.",
            filepath,
            self.total_docs,
            len(self.vocab),
        )
