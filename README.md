# Quotes Search Engine

> A TF-IDF search engine with Boolean queries, phrase search, and spelling suggestions — built for [quotes.toscrape.com](https://quotes.toscrape.com).

**COMP3011 Web Services and Web Data — Coursework 2**

---

## Features

| Feature | Description |
|---------|-------------|
| **TF-IDF Ranked Search** | Multi-word queries with AND semantics, ranked by combined TF-IDF score |
| **Exact Phrase Search** | Positional index enables `"good friends"` to match only adjacent words |
| **Boolean Queries** | Full `AND` / `OR` / `NOT` with parenthesised sub-expressions |
| **Spelling Suggestions** | Levenshtein edit-distance corrections for misspelled queries |
| **Polite Crawling** | Configurable rate limiting (6 s delay) with exponential-backoff retry |
| **Persistent Index** | JSON serialisation with full round-trip fidelity |

---

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌─────────┐
│  crawler.py  │────▶│  indexer.py   │────▶│    search.py     │◀───│ main.py │
│              │     │              │     │                  │     │  (CLI)  │
│ • HTTP fetch │     │ • Tokenize   │     │ • TF-IDF rank   │     │         │
│ • HTML parse │     │ • Build index│     │ • Boolean parse  │     │ • build │
│ • Pagination │     │ • TF/IDF calc│     │ • Phrase search  │     │ • load  │
│ • Retry logic│     │ • Save/Load  │     │ • Suggestions    │     │ • find  │
└──────────────┘     └──────────────┘     └──────────────────┘     │ • print │
                            │                                       └─────────┘
                            ▼
                     data/index.json
```

**Data flow:** The crawler fetches each page of the website and returns a list of `Page` objects (each containing multiple `Quote` objects). The indexer tokenizes the combined text of all quotes on each page, builds a positional inverted index with TF-IDF weights, and persists it to JSON. The search engine queries the index using ranked, phrase, or Boolean retrieval, and the CLI provides the interactive interface.

---

## Project Structure

```
quotes-search-engine/
├── src/
│   ├── __init__.py
│   ├── crawler.py          # Web crawler with retry & politeness
│   ├── indexer.py           # Positional inverted index with TF-IDF
│   ├── search.py            # Ranked, phrase, and Boolean search
│   └── main.py              # Interactive CLI
├── tests/
│   ├── conftest.py          # Shared fixtures
│   ├── test_crawler.py      # 34 tests — parsing, pagination, retries
│   ├── test_indexer.py      # 44 tests — tokenization, indexing, persistence
│   ├── test_search.py       # 68 tests — ranking, Boolean, phrase, benchmarks
│   └── test_main.py         # 33 tests — CLI commands, edge cases
├── data/
│   └── index.json           # Generated inverted index
├── requirements.txt
├── pyproject.toml           # Project config & pytest settings
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.9+

```bash
# Clone and set up virtual environment
git clone <repo-url>
cd quotes-search-engine

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Usage

```bash
python -m src.main
```

### Commands

| Command | Description |
|---------|-------------|
| `build` | Crawl the website, build the inverted index, and save to disk |
| `load` | Load a previously saved index from disk |
| `print <word>` | Display the inverted index entry for a word (page URLs, frequency, TF, positions) |
| `find <query>` | Search using ranked, phrase, or Boolean mode (auto-detected) |
| `suggest <query>` | Get spelling suggestions for misspelled terms |
| `help` | Show available commands |
| `quit` | Exit the search engine |

### Search Examples

```
search> build                          # Crawl and build index
search> load                           # Load saved index

search> find indifference              # Ranked search — single word
search> find good friends              # Ranked search — AND semantics
search> find "good friends"            # Phrase search — exact adjacency
search> find love AND NOT hate         # Boolean search
search> find (love OR life) AND world  # Boolean with grouping

search> print nonsense                 # Inspect index entry
search> suggest happness               # → "happiness"
```

**Query type auto-detection:**
- `"quoted text"` → exact phrase search (uses positional index)
- Queries with `AND` / `OR` / `NOT` / `()` → Boolean evaluation
- Everything else → TF-IDF ranked search with AND semantics

---

## Algorithm Details

### TF-IDF Scoring

Each term's weight in a document is the product of:

- **Term Frequency (TF):** `frequency / document_length` — normalised to prevent bias toward longer pages
- **Inverse Document Frequency (IDF):** `log((1 + N) / (1 + df)) + 1` — smoothed to avoid zero-division

Multi-word queries require **all** terms to appear in a page (AND semantics). Pages are ranked by the sum of per-term TF-IDF scores.

### Positional Index & Phrase Search

The inverted index stores the exact position of every term occurrence. Phrase search exploits this:

1. Verify all phrase terms appear in the candidate page
2. For each occurrence of the first term, check whether subsequent terms appear at consecutive positions
3. Only pages with a full positional match are returned

This enables `"good friends"` to match pages where these words are adjacent while `good friends` (unquoted) returns any page containing both words anywhere.

### Boolean Query Parser

A hand-written **recursive-descent parser** implements the grammar:

```
expression := and_expr ( 'OR' and_expr )*
and_expr   := not_expr ( 'AND' not_expr )*
not_expr   := 'NOT' not_expr | primary
primary    := '(' expression ')' | WORD
```

Operator precedence: `NOT` > `AND` > `OR` (standard Boolean algebra).

### Spelling Suggestions

Uses **Levenshtein edit distance** with a length pre-filter to efficiently prune vocabulary candidates. Words within edit distance ≤ 2 are offered as suggestions, sorted by distance then alphabetically.

---

## Complexity Analysis

| Operation | Time Complexity | Notes |
|-----------|----------------|-------|
| `crawl()` | O(P · (H + D)) | P pages, H = HTML size, D = delay |
| `build_index()` | O(N · L) | N pages, L = avg tokens per page |
| `search()` | O(Q · P) | Q query terms, P = max postings |
| `phrase_search()` | O(Q · P · H) | H = positions per posting |
| `boolean_search()` | O(Q · D) | D = total documents |
| `suggest()` | O(T · V) | T query terms, V = vocabulary size |
| `_edit_distance()` | O(\|a\| · \|b\|) | Single-row DP optimisation for space |

---

## Testing

```bash
# Run all tests with coverage
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# Run only fast tests (skip benchmarks)
python -m pytest tests/ -m "not slow"

# Run integration tests only
python -m pytest tests/ -m integration

# Run performance benchmarks only
python -m pytest tests/ -m performance
```

**Test suite:** 178 tests | 98% coverage | <5 s runtime

The test suite uses:
- **pytest fixtures** for shared setup (see `conftest.py`)
- **Parametrized tests** for systematic edge-case coverage
- **Mocking** (`unittest.mock`) to isolate crawler from network
- **Performance benchmarks** validating search speed over 500+ page indexes
- **Integration tests** covering the full pipeline: crawl → index → persist → load → search

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Page-based indexing** | Each web page is one document — results show the page URL and all quotes, matching the brief's requirement |
| **AND semantics for multi-word** | `find good friends` returns only pages containing *both* words, as specified in the brief |
| **Augmented TF** | Normalising by document length prevents longer pages from dominating results |
| **Smoothed IDF** | The `+1` terms prevent log(0) and reduce extreme weights for rare terms |
| **Positional index** | Storing token positions enables exact phrase matching with zero additional crawling |
| **Case-sensitive Boolean detection** | Only uppercase `AND`/`OR`/`NOT` trigger Boolean mode — lowercase "not" is searched as a regular word |
| **Polite 6 s delay** | Respects the target website while still completing a full crawl in ~1 minute |
| **Exponential backoff** | Gracefully handles transient network errors without aggressive retrying |

---

## References

- Manning, C.D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. — TF-IDF weighting, positional indexing, Boolean retrieval model.
- Levenshtein, V.I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals." — Edit distance algorithm for spelling suggestions.
- Robertson, S.E. & Zaragoza, H. (2009). "The Probabilistic Relevance Framework: BM25 and Beyond." — Context for understanding TF-IDF's place among ranking functions (BM25 as a potential future enhancement).
- Zobel, J. & Moffat, A. (2006). "Inverted Files for Text Search Engines." *ACM Computing Surveys*. — Inverted index design patterns and optimisation strategies.
