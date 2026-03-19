"""
main.py — Interactive CLI front-end for the search engine.

Provides a shell-like interface with commands to build the index (crawl +
index + save), load a previously saved index, print index entries, and
search with ranked or Boolean queries.
"""

import json
import logging
import os
import textwrap
from typing import Optional

from src.crawler import Crawler
from src.indexer import Indexer
from src.search import SearchEngine

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

# Configure the root logger once at application startup.  All modules that
# call logging.getLogger(__name__) will inherit this handler and format.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# os.path.join uses the correct path separator for the current OS
# (forward-slash on macOS/Linux, backslash on Windows) so the path
# is correct regardless of where the project is run.
INDEX_PATH = os.path.join("data", "index.json")
BANNER = r"""
╔══════════════════════════════════════════════════════╗
║          Quotes Search Engine v1.0                   ║
║  Crawls, indexes, and searches quotes.toscrape.com   ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = textwrap.dedent("""\
    Available commands:
      build                 Crawl the website, build the index, and save to disk.
      load                  Load a previously saved index from disk.
      print <word>          Display the full index entry for a word.
      find <query>          Search for quotes matching the query.
                            • Multi-word: AND semantics (e.g. find good friends)
                            • Phrase:     wrap in quotes  (e.g. find "good friends")
                            • Boolean:    AND / OR / NOT  (e.g. find love AND NOT hate)
      suggest <query>       Get spelling suggestions for query terms.
      help                  Show this help message.
      quit / exit           Exit the search engine.
""")

# ---------------------------------------------------------------------------
# CLI class
# ---------------------------------------------------------------------------


class CLI:
    """Interactive command-line interface for the search engine.

    Attributes:
        indexer: The indexer managing the inverted index.
        engine: The search engine (initialised after loading/building).
    """

    def __init__(self) -> None:
        self.indexer: Indexer = Indexer()
        self.engine: Optional[SearchEngine] = None

    # ----- command handlers -----

    def cmd_build(self) -> None:
        """Crawl, build index, and save to disk."""
        print("\n[*] Starting crawl of quotes.toscrape.com …")
        crawler = Crawler()
        pages = crawler.crawl()

        if not pages:
            print("[!] No pages were retrieved. Aborting build.")
            return

        # Use sum() with a generator expression to count total quotes across
        # all crawled pages without building an intermediate list.
        total_quotes = sum(len(p.quotes) for p in pages)
        print(f"[+] Crawled {len(pages)} pages ({total_quotes} quotes).")
        print("[*] Building inverted index …")
        self.indexer.build_index(pages)
        print(f"[+] Index built: {len(self.indexer.vocab)} unique terms.")

        print(f"[*] Saving index to {INDEX_PATH} …")
        self.indexer.save_to_disk(INDEX_PATH)
        print("[+] Index saved successfully.")

        self.engine = SearchEngine(self.indexer)
        print("[+] Search engine ready.\n")

    def cmd_load(self) -> None:
        """Load index from disk."""
        if not os.path.exists(INDEX_PATH):
            print(f"[!] Index file not found at {INDEX_PATH}.")
            print("    Run 'build' first to create the index.")
            return

        try:
            print(f"[*] Loading index from {INDEX_PATH} …")
            self.indexer.load_from_disk(INDEX_PATH)
            self.engine = SearchEngine(self.indexer)
            print(
                f"[+] Loaded {self.indexer.total_docs} pages, "
                f"{len(self.indexer.vocab)} terms."
            )
            print("[+] Search engine ready.\n")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[!] Failed to load index: {exc}")
        except FileNotFoundError:
            print(f"[!] Index file not found at {INDEX_PATH}.")

    def cmd_print(self, word: str) -> None:
        """Print the full index entry for a word."""
        if not self._ensure_loaded():
            return

        term = word.lower().strip()
        if not term:
            print("[!] Usage: print <word>")
            return

        entry = self.indexer.get_term_entry(term)
        if entry is None:
            print(f"[!] Term '{term}' not found in index.")
            # Offer suggestions
            if self.engine:
                suggestions = self.engine.suggest(term)
                if suggestions:
                    print(f"    Did you mean: {', '.join(suggestions)}?")
            return

        print(f"\n--- Index entry for '{term}' ---")
        print(f"  IDF: {entry['idf']:.4f}")
        print(f"  Appears in {len(entry['postings'])} page(s):\n")

        for doc_id, posting in sorted(entry["postings"].items()):
            # sorted() gives deterministic output order (low→high doc_id)
            # instead of Python's arbitrary dict-insertion order.
            doc = self.indexer.get_document(doc_id)
            url = doc.get("url", "N/A") if doc else "N/A"
            print(
                f"    Page {doc_id} ({url})\n"
                f"      Frequency : {posting['frequency']}\n"
                f"      TF        : {posting['tf']:.4f}\n"
                f"      Positions : {posting['positions']}"
            )
        print()

    def cmd_find(self, query: str) -> None:
        """Execute a search query and display ranked results."""
        if not self._ensure_loaded():
            return

        if not query.strip():
            print("[!] Usage: find <query>")
            return

        results = self.engine.find(query)

        if not results:
            print(f"[!] No results found for '{query}'.")
            suggestions = self.engine.suggest(query)
            if suggestions:
                print(f"    Did you mean: {', '.join(suggestions)}?")
            return

        print(f"\n--- Search results for '{query}' ({len(results)} page(s) found) ---\n")
        for i, result in enumerate(results, 1):
            doc = result.document
            url = doc.get("url", "N/A")
            print(f"  {i}. [Score: {result.score:.4f}] Page {result.doc_id}: {url}")
            for q in doc.get("quotes", []):
                print(f"     \"{q['text']}\"")
                # \u2014 is the Unicode em-dash (—), used as an attribution
                # separator between the quote text and the author name.
                print(f"     \u2014 {q['author']}  |  Tags: {', '.join(q['tags'])}")
            print()

    def cmd_suggest(self, query: str) -> None:
        """Show spelling suggestions for query terms."""
        if not self._ensure_loaded():
            return

        if not query.strip():
            print("[!] Usage: suggest <query>")
            return

        suggestions = self.engine.suggest(query)
        if suggestions:
            print(f"  Suggestions: {', '.join(suggestions)}")
        else:
            print("  No suggestions available (all terms look correct).")

    # ----- helpers -----

    def _ensure_loaded(self) -> bool:
        """Check that the index is loaded and the engine is initialised."""
        if self.engine is None:
            print("[!] No index loaded. Run 'build' or 'load' first.")
            return False
        return True

    # ----- main loop -----

    def run(self) -> None:
        """Start the interactive CLI event loop."""
        print(BANNER)
        print(HELP_TEXT)

        while True:
            try:
                raw = input("search> ").strip()
            except (EOFError, KeyboardInterrupt):
                # EOFError is raised when stdin is closed (e.g. piped input
                # or a non-interactive shell). KeyboardInterrupt handles
                # Ctrl-C. Both are treated as a clean exit.
                print("\n[*] Goodbye!")
                break

            if not raw:
                continue

            # maxsplit=1 splits only on the first space so multi-word
            # arguments like  find good friends  are kept intact as a single
            # string in parts[1] rather than being split into separate words.
            parts = raw.split(maxsplit=1)
            command = parts[0].lower()
            # Default argument to empty string when no argument is provided.
            argument = parts[1] if len(parts) > 1 else ""

            if command in ("quit", "exit"):
                print("[*] Goodbye!")
                break
            elif command == "build":
                self.cmd_build()
            elif command == "load":
                self.cmd_load()
            elif command == "print":
                self.cmd_print(argument)
            elif command == "find":
                self.cmd_find(argument)
            elif command == "suggest":
                self.cmd_suggest(argument)
            elif command == "help":
                print(HELP_TEXT)
            else:
                print(f"[!] Unknown command: '{command}'. Type 'help' for usage.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    cli = CLI()
    cli.run()


if __name__ == "__main__":
    main()
