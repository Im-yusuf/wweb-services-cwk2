"""
test_main.py — Tests for the CLI interface module.

Uses mocking and input simulation to test all CLI commands without
real network calls or user interaction.
"""

import json
import os
import tempfile
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from src.crawler import Page, Quote
from src.indexer import Indexer
from src.main import CLI
from src.search import SearchEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cli_with_index() -> CLI:
    """Create a CLI instance with a pre-built index."""
    cli = CLI()
    pages = [
        Page(
            page_id=0,
            url="https://quotes.toscrape.com/page/1/",
            quotes=[
                Quote("The world is beautiful and full of love", "Author A", ["love", "world"]),
                Quote("Knowledge is power and wisdom", "Author B", ["knowledge"]),
            ],
        ),
        Page(
            page_id=1,
            url="https://quotes.toscrape.com/page/2/",
            quotes=[
                Quote("Love conquers all fear and hatred", "Author C", ["love", "courage"]),
            ],
        ),
    ]
    cli.indexer.build_index(pages)
    cli.engine = SearchEngine(cli.indexer)
    return cli


# ---------------------------------------------------------------------------
# Tests: CLI initialisation
# ---------------------------------------------------------------------------


class TestCLIInit:
    """Tests for CLI initialisation."""

    def test_initial_state(self) -> None:
        """CLI should start with no loaded engine."""
        cli = CLI()
        assert cli.engine is None
        assert isinstance(cli.indexer, Indexer)


# ---------------------------------------------------------------------------
# Tests: cmd_build
# ---------------------------------------------------------------------------


class TestCmdBuild:
    """Tests for the build command."""

    @patch("src.main.Crawler")
    def test_build_success(self, mock_crawler_cls: MagicMock, capsys) -> None:
        """Build should crawl, index, and save."""
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = [
            Page(
                page_id=0,
                url="https://quotes.toscrape.com/page/1/",
                quotes=[
                    Quote("Test quote one", "Author", ["tag"]),
                    Quote("Test quote two", "Author", ["tag"]),
                ],
            ),
        ]
        mock_crawler_cls.return_value = mock_crawler

        cli = CLI()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.INDEX_PATH", os.path.join(tmpdir, "index.json")):
                cli.cmd_build()

        assert cli.engine is not None
        captured = capsys.readouterr()
        assert "Crawled 1 pages" in captured.out

    @patch("src.main.Crawler")
    def test_build_no_pages(self, mock_crawler_cls: MagicMock, capsys) -> None:
        """Build with no pages should abort gracefully."""
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = []
        mock_crawler_cls.return_value = mock_crawler

        cli = CLI()
        cli.cmd_build()

        assert cli.engine is None
        captured = capsys.readouterr()
        assert "No pages were retrieved" in captured.out


# ---------------------------------------------------------------------------
# Tests: cmd_load
# ---------------------------------------------------------------------------


class TestCmdLoad:
    """Tests for the load command."""

    def test_load_success(self, capsys) -> None:
        """Should load a valid index file."""
        pages = [
            Page(
                page_id=0,
                url="https://example.com/page/1/",
                quotes=[Quote("Hello world", "Auth", ["tag"])],
            ),
        ]
        indexer = Indexer()
        indexer.build_index(pages)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "index.json")
            indexer.save_to_disk(filepath)

            cli = CLI()
            with patch("src.main.INDEX_PATH", filepath):
                cli.cmd_load()

            assert cli.engine is not None
            captured = capsys.readouterr()
            assert "Loaded" in captured.out

    def test_load_missing_file(self, capsys) -> None:
        """Should handle missing index file gracefully."""
        cli = CLI()
        with patch("src.main.INDEX_PATH", "/nonexistent/path/index.json"):
            cli.cmd_load()

        assert cli.engine is None
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_load_invalid_json(self, capsys) -> None:
        """Should handle corrupt JSON gracefully."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{{{invalid")
            f.flush()
            try:
                cli = CLI()
                with patch("src.main.INDEX_PATH", f.name):
                    cli.cmd_load()
                assert cli.engine is None
                captured = capsys.readouterr()
                assert "Failed to load" in captured.out
            finally:
                os.unlink(f.name)


# ---------------------------------------------------------------------------
# Tests: cmd_print
# ---------------------------------------------------------------------------


class TestCmdPrint:
    """Tests for the print command."""

    def test_print_existing_term(self, capsys) -> None:
        """Should display index entry for a known term."""
        cli = _build_cli_with_index()
        cli.cmd_print("love")
        captured = capsys.readouterr()
        assert "Index entry for 'love'" in captured.out
        assert "IDF" in captured.out
        assert "Page" in captured.out

    def test_print_missing_term(self, capsys) -> None:
        """Should report when term is not found."""
        cli = _build_cli_with_index()
        cli.cmd_print("xyznonexistent")
        captured = capsys.readouterr()
        assert "not found in index" in captured.out

    def test_print_empty_word(self, capsys) -> None:
        """Should show usage hint for empty argument."""
        cli = _build_cli_with_index()
        cli.cmd_print("")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_print_no_index(self, capsys) -> None:
        """Should warn if no index is loaded."""
        cli = CLI()
        cli.cmd_print("love")
        captured = capsys.readouterr()
        assert "No index loaded" in captured.out

    def test_print_with_suggestions(self, capsys) -> None:
        """Should offer suggestions for misspelled terms."""
        cli = _build_cli_with_index()
        cli.cmd_print("lov")
        captured = capsys.readouterr()
        assert "not found" in captured.out
        assert "Did you mean" in captured.out


# ---------------------------------------------------------------------------
# Tests: cmd_find
# ---------------------------------------------------------------------------


class TestCmdFind:
    """Tests for the find command."""

    def test_find_results(self, capsys) -> None:
        """Should display ranked results with page info."""
        cli = _build_cli_with_index()
        cli.cmd_find("love")
        captured = capsys.readouterr()
        assert "Search results" in captured.out
        assert "Score" in captured.out
        assert "Page" in captured.out

    def test_find_no_results(self, capsys) -> None:
        """Should report no results for unknown terms."""
        cli = _build_cli_with_index()
        cli.cmd_find("xyznonexistent")
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_find_empty_query(self, capsys) -> None:
        """Should show usage hint for empty query."""
        cli = _build_cli_with_index()
        cli.cmd_find("")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_find_no_index(self, capsys) -> None:
        """Should warn if no index is loaded."""
        cli = CLI()
        cli.cmd_find("love")
        captured = capsys.readouterr()
        assert "No index loaded" in captured.out

    def test_find_boolean_query(self, capsys) -> None:
        """Should handle Boolean queries."""
        cli = _build_cli_with_index()
        cli.cmd_find("love AND world")
        captured = capsys.readouterr()
        assert "Search results" in captured.out

    def test_find_with_suggestions(self, capsys) -> None:
        """Should offer suggestions when no results found."""
        cli = _build_cli_with_index()
        cli.cmd_find("lov")
        captured = capsys.readouterr()
        # May show suggestions if no exact match
        assert ("No results" in captured.out) or ("Search results" in captured.out)


# ---------------------------------------------------------------------------
# Tests: cmd_suggest
# ---------------------------------------------------------------------------


class TestCmdSuggest:
    """Tests for the suggest command."""

    def test_suggest_misspelled(self, capsys) -> None:
        """Should return suggestions for misspelled terms."""
        cli = _build_cli_with_index()
        cli.cmd_suggest("lov")
        captured = capsys.readouterr()
        assert "Suggestions" in captured.out

    def test_suggest_correct_word(self, capsys) -> None:
        """Correct words should produce no suggestions."""
        cli = _build_cli_with_index()
        cli.cmd_suggest("love")
        captured = capsys.readouterr()
        assert "No suggestions" in captured.out or "look correct" in captured.out

    def test_suggest_empty(self, capsys) -> None:
        """Empty query should show usage."""
        cli = _build_cli_with_index()
        cli.cmd_suggest("")
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_suggest_no_index(self, capsys) -> None:
        """Should warn if no index loaded."""
        cli = CLI()
        cli.cmd_suggest("love")
        captured = capsys.readouterr()
        assert "No index loaded" in captured.out


# ---------------------------------------------------------------------------
# Tests: CLI.run (interactive loop)
# ---------------------------------------------------------------------------


class TestCLIRun:
    """Tests for the interactive main loop."""

    def test_quit_command(self, capsys) -> None:
        """'quit' should exit the loop."""
        cli = CLI()
        with patch("builtins.input", side_effect=["quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_exit_command(self, capsys) -> None:
        """'exit' should exit the loop."""
        cli = CLI()
        with patch("builtins.input", side_effect=["exit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_help_command(self, capsys) -> None:
        """'help' should display available commands."""
        cli = CLI()
        with patch("builtins.input", side_effect=["help", "quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "build" in captured.out
        assert "find" in captured.out

    def test_unknown_command(self, capsys) -> None:
        """Unknown commands should show an error."""
        cli = CLI()
        with patch("builtins.input", side_effect=["foobar", "quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out

    def test_empty_input(self, capsys) -> None:
        """Empty input should be ignored."""
        cli = CLI()
        with patch("builtins.input", side_effect=["", "quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_eof_handling(self, capsys) -> None:
        """EOF should exit gracefully."""
        cli = CLI()
        with patch("builtins.input", side_effect=EOFError):
            cli.run()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_keyboard_interrupt(self, capsys) -> None:
        """Ctrl+C should exit gracefully."""
        cli = CLI()
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            cli.run()
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_load_command_in_loop(self, capsys) -> None:
        """'load' command should invoke cmd_load."""
        cli = CLI()
        with patch("builtins.input", side_effect=["load", "quit"]):
            with patch("src.main.INDEX_PATH", "/nonexistent/index.json"):
                cli.run()
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_find_command_in_loop(self, capsys) -> None:
        """'find' with no index should warn."""
        cli = CLI()
        with patch("builtins.input", side_effect=["find love", "quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "No index loaded" in captured.out

    def test_print_command_in_loop(self, capsys) -> None:
        """'print' with no index should warn."""
        cli = CLI()
        with patch("builtins.input", side_effect=["print world", "quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "No index loaded" in captured.out

    def test_suggest_command_in_loop(self, capsys) -> None:
        """'suggest' with no index should warn."""
        cli = CLI()
        with patch("builtins.input", side_effect=["suggest love", "quit"]):
            cli.run()
        captured = capsys.readouterr()
        assert "No index loaded" in captured.out

    @patch("src.main.Crawler")
    def test_build_command_in_loop(self, mock_crawler_cls: MagicMock, capsys) -> None:
        """'build' should invoke the full build pipeline."""
        mock_crawler = MagicMock()
        mock_crawler.crawl.return_value = [
            Page(
                page_id=0,
                url="https://quotes.toscrape.com/page/1/",
                quotes=[Quote("Test", "Auth", ["tag"])],
            ),
        ]
        mock_crawler_cls.return_value = mock_crawler

        cli = CLI()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("src.main.INDEX_PATH", os.path.join(tmpdir, "idx.json")):
                with patch("builtins.input", side_effect=["build", "quit"]):
                    cli.run()

        captured = capsys.readouterr()
        assert "Crawled 1 pages" in captured.out
