"""Tests for the CLI module."""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from afterimage.cli import (
    cmd_clear,
    cmd_config,
    cmd_export,
    cmd_ingest,
    cmd_recent,
    cmd_search,
    cmd_stats,
    main,
    _format_bytes,
)


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Set up isolated home directory for tests."""
    # Create .afterimage directory
    afterimage_dir = tmp_path / ".afterimage"
    afterimage_dir.mkdir()

    # Set HOME environment
    monkeypatch.setenv("HOME", str(tmp_path))

    return tmp_path


class TestFormatBytes:
    """Tests for _format_bytes helper."""

    def test_bytes(self):
        """Test byte formatting."""
        assert _format_bytes(500) == "500.0 B"

    def test_kilobytes(self):
        """Test kilobyte formatting."""
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(1536) == "1.5 KB"

    def test_megabytes(self):
        """Test megabyte formatting."""
        assert _format_bytes(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self):
        """Test gigabyte formatting."""
        assert _format_bytes(1024 * 1024 * 1024) == "1.0 GB"

    def test_terabytes(self):
        """Test terabyte formatting."""
        assert _format_bytes(1024 * 1024 * 1024 * 1024) == "1.0 TB"

    def test_zero_bytes(self):
        """Test zero byte formatting."""
        assert _format_bytes(0) == "0.0 B"


class TestCmdStats:
    """Tests for stats command."""

    def test_stats_json_empty(self, temp_home, capsys):
        """Test stats command with JSON output on empty KB."""
        args = argparse.Namespace(json=True)
        result = cmd_stats(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_entries" in data
        assert data["total_entries"] == 0

    def test_stats_text_empty(self, temp_home, capsys):
        """Test stats command with text output on empty KB."""
        args = argparse.Namespace(json=False)
        result = cmd_stats(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Total entries:" in captured.out
        assert "0" in captured.out

    def test_stats_with_data(self, temp_home, capsys):
        """Test stats command with data in KB."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code", session_id="sess1")
        kb.store(file_path="/other.py", new_code="code2", session_id="sess2")

        args = argparse.Namespace(json=True)
        result = cmd_stats(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["total_entries"] == 2
        assert data["unique_files"] == 2
        assert data["unique_sessions"] == 2


class TestCmdRecent:
    """Tests for recent command."""

    def test_recent_empty_kb(self, temp_home, capsys):
        """Test recent command with empty KB."""
        args = argparse.Namespace(limit=10, json=False)
        result = cmd_recent(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No entries found" in captured.out

    def test_recent_with_data(self, temp_home, capsys):
        """Test recent command with data."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="def hello(): pass")

        args = argparse.Namespace(limit=10, json=False)
        result = cmd_recent(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "test.py" in captured.out
        assert "def hello(): pass" in captured.out

    def test_recent_json(self, temp_home, capsys):
        """Test recent command with JSON output."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code")

        args = argparse.Namespace(limit=10, json=True)
        result = cmd_recent(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["file_path"] == "/test.py"

    def test_recent_limit(self, temp_home, capsys):
        """Test recent command respects limit."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        for i in range(5):
            kb.store(file_path=f"/file{i}.py", new_code=f"code{i}")

        args = argparse.Namespace(limit=2, json=True)
        result = cmd_recent(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) == 2

    def test_recent_truncates_long_code(self, temp_home, capsys):
        """Test recent command truncates long code."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        long_code = "x = 1\n" * 200  # Over 500 chars
        kb.store(file_path="/test.py", new_code=long_code)

        args = argparse.Namespace(limit=10, json=False)
        result = cmd_recent(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "(truncated)" in captured.out


class TestCmdExport:
    """Tests for export command."""

    def test_export_stdout(self, temp_home, capsys):
        """Test export to stdout."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code")

        args = argparse.Namespace(output=None)
        result = cmd_export(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "exported_at" in data
        assert data["count"] == 1
        assert "entries" in data

    def test_export_to_file(self, temp_home, capsys):
        """Test export to file."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code")

        output_file = temp_home / "export.json"
        args = argparse.Namespace(output=str(output_file))
        result = cmd_export(args)

        assert result == 0
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)
        assert data["count"] == 1


class TestCmdClear:
    """Tests for clear command."""

    def test_clear_with_yes_flag(self, temp_home, capsys):
        """Test clear with --yes flag (non-interactive)."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code")

        args = argparse.Namespace(yes=True)
        result = cmd_clear(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Cleared 1 entries" in captured.out

        # Verify KB is empty
        kb2 = KnowledgeBase()
        stats = kb2.stats()
        assert stats["total_entries"] == 0

    def test_clear_without_yes_aborted(self, temp_home, capsys, monkeypatch):
        """Test clear without --yes flag, user aborts."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code")

        # Mock input to return 'n'
        monkeypatch.setattr("builtins.input", lambda _: "n")

        args = argparse.Namespace(yes=False)
        result = cmd_clear(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Aborted" in captured.out

        # Verify KB is not cleared
        kb2 = KnowledgeBase()
        stats = kb2.stats()
        assert stats["total_entries"] == 1

    def test_clear_without_yes_confirmed(self, temp_home, capsys, monkeypatch):
        """Test clear without --yes flag, user confirms."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="code")

        # Mock input to return 'y'
        monkeypatch.setattr("builtins.input", lambda _: "y")

        args = argparse.Namespace(yes=False)
        result = cmd_clear(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Cleared 1 entries" in captured.out


class TestCmdConfig:
    """Tests for config command."""

    def test_config_show_no_file(self, temp_home, capsys):
        """Test config command when no config file exists."""
        args = argparse.Namespace(init=False, force=False)
        result = cmd_config(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No config file found" in captured.out
        assert "afterimage config --init" in captured.out

    def test_config_init(self, temp_home, capsys):
        """Test config --init creates config file."""
        args = argparse.Namespace(init=True, force=False)
        result = cmd_config(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Created config" in captured.out

        config_path = temp_home / ".afterimage" / "config.yaml"
        assert config_path.exists()

        content = config_path.read_text()
        assert "search:" in content
        assert "filter:" in content
        assert "embeddings:" in content

    def test_config_init_existing_no_force(self, temp_home, capsys):
        """Test config --init fails if file exists without --force."""
        # Create existing config
        config_dir = temp_home / ".afterimage"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text("existing config")

        args = argparse.Namespace(init=True, force=False)
        result = cmd_config(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "already exists" in captured.out
        assert "--force" in captured.out

    def test_config_init_existing_with_force(self, temp_home, capsys):
        """Test config --init overwrites with --force."""
        # Create existing config
        config_dir = temp_home / ".afterimage"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text("existing config")

        args = argparse.Namespace(init=True, force=True)
        result = cmd_config(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Created config" in captured.out

        # Verify content changed
        content = config_path.read_text()
        assert "AI-AfterImage Configuration" in content

    def test_config_show_existing(self, temp_home, capsys):
        """Test config command shows existing config."""
        # Create config
        config_dir = temp_home / ".afterimage"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.yaml"
        config_path.write_text("test: value\n")

        args = argparse.Namespace(init=False, force=False)
        result = cmd_config(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Config location:" in captured.out
        assert "test: value" in captured.out


class TestCmdSearch:
    """Tests for search command."""

    def test_search_no_results(self, temp_home, capsys):
        """Test search with no results."""
        args = argparse.Namespace(
            query="nonexistent",
            limit=5,
            threshold=0.3,
            path=None,
            json=False
        )
        result = cmd_search(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No results found" in captured.out

    def test_search_with_results(self, temp_home, capsys):
        """Test search with results."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="def authenticate(): pass")

        args = argparse.Namespace(
            query="authenticate",
            limit=5,
            threshold=0.1,  # Low threshold for FTS match
            path=None,
            json=False
        )
        result = cmd_search(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Found" in captured.out
        assert "result" in captured.out

    def test_search_json_output(self, temp_home, capsys):
        """Test search with JSON output."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/test.py", new_code="def validate(): pass")

        args = argparse.Namespace(
            query="validate",
            limit=5,
            threshold=0.1,
            path=None,
            json=True
        )
        result = cmd_search(args)

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_search_with_path_filter(self, temp_home, capsys):
        """Test search with path filter."""
        from afterimage.kb import KnowledgeBase
        kb = KnowledgeBase()
        kb.store(file_path="/src/auth/login.py", new_code="def login(): pass")
        kb.store(file_path="/src/db/models.py", new_code="def login(): pass")

        args = argparse.Namespace(
            query="login",
            limit=5,
            threshold=0.1,
            path="auth",
            json=True
        )
        result = cmd_search(args)

        # Results should be filtered to auth path
        if result == 0:
            captured = capsys.readouterr()
            data = json.loads(captured.out)
            for item in data:
                assert "auth" in item.get("file_path", "")


class TestCmdIngest:
    """Tests for ingest command."""

    def test_ingest_no_files(self, temp_home, capsys):
        """Test ingest with no transcript files found."""
        args = argparse.Namespace(
            file=None,
            directory=str(temp_home / "empty"),
            no_embeddings=True,
            verbose=False
        )

        # Create empty directory
        (temp_home / "empty").mkdir()

        result = cmd_ingest(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No transcript files found" in captured.out

    def test_ingest_verbose(self, temp_home, capsys):
        """Test ingest with verbose flag."""
        # Create a minimal transcript file
        transcript_dir = temp_home / "transcripts"
        transcript_dir.mkdir()

        # Simple JSON file that might be recognized
        transcript_file = transcript_dir / "transcript.json"
        transcript_file.write_text('{"messages": []}')

        args = argparse.Namespace(
            file=str(transcript_file),
            directory=None,
            no_embeddings=True,
            verbose=True
        )

        # This will process the file (likely finding 0 changes)
        result = cmd_ingest(args)

        captured = capsys.readouterr()
        assert "Processing:" in captured.out or "transcript file" in captured.out


class TestMain:
    """Tests for main entry point."""

    def test_main_no_args(self, monkeypatch, capsys):
        """Test main with no arguments shows help."""
        monkeypatch.setattr(sys, "argv", ["afterimage"])

        result = main()

        assert result == 0
        captured = capsys.readouterr()
        # Help output should contain usage info
        assert "afterimage" in captured.out.lower() or "usage" in captured.out.lower()

    def test_main_version(self, monkeypatch, capsys):
        """Test main with --version."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "--version"])

        with pytest.raises(SystemExit) as exc_info:
            main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "0.3.0" in captured.out

    def test_main_unknown_command(self, monkeypatch, capsys):
        """Test main with unknown command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "unknown"])

        with pytest.raises(SystemExit):
            main()

    def test_main_stats_command(self, temp_home, monkeypatch, capsys):
        """Test main dispatches to stats command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "stats", "--json"])

        result = main()

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "total_entries" in data

    def test_main_recent_command(self, temp_home, monkeypatch, capsys):
        """Test main dispatches to recent command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "recent", "-l", "5"])

        result = main()

        # Returns 1 because KB is empty
        assert result == 1

    def test_main_config_command(self, temp_home, monkeypatch, capsys):
        """Test main dispatches to config command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "config"])

        result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "No config file found" in captured.out or "config" in captured.out.lower()

    def test_main_search_command(self, temp_home, monkeypatch, capsys):
        """Test main dispatches to search command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "search", "test"])

        result = main()

        # Returns 1 because no results
        assert result == 1

    def test_main_export_command(self, temp_home, monkeypatch, capsys):
        """Test main dispatches to export command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "export"])

        result = main()

        assert result == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "entries" in data

    def test_main_clear_command(self, temp_home, monkeypatch, capsys):
        """Test main dispatches to clear command."""
        monkeypatch.setattr(sys, "argv", ["afterimage", "clear", "-y"])

        result = main()

        assert result == 0
