"""Tests for standalone Claude Code hook configuration helpers."""

import importlib.util
from pathlib import Path


def load_hook_module():
    hook_path = Path(__file__).parent.parent / "hooks" / "afterimage_hook.py"
    spec = importlib.util.spec_from_file_location("afterimage_hook_test", hook_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_seen_write_key_file_mode(monkeypatch):
    """File mode should show context once per file regardless of content."""
    monkeypatch.setenv("AFTERIMAGE_HOOK_SEEN_WRITE_KEY", "file")
    hook = load_hook_module()

    first = hook.get_content_hash("/tmp/app.py", "def one(): pass")
    second = hook.get_content_hash("/tmp/app.py", "def two(): pass")

    assert first == second


def test_seen_write_key_content_mode(monkeypatch):
    """Content mode should distinguish different write attempts."""
    monkeypatch.setenv("AFTERIMAGE_HOOK_SEEN_WRITE_KEY", "content")
    hook = load_hook_module()

    first = hook.get_content_hash("/tmp/app.py", "def one(): pass")
    second = hook.get_content_hash("/tmp/app.py", "def two(): pass")

    assert first != second


def test_seen_write_key_session_file_mode(monkeypatch):
    """Session-file mode should reset once the Claude session changes."""
    monkeypatch.setenv("AFTERIMAGE_HOOK_SEEN_WRITE_KEY", "session_file")
    hook = load_hook_module()

    monkeypatch.setenv("CLAUDE_SESSION_ID", "session-a")
    first = hook.get_content_hash("/tmp/app.py", "def one(): pass")

    monkeypatch.setenv("CLAUDE_SESSION_ID", "session-b")
    second = hook.get_content_hash("/tmp/app.py", "def one(): pass")

    assert first != second


def test_seen_write_key_invalid_mode_falls_back_to_file(monkeypatch):
    """Invalid modes should fall back to file mode."""
    monkeypatch.setenv("AFTERIMAGE_HOOK_SEEN_WRITE_KEY", "nonsense")
    hook = load_hook_module()

    first = hook.get_content_hash("/tmp/app.py", "def one(): pass")
    second = hook.get_content_hash("/tmp/app.py", "def two(): pass")

    assert first == second
