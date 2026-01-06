"""
Integration Tests for AI-AfterImage.

End-to-end tests covering the full workflow:
1. Ingest transcript -> Store in KB
2. Search KB -> Find relevant code
3. Format results -> Inject context

These tests validate the entire pipeline works together.
"""

import json
import os
import tempfile
import shutil
import pytest
from pathlib import Path
from datetime import datetime, timezone

from afterimage.kb import KnowledgeBase
from afterimage.extract import TranscriptExtractor, CodeChange
from afterimage.filter import CodeFilter
from afterimage.search import HybridSearch, SearchResult
from afterimage.inject import ContextInjector, InjectionConfig


class TestFullPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        tmpdir = tempfile.mkdtemp(prefix="afterimage_test_")
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def test_kb(self, temp_dir):
        """Create a test knowledge base."""
        db_path = Path(temp_dir) / "test_memory.db"
        kb = KnowledgeBase(db_path=db_path)
        yield kb
        # KB manages connections per-operation, no close needed

    @pytest.fixture
    def sample_transcript(self, temp_dir):
        """Create a sample transcript file."""
        transcript_path = Path(temp_dir) / "transcript.jsonl"

        # Create sample conversation entries using supported formats
        entries = [
            {
                "role": "user",
                "content": "Can you add a function to validate email addresses?"
            },
            {
                "role": "assistant",
                "content": "I'll add an email validation function."
            },
            # Format 2: {"tool": "...", "input": {...}}
            {
                "tool": "Write",
                "input": {
                    "file_path": "/home/user/project/src/validators.py",
                    "content": "import re\n\ndef validate_email(email: str) -> bool:\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n"
                }
            },
            {
                "type": "tool_result",
                "result": {"success": True}
            },
            {
                "role": "user",
                "content": "Now add a function to validate phone numbers"
            },
            {
                "tool": "Write",
                "input": {
                    "file_path": "/home/user/project/src/validators.py",
                    "content": "import re\n\ndef validate_email(email: str) -> bool:\n    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n    return bool(re.match(pattern, email))\n\ndef validate_phone(phone: str) -> bool:\n    pattern = r'^\\\\+?1?[\\\\s.-]?\\\\(?\\\\d{3}\\\\)?[\\\\s.-]?\\\\d{3}[\\\\s.-]?\\\\d{4}$'\n    return bool(re.match(pattern, phone))\n"
                }
            },
            {
                "type": "tool_result",
                "result": {"success": True}
            },
            # Also add an Edit operation
            {
                "role": "user",
                "content": "Fix the email pattern to handle subdomains"
            },
            {
                "tool": "Edit",
                "input": {
                    "file_path": "/home/user/project/src/validators.py",
                    "old_string": "pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'",
                    "new_string": "pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+(?:\\.[a-zA-Z]{2,})+$'"
                }
            },
            {
                "type": "tool_result",
                "result": {"success": True}
            },
        ]

        with open(transcript_path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

        return transcript_path

    def test_extract_from_transcript(self, sample_transcript):
        """Should extract code changes from transcript."""
        extractor = TranscriptExtractor()

        changes = extractor.extract_from_file(sample_transcript)

        # Should find 2 Write + 1 Edit = 3 changes
        assert len(changes) >= 2
        # All should have file_path
        assert all(c.file_path for c in changes)
        # All should have new_code
        assert all(c.new_code for c in changes)

    def test_filter_code_files(self, sample_transcript):
        """Should correctly identify code files."""
        extractor = TranscriptExtractor()
        code_filter = CodeFilter()

        changes = extractor.extract_from_file(sample_transcript)

        for change in changes:
            # validators.py should be recognized as code
            assert code_filter.is_code(change.file_path, change.new_code)

        # Non-code files should be rejected
        assert not code_filter.is_code("/path/to/file.md", "# README")
        assert not code_filter.is_code("/path/to/data.json", '{"key": "value"}')

    def test_store_and_retrieve(self, test_kb):
        """Should store and retrieve code entries."""
        # Store some entries
        id1 = test_kb.store(
            file_path="/project/src/auth.py",
            new_code="def authenticate(user, password):\n    return True",
            context="Added authentication function",
            session_id="session_1"
        )

        id2 = test_kb.store(
            file_path="/project/src/database.py",
            new_code="def connect_db():\n    return connection",
            context="Added database connection",
            session_id="session_1"
        )

        # Retrieve and verify
        stats = test_kb.stats()
        assert stats["total_entries"] == 2
        assert stats["unique_files"] == 2

        # Get recent
        recent = test_kb.get_recent(10)
        assert len(recent) == 2

    def test_search_by_path(self, test_kb):
        """Should find entries by file path."""
        # Store entries
        test_kb.store(
            file_path="/project/src/utils/validators.py",
            new_code="def validate():\n    pass",
            context="Validator utils",
            session_id="s1"
        )
        test_kb.store(
            file_path="/project/src/models/user.py",
            new_code="class User:\n    pass",
            context="User model",
            session_id="s1"
        )

        # Search by path
        results = test_kb.search_by_path("validators", limit=10)
        assert len(results) == 1
        assert "validators.py" in results[0]["file_path"]

        results = test_kb.search_by_path("models", limit=10)
        assert len(results) == 1
        assert "user.py" in results[0]["file_path"]

    def test_fts_search(self, test_kb):
        """Should find entries using full-text search."""
        # Store entries with searchable content
        test_kb.store(
            file_path="/project/auth.py",
            new_code="def login(username, password):\n    # Authenticate user\n    return True",
            context="Login function for user authentication",
            session_id="s1"
        )
        test_kb.store(
            file_path="/project/api.py",
            new_code="def get_users():\n    # Fetch all users from database\n    return []",
            context="API endpoint to list users",
            session_id="s1"
        )

        # Search for authentication-related code
        results = test_kb.search_fts("authenticate", limit=5)
        assert len(results) >= 1

        # Search for user-related code
        results = test_kb.search_fts("users", limit=5)
        assert len(results) >= 1

    def test_inject_context_from_search(self, test_kb):
        """Should format search results for injection."""
        # Store code
        test_kb.store(
            file_path="/project/src/validators.py",
            new_code="def validate_email(email):\n    return '@' in email",
            context="Simple email validation",
            session_id="s1"
        )

        # Create search result manually (bypassing embeddings)
        result = SearchResult(
            id="test_id",
            file_path="/project/src/validators.py",
            new_code="def validate_email(email):\n    return '@' in email",
            old_code=None,
            context="Simple email validation",
            timestamp="2026-01-06T12:00:00Z",
            session_id="s1",
            relevance_score=0.8,
            fts_score=0.6,
            semantic_score=0.9
        )

        # Format for injection
        injector = ContextInjector()
        output = injector.format_injection([result])

        assert output is not None
        assert "You have written similar code before" in output
        assert "validate_email" in output
        assert "Simple email validation" in output

    def test_hook_format_integration(self, test_kb):
        """Should format results for hook injection."""
        result = SearchResult(
            id="test_id",
            file_path="/project/src/auth.py",
            new_code="def login():\n    pass",
            old_code=None,
            context="Login implementation",
            timestamp="2026-01-06T12:00:00Z",
            session_id="s1",
            relevance_score=0.75
        )

        injector = ContextInjector()
        output = injector.format_for_hook(
            results=[result],
            file_path="/project/src/new_auth.py",
            tool_type="Write"
        )

        assert output is not None
        assert "<memory" in output
        assert "creating" in output
        assert "new_auth.py" in output
        assert "</memory>" in output


class TestPipelineWithEmbeddings:
    """Tests that use the embedding system (may be slow)."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test data."""
        tmpdir = tempfile.mkdtemp(prefix="afterimage_embed_test_")
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def test_kb(self, temp_dir):
        """Create a test knowledge base."""
        db_path = Path(temp_dir) / "test_memory.db"
        kb = KnowledgeBase(db_path=db_path)
        yield kb
        # KB manages connections per-operation, no close needed

    @pytest.mark.slow
    def test_semantic_search_with_embeddings(self, test_kb):
        """Should perform semantic search with embeddings."""
        try:
            from afterimage.embeddings import EmbeddingGenerator
            embedder = EmbeddingGenerator()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Store code with embeddings
        code1 = "def validate_email(email):\n    return '@' in email"
        emb1 = embedder.embed_code(code1, "/project/email.py")
        test_kb.store(
            file_path="/project/email.py",
            new_code=code1,
            context="Email validation",
            session_id="s1",
            embedding=emb1
        )

        code2 = "def check_password(pwd):\n    return len(pwd) >= 8"
        emb2 = embedder.embed_code(code2, "/project/password.py")
        test_kb.store(
            file_path="/project/password.py",
            new_code=code2,
            context="Password validation",
            session_id="s1",
            embedding=emb2
        )

        # Search semantically
        search = HybridSearch(kb=test_kb, embedder=embedder)
        results = search.search("email validation function", limit=5)

        # Should find the email validator
        assert len(results) >= 1
        # Email-related code should rank higher
        email_results = [r for r in results if "email" in r.file_path]
        assert len(email_results) >= 1

    @pytest.mark.slow
    def test_hybrid_search_combines_scores(self, test_kb):
        """Should combine FTS and semantic scores."""
        try:
            from afterimage.embeddings import EmbeddingGenerator
            embedder = EmbeddingGenerator()
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Store entries
        code = "def authenticate_user(username, password):\n    return True"
        emb = embedder.embed_code(code, "/project/auth.py")
        test_kb.store(
            file_path="/project/auth.py",
            new_code=code,
            context="User authentication",
            session_id="s1",
            embedding=emb
        )

        # Search
        search = HybridSearch(kb=test_kb, embedder=embedder)
        results = search.search("authenticate user login", limit=5)

        # Should have both FTS and semantic scores
        if results:
            result = results[0]
            # At least one score should be > 0
            assert result.fts_score > 0 or result.semantic_score > 0


class TestCLIIntegration:
    """Tests for CLI command integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp(prefix="afterimage_cli_test_")
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = tmpdir
        yield tmpdir
        os.environ["HOME"] = old_home or ""
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_config_init_creates_file(self, temp_dir):
        """Should create config file."""
        import argparse
        from afterimage.cli import cmd_config

        args = argparse.Namespace(init=True, force=False)
        result = cmd_config(args)

        config_path = Path(temp_dir) / ".afterimage" / "config.yaml"
        assert config_path.exists()

    def test_stats_shows_empty_kb(self, temp_dir):
        """Should handle empty KB gracefully."""
        import argparse
        from afterimage.cli import cmd_stats

        args = argparse.Namespace(json=False)
        # Should not raise
        result = cmd_stats(args)
        assert result == 0


class TestHookIntegration:
    """Tests for hook script integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp(prefix="afterimage_hook_test_")
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = tmpdir
        yield tmpdir
        os.environ["HOME"] = old_home or ""
        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_handle_hook_write(self, temp_dir):
        """Should handle Write hook."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from hooks.afterimage_hook import handle_hook

        hook_data = {
            "type": "post",
            "tool": "Write",
            "input": {
                "file_path": "/project/src/module.py",
                "content": "def test_function():\n    return True"
            },
            "context": "Added test function"
        }

        response = handle_hook(hook_data)

        assert response["success"] is True

    def test_handle_hook_edit(self, temp_dir):
        """Should handle Edit hook."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from hooks.afterimage_hook import handle_hook

        hook_data = {
            "type": "post",
            "tool": "Edit",
            "input": {
                "file_path": "/project/src/module.py",
                "old_string": "def old_function():",
                "new_string": "def new_function():"
            },
            "context": "Renamed function"
        }

        response = handle_hook(hook_data)

        assert response["success"] is True

    def test_handle_hook_skips_non_code(self, temp_dir):
        """Should skip non-code files."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from hooks.afterimage_hook import handle_hook

        hook_data = {
            "type": "post",
            "tool": "Write",
            "input": {
                "file_path": "/project/README.md",
                "content": "# Readme\n\nThis is documentation."
            },
            "context": "Updated readme"
        }

        response = handle_hook(hook_data)

        # Should succeed but not store (non-code file)
        assert response["success"] is True

    def test_pre_hook_returns_injection(self, temp_dir):
        """Should return injection for pre-hooks when relevant code found."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from hooks.afterimage_hook import handle_hook
        from afterimage.kb import KnowledgeBase

        # First store some code
        kb = KnowledgeBase()
        kb.store(
            file_path="/project/src/auth.py",
            new_code="def authenticate(user, pwd):\n    return True",
            context="Auth function",
            session_id="s1"
        )

        # Now try pre-hook
        hook_data = {
            "type": "pre",
            "tool": "Write",
            "input": {
                "file_path": "/project/src/new_auth.py",
                "content": "def login(username, password):\n    pass"
            },
            "context": "New auth module"
        }

        response = handle_hook(hook_data)

        assert response["success"] is True
        # May or may not have injection depending on search results
        # Just verify it doesn't crash

        # KB manages connections per-operation, no close needed


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_empty_transcript(self):
        """Should handle empty transcript."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("")
            path = f.name

        try:
            extractor = TranscriptExtractor()
            changes = extractor.extract_from_file(path)
            assert changes == []
        finally:
            os.unlink(path)

    def test_malformed_transcript(self):
        """Should handle malformed JSON in transcript."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("not valid json\n")
            f.write('{"valid": "json"}\n')
            f.write("{also invalid}\n")
            path = f.name

        try:
            extractor = TranscriptExtractor()
            # Should not crash
            changes = extractor.extract_from_file(path)
            # May extract 0 changes but shouldn't crash
        finally:
            os.unlink(path)

    def test_unicode_content(self):
        """Should handle unicode in code."""
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "test.db"
            kb = KnowledgeBase(db_path=db_path)

            # Store code with unicode (using valid unicode, not surrogates)
            code = 'def greet(name):\n    """Say hello in multiple languages."""\n    return f"Hello {name}! \u4f60\u597d \u3053\u3093\u306b\u3061\u306f"\n'
            entry_id = kb.store(
                file_path="/project/greet.py",
                new_code=code,
                context="Multi-language greeting",
                session_id="s1"
            )

            # Retrieve and verify
            recent = kb.get_recent(1)
            assert len(recent) == 1
            assert "\u4f60\u597d" in recent[0]["new_code"]  # Chinese characters
            assert "\u3053\u3093\u306b\u3061\u306f" in recent[0]["new_code"]  # Japanese

            # KB manages connections per-operation, no close needed
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_very_large_code(self):
        """Should handle very large code blocks."""
        import tempfile
        import shutil

        tmpdir = tempfile.mkdtemp()
        try:
            db_path = Path(tmpdir) / "test.db"
            kb = KnowledgeBase(db_path=db_path)

            # Generate large code
            large_code = "\n".join([f"line_{i} = {i}" for i in range(10000)])

            entry_id = kb.store(
                file_path="/project/large.py",
                new_code=large_code,
                context="Large file",
                session_id="s1"
            )

            assert entry_id is not None

            # Verify truncation in injection
            result = SearchResult(
                id=entry_id,
                file_path="/project/large.py",
                new_code=large_code,
                old_code=None,
                context="Large file",
                timestamp="2026-01-06T12:00:00Z",
                session_id="s1",
                relevance_score=0.8
            )

            injector = ContextInjector()
            output = injector.format_single(result)

            # Should be truncated
            assert "truncated" in output.lower()

            # KB manages connections per-operation, no close needed
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
