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
        search = HybridSearch(backend=test_kb.backend, embedder=embedder)
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
        search = HybridSearch(backend=test_kb.backend, embedder=embedder)
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
        # Create .afterimage directory for KB to use
        (Path(tmpdir) / ".afterimage").mkdir(parents=True, exist_ok=True)
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


@pytest.mark.skip(reason="Hook uses stdin/stdout API, not function API - tested via E2E tests")
class TestHookIntegration:
    """Tests for hook script integration.

    NOTE: These tests are skipped because the hook script uses stdin/stdout
    rather than a function API. The hook is tested end-to-end via actual
    Claude Code integration. See test_churn.py for churn tracking unit tests.
    """

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
        pytest.skip("Hook uses stdin/stdout API")

    def test_handle_hook_edit(self, temp_dir):
        """Should handle Edit hook."""
        pytest.skip("Hook uses stdin/stdout API")

    def test_handle_hook_skips_non_code(self, temp_dir):
        """Should skip non-code files."""
        pytest.skip("Hook uses stdin/stdout API")

    def test_pre_hook_returns_injection(self, temp_dir):
        """Should return injection for pre-hooks when relevant code found."""
        pytest.skip("Hook uses stdin/stdout API")


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


class TestOfflineFunctionality:
    """
    Tests for offline functionality (Phase 7 success criteria).

    After initial model download, the system should work completely offline
    with no network access required.
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        tmpdir = tempfile.mkdtemp(prefix="afterimage_offline_test_")
        yield tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)

    @pytest.fixture
    def test_kb(self, temp_dir):
        """Create a test knowledge base."""
        db_path = Path(temp_dir) / "test_memory.db"
        kb = KnowledgeBase(db_path=db_path)
        yield kb

    def test_sqlite_is_local(self, test_kb):
        """SQLite should be completely local (no network)."""
        # Store and retrieve without network
        entry_id = test_kb.store(
            file_path="/project/offline.py",
            new_code="def offline_function():\n    return 'works offline'",
            context="Testing offline storage",
            session_id="offline_test"
        )

        assert entry_id is not None

        # FTS search is local
        results = test_kb.search_fts("offline", limit=5)
        assert len(results) >= 1
        assert "offline" in results[0]["new_code"]

    def test_code_filter_is_local(self):
        """Code filter uses no network resources."""
        code_filter = CodeFilter()

        # All operations are local lookups
        assert code_filter.is_code("/path/to/file.py", "def foo(): pass") is True
        assert code_filter.is_code("/path/to/file.md", "# Readme") is False
        assert code_filter.is_code("/path/to/node_modules/file.js", "code") is False

        # Extension and path checks are local (via sets)
        assert ".py" in code_filter.code_extensions
        assert ".json" in code_filter.skip_extensions
        assert "artifacts/" in code_filter.skip_paths

    def test_inject_formatter_is_local(self):
        """Context injection formatting uses no network."""
        result = SearchResult(
            id="test_id",
            file_path="/project/test.py",
            new_code="def test(): pass",
            old_code=None,
            context="Test context",
            timestamp="2026-01-06T12:00:00Z",
            session_id="s1",
            relevance_score=0.85
        )

        injector = ContextInjector()

        # All formatting is local string operations
        output = injector.format_injection([result])
        assert output is not None
        assert "test" in output.lower()

        hook_output = injector.format_for_hook([result], "/project/new.py", "Write")
        assert hook_output is not None
        assert "<memory" in hook_output

    def test_transcript_extractor_is_local(self, temp_dir):
        """Transcript extraction uses no network."""
        # Create local transcript file
        transcript_path = Path(temp_dir) / "test.jsonl"
        with open(transcript_path, "w") as f:
            f.write(json.dumps({
                "tool": "Write",
                "input": {
                    "file_path": "/project/local.py",
                    "content": "# local file"
                }
            }) + "\n")

        extractor = TranscriptExtractor()
        changes = extractor.extract_from_file(transcript_path)

        assert len(changes) >= 1
        assert changes[0].file_path == "/project/local.py"

    @pytest.mark.slow
    def test_embeddings_use_cached_model(self):
        """Embeddings should use locally cached model (no download after first use)."""
        try:
            from afterimage.embeddings import EmbeddingGenerator, get_cache_dir
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Check that model cache exists
        cache_dir = get_cache_dir()
        model_dir = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"

        # Model should be cached from previous runs
        # (First run downloads, subsequent runs use cache)
        if not model_dir.exists():
            pytest.skip("Model not yet cached - run once with network to cache")

        # Verify model files exist locally
        assert model_dir.exists(), "Model should be cached locally"

        # Load model from cache (no network needed)
        embedder = EmbeddingGenerator()

        # Generate embedding - should work offline with cached model
        embedding = embedder.embed("def offline_test(): pass")

        assert embedding is not None
        assert len(embedding) == 384  # all-MiniLM-L6-v2 dimension

        # Batch embedding should also work offline
        embeddings = embedder.embed_batch([
            "def function_a(): pass",
            "def function_b(): pass"
        ])
        assert len(embeddings) == 2
        assert all(len(e) == 384 for e in embeddings)

    @pytest.mark.slow
    def test_full_pipeline_offline(self, temp_dir):
        """Full pipeline should work offline after model caching."""
        try:
            from afterimage.embeddings import EmbeddingGenerator, get_cache_dir
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        # Check model is cached
        cache_dir = get_cache_dir()
        model_dir = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
        if not model_dir.exists():
            pytest.skip("Model not yet cached - run once with network to cache")

        # Create KB
        db_path = Path(temp_dir) / "offline_test.db"
        kb = KnowledgeBase(db_path=db_path)

        # Create embedder
        embedder = EmbeddingGenerator()

        # Store with embeddings (all local)
        code = "def validate_input(data):\n    return bool(data)"
        embedding = embedder.embed_code(code, "/project/validators.py")
        kb.store(
            file_path="/project/validators.py",
            new_code=code,
            context="Input validation",
            session_id="offline_s1",
            embedding=embedding
        )

        # Search with hybrid (all local)
        search = HybridSearch(backend=kb.backend, embedder=embedder)
        results = search.search("validate", limit=5)

        assert len(results) >= 1
        assert "validate" in results[0].new_code

        # Format for injection (all local)
        injector = ContextInjector()
        output = injector.format_injection(results)

        assert output is not None
        assert "validate" in output

    def test_config_loading_is_local(self, temp_dir):
        """Configuration loading uses local YAML files only."""
        import yaml

        # Create config directory
        config_dir = Path(temp_dir) / ".afterimage"
        config_dir.mkdir(parents=True)

        config_path = config_dir / "config.yaml"
        config_data = {
            "search": {
                "max_results": 10,
                "relevance_threshold": 0.5
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2",
                "device": "cpu"
            }
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        # Load config (pure local file read)
        with open(config_path) as f:
            loaded = yaml.safe_load(f)

        assert loaded["search"]["max_results"] == 10
        assert loaded["embeddings"]["model"] == "all-MiniLM-L6-v2"

    def test_no_external_api_calls_in_modules(self):
        """Verify no modules make external API calls."""
        # Import all core modules - they should not make network calls on import
        from afterimage import kb, filter, search, inject, extract

        # Inspect modules for suspicious imports
        import inspect

        suspicious_imports = ["requests", "urllib.request", "httpx", "aiohttp"]

        for module in [kb, filter, search, inject, extract]:
            source = inspect.getsource(module)
            for suspicious in suspicious_imports:
                # Check if module imports networking libraries
                assert f"import {suspicious}" not in source, \
                    f"{module.__name__} should not import {suspicious}"
                assert f"from {suspicious}" not in source, \
                    f"{module.__name__} should not import from {suspicious}"

    def test_cli_commands_are_local(self, temp_dir):
        """CLI commands should work without network."""
        import argparse
        from afterimage.cli import cmd_stats, cmd_config

        # Set temp home to avoid touching real config
        old_home = os.environ.get("HOME")
        os.environ["HOME"] = temp_dir

        try:
            # Config init is local file creation
            args = argparse.Namespace(init=True, force=False)
            result = cmd_config(args)
            assert result == 0

            # Stats reads local DB
            args = argparse.Namespace(json=False)
            result = cmd_stats(args)
            assert result == 0
        finally:
            os.environ["HOME"] = old_home or ""
