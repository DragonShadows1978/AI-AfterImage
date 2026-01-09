#!/usr/bin/env python3
"""
AI-AfterImage Hook for Claude Code v0.2.0

Provides episodic code memory through Claude Code's hook system:
1. Pre-Write/Edit: DENY first attempt with similar code context (Claude sees this!)
2. Pre-Write/Edit: ALLOW retry attempts (after Claude has seen the context)
3. Post-Write/Edit: Store the code change in KB for future recall

v0.2.0 Changes:
- Config-based backend loading (PostgreSQL or SQLite)
- Graceful fallback from PostgreSQL to SQLite
- Connection pool reuse across hook invocations
- Environment variable support for AFTERIMAGE_PG_PASSWORD

The deny-then-allow pattern ensures Claude sees relevant past code BEFORE writing.
"""

import json
import sys
import os
import re
import hashlib
import atexit
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

# Configurable path - users should update this or set AFTERIMAGE_PATH env var
AFTERIMAGE_PATH = os.environ.get(
    "AFTERIMAGE_PATH",
    str(Path.home() / "AI-AfterImage")  # Default install location
)

# Search paths for the afterimage package (in priority order)
SEARCH_PATHS = [
    AFTERIMAGE_PATH,
    str(Path.home() / "Shared" / "AI-AfterImage"),  # Shared location
    str(Path.home() / "mini-mind-v2" / "workspace" / "afterimage"),  # Dev workspace
    str(Path.home() / "mini-mind-v2" / "workspace" / "AI-AfterImage"),  # Alt dev location
    str(Path.home() / ".local" / "lib" / "afterimage"),
    "/usr/local/lib/afterimage",
]

# Track which operations we've already shown context for (deny once, allow after)
# Uses file content hash to identify unique write attempts
SEEN_WRITES_FILE = Path.home() / ".afterimage" / ".seen_writes"

# =============================================================================
# MODULE-LEVEL CACHING FOR CONNECTION POOL REUSE
# =============================================================================

# Cached backend instance - reused across hook invocations to avoid
# creating new connection pools on every call
_cached_backend = None
_cached_backend_type = None  # "postgresql" or "sqlite"

# Cached module imports
_kb_class = None
_code_filter_class = None
_hybrid_search_class = None
_embedding_generator_class = None
_config_module = None

AFTERIMAGE_AVAILABLE = False

# =============================================================================
# PACKAGE LOADING
# =============================================================================

def _load_afterimage():
    """
    Load the afterimage package from search paths.
    Returns True if loaded successfully.
    """
    global AFTERIMAGE_AVAILABLE, _kb_class, _code_filter_class
    global _hybrid_search_class, _embedding_generator_class, _config_module

    if AFTERIMAGE_AVAILABLE:
        return True

    for path in SEARCH_PATHS:
        if Path(path).exists():
            if path not in sys.path:
                sys.path.insert(0, path)
            try:
                from afterimage.kb import KnowledgeBase
                from afterimage.filter import CodeFilter
                from afterimage.search import HybridSearch
                from afterimage import config as afterimage_config

                _kb_class = KnowledgeBase
                _code_filter_class = CodeFilter
                _hybrid_search_class = HybridSearch
                _config_module = afterimage_config

                # Optional: embeddings
                try:
                    from afterimage.embeddings import EmbeddingGenerator
                    _embedding_generator_class = EmbeddingGenerator
                except ImportError:
                    _embedding_generator_class = None

                AFTERIMAGE_AVAILABLE = True
                return True
            except ImportError:
                continue

    return False


# Attempt to load on module import
_load_afterimage()


# =============================================================================
# BACKEND MANAGEMENT WITH FALLBACK
# =============================================================================

def get_backend():
    """
    Get or create the storage backend with caching and fallback.

    Priority:
    1. Return cached backend if available
    2. Try PostgreSQL if configured
    3. Fall back to SQLite if PostgreSQL fails

    The backend is cached at module level to reuse connection pools
    across hook invocations within the same process.
    """
    global _cached_backend, _cached_backend_type

    if _cached_backend is not None:
        return _cached_backend, _cached_backend_type

    if not AFTERIMAGE_AVAILABLE or _config_module is None:
        return None, None

    try:
        # Load configuration
        config = _config_module.load_config()

        # Check environment variable override for backend
        env_backend = os.environ.get("AFTERIMAGE_BACKEND")
        if env_backend:
            config.backend = env_backend

        # Check for PostgreSQL password in environment
        if os.environ.get("AFTERIMAGE_PG_PASSWORD"):
            config.postgresql.password = os.environ["AFTERIMAGE_PG_PASSWORD"]

        # Try PostgreSQL if configured
        if config.backend == "postgresql":
            try:
                backend = _create_postgresql_backend(config)
                if backend is not None:
                    _cached_backend = backend
                    _cached_backend_type = "postgresql"
                    return _cached_backend, _cached_backend_type
            except Exception as e:
                print(f"[AfterImage] PostgreSQL unavailable ({e}), falling back to SQLite",
                      file=sys.stderr)

        # Fall back to SQLite
        backend = _create_sqlite_backend(config)
        _cached_backend = backend
        _cached_backend_type = "sqlite"
        return _cached_backend, _cached_backend_type

    except Exception as e:
        print(f"[AfterImage] Backend initialization failed: {e}", file=sys.stderr)
        return None, None


def _create_postgresql_backend(config):
    """Create and initialize PostgreSQL backend."""
    try:
        # Import psycopg to check availability
        import psycopg
    except ImportError:
        raise ImportError("psycopg not installed")

    from afterimage.storage import SyncPostgreSQLBackend

    # Check for password
    password = config.postgresql.password
    if not password:
        password = os.environ.get("AFTERIMAGE_PG_PASSWORD")

    if not password and not config.postgresql.connection_string:
        raise ValueError("PostgreSQL password not configured")

    backend = SyncPostgreSQLBackend(
        host=config.postgresql.host,
        port=config.postgresql.port,
        database=config.postgresql.database,
        user=config.postgresql.user,
        password=password,
        connection_string=config.postgresql.connection_string,
        embedding_dim=config.embeddings.embedding_dim
    )

    # Test connection by initializing
    backend.initialize()

    return backend


def _create_sqlite_backend(config):
    """Create and initialize SQLite backend."""
    from afterimage.storage import SQLiteBackend

    # Ensure directory exists
    config.sqlite.path.parent.mkdir(parents=True, exist_ok=True)

    backend = SQLiteBackend(db_path=config.sqlite.path)
    backend.initialize()

    return backend


def cleanup_backend():
    """Cleanup backend on process exit."""
    global _cached_backend
    if _cached_backend is not None:
        try:
            _cached_backend.close()
        except:
            pass
        _cached_backend = None


# Register cleanup handler
atexit.register(cleanup_backend)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_content_hash(file_path: str, content: str) -> str:
    """Generate hash to identify unique write attempts."""
    key = f"{file_path}:{content[:500]}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def was_already_shown(content_hash: str) -> bool:
    """Check if we already showed context for this write attempt."""
    if not SEEN_WRITES_FILE.exists():
        return False
    try:
        seen = SEEN_WRITES_FILE.read_text().strip().split("\n")
        # Keep only recent entries (last 100)
        return content_hash in seen[-100:]
    except:
        return False


def mark_as_shown(content_hash: str):
    """Mark this write attempt as having been shown context."""
    SEEN_WRITES_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = []
        if SEEN_WRITES_FILE.exists():
            existing = SEEN_WRITES_FILE.read_text().strip().split("\n")[-99:]
        existing.append(content_hash)
        SEEN_WRITES_FILE.write_text("\n".join(existing))
    except:
        pass


def get_session_id() -> str:
    """Get session ID from environment or generate one."""
    return os.environ.get(
        "CLAUDE_SESSION_ID",
        f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )


def extract_search_terms(content: str, file_path: str) -> str:
    """Extract meaningful search terms from code content."""
    terms = []

    # Get imports
    for m in re.finditer(r'from\s+(\w+)|import\s+(\w+)', content):
        terms.append(m.group(1) or m.group(2))

    # Get function/class names
    for m in re.finditer(r'def\s+(\w+)|class\s+(\w+)', content):
        terms.append(m.group(1) or m.group(2))

    # Get decorators (often indicate patterns like @app.route)
    for m in re.finditer(r'@(\w+)', content):
        terms.append(m.group(1))

    # Add file stem
    terms.append(Path(file_path).stem)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in terms:
        if t not in seen and len(t) > 2:
            seen.add(t)
            unique.append(t)

    return " ".join(unique[:8])


# =============================================================================
# SEARCH AND STORE OPERATIONS
# =============================================================================

def search_similar_code(file_path: str, content: str) -> Optional[str]:
    """Search KB for similar code and format injection context."""
    if not AFTERIMAGE_AVAILABLE:
        return None

    try:
        # Use code filter to check if this is code
        code_filter = _code_filter_class()
        if not code_filter.is_code(file_path, content):
            return None

        query = extract_search_terms(content, file_path)
        if not query.strip():
            return None

        # Get backend for search
        backend, backend_type = get_backend()
        if backend is None:
            return None

        # Create search instance with cached backend
        search = _hybrid_search_class(backend=backend)
        results = search.search(query, limit=5, threshold=0.01)

        if not results:
            return None

        # Format the injection - this is what Claude will see!
        lines = [
            "",
            "=" * 60,
            f"📚 AFTERIMAGE [{backend_type.upper()}]: You've written similar code before!",
            "=" * 60,
            "",
            "Review these patterns before proceeding:",
            ""
        ]

        seen_paths = set()
        for r in results:
            short_path = "/".join(Path(r.file_path).parts[-3:])
            if short_path in seen_paths:
                continue
            seen_paths.add(short_path)

            if len(seen_paths) > 3:
                break

            code_preview = r.new_code[:400]
            lines.append(f"**From:** `{short_path}`")
            lines.append("```")
            lines.append(code_preview)
            if len(r.new_code) > 400:
                lines.append("... (truncated)")
            lines.append("```")
            lines.append("")

        lines.extend([
            "Consider these patterns. Retry your write now.",
            "=" * 60,
            ""
        ])

        return "\n".join(lines)

    except Exception as e:
        print(f"[AfterImage] Search error: {e}", file=sys.stderr)
        return None


def store_code(file_path: str, new_code: str, old_code: Optional[str] = None) -> bool:
    """Store code in KB for future recall."""
    if not AFTERIMAGE_AVAILABLE:
        return False

    try:
        # Use code filter
        code_filter = _code_filter_class()
        if not code_filter.is_code(file_path, new_code):
            return False

        # Get backend
        backend, backend_type = get_backend()
        if backend is None:
            return False

        # Create KB with cached backend
        kb = _kb_class(backend=backend)
        session_id = get_session_id()

        # Try to generate embedding (optional)
        embedding = None
        if _embedding_generator_class is not None:
            try:
                embedder = _embedding_generator_class()
                embedding = embedder.embed_code(new_code, file_path)
            except:
                pass

        kb.store(
            file_path=file_path,
            new_code=new_code,
            old_code=old_code,
            context="",
            session_id=session_id,
            embedding=embedding
        )
        return True

    except Exception as e:
        print(f"[AfterImage] Store error: {e}", file=sys.stderr)
        return False


# =============================================================================
# MAIN HOOK HANDLER
# =============================================================================

def main():
    """Process Claude Code hook."""
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        sys.exit(0)

    hook_event = input_data.get("hook_event_name", "")
    tool_name = input_data.get("tool_name", "")
    tool_input = input_data.get("tool_input", {})

    # Only process Write and Edit
    if tool_name not in ("Write", "Edit"):
        sys.exit(0)

    file_path = tool_input.get("file_path", "")
    if not file_path:
        sys.exit(0)

    # =========================================================================
    # PRE-TOOL: Search for similar code and inject via DENY (first time only)
    # =========================================================================
    if hook_event == "PreToolUse":
        content = tool_input.get("content", "") or tool_input.get("new_string", "")

        if content and AFTERIMAGE_AVAILABLE:
            content_hash = get_content_hash(file_path, content)

            # First attempt: DENY with context (Claude sees this!)
            if not was_already_shown(content_hash):
                injection = search_similar_code(file_path, content)

                if injection:
                    mark_as_shown(content_hash)

                    output = {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": injection
                        }
                    }
                    print(json.dumps(output))
                    sys.exit(0)

            # Already shown or no results: allow the write
            # (Claude will retry after seeing context)

    # =========================================================================
    # POST-TOOL: Store the code for future recall
    # =========================================================================
    elif hook_event == "PostToolUse":
        if tool_name == "Write":
            content = tool_input.get("content", "")
            if content:
                stored = store_code(file_path, content)
                if stored:
                    _, backend_type = get_backend()
                    print(f"[AfterImage] Stored ({backend_type}): {Path(file_path).name}",
                          file=sys.stderr)

        elif tool_name == "Edit":
            old_string = tool_input.get("old_string", "")
            new_string = tool_input.get("new_string", "")
            if new_string:
                stored = store_code(file_path, new_string, old_string)
                if stored:
                    _, backend_type = get_backend()
                    print(f"[AfterImage] Stored ({backend_type}): {Path(file_path).name}",
                          file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
