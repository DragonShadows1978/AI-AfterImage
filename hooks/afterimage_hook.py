#!/usr/bin/env python3
"""
AI-AfterImage Hook for Claude Code.

Provides episodic code memory through Claude Code's hook system:
1. Pre-Write/Edit: DENY first attempt with similar code context (Claude sees this!)
2. Pre-Write/Edit: ALLOW retry attempts (after Claude has seen the context)
3. Post-Write/Edit: Store the code change in KB for future recall

The deny-then-allow pattern ensures Claude sees relevant past code BEFORE writing.
"""

import json
import sys
import os
import re
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

# Configurable path - users should update this
AFTERIMAGE_PATH = os.environ.get(
    "AFTERIMAGE_PATH",
    str(Path.home() / "AI-AfterImage")  # Default install location
)

# Try local install first, then fall back to common locations
SEARCH_PATHS = [
    AFTERIMAGE_PATH,
    str(Path.home() / "mini-mind-v2" / "workspace" / "AI-AfterImage"),  # Dev location
    str(Path.home() / ".local" / "lib" / "afterimage"),
    "/usr/local/lib/afterimage",
]

AFTERIMAGE_AVAILABLE = False
for path in SEARCH_PATHS:
    if Path(path).exists():
        sys.path.insert(0, path)
        try:
            from afterimage.kb import KnowledgeBase
            from afterimage.filter import CodeFilter
            from afterimage.search import HybridSearch
            AFTERIMAGE_AVAILABLE = True
            break
        except ImportError:
            continue

# Track which operations we've already shown context for (deny once, allow after)
# Uses file content hash to identify unique write attempts
SEEN_WRITES_FILE = Path.home() / ".afterimage" / ".seen_writes"


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


def search_similar_code(file_path: str, content: str) -> Optional[str]:
    """Search KB for similar code and format injection context."""
    if not AFTERIMAGE_AVAILABLE:
        return None

    try:
        code_filter = CodeFilter()
        if not code_filter.is_code(file_path, content):
            return None

        query = extract_search_terms(content, file_path)
        if not query.strip():
            return None

        search = HybridSearch()
        results = search.search(query, limit=5, threshold=0.01)

        if not results:
            return None

        # Format the injection - this is what Claude will see!
        lines = [
            "",
            "=" * 60,
            "📚 AFTERIMAGE: You've written similar code before!",
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


def store_code(file_path: str, new_code: str, old_code: Optional[str] = None):
    """Store code in KB for future recall."""
    if not AFTERIMAGE_AVAILABLE:
        return False

    try:
        code_filter = CodeFilter()
        if not code_filter.is_code(file_path, new_code):
            return False

        kb = KnowledgeBase()
        session_id = get_session_id()

        # Try to generate embedding (optional)
        embedding = None
        try:
            from afterimage.embeddings import EmbeddingGenerator
            embedder = EmbeddingGenerator()
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
            if content and store_code(file_path, content):
                print(f"[AfterImage] Stored: {Path(file_path).name}", file=sys.stderr)

        elif tool_name == "Edit":
            old_string = tool_input.get("old_string", "")
            new_string = tool_input.get("new_string", "")
            if new_string and store_code(file_path, new_string, old_string):
                print(f"[AfterImage] Stored: {Path(file_path).name}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
