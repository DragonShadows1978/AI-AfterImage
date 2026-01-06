#!/usr/bin/env python3
"""
AI-AfterImage Hook for Claude Code.

This hook integrates with Claude Code's hook system to:
1. Pre-Write/Edit: Search KB for similar code and inject context
2. Post-Write/Edit: Store the code change in KB for future recall

Hook Configuration:
    ~/.claude/hooks/afterimage.json:
    {
        "name": "afterimage",
        "description": "Episodic memory for code",
        "pre_tool": ["Write", "Edit"],
        "post_tool": ["Write", "Edit"]
    }

This script is invoked by Claude Code with specific arguments
based on the hook type and tool being executed.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# Add parent directory to path for imports when run as script
if __name__ == "__main__":
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))

from afterimage.kb import KnowledgeBase
from afterimage.filter import CodeFilter
from afterimage.search import HybridSearch
from afterimage.inject import ContextInjector


def get_session_id() -> str:
    """Get or create session ID for this Claude Code session."""
    # Try to get from environment (Claude Code may set this)
    session_id = os.environ.get("CLAUDE_SESSION_ID")
    if session_id:
        return session_id

    # Fall back to a generated ID based on current time
    return f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def pre_write_hook(
    file_path: str,
    content: str,
    context: Optional[str] = None
) -> Optional[str]:
    """
    Pre-write hook: Search for similar code and return injection context.

    Args:
        file_path: Path to file being written
        content: Content about to be written
        context: Optional conversation context

    Returns:
        Injection context string if similar code found, None otherwise
    """
    # Filter: only process code files
    code_filter = CodeFilter()
    if not code_filter.is_code(file_path, content):
        return None

    # Search for similar code
    search = HybridSearch()

    # Try semantic search on the content
    results = search.search_by_code(
        code=content,
        file_path=file_path,
        limit=3,
        threshold=0.5  # Higher threshold for pre-write
    )

    if not results:
        # Try path-based search as fallback
        path_parts = Path(file_path).parts[-2:]
        if path_parts:
            results = search.search_by_path("/".join(path_parts), limit=2)

    if not results:
        return None

    # Format injection
    injector = ContextInjector()
    injection = injector.format_for_hook(
        results=results,
        file_path=file_path,
        tool_type="Write"
    )

    return injection


def pre_edit_hook(
    file_path: str,
    old_string: str,
    new_string: str,
    context: Optional[str] = None
) -> Optional[str]:
    """
    Pre-edit hook: Search for similar edits and return injection context.

    Args:
        file_path: Path to file being edited
        old_string: String being replaced
        new_string: Replacement string
        context: Optional conversation context

    Returns:
        Injection context string if similar code found, None otherwise
    """
    # Filter: only process code files
    code_filter = CodeFilter()
    if not code_filter.is_code(file_path, new_string):
        return None

    # Search for similar code edits
    search = HybridSearch()

    # Search based on new content
    results = search.search_by_code(
        code=new_string,
        file_path=file_path,
        limit=3,
        threshold=0.5
    )

    if not results:
        return None

    # Format injection
    injector = ContextInjector()
    injection = injector.format_for_hook(
        results=results,
        file_path=file_path,
        tool_type="Edit"
    )

    return injection


def post_write_hook(
    file_path: str,
    content: str,
    context: Optional[str] = None
) -> bool:
    """
    Post-write hook: Store the written code in KB.

    Args:
        file_path: Path to file that was written
        content: Content that was written
        context: Optional conversation context

    Returns:
        True if stored successfully
    """
    # Filter: only store code files
    code_filter = CodeFilter()
    if not code_filter.is_code(file_path, content):
        return False

    kb = KnowledgeBase()
    session_id = get_session_id()

    # Generate embedding (if available)
    embedding = None
    try:
        from afterimage.embeddings import EmbeddingGenerator
        embedder = EmbeddingGenerator()
        embedding = embedder.embed_code(content, file_path, context)
    except ImportError:
        pass  # Embeddings not available
    except Exception:
        pass  # Failed to generate embedding

    # Store in KB
    try:
        kb.store(
            file_path=file_path,
            new_code=content,
            old_code=None,
            context=context,
            session_id=session_id,
            embedding=embedding
        )
        return True
    except Exception as e:
        # Log error but don't block the write
        print(f"AfterImage: Failed to store code: {e}", file=sys.stderr)
        return False


def post_edit_hook(
    file_path: str,
    old_string: str,
    new_string: str,
    context: Optional[str] = None
) -> bool:
    """
    Post-edit hook: Store the code edit in KB.

    Args:
        file_path: Path to file that was edited
        old_string: String that was replaced
        new_string: Replacement string
        context: Optional conversation context

    Returns:
        True if stored successfully
    """
    # Filter: only store code files
    code_filter = CodeFilter()
    if not code_filter.is_code(file_path, new_string):
        return False

    kb = KnowledgeBase()
    session_id = get_session_id()

    # Generate embedding
    embedding = None
    try:
        from afterimage.embeddings import EmbeddingGenerator
        embedder = EmbeddingGenerator()
        embedding = embedder.embed_code(new_string, file_path, context)
    except ImportError:
        pass
    except Exception:
        pass

    # Store in KB
    try:
        kb.store(
            file_path=file_path,
            new_code=new_string,
            old_code=old_string,
            context=context,
            session_id=session_id,
            embedding=embedding
        )
        return True
    except Exception as e:
        print(f"AfterImage: Failed to store edit: {e}", file=sys.stderr)
        return False


def handle_hook(hook_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main hook handler that processes Claude Code hook invocations.

    Args:
        hook_data: Dictionary with hook type, tool, and parameters

    Returns:
        Response dictionary with any injection content
    """
    hook_type = hook_data.get("type")  # "pre" or "post"
    tool_name = hook_data.get("tool")  # "Write" or "Edit"
    tool_input = hook_data.get("input", {})
    context = hook_data.get("context")

    response = {"success": True, "injection": None}

    try:
        if tool_name == "Write":
            file_path = tool_input.get("file_path", "")
            content = tool_input.get("content", "")

            if hook_type == "pre":
                injection = pre_write_hook(file_path, content, context)
                if injection:
                    response["injection"] = injection
            elif hook_type == "post":
                post_write_hook(file_path, content, context)

        elif tool_name == "Edit":
            file_path = tool_input.get("file_path", "")
            old_string = tool_input.get("old_string", "")
            new_string = tool_input.get("new_string", "")

            if hook_type == "pre":
                injection = pre_edit_hook(file_path, old_string, new_string, context)
                if injection:
                    response["injection"] = injection
            elif hook_type == "post":
                post_edit_hook(file_path, old_string, new_string, context)

    except Exception as e:
        response["success"] = False
        response["error"] = str(e)

    return response


def main():
    """
    CLI entry point for the hook script.

    Claude Code invokes this script with JSON input on stdin.
    """
    parser = argparse.ArgumentParser(
        description="AI-AfterImage hook for Claude Code"
    )
    parser.add_argument(
        "--type", choices=["pre", "post"],
        help="Hook type (pre or post tool execution)"
    )
    parser.add_argument(
        "--tool", choices=["Write", "Edit"],
        help="Tool being executed"
    )
    parser.add_argument(
        "--input", help="Tool input as JSON"
    )
    parser.add_argument(
        "--context", help="Conversation context"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Read full hook data from stdin as JSON"
    )

    args = parser.parse_args()

    # Get hook data
    if args.json:
        # Read from stdin
        hook_data = json.load(sys.stdin)
    else:
        # Build from arguments
        tool_input = {}
        if args.input:
            tool_input = json.loads(args.input)

        hook_data = {
            "type": args.type,
            "tool": args.tool,
            "input": tool_input,
            "context": args.context
        }

    # Handle the hook
    response = handle_hook(hook_data)

    # Output response
    print(json.dumps(response))
    return 0 if response["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
