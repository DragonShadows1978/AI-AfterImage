"""
Transcript Extractor: Parses Claude Code JSONL transcripts.

Extracts Write/Edit tool calls along with surrounding context
for storage in the knowledge base.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass


@dataclass
class CodeChange:
    """Represents a code change extracted from a transcript."""
    file_path: str
    new_code: str
    old_code: Optional[str]  # For Edit operations
    tool_type: str  # "Write" or "Edit"
    context: str  # Surrounding conversation
    timestamp: str
    session_id: str
    raw_entry: Dict[str, Any]  # Original transcript entry


def get_transcripts_dir() -> Path:
    """Get the Claude Code transcripts directory."""
    # Default location for Claude Code transcripts
    return Path.home() / ".claude" / "projects"


def find_transcript_files(
    base_dir: Optional[Path] = None,
    since: Optional[datetime] = None
) -> List[Path]:
    """
    Find all JSONL transcript files.

    Args:
        base_dir: Directory to search (defaults to ~/.claude/projects)
        since: Only return files modified after this time

    Returns:
        List of paths to transcript files
    """
    base_dir = base_dir or get_transcripts_dir()

    if not base_dir.exists():
        return []

    transcript_files = []

    # Claude Code stores transcripts in project-specific directories
    for jsonl_file in base_dir.rglob("*.jsonl"):
        # Skip non-transcript files
        if "transcript" not in jsonl_file.name.lower() and \
           not jsonl_file.name.endswith(".jsonl"):
            continue

        if since is not None:
            mtime = datetime.fromtimestamp(jsonl_file.stat().st_mtime)
            if mtime < since:
                continue

        transcript_files.append(jsonl_file)

    return sorted(transcript_files, key=lambda p: p.stat().st_mtime)


class TranscriptExtractor:
    """
    Extracts code changes from Claude Code transcripts.

    Parses JSONL files and extracts Write/Edit tool calls
    along with surrounding conversation context.
    """

    def __init__(self, context_lines: int = 5):
        """
        Initialize the extractor.

        Args:
            context_lines: Number of conversation entries to capture
                          before and after each code change
        """
        self.context_lines = context_lines

    def extract_from_file(self, file_path: Path) -> List[CodeChange]:
        """
        Extract all code changes from a transcript file.

        Args:
            file_path: Path to the JSONL transcript file

        Returns:
            List of CodeChange objects
        """
        entries = self._load_jsonl(file_path)
        if not entries:
            return []

        # Derive session ID from file path or first entry
        session_id = self._get_session_id(file_path, entries)

        changes = []
        for i, entry in enumerate(entries):
            code_change = self._extract_code_change(
                entry, entries, i, session_id
            )
            if code_change:
                changes.append(code_change)

        return changes

    def extract_from_directory(
        self,
        directory: Optional[Path] = None,
        since: Optional[datetime] = None
    ) -> Iterator[CodeChange]:
        """
        Extract code changes from all transcripts in a directory.

        Args:
            directory: Directory to search (defaults to Claude Code transcripts)
            since: Only process files modified after this time

        Yields:
            CodeChange objects
        """
        transcript_files = find_transcript_files(directory, since)

        for file_path in transcript_files:
            try:
                changes = self.extract_from_file(file_path)
                yield from changes
            except Exception as e:
                # Log but continue with other files
                print(f"Error processing {file_path}: {e}")
                continue

    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load entries from a JSONL file."""
        entries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            return []
        return entries

    def _get_session_id(
        self,
        file_path: Path,
        entries: List[Dict[str, Any]]
    ) -> str:
        """Get or generate a session ID."""
        # Try to get from first entry
        if entries:
            first = entries[0]
            if "session_id" in first:
                return first["session_id"]
            if "sessionId" in first:
                return first["sessionId"]

        # Fall back to file name - ensure file_path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        return file_path.stem

    def _extract_code_change(
        self,
        entry: Dict[str, Any],
        all_entries: List[Dict[str, Any]],
        index: int,
        session_id: str
    ) -> Optional[CodeChange]:
        """
        Extract a code change from a transcript entry if applicable.

        Returns None if entry is not a Write/Edit tool call.
        """
        # Check if this is a tool use entry
        tool_info = self._get_tool_info(entry)
        if not tool_info:
            return None

        tool_name, tool_input = tool_info

        # Only process Write and Edit tools
        if tool_name not in ("Write", "Edit"):
            return None

        # Extract code change details based on tool type
        if tool_name == "Write":
            return self._extract_write_change(
                entry, tool_input, all_entries, index, session_id
            )
        else:  # Edit
            return self._extract_edit_change(
                entry, tool_input, all_entries, index, session_id
            )

    def _get_tool_info(
        self,
        entry: Dict[str, Any]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Extract tool name and input from various transcript formats.

        Returns (tool_name, tool_input) or None if not a tool use.
        """
        # Format 1: {"type": "tool_use", "name": "...", "input": {...}}
        if entry.get("type") == "tool_use":
            return entry.get("name"), entry.get("input", {})

        # Format 2: {"tool": "...", "input": {...}} or {"tool": "...", ...}
        if "tool" in entry:
            tool_name = entry["tool"]
            tool_input = entry.get("input", entry)
            return tool_name, tool_input

        # Format 3: Nested in content array
        if "content" in entry and isinstance(entry["content"], list):
            for item in entry["content"]:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    return item.get("name"), item.get("input", {})

        # Format 3b: Claude Code format - nested in message.content
        if "message" in entry and isinstance(entry["message"], dict):
            message = entry["message"]
            if "content" in message and isinstance(message["content"], list):
                for item in message["content"]:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        return item.get("name"), item.get("input", {})

        # Format 4: {"toolName": "...", "toolInput": {...}}
        if "toolName" in entry:
            return entry["toolName"], entry.get("toolInput", {})

        return None

    def _extract_write_change(
        self,
        entry: Dict[str, Any],
        tool_input: Dict[str, Any],
        all_entries: List[Dict[str, Any]],
        index: int,
        session_id: str
    ) -> Optional[CodeChange]:
        """Extract a Write tool change."""
        file_path = tool_input.get("file_path") or tool_input.get("path")
        content = tool_input.get("content")

        if not file_path or not content:
            return None

        context = self._get_context(all_entries, index)
        timestamp = self._get_timestamp(entry)

        return CodeChange(
            file_path=file_path,
            new_code=content,
            old_code=None,
            tool_type="Write",
            context=context,
            timestamp=timestamp,
            session_id=session_id,
            raw_entry=entry
        )

    def _extract_edit_change(
        self,
        entry: Dict[str, Any],
        tool_input: Dict[str, Any],
        all_entries: List[Dict[str, Any]],
        index: int,
        session_id: str
    ) -> Optional[CodeChange]:
        """Extract an Edit tool change."""
        file_path = tool_input.get("file_path") or tool_input.get("path")
        old_string = tool_input.get("old_string") or tool_input.get("old")
        new_string = tool_input.get("new_string") or tool_input.get("new")

        if not file_path or new_string is None:
            return None

        context = self._get_context(all_entries, index)
        timestamp = self._get_timestamp(entry)

        return CodeChange(
            file_path=file_path,
            new_code=new_string,
            old_code=old_string,
            tool_type="Edit",
            context=context,
            timestamp=timestamp,
            session_id=session_id,
            raw_entry=entry
        )

    def _get_context(
        self,
        all_entries: List[Dict[str, Any]],
        index: int
    ) -> str:
        """
        Get conversation context around a code change.

        Returns a string summarizing what was being discussed.
        """
        context_parts = []

        # Get entries before
        start = max(0, index - self.context_lines)
        for i in range(start, index):
            text = self._extract_text(all_entries[i])
            if text:
                context_parts.append(text)

        # Get entries after (up to context_lines, stopping at next tool use)
        end = min(len(all_entries), index + self.context_lines + 1)
        for i in range(index + 1, end):
            # Stop if we hit another tool use
            if self._get_tool_info(all_entries[i]):
                break
            text = self._extract_text(all_entries[i])
            if text:
                context_parts.append(text)

        # Combine and truncate
        context = "\n".join(context_parts)

        # Limit context length
        max_context_chars = 1000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "..."

        return context

    def _extract_text(self, entry: Dict[str, Any]) -> Optional[str]:
        """Extract human-readable text from an entry."""
        # Direct text content
        if "text" in entry:
            return entry["text"]

        # Role-based message
        if entry.get("role") in ("user", "assistant", "human"):
            content = entry.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for item in content:
                    if isinstance(item, str):
                        texts.append(item)
                    elif isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                return " ".join(texts)

        # Message content
        if "message" in entry:
            return str(entry["message"])

        return None

    def _get_timestamp(self, entry: Dict[str, Any]) -> str:
        """Get timestamp from entry or generate current time."""
        # Try various timestamp fields
        for field in ("timestamp", "created_at", "time", "ts"):
            if field in entry:
                ts = entry[field]
                if isinstance(ts, str):
                    return ts
                if isinstance(ts, (int, float)):
                    return datetime.fromtimestamp(ts).isoformat()

        # Fall back to current time
        return datetime.now(timezone.utc).isoformat()


def extract_code_symbols(code: str) -> List[str]:
    """
    Extract function and class names from code.

    Useful for creating searchable keywords.
    """
    symbols = []

    # Python functions and methods
    for match in re.finditer(r'\bdef\s+(\w+)\s*\(', code):
        symbols.append(match.group(1))

    # Python classes
    for match in re.finditer(r'\bclass\s+(\w+)', code):
        symbols.append(match.group(1))

    # JavaScript/TypeScript functions
    for match in re.finditer(r'\bfunction\s+(\w+)\s*\(', code):
        symbols.append(match.group(1))

    # Arrow function assignments: const foo = () =>
    for match in re.finditer(r'\b(?:const|let|var)\s+(\w+)\s*=\s*(?:\([^)]*\)|[^=])\s*=>', code):
        symbols.append(match.group(1))

    # Method definitions in classes
    for match in re.finditer(r'^\s*(\w+)\s*\([^)]*\)\s*{', code, re.MULTILINE):
        symbols.append(match.group(1))

    # Rust functions
    for match in re.finditer(r'\bfn\s+(\w+)\s*[<(]', code):
        symbols.append(match.group(1))

    # Go functions
    for match in re.finditer(r'\bfunc\s+(?:\([^)]*\)\s*)?(\w+)\s*\(', code):
        symbols.append(match.group(1))

    return list(set(symbols))  # Deduplicate
