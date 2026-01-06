"""
CLI: Command-line interface for AI-AfterImage.

Provides commands for searching, ingesting transcripts, and managing
the knowledge base.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .kb import KnowledgeBase
from .search import HybridSearch, SearchResult
from .extract import TranscriptExtractor, find_transcript_files
from .filter import CodeFilter
from .inject import ContextInjector


def cmd_search(args):
    """Search the knowledge base."""
    search = HybridSearch()

    results = search.search(
        query=args.query,
        limit=args.limit,
        threshold=args.threshold,
        path_filter=args.path
    )

    if not results:
        print("No results found.")
        return 1

    if args.json:
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2))
    else:
        injector = ContextInjector()
        for i, result in enumerate(results, 1):
            print(f"\n{'='*60}")
            print(injector.format_single(result))

        print(f"\n{'='*60}")
        print(f"Found {len(results)} result(s)")

    return 0


def cmd_ingest(args):
    """Ingest transcripts into the knowledge base."""
    kb = KnowledgeBase()
    extractor = TranscriptExtractor()
    code_filter = CodeFilter()

    # Determine source
    if args.file:
        files = [Path(args.file)]
    elif args.directory:
        files = find_transcript_files(Path(args.directory))
    else:
        # Default: Claude Code transcripts
        files = find_transcript_files()

    if not files:
        print("No transcript files found.")
        return 1

    print(f"Found {len(files)} transcript file(s)")

    # Track stats
    total_changes = 0
    code_changes = 0
    stored = 0

    # Import embedder only if needed
    embedder = None
    if not args.no_embeddings:
        try:
            from .embeddings import EmbeddingGenerator
            embedder = EmbeddingGenerator()
            print("Embedding model loaded.")
        except ImportError:
            print("Warning: sentence-transformers not installed. Skipping embeddings.")

    for file_path in files:
        if args.verbose:
            print(f"\nProcessing: {file_path}")

        try:
            changes = extractor.extract_from_file(file_path)
            total_changes += len(changes)

            for change in changes:
                # Filter for code files only
                if not code_filter.is_code(change.file_path, change.new_code):
                    if args.verbose:
                        print(f"  Skipped (not code): {change.file_path}")
                    continue

                code_changes += 1

                # Generate embedding
                embedding = None
                if embedder:
                    try:
                        embedding = embedder.embed_code(
                            change.new_code,
                            change.file_path,
                            change.context
                        )
                    except Exception as e:
                        if args.verbose:
                            print(f"  Warning: Failed to generate embedding: {e}")

                # Store in KB
                try:
                    entry_id = kb.store(
                        file_path=change.file_path,
                        new_code=change.new_code,
                        old_code=change.old_code,
                        context=change.context,
                        session_id=change.session_id,
                        embedding=embedding,
                        timestamp=change.timestamp
                    )
                    stored += 1
                    if args.verbose:
                        print(f"  Stored: {change.file_path} ({entry_id[:8]}...)")
                except Exception as e:
                    if args.verbose:
                        print(f"  Error storing {change.file_path}: {e}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    print(f"\nIngestion complete:")
    print(f"  Total changes found: {total_changes}")
    print(f"  Code changes: {code_changes}")
    print(f"  Stored in KB: {stored}")

    return 0


def cmd_stats(args):
    """Show knowledge base statistics."""
    kb = KnowledgeBase()
    stats = kb.stats()

    if args.json:
        print(json.dumps(stats, indent=2))
    else:
        print("\nAI-AfterImage Knowledge Base Statistics")
        print("=" * 40)
        print(f"Total entries:        {stats['total_entries']}")
        print(f"With embeddings:      {stats['entries_with_embeddings']}")
        print(f"Unique files:         {stats['unique_files']}")
        print(f"Unique sessions:      {stats['unique_sessions']}")
        print(f"Database size:        {_format_bytes(stats['db_size_bytes'])}")

        if stats['oldest_entry']:
            print(f"\nDate range:")
            print(f"  Oldest: {stats['oldest_entry'][:19]}")
            print(f"  Newest: {stats['newest_entry'][:19]}")

    return 0


def cmd_export(args):
    """Export knowledge base to JSON."""
    kb = KnowledgeBase()
    entries = kb.export()

    output = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "count": len(entries),
        "entries": entries
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Exported {len(entries)} entries to {args.output}")
    else:
        print(json.dumps(output, indent=2))

    return 0


def cmd_clear(args):
    """Clear the knowledge base."""
    if not args.yes:
        confirm = input("Are you sure you want to clear the knowledge base? [y/N] ")
        if confirm.lower() != "y":
            print("Aborted.")
            return 1

    kb = KnowledgeBase()
    count = kb.clear()
    print(f"Cleared {count} entries from the knowledge base.")
    return 0


def cmd_recent(args):
    """Show recent entries."""
    kb = KnowledgeBase()
    entries = kb.get_recent(args.limit)

    if not entries:
        print("No entries found.")
        return 1

    if args.json:
        print(json.dumps(entries, indent=2))
    else:
        for entry in entries:
            print(f"\n{'='*60}")
            print(f"File: {entry['file_path']}")
            print(f"Time: {entry['timestamp']}")
            if entry.get('session_id'):
                print(f"Session: {entry['session_id'][:20]}...")

            code = entry['new_code']
            if len(code) > 500:
                code = code[:500] + "\n... (truncated)"
            print(f"\n{code}")

    return 0


def cmd_config(args):
    """Show or create configuration."""
    config_path = Path.home() / ".afterimage" / "config.yaml"

    if args.init:
        # Create default config
        default_config = """# AI-AfterImage Configuration

# Search settings
search:
  max_results: 5
  relevance_threshold: 0.6
  max_injection_tokens: 2000

# Filter settings
filter:
  code_extensions:
    - .py
    - .js
    - .ts
    - .jsx
    - .tsx
    - .rs
    - .go
    - .java
    - .c
    - .cpp
    - .h
    - .rb
    - .php
    - .swift
    - .kt
  skip_extensions:
    - .md
    - .json
    - .yaml
    - .yml
    - .txt
    - .log
    - .env
  skip_paths:
    - artifacts/
    - docs/
    - research/
    - test_data/
    - __pycache__/
    - node_modules/

# Embedding model
embeddings:
  model: all-MiniLM-L6-v2
  device: cpu  # or cuda
"""
        config_path.parent.mkdir(parents=True, exist_ok=True)

        if config_path.exists() and not args.force:
            print(f"Config already exists at {config_path}")
            print("Use --force to overwrite.")
            return 1

        with open(config_path, "w") as f:
            f.write(default_config)
        print(f"Created config at {config_path}")
        return 0

    # Show current config
    if config_path.exists():
        print(f"Config location: {config_path}\n")
        with open(config_path) as f:
            print(f.read())
    else:
        print(f"No config file found at {config_path}")
        print("Run 'afterimage config --init' to create one.")

    return 0


def _format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="afterimage",
        description="AI-AfterImage: Episodic memory for Claude Code"
    )
    parser.add_argument(
        "--version", action="version",
        version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # search command
    search_parser = subparsers.add_parser(
        "search", help="Search the knowledge base"
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "-l", "--limit", type=int, default=5,
        help="Maximum results (default: 5)"
    )
    search_parser.add_argument(
        "-t", "--threshold", type=float, default=0.3,
        help="Minimum relevance threshold (default: 0.3)"
    )
    search_parser.add_argument(
        "-p", "--path", help="Filter by file path pattern"
    )
    search_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest transcripts into the knowledge base"
    )
    ingest_parser.add_argument(
        "-f", "--file", help="Specific transcript file to ingest"
    )
    ingest_parser.add_argument(
        "-d", "--directory", help="Directory to search for transcripts"
    )
    ingest_parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Skip embedding generation"
    )
    ingest_parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )

    # stats command
    stats_parser = subparsers.add_parser(
        "stats", help="Show knowledge base statistics"
    )
    stats_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )

    # export command
    export_parser = subparsers.add_parser(
        "export", help="Export knowledge base to JSON"
    )
    export_parser.add_argument(
        "-o", "--output", help="Output file (default: stdout)"
    )

    # clear command
    clear_parser = subparsers.add_parser(
        "clear", help="Clear the knowledge base"
    )
    clear_parser.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation"
    )

    # recent command
    recent_parser = subparsers.add_parser(
        "recent", help="Show recent entries"
    )
    recent_parser.add_argument(
        "-l", "--limit", type=int, default=10,
        help="Number of entries (default: 10)"
    )
    recent_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )

    # config command
    config_parser = subparsers.add_parser(
        "config", help="Show or create configuration"
    )
    config_parser.add_argument(
        "--init", action="store_true",
        help="Create default config file"
    )
    config_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing config"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Dispatch to command handler
    commands = {
        "search": cmd_search,
        "ingest": cmd_ingest,
        "stats": cmd_stats,
        "export": cmd_export,
        "clear": cmd_clear,
        "recent": cmd_recent,
        "config": cmd_config,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
