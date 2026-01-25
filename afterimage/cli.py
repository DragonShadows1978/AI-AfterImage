"""
CLI: Command-line interface for AI-AfterImage.

Provides commands for searching, ingesting transcripts, and managing
the knowledge base.
"""

import argparse
import json
import os
import shutil
import stat
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


def cmd_setup(args):
    """Set up AfterImage: config, hook, settings, and model."""
    home = Path.home()
    afterimage_dir = home / ".afterimage"
    claude_dir = home / ".claude"
    hooks_dir = claude_dir / "hooks"
    settings_path = claude_dir / "settings.json"
    config_path = afterimage_dir / "config.yaml"

    print("AI-AfterImage Setup")
    print("=" * 40)

    # Step 1: Create ~/.afterimage directory
    print("\n[1/5] Creating configuration directory...")
    afterimage_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Created: {afterimage_dir}")

    # Step 2: Create config file
    print("\n[2/5] Creating configuration file...")
    if not config_path.exists() or args.force:
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
        with open(config_path, "w") as f:
            f.write(default_config)
        print(f"  Created: {config_path}")
    else:
        print(f"  Exists: {config_path} (use --force to overwrite)")

    # Step 3: Install hook
    print("\n[3/5] Installing Claude Code hook...")
    hooks_dir.mkdir(parents=True, exist_ok=True)

    # Find the hook script from installed package or source
    hook_source = None
    try:
        import afterimage
        package_dir = Path(afterimage.__file__).parent.parent
        candidate = package_dir / "hooks" / "afterimage_hook.py"
        if candidate.exists():
            hook_source = candidate
    except:
        pass

    if not hook_source:
        # Try common locations
        for loc in [
            Path.home() / "AI-AfterImage" / "hooks" / "afterimage_hook.py",
            Path("/usr/local/lib/afterimage/hooks/afterimage_hook.py"),
        ]:
            if loc.exists():
                hook_source = loc
                break

    hook_dest = hooks_dir / "afterimage_hook.py"

    if hook_source and hook_source.exists():
        shutil.copy2(hook_source, hook_dest)
        # Make executable
        hook_dest.chmod(hook_dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  Installed: {hook_dest}")
    else:
        # Create a minimal hook that imports from the installed package
        hook_content = '''#!/usr/bin/env python3
"""AI-AfterImage hook for Claude Code - auto-generated by setup."""
import sys
try:
    from afterimage.hook import main
    main()
except ImportError:
    # Fallback: try to run the full hook inline
    import json
    input_data = json.load(sys.stdin)
    # Just pass through if package not found
    sys.exit(0)
'''
        with open(hook_dest, "w") as f:
            f.write(hook_content)
        hook_dest.chmod(hook_dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"  Created minimal hook: {hook_dest}")
        print("  Note: For full functionality, ensure afterimage package is importable")

    # Step 4: Update settings.json
    print("\n[4/5] Configuring Claude Code settings...")
    hook_command = str(hook_dest)

    settings = {}
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)
        except json.JSONDecodeError:
            print(f"  Warning: Could not parse {settings_path}, creating new")
            settings = {}

    # Ensure hooks structure exists
    if "hooks" not in settings:
        settings["hooks"] = {}
    if "PreToolUse" not in settings["hooks"]:
        settings["hooks"]["PreToolUse"] = []
    if "PostToolUse" not in settings["hooks"]:
        settings["hooks"]["PostToolUse"] = []

    # Check if afterimage hook already configured
    def has_afterimage_hook(hook_list):
        for entry in hook_list:
            if entry.get("matcher") == "Write|Edit":
                for hook in entry.get("hooks", []):
                    if "afterimage" in hook.get("command", ""):
                        return True
        return False

    # Add PreToolUse hook if not present
    if not has_afterimage_hook(settings["hooks"]["PreToolUse"]):
        settings["hooks"]["PreToolUse"].append({
            "matcher": "Write|Edit",
            "hooks": [{"type": "command", "command": hook_command}]
        })
        print("  Added PreToolUse hook")
    else:
        print("  PreToolUse hook already configured")

    # Add PostToolUse hook if not present
    if not has_afterimage_hook(settings["hooks"]["PostToolUse"]):
        settings["hooks"]["PostToolUse"].append({
            "matcher": "Write|Edit",
            "hooks": [{"type": "command", "command": hook_command}]
        })
        print("  Added PostToolUse hook")
    else:
        print("  PostToolUse hook already configured")

    # Write settings
    claude_dir.mkdir(parents=True, exist_ok=True)
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"  Updated: {settings_path}")

    # Step 5: Download embedding model
    print("\n[5/5] Downloading embedding model...")
    try:
        os.environ["TRANSFORMERS_CACHE"] = str(afterimage_dir / "models")
        os.environ["HF_HOME"] = str(afterimage_dir / "models")
        from sentence_transformers import SentenceTransformer
        print("  Downloading all-MiniLM-L6-v2 (~90MB)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"  Model cached to: {afterimage_dir / 'models'}")
    except ImportError:
        print("  Skipped: sentence-transformers not installed")
        print("  Run: pip install sentence-transformers")
    except Exception as e:
        print(f"  Warning: Could not download model: {e}")
        print("  Model will download on first search")

    print("\n" + "=" * 40)
    print("Setup complete!")
    print("\nNext steps:")
    print("  1. Restart Claude Code")
    print("  2. Run 'afterimage stats' to verify")
    print("  3. AfterImage will automatically capture your code")

    return 0


def cmd_uninstall(args):
    """Uninstall AfterImage hook and optionally all data."""
    home = Path.home()
    afterimage_dir = home / ".afterimage"
    claude_dir = home / ".claude"
    hooks_dir = claude_dir / "hooks"
    settings_path = claude_dir / "settings.json"
    hook_path = hooks_dir / "afterimage_hook.py"

    print("AI-AfterImage Uninstall")
    print("=" * 40)

    # Step 1: Remove hook from settings.json
    print("\n[1/3] Removing from Claude Code settings...")
    if settings_path.exists():
        try:
            with open(settings_path) as f:
                settings = json.load(f)

            modified = False

            # Remove afterimage hooks from PreToolUse
            if "hooks" in settings and "PreToolUse" in settings["hooks"]:
                original_len = len(settings["hooks"]["PreToolUse"])
                settings["hooks"]["PreToolUse"] = [
                    entry for entry in settings["hooks"]["PreToolUse"]
                    if not (entry.get("matcher") == "Write|Edit" and
                            any("afterimage" in h.get("command", "") for h in entry.get("hooks", [])))
                ]
                if len(settings["hooks"]["PreToolUse"]) < original_len:
                    modified = True
                    print("  Removed PreToolUse hook")

            # Remove afterimage hooks from PostToolUse
            if "hooks" in settings and "PostToolUse" in settings["hooks"]:
                original_len = len(settings["hooks"]["PostToolUse"])
                settings["hooks"]["PostToolUse"] = [
                    entry for entry in settings["hooks"]["PostToolUse"]
                    if not (entry.get("matcher") == "Write|Edit" and
                            any("afterimage" in h.get("command", "") for h in entry.get("hooks", [])))
                ]
                if len(settings["hooks"]["PostToolUse"]) < original_len:
                    modified = True
                    print("  Removed PostToolUse hook")

            if modified:
                with open(settings_path, "w") as f:
                    json.dump(settings, f, indent=2)
                print(f"  Updated: {settings_path}")
            else:
                print("  No AfterImage hooks found in settings")

        except Exception as e:
            print(f"  Warning: Could not update settings: {e}")
    else:
        print(f"  Settings file not found: {settings_path}")

    # Step 2: Remove hook script
    print("\n[2/3] Removing hook script...")
    if hook_path.exists():
        hook_path.unlink()
        print(f"  Removed: {hook_path}")
    else:
        print(f"  Hook not found: {hook_path}")

    # Step 3: Optionally remove all data
    if args.purge:
        print("\n[3/3] Purging all data...")
        if afterimage_dir.exists():
            # Show what will be deleted
            db_path = afterimage_dir / "memory.db"
            if db_path.exists():
                db_size = db_path.stat().st_size
                print(f"  Removing database ({_format_bytes(db_size)})")

            shutil.rmtree(afterimage_dir)
            print(f"  Removed: {afterimage_dir}")
        else:
            print(f"  Directory not found: {afterimage_dir}")
    else:
        print("\n[3/3] Keeping data...")
        print(f"  Data preserved at: {afterimage_dir}")
        print("  Use --purge to remove all data")

    print("\n" + "=" * 40)
    print("Uninstall complete!")

    if not args.purge:
        print("\nNote: Your code memory is still saved at ~/.afterimage/")
        print("Run 'afterimage uninstall --purge' to remove everything")

    return 0


def _format_bytes(size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# =============================================================================
# CHURN TRACKING COMMANDS (v0.3.0)
# =============================================================================

def cmd_churn(args):
    """Show churn statistics for a file."""
    try:
        from .churn import ChurnTracker, format_tier_badge
    except ImportError:
        print("Error: Churn tracking module not available")
        return 1

    tracker = ChurnTracker()
    tracker.initialize()

    file_path = args.file

    if args.json:
        import json as json_module
        stats = tracker.get_file_stats(file_path)
        output = {"file_stats": stats.to_dict()}

        if args.functions:
            func_stats = tracker.get_function_stats(file_path)
            output["function_stats"] = [f.to_dict() for f in func_stats]

        if args.history:
            history = tracker.get_edit_history(file_path, limit=args.history_limit)
            output["history"] = [h.to_dict() for h in history]

        print(json_module.dumps(output, indent=2))
    else:
        stats = tracker.get_file_stats(file_path)

        print(f"\nChurn Statistics: {file_path}")
        print("=" * 60)
        print(f"Tier:            {format_tier_badge(stats.tier)}")
        print(f"Total edits:     {stats.total_edits}")
        print(f"Last 24h:        {stats.edits_last_24h}")
        print(f"Last 7 days:     {stats.edits_last_7d}")
        print(f"Last 30 days:    {stats.edits_last_30d}")

        if stats.first_edit:
            print(f"\nFirst edit:      {stats.first_edit[:19]}")
        if stats.last_edit:
            print(f"Last edit:       {stats.last_edit[:19]}")

        if args.functions:
            func_stats = tracker.get_function_stats(file_path)
            if func_stats:
                print(f"\nFunction-level churn:")
                print("-" * 40)
                for f in func_stats[:10]:
                    change_summary = ", ".join(f.change_types[:3]) if f.change_types else "unknown"
                    print(f"  {f.function_name}(): {f.edit_count} edits ({change_summary})")

        if args.history:
            history = tracker.get_edit_history(file_path, limit=args.history_limit)
            if history:
                print(f"\nRecent edit history:")
                print("-" * 40)
                for h in history:
                    func_name = f" [{h.function_name}]" if h.function_name else ""
                    print(f"  {h.timestamp[:19]} - {h.change_type.value}{func_name}")

    return 0


def cmd_hotspots(args):
    """Show files ranked by churn hotspot score."""
    try:
        from .churn import ChurnTracker, format_tier_badge
    except ImportError:
        print("Error: Churn tracking module not available")
        return 1

    tracker = ChurnTracker()
    tracker.initialize()

    hotspots = tracker.get_hotspots(limit=args.limit)

    if not hotspots:
        print("No churn data found. Edit some files first!")
        return 0

    if args.json:
        import json as json_module
        output = [
            {"file_stats": stats.to_dict(), "score": score}
            for stats, score in hotspots
        ]
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\nCode Churn Hotspots (Top {args.limit})")
        print("=" * 70)
        print(f"{'Rank':<5} {'Score':<8} {'Tier':<10} {'30d':<6} {'File'}")
        print("-" * 70)

        for i, (stats, score) in enumerate(hotspots, 1):
            # Truncate file path for display
            display_path = stats.file_path
            if len(display_path) > 40:
                display_path = "..." + display_path[-37:]

            tier_short = stats.tier.value.upper()[:4]
            print(f"{i:<5} {score:<8.1f} {tier_short:<10} {stats.edits_last_30d:<6} {display_path}")

    return 0


def cmd_files_by_tier(args):
    """Show files filtered by tier."""
    try:
        from .churn import ChurnTracker, ChurnTier, format_tier_badge
    except ImportError:
        print("Error: Churn tracking module not available")
        return 1

    tracker = ChurnTracker()
    tracker.initialize()

    # Parse tier argument
    tier_map = {
        "gold": ChurnTier.GOLD,
        "silver": ChurnTier.SILVER,
        "bronze": ChurnTier.BRONZE,
        "red": ChurnTier.RED,
    }

    tier = tier_map.get(args.tier.lower())
    if not tier:
        print(f"Invalid tier: {args.tier}. Use: gold, silver, bronze, red")
        return 1

    files = tracker.get_files_by_tier(tier)

    if not files:
        print(f"No files in {args.tier.upper()} tier")
        return 0

    if args.json:
        import json as json_module
        output = [f.to_dict() for f in files]
        print(json_module.dumps(output, indent=2))
    else:
        print(f"\n{format_tier_badge(tier)} Files ({len(files)} total)")
        print("=" * 60)

        for stats in files[:args.limit]:
            # Truncate file path for display
            display_path = stats.file_path
            if len(display_path) > 50:
                display_path = "..." + display_path[-47:]

            print(f"  {stats.edits_last_30d:>3} edits | {display_path}")

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="afterimage",
        description="AI-AfterImage: Episodic memory for Claude Code"
    )
    from importlib.metadata import version as pkg_version
    parser.add_argument(
        "--version", action="version",
        version=f"%(prog)s {pkg_version('ai-afterimage')}"
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

    # setup command
    setup_parser = subparsers.add_parser(
        "setup", help="First-time setup: config, hook, settings, model"
    )
    setup_parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing configuration"
    )

    # uninstall command
    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Remove AfterImage hook and optionally all data"
    )
    uninstall_parser.add_argument(
        "--purge", action="store_true",
        help="Also remove all data (~/.afterimage)"
    )

    # churn command (v0.3.0)
    churn_parser = subparsers.add_parser(
        "churn", help="Show churn statistics for a file"
    )
    churn_parser.add_argument("file", help="File path to analyze")
    churn_parser.add_argument(
        "-f", "--functions", action="store_true",
        help="Show function-level statistics"
    )
    churn_parser.add_argument(
        "-H", "--history", action="store_true",
        help="Show edit history"
    )
    churn_parser.add_argument(
        "--history-limit", type=int, default=20,
        help="Number of history entries (default: 20)"
    )
    churn_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )

    # hotspots command (v0.3.0)
    hotspots_parser = subparsers.add_parser(
        "hotspots", help="Show files ranked by churn hotspot score"
    )
    hotspots_parser.add_argument(
        "-l", "--limit", type=int, default=20,
        help="Number of files to show (default: 20)"
    )
    hotspots_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
    )

    # files command (v0.3.0) - filter by tier
    files_parser = subparsers.add_parser(
        "files", help="Show files by stability tier"
    )
    files_parser.add_argument(
        "--tier", required=True,
        choices=["gold", "silver", "bronze", "red"],
        help="Filter by tier (gold/silver/bronze/red)"
    )
    files_parser.add_argument(
        "-l", "--limit", type=int, default=50,
        help="Maximum files to show (default: 50)"
    )
    files_parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON"
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
        "setup": cmd_setup,
        "uninstall": cmd_uninstall,
        # Churn tracking commands (v0.3.0)
        "churn": cmd_churn,
        "hotspots": cmd_hotspots,
        "files": cmd_files_by_tier,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
