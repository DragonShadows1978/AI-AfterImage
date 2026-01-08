# AI-AfterImage

[![Tests](https://github.com/DragonShadows1978/AI-AfterImage/actions/workflows/test.yml/badge.svg)](https://github.com/DragonShadows1978/AI-AfterImage/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/DragonShadows1978/AI-AfterImage/graph/badge.svg)](https://codecov.io/gh/DragonShadows1978/AI-AfterImage)
[![PyPI](https://img.shields.io/pypi/v/ai-afterimage)](https://pypi.org/project/ai-afterimage/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Give Claude Code memory of code it wrote in past sessions.**

Claude starts every session with amnesia. AfterImage fixes this by intercepting Write/Edit operations and showing Claude similar code it wrote before - using a clever deny-then-allow hook pattern that injects context directly into Claude's view.

## Quick Start (< 5 minutes)

### Step 1: Install

```bash
pip install ai-afterimage
```

### Step 2: Set Up Hook

```bash
afterimage setup
```

This automatically:
- Creates `~/.afterimage/` configuration
- Installs the hook to `~/.claude/hooks/`
- Configures `~/.claude/settings.json`
- Downloads the embedding model (~90MB)

### Step 3: You're Done

Start Claude Code. AfterImage now works invisibly in the background:
- Before writes: Shows similar past code (if found)
- After writes: Stores the code for future recall

## How It Actually Works

The magic is in the **deny-then-allow pattern**:

1. Claude tries to Write/Edit a file
2. Hook searches knowledge base for similar past code
3. If found: **DENY** with past code in the reason message
4. Claude **sees** the deny reason (this is documented Claude Code behavior!)
5. Claude retries the same write
6. Hook recognizes retry (same content hash) → **ALLOW**
7. File is written
8. Post-hook stores the new code in KB

This is the only way to inject context into Claude's view before a write. The deny reason IS the injection mechanism.

## Verify It's Working

```bash
# Check KB stats
afterimage stats

# Search your code memory
afterimage search "authentication"

# See recent code stored
afterimage recent
```

## All Your Code Stays Local

- SQLite database: `~/.afterimage/memory.db`
- No cloud sync, no API calls
- Works fully offline after setup
- Your code never leaves your machine

## CLI Reference

| Command | Description |
|---------|-------------|
| `afterimage setup` | First-time setup |
| `afterimage search <query>` | Search past code |
| `afterimage stats` | Show KB statistics |
| `afterimage recent` | Recent stored entries |
| `afterimage ingest` | Import existing transcripts |
| `afterimage config` | View/edit configuration |
| `afterimage uninstall` | Clean removal |

## Requirements

- Python 3.10+
- Claude Code CLI
- Linux or macOS (Windows support planned)

## Uninstall

```bash
afterimage uninstall          # Remove hook, keep data
afterimage uninstall --purge  # Remove everything
```

## License

MIT License - Do whatever you want with it.

## Support

If AfterImage saves you time, consider starring the repo - it helps others find it.

---

**The ghost of what was written, persisting across sessions.**
