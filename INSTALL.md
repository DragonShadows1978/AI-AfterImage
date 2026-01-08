# AI-AfterImage Installation Guide

## Prerequisites

- Python 3.10 or higher
- Claude Code CLI installed and working
- ~150MB disk space (90MB for embedding model + database growth)

## Method 1: Automated Setup (Recommended)

```bash
# Install the package
pip install ai-afterimage

# Run setup (creates config, installs hook, downloads model)
afterimage setup
```

That's it. Setup handles everything automatically.

## Method 2: Manual Installation

If you need more control or automated setup fails:

### 2.1 Install Package

```bash
# From PyPI
pip install ai-afterimage

# OR from source
git clone https://github.com/DragonShadows1978/AI-AfterImage.git
cd AI-AfterImage
pip install -e ".[embeddings]"
```

### 2.2 Create Configuration Directory

```bash
mkdir -p ~/.afterimage
```

### 2.3 Initialize Config

```bash
afterimage config --init
```

This creates `~/.afterimage/config.yaml` with defaults.

### 2.4 Install the Hook

```bash
# Create hooks directory if needed
mkdir -p ~/.claude/hooks

# Copy hook script
cp hooks/afterimage_hook.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/afterimage_hook.py
```

### 2.5 Configure Claude Code Settings

Edit `~/.claude/settings.json` (create if doesn't exist):

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/afterimage_hook.py"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "$HOME/.claude/hooks/afterimage_hook.py"
          }
        ]
      }
    ]
  }
}
```

**Note:** Replace `$HOME` with your actual home path (e.g., `/home/username`).

### 2.6 Set Environment Variable

Add to your shell config (~/.bashrc, ~/.zshrc):

```bash
export AFTERIMAGE_PATH="$HOME/.local/lib/afterimage"
```

Or if installed from source:

```bash
export AFTERIMAGE_PATH="$HOME/AI-AfterImage"
```

### 2.7 Download Embedding Model

First search triggers download (~90MB):

```bash
afterimage search "test"
```

Model caches to `~/.afterimage/models/`.

## Verify Installation

```bash
# Check hook is configured
cat ~/.claude/settings.json | grep afterimage

# Check KB is accessible
afterimage stats

# Test search
afterimage search "function"
```

## Merging with Existing Hooks

If you already have hooks in settings.json, merge the matchers:

**Before (your existing hook):**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "your_hook.py"}]
      }
    ]
  }
}
```

**After (merged):**
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [{"type": "command", "command": "your_hook.py"}]
      },
      {
        "matcher": "Write|Edit",
        "hooks": [{"type": "command", "command": "$HOME/.claude/hooks/afterimage_hook.py"}]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{"type": "command", "command": "$HOME/.claude/hooks/afterimage_hook.py"}]
      }
    ]
  }
}
```

## Troubleshooting

### Hook Not Firing

1. Check hook path is absolute (not `~`, use `/home/username`)
2. Check hook is executable: `chmod +x ~/.claude/hooks/afterimage_hook.py`
3. Check settings.json syntax: `python -m json.tool ~/.claude/settings.json`
4. Check hook logs: `tail ~/.afterimage/afterimage.log`

### "Module not found" Errors

```bash
# Verify AFTERIMAGE_PATH is set
echo $AFTERIMAGE_PATH

# Verify package is importable
python -c "from afterimage.kb import KnowledgeBase; print('OK')"
```

### Embedding Model Won't Download

```bash
# Check network access to HuggingFace
curl -I https://huggingface.co

# Manual model download
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### KB Not Growing

Check that code files are being written (not just .md/.json):

```bash
afterimage stats
# Should show entries growing after Claude writes .py/.js/etc
```

## Complete Uninstall

```bash
# Remove hook from settings
afterimage uninstall

# Remove all data (optional)
rm -rf ~/.afterimage

# Uninstall package
pip uninstall ai-afterimage
```
