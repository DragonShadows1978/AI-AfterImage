# AI-AfterImage Claude Code Hook

This directory contains the hook script for integrating AI-AfterImage with Claude Code.

## Installation

### 1. Install the afterimage package

```bash
pip install -e /path/to/AI-AfterImage
```

Or install from pip (when published):

```bash
pip install ai-afterimage
```

### 2. Create the hooks directory

```bash
mkdir -p ~/.claude/hooks
```

### 3. Copy or symlink the hook script

```bash
# Option A: Symlink (recommended for development)
ln -s /path/to/AI-AfterImage/hooks/afterimage_hook.py ~/.claude/hooks/

# Option B: Copy
cp /path/to/AI-AfterImage/hooks/afterimage_hook.py ~/.claude/hooks/
```

### 4. Configure Claude Code

Create or edit `~/.claude/settings.json`:

```json
{
  "hooks": {
    "afterimage": {
      "enabled": true,
      "script": "~/.claude/hooks/afterimage_hook.py",
      "pre_tool": ["Write", "Edit"],
      "post_tool": ["Write", "Edit"],
      "timeout_ms": 5000
    }
  }
}
```

## How It Works

### Pre-Write/Edit Hook

Before Claude writes or edits a file:

1. The hook checks if the file is a code file (not .md, .json, etc.)
2. If it's code, it searches the knowledge base for similar past code
3. If similar code is found, it injects context: "You have written similar code before..."
4. This context helps Claude maintain consistency and avoid reinventing solutions

### Post-Write/Edit Hook

After Claude writes or edits a file:

1. The hook checks if the file is a code file
2. If it's code, it extracts the diff (for edits) or full content (for writes)
3. It generates a vector embedding for semantic search
4. It stores the code in the knowledge base with context

## Configuration

Create `~/.afterimage/config.yaml` to customize behavior:

```yaml
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
    # ... add more as needed
  skip_extensions:
    - .md
    - .json
    # ... add more as needed
  skip_paths:
    - artifacts/
    - docs/
    # ... add more as needed

# Embedding model
embeddings:
  model: all-MiniLM-L6-v2
  device: cpu  # or cuda
```

## Manual Testing

You can test the hook manually:

```bash
# Test pre-write hook
echo '{"type":"pre","tool":"Write","input":{"file_path":"test.py","content":"def hello(): pass"}}' | \
  python ~/.claude/hooks/afterimage_hook.py --json

# Test post-write hook
echo '{"type":"post","tool":"Write","input":{"file_path":"test.py","content":"def hello(): pass"}}' | \
  python ~/.claude/hooks/afterimage_hook.py --json
```

## Troubleshooting

### Hook not firing

- Check that `~/.claude/settings.json` has hooks enabled
- Verify the script path is correct and executable
- Check Claude Code logs for hook errors

### Slow pre-write searches

- The first search may be slow while loading the embedding model
- Subsequent searches should be faster (model is cached)
- Consider increasing `timeout_ms` in settings

### No results found

- Run `afterimage stats` to see if the KB has entries
- Run `afterimage ingest` to import existing transcripts
- Check that the file type is in `code_extensions`

### Embeddings not working

- Install sentence-transformers: `pip install sentence-transformers`
- Check GPU availability with: `python -c "import torch; print(torch.cuda.is_available())"`
