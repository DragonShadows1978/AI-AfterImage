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
    "PreToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/home/.claude/hooks/afterimage_hook.py"
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
            "command": "/path/to/home/.claude/hooks/afterimage_hook.py"
          }
        ]
      }
    ]
  }
}
```

**Important**: Update the path to match your home directory (e.g., `/home/username/.claude/hooks/afterimage_hook.py`).

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

### SQLite Mode (Default)

Create `~/.afterimage/config.yaml` to customize behavior:

```yaml
# Storage backend
storage:
  backend: sqlite  # Default
  sqlite:
    path: ~/.afterimage/memory.db

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

### PostgreSQL Mode (v0.2.0+)

For better concurrent access and vector search performance, use PostgreSQL with pgvector:

#### Prerequisites

1. **Install PostgreSQL 14+ with pgvector extension**:
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib

   # Install pgvector extension
   sudo apt install postgresql-server-dev-all
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector && make && sudo make install
   ```

2. **Create the AfterImage database**:
   ```bash
   sudo -u postgres psql <<EOF
   CREATE USER afterimage WITH PASSWORD 'your_secure_password';
   CREATE DATABASE afterimage OWNER afterimage;
   \c afterimage
   CREATE EXTENSION vector;
   EOF
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -e ".[postgresql]"
   # or manually:
   pip install asyncpg psycopg[binary] pgvector
   ```

#### Configuration

Update `~/.afterimage/config.yaml`:

```yaml
storage:
  backend: postgresql

  sqlite:
    path: ~/.afterimage/memory.db  # Fallback location

  postgresql:
    host: localhost
    port: 5432
    database: afterimage
    user: afterimage
    # Password via environment variable (recommended)
    # Or set here: password: your_secure_password
    min_pool_size: 2
    max_pool_size: 10

# Other settings remain the same
search:
  max_results: 5
  relevance_threshold: 0.6

embeddings:
  model: all-MiniLM-L6-v2
  device: cpu
```

#### Environment Variables

The hook supports these environment variables for PostgreSQL configuration:

| Variable | Description |
|----------|-------------|
| `AFTERIMAGE_BACKEND` | Override backend: `sqlite` or `postgresql` |
| `AFTERIMAGE_PG_PASSWORD` | PostgreSQL password (recommended over config file) |
| `AFTERIMAGE_PG_HOST` | PostgreSQL host |
| `AFTERIMAGE_PG_PORT` | PostgreSQL port |
| `AFTERIMAGE_PG_DATABASE` | Database name |
| `AFTERIMAGE_PG_USER` | Database user |
| `AFTERIMAGE_DATABASE_URL` | Full connection string (overrides individual params) |
| `AFTERIMAGE_GENERATE_EMBEDDINGS` | Set to `1` to enable embedding generation during store (slow on cold start, ~6s) |

### Performance Notes

The v0.2.0 hook is optimized for Claude Code's 5000ms timeout:

| Operation | Time | Notes |
|-----------|------|-------|
| Pre-hook search (FTS) | ~100-250ms | Uses FTS-only search, no embeddings |
| Post-hook store | ~230-270ms | Embeddings disabled by default |
| Cold start | ~220-280ms | Module loading, backend connection |

**Why embeddings are disabled by default:**

The sentence-transformers embedding model takes ~6 seconds to load on first use (cold start). Since each hook invocation is a separate process, this would cause every first write to timeout.

**To enable embeddings:**

1. Set `AFTERIMAGE_GENERATE_EMBEDDINGS=1` in your environment
2. Accept the ~6s delay on first write per session
3. Or run `afterimage reindex` periodically to generate embeddings offline

**Recommended approach:**

Keep embeddings disabled for real-time hooks. Run `afterimage reindex` as a cron job or after coding sessions to generate embeddings for semantic search.

Example shell configuration:

```bash
# Add to ~/.bashrc or ~/.zshrc
export AFTERIMAGE_BACKEND=postgresql
export AFTERIMAGE_PG_PASSWORD=your_secure_password
```

#### Graceful Fallback

The v0.2.0 hook automatically falls back to SQLite if PostgreSQL is unavailable:

1. If `backend: postgresql` is configured but PostgreSQL is unreachable
2. If `psycopg` is not installed
3. If authentication fails

You'll see a message in stderr: `[AfterImage] PostgreSQL unavailable (...), falling back to SQLite`

#### Migrating from SQLite to PostgreSQL

Use the migration command to copy existing entries:

```bash
# Migrate all entries
afterimage migrate --sqlite ~/.afterimage/memory.db --postgresql

# Validate after migration
afterimage stats --backend postgresql
```

## Manual Testing

You can test the hook manually:

```bash
# Test pre-write hook (SQLite)
echo '{"hook_event_name":"PreToolUse","tool_name":"Write","tool_input":{"file_path":"test.py","content":"def hello(): pass"}}' | \
  python ~/.claude/hooks/afterimage_hook.py

# Test with PostgreSQL
AFTERIMAGE_BACKEND=postgresql AFTERIMAGE_PG_PASSWORD=yourpassword \
  echo '{"hook_event_name":"PreToolUse","tool_name":"Write","tool_input":{"file_path":"test.py","content":"def hello(): pass"}}' | \
  python ~/.claude/hooks/afterimage_hook.py

# Test post-write hook
echo '{"hook_event_name":"PostToolUse","tool_name":"Write","tool_input":{"file_path":"test.py","content":"def hello(): pass"}}' | \
  python ~/.claude/hooks/afterimage_hook.py
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

### PostgreSQL connection issues

- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify credentials: `psql -h localhost -U afterimage -d afterimage`
- Check pgvector extension: `psql -d afterimage -c "SELECT * FROM pg_extension WHERE extname = 'vector'"`
- Verify password is set: `echo $AFTERIMAGE_PG_PASSWORD`

### Backend fallback happening unexpectedly

Check stderr for the reason:
```bash
# Run with stderr visible
python ~/.claude/hooks/afterimage_hook.py 2>&1 <<< '{"hook_event_name":"PreToolUse","tool_name":"Write","tool_input":{"file_path":"test.py","content":"def test(): pass"}}'
```

Common causes:
- `psycopg not installed` - Run `pip install psycopg[binary]`
- `PostgreSQL password not configured` - Set `AFTERIMAGE_PG_PASSWORD`
- Connection refused - Check PostgreSQL is running and accessible
