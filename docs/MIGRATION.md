# AI-AfterImage Migration Guide

This guide covers migrating from SQLite to PostgreSQL backend for AI-AfterImage v0.2.0.

## Why Migrate?

PostgreSQL with pgvector offers several advantages over SQLite:

| Feature | SQLite | PostgreSQL |
|---------|--------|------------|
| Concurrent Writes | ❌ Single-writer | ✅ Multi-writer |
| Vector Search | In-memory (slow) | HNSW index (fast) |
| FTS | FTS5 | tsvector + GIN |
| Scalability | Single file | Server-based |
| Multi-agent Support | ❌ | ✅ |

## Prerequisites

### Install PostgreSQL and pgvector

```bash
# Ubuntu/Debian
sudo apt install postgresql postgresql-contrib

# Install pgvector extension
sudo apt install postgresql-16-pgvector
```

### Create Database and User

```bash
# Connect to PostgreSQL
sudo -u postgres psql

-- Create user and database
CREATE USER afterimage WITH PASSWORD 'yourpassword';
CREATE DATABASE afterimage OWNER afterimage;
GRANT ALL PRIVILEGES ON DATABASE afterimage TO afterimage;

-- Connect to database and enable pgvector
\c afterimage
CREATE EXTENSION IF NOT EXISTS vector;

\q
```

### Install Python Dependencies

```bash
pip install "ai-afterimage[postgresql]"
# Or manually:
pip install asyncpg psycopg[binary] pgvector numpy
```

## Migration Steps

### 1. Update Configuration

Edit `~/.afterimage/config.yaml`:

```yaml
storage:
  backend: postgresql  # Change from sqlite

  sqlite:
    path: ~/.afterimage/memory.db

  postgresql:
    host: localhost
    port: 5432
    database: afterimage
    user: afterimage
    # Password from environment: AFTERIMAGE_PG_PASSWORD
    min_pool_size: 2
    max_pool_size: 10
```

### 2. Set Password Environment Variable

```bash
export AFTERIMAGE_PG_PASSWORD="yourpassword"
# Or add to ~/.bashrc for persistence
```

### 3. Run Migration

The migration script transfers all entries from SQLite to PostgreSQL:

```bash
cd /path/to/AI-AfterImage
AFTERIMAGE_PG_PASSWORD=yourpassword python -m afterimage.migrate \
    --sqlite ~/.afterimage/memory.db \
    --validate
```

Expected output:
```
Starting migration...
Reading source data: 5000/5000 (100.0%)
Migrating entries: 5000/5000 (100.0%)
Validating migration: 1/1 (100.0%)

============================================================
AI-AfterImage Migration Report
============================================================
Source entries:     5000
Migrated:           5000
Skipped (existing): 0
Failed:             0
Success rate:       100.0%
Elapsed time:       42.3s
============================================================
```

### 4. Verify Migration

```python
from afterimage import get_storage_backend

backend = get_storage_backend()
stats = backend.stats()
print(f"Backend: {stats.backend_type}")
print(f"Entries: {stats.total_entries}")
```

## Rollback

To revert to SQLite:

1. Edit `~/.afterimage/config.yaml`:
   ```yaml
   storage:
     backend: sqlite
   ```

2. Your SQLite database remains at `~/.afterimage/memory.db`

## Troubleshooting

### Connection Refused

Check PostgreSQL is running:
```bash
sudo systemctl status postgresql
sudo systemctl start postgresql
```

### Authentication Failed

1. Check password in environment:
   ```bash
   echo $AFTERIMAGE_PG_PASSWORD
   ```

2. Verify pg_hba.conf allows password authentication:
   ```bash
   sudo nano /etc/postgresql/16/main/pg_hba.conf
   # Ensure this line exists:
   # host    afterimage    afterimage    127.0.0.1/32    md5
   sudo systemctl reload postgresql
   ```

### pgvector Extension Missing

```sql
sudo -u postgres psql -d afterimage -c "CREATE EXTENSION IF NOT EXISTS vector"
```

### Migration Interrupted

The migration is resumable. Re-run the command and it will skip already-migrated entries.

## Performance Comparison

Based on testing with 5,000 entries:

| Operation | SQLite | PostgreSQL | Speedup |
|-----------|--------|------------|---------|
| FTS Search | 45ms | 12ms | 3.8x |
| Semantic Search (10 results) | 180ms | 15ms | 12x |
| Concurrent Writes (10 threads) | Fails | 25ms each | ∞ |

## Support

For issues, visit: https://github.com/DragonShadows1978/AI-AfterImage/issues
