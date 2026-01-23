"""
Configuration loader for AI-AfterImage.

Supports dual backend configuration (SQLite and PostgreSQL) via YAML config
and environment variables.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass, field

import yaml


@dataclass
class SQLiteConfig:
    """SQLite backend configuration."""
    path: Path = field(default_factory=lambda: Path.home() / ".afterimage" / "memory.db")


@dataclass
class PostgreSQLConfig:
    """PostgreSQL backend configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "afterimage"
    user: str = "afterimage"
    password: Optional[str] = None
    connection_string: Optional[str] = None
    min_pool_size: int = 2
    max_pool_size: int = 10


@dataclass
class SearchConfig:
    """Search configuration."""
    max_results: int = 5
    relevance_threshold: float = 0.6
    max_injection_tokens: int = 2000
    fts_weight: float = 0.4
    semantic_weight: float = 0.6


@dataclass
class EmbeddingsConfig:
    """Embeddings configuration."""
    model: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # or "cuda"
    embedding_dim: int = 384


@dataclass
class FilterConfig:
    """Code filter configuration."""
    code_extensions: list = field(default_factory=lambda: [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".rs", ".go",
        ".java", ".c", ".cpp", ".h", ".rb", ".php", ".swift", ".kt"
    ])
    skip_extensions: list = field(default_factory=lambda: [
        ".md", ".json", ".yaml", ".yml", ".txt", ".log", ".env"
    ])
    skip_paths: list = field(default_factory=lambda: [
        "artifacts/", "docs/", "research/", "test_data/",
        "__pycache__/", "node_modules/"
    ])


@dataclass
class AfterImageConfig:
    """Main AfterImage configuration."""
    backend: str = "sqlite"  # "sqlite" or "postgresql"
    sqlite: SQLiteConfig = field(default_factory=SQLiteConfig)
    postgresql: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


def get_config_path() -> Path:
    """Get path to config file."""
    return Path.home() / ".afterimage" / "config.yaml"


def load_config(config_path: Optional[Path] = None) -> AfterImageConfig:
    """
    Load configuration from YAML file and environment variables.

    Priority (highest to lowest):
    1. Environment variables (AFTERIMAGE_*)
    2. Config file values
    3. Default values

    Args:
        config_path: Path to config file. Defaults to ~/.afterimage/config.yaml

    Returns:
        Loaded configuration
    """
    if config_path is None:
        config_path = get_config_path()

    config = AfterImageConfig()

    # Load from file if exists
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        config = _merge_config(config, data)

    # Override with environment variables
    config = _apply_env_overrides(config)

    return config


def _merge_config(config: AfterImageConfig, data: Dict[str, Any]) -> AfterImageConfig:
    """Merge YAML data into config dataclass."""
    # Storage backend
    if "storage" in data:
        storage = data["storage"]
        if "backend" in storage:
            config.backend = storage["backend"]
        if "sqlite" in storage:
            sqlite = storage["sqlite"]
            if "path" in sqlite:
                config.sqlite.path = Path(sqlite["path"]).expanduser()
        if "postgresql" in storage:
            pg = storage["postgresql"]
            if "host" in pg:
                config.postgresql.host = pg["host"]
            if "port" in pg:
                config.postgresql.port = int(pg["port"])
            if "database" in pg:
                config.postgresql.database = pg["database"]
            if "user" in pg:
                config.postgresql.user = pg["user"]
            if "password" in pg:
                config.postgresql.password = pg["password"]
            if "connection_string" in pg:
                config.postgresql.connection_string = pg["connection_string"]
            if "min_pool_size" in pg:
                config.postgresql.min_pool_size = int(pg["min_pool_size"])
            if "max_pool_size" in pg:
                config.postgresql.max_pool_size = int(pg["max_pool_size"])

    # Search settings
    if "search" in data:
        search = data["search"]
        if "max_results" in search:
            config.search.max_results = int(search["max_results"])
        if "relevance_threshold" in search:
            config.search.relevance_threshold = float(search["relevance_threshold"])
        if "max_injection_tokens" in search:
            config.search.max_injection_tokens = int(search["max_injection_tokens"])
        if "fts_weight" in search:
            config.search.fts_weight = float(search["fts_weight"])
        if "semantic_weight" in search:
            config.search.semantic_weight = float(search["semantic_weight"])

    # Embeddings settings
    if "embeddings" in data:
        emb = data["embeddings"]
        if "model" in emb:
            config.embeddings.model = emb["model"]
        if "device" in emb:
            config.embeddings.device = emb["device"]
        if "embedding_dim" in emb:
            config.embeddings.embedding_dim = int(emb["embedding_dim"])

    # Filter settings
    if "filter" in data:
        flt = data["filter"]
        if "code_extensions" in flt:
            config.filter.code_extensions = flt["code_extensions"]
        if "skip_extensions" in flt:
            config.filter.skip_extensions = flt["skip_extensions"]
        if "skip_paths" in flt:
            config.filter.skip_paths = flt["skip_paths"]

    return config


def _apply_env_overrides(config: AfterImageConfig) -> AfterImageConfig:
    """Apply environment variable overrides."""
    # Backend selection
    if os.environ.get("AFTERIMAGE_BACKEND"):
        config.backend = os.environ["AFTERIMAGE_BACKEND"]

    # SQLite path
    if os.environ.get("AFTERIMAGE_SQLITE_PATH"):
        config.sqlite.path = Path(os.environ["AFTERIMAGE_SQLITE_PATH"]).expanduser()

    # PostgreSQL settings
    if os.environ.get("AFTERIMAGE_DATABASE_URL"):
        config.postgresql.connection_string = os.environ["AFTERIMAGE_DATABASE_URL"]
    if os.environ.get("AFTERIMAGE_PG_HOST"):
        config.postgresql.host = os.environ["AFTERIMAGE_PG_HOST"]
    if os.environ.get("AFTERIMAGE_PG_PORT"):
        config.postgresql.port = int(os.environ["AFTERIMAGE_PG_PORT"])
    if os.environ.get("AFTERIMAGE_PG_DATABASE"):
        config.postgresql.database = os.environ["AFTERIMAGE_PG_DATABASE"]
    if os.environ.get("AFTERIMAGE_PG_USER"):
        config.postgresql.user = os.environ["AFTERIMAGE_PG_USER"]

    # PostgreSQL password - try environment first, then bashrc
    pg_password = os.environ.get("AFTERIMAGE_PG_PASSWORD")
    if not pg_password:
        bashrc_path = Path.home() / ".bashrc"
        if bashrc_path.exists():
            try:
                with open(bashrc_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('export AFTERIMAGE_PG_PASSWORD='):
                            pg_password = line.split('=', 1)[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass
    if pg_password:
        config.postgresql.password = pg_password

    # Embeddings
    if os.environ.get("AFTERIMAGE_EMBEDDING_MODEL"):
        config.embeddings.model = os.environ["AFTERIMAGE_EMBEDDING_MODEL"]
    if os.environ.get("AFTERIMAGE_EMBEDDING_DEVICE"):
        config.embeddings.device = os.environ["AFTERIMAGE_EMBEDDING_DEVICE"]

    return config


def create_default_config(config_path: Optional[Path] = None, force: bool = False) -> Path:
    """
    Create default configuration file.

    Args:
        config_path: Path to config file. Defaults to ~/.afterimage/config.yaml
        force: Overwrite existing config

    Returns:
        Path to created config file
    """
    if config_path is None:
        config_path = get_config_path()

    if config_path.exists() and not force:
        raise FileExistsError(f"Config already exists at {config_path}")

    config_path.parent.mkdir(parents=True, exist_ok=True)

    default_config = """# AI-AfterImage Configuration

# Storage backend configuration
storage:
  backend: sqlite  # Options: sqlite, postgresql

  sqlite:
    path: ~/.afterimage/memory.db

  postgresql:
    host: localhost
    port: 5432
    database: afterimage
    user: afterimage
    # Password from env: AFTERIMAGE_PG_PASSWORD
    # Or connection string: AFTERIMAGE_DATABASE_URL
    min_pool_size: 2
    max_pool_size: 10

# Search settings
search:
  max_results: 5
  relevance_threshold: 0.6
  max_injection_tokens: 2000
  fts_weight: 0.4
  semantic_weight: 0.6

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
  embedding_dim: 384
"""

    with open(config_path, "w") as f:
        f.write(default_config)

    return config_path


def get_storage_backend(config: Optional[AfterImageConfig] = None):
    """
    Get configured storage backend instance.

    Args:
        config: Configuration to use. Loads default if None.

    Returns:
        StorageBackend instance (SQLiteBackend or PostgreSQLBackend)
    """
    if config is None:
        config = load_config()

    if config.backend == "postgresql":
        from .storage import PostgreSQLBackend
        backend = PostgreSQLBackend(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            connection_string=config.postgresql.connection_string,
            min_pool_size=config.postgresql.min_pool_size,
            max_pool_size=config.postgresql.max_pool_size,
            embedding_dim=config.embeddings.embedding_dim
        )
    else:
        from .storage import SQLiteBackend
        backend = SQLiteBackend(db_path=config.sqlite.path)

    backend.initialize()
    return backend
