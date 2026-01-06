# Changelog

All notable changes to AI-AfterImage will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PyPI publish workflow with trusted publishing
- CONTRIBUTING.md with development guidelines

### Fixed
- Deprecated `datetime.utcnow()` replaced with `datetime.now(timezone.utc)`

## [0.1.0] - 2026-01-06

### Added

#### Core Features
- **Episodic memory system** for Claude Code with persistent cross-session recall
- **Knowledge Base** (`kb.py`) with SQLite + FTS5 full-text search
- **Hybrid search** (`search.py`) combining keyword (BM25) and semantic similarity
- **Embedding generation** (`embeddings.py`) using sentence-transformers (all-MiniLM-L6-v2)
- **Code filter** (`filter.py`) with 50+ recognized code extensions
- **Context injection** (`inject.py`) with configurable token limits
- **Transcript extractor** (`extract.py`) for Claude Code JSONL files

#### CLI Commands
- `afterimage search` - Search the knowledge base with hybrid ranking
- `afterimage ingest` - Ingest Claude Code transcripts into KB
- `afterimage stats` - Show knowledge base statistics
- `afterimage recent` - List recent code memories
- `afterimage export` - Export KB to JSON format
- `afterimage clear` - Clear the knowledge base
- `afterimage config` - Show or initialize configuration

#### Claude Code Integration
- Hook script (`hooks/afterimage_hook.py`) for pre/post Write/Edit
- Pre-tool hook: searches KB and injects relevant past code
- Post-tool hook: extracts and stores new code with context
- JSON-based hook configuration

#### Configuration
- YAML configuration at `~/.afterimage/config.yaml`
- Configurable search thresholds and limits
- Customizable code extension lists
- Path blacklist support for artifacts, docs, etc.

### Technical Details

#### Knowledge Base Schema
- `id`: UUID primary key
- `file_path`: Source file path
- `old_code`: Previous content (for edits)
- `new_code`: The written code
- `context`: Conversation context
- `timestamp`: ISO 8601 timestamp
- `session_id`: Claude Code session identifier
- `embedding`: 384-dimensional vector (BLOB)

#### Search Algorithm
- FTS5 BM25 scoring for keyword relevance
- Cosine similarity for semantic matching
- Configurable weights (default: 40% FTS, 60% semantic)
- Score normalization and threshold filtering

#### Performance Targets
| Operation | Target | Typical |
|-----------|--------|---------|
| Model load | <5s | 2-3s |
| Embedding generation | <50ms | 20-30ms |
| Hybrid search | <100ms | 30-50ms |
| FTS search only | <10ms | 2-5ms |

### Testing
- 154 tests across 6 test modules
- 88% code coverage
- GitHub Actions CI with Python 3.10, 3.11, 3.12
- Codecov integration for coverage tracking

[Unreleased]: https://github.com/DragonShadows1978/AI-AfterImage/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DragonShadows1978/AI-AfterImage/releases/tag/v0.1.0
