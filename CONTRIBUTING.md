# Contributing to AI-AfterImage

Thank you for your interest in contributing to AI-AfterImage! This document provides guidelines for contributing to the project.

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/AI-AfterImage.git
cd AI-AfterImage
```

### Development Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

# Install in development mode with all dependencies
pip install -e ".[dev,embeddings]"
```

### Verify Setup

```bash
# Run tests to ensure everything works
pytest

# Run with coverage
pytest --cov=afterimage
```

## Code Style

We use automated formatting and linting:

### Formatting

- **black** for code formatting (line length 88)
- **isort** for import sorting

```bash
# Format code
black afterimage tests
isort afterimage tests
```

### Linting

- **ruff** for fast linting

```bash
# Run linter
ruff check afterimage tests
```

### Type Hints

- Use type hints for function signatures
- Run mypy for type checking (optional but recommended)

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_search.py

# Run tests matching a pattern
pytest -k "test_search"

# Run with coverage report
pytest --cov=afterimage --cov-report=html
```

### Coverage Requirements

- New code should maintain or improve coverage
- Target: 80%+ coverage for new features
- All public functions should have tests

### Test Structure

```
tests/
  test_kb.py           # Knowledge base tests
  test_filter.py       # Code filter tests
  test_extract.py      # Transcript extraction tests
  test_search.py       # Search functionality tests
  test_inject.py       # Context injection tests
  test_integration.py  # End-to-end integration tests
```

### Writing Tests

```python
import pytest
from afterimage.kb import KnowledgeBase

def test_kb_store_basic(tmp_path):
    """Test basic storage functionality."""
    kb = KnowledgeBase(db_path=tmp_path / "test.db")

    entry_id = kb.store(
        file_path="test.py",
        new_code="def hello(): pass",
        context="Added hello function"
    )

    assert entry_id is not None
    assert len(entry_id) > 0
```

## Pull Request Process

### Before Submitting

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and add tests
4. **Run the test suite**:
   ```bash
   pytest
   ```
5. **Format your code**:
   ```bash
   black afterimage tests
   isort afterimage tests
   ```
6. **Commit** with a clear message

### Commit Messages

Use clear, descriptive commit messages:

```
Add hybrid search with semantic similarity

- Implement cosine similarity for embeddings
- Add configurable FTS/semantic weights
- Add tests for search scoring
```

### Submitting

1. Push your branch to your fork
2. Open a Pull Request against `main`
3. Fill out the PR template
4. Wait for CI checks to pass
5. Address any review feedback

### PR Requirements

- [ ] All tests pass
- [ ] New features have tests
- [ ] Code is formatted (black, isort)
- [ ] No new linting errors
- [ ] Documentation updated if needed

## Issue Reporting

### Bug Reports

When reporting bugs, include:

1. **Python version** (`python --version`)
2. **OS** (Linux, macOS, Windows)
3. **Steps to reproduce**
4. **Expected vs actual behavior**
5. **Error messages** (full traceback)

### Feature Requests

For feature requests:

1. **Use case** - Why do you need this?
2. **Proposed solution** - How should it work?
3. **Alternatives** - Other approaches considered

## Development Tips

### Local Testing with Transcripts

```bash
# Ingest test transcripts
afterimage ingest -d tests/fixtures/transcripts -v

# Search your test data
afterimage search "test query"

# Clear and reset
afterimage clear -y
```

### Testing Hooks

The hook script can be tested locally:

```bash
# Simulate pre-tool hook
echo '{"tool": "Write", "params": {"file_path": "test.py"}}' | \
  python hooks/afterimage_hook.py --stage pre

# Simulate post-tool hook
echo '{"tool": "Write", "params": {"file_path": "test.py", "content": "print(1)"}}' | \
  python hooks/afterimage_hook.py --stage post
```

### Debugging Embeddings

```python
from afterimage.embeddings import EmbeddingGenerator

gen = EmbeddingGenerator()
print(f"Model: {gen.model_name}")
print(f"Device: {gen.device}")

# Test embedding
vec = gen.embed_text("test code")
print(f"Embedding shape: {vec.shape}")
```

## Questions?

- Open an issue for questions
- Check existing issues first
- Be patient and respectful

Thank you for contributing!
