"""Tests for the Transcript Extractor module."""

import json
import tempfile
import pytest
from pathlib import Path

from afterimage.extract import (
    TranscriptExtractor,
    CodeChange,
    extract_code_symbols,
)


class TestExtractCodeSymbols:
    """Tests for the extract_code_symbols function."""

    def test_python_functions(self):
        """Test extraction of Python function names."""
        code = """
def hello():
    pass

def process_data(items):
    return items
"""
        symbols = extract_code_symbols(code)
        assert "hello" in symbols
        assert "process_data" in symbols

    def test_python_classes(self):
        """Test extraction of Python class names."""
        code = """
class UserService:
    def __init__(self):
        pass

class DataProcessor(BaseClass):
    pass
"""
        symbols = extract_code_symbols(code)
        assert "UserService" in symbols
        assert "DataProcessor" in symbols
        assert "__init__" in symbols

    def test_javascript_functions(self):
        """Test extraction of JavaScript function names."""
        code = """
function processData(items) {
    return items.map(x => x * 2);
}

const handleClick = () => {
    console.log('clicked');
}
"""
        symbols = extract_code_symbols(code)
        assert "processData" in symbols

    def test_rust_functions(self):
        """Test extraction of Rust function names."""
        code = """
fn main() {
    println!("Hello");
}

fn process_data<T>(items: Vec<T>) -> Vec<T> {
    items
}
"""
        symbols = extract_code_symbols(code)
        assert "main" in symbols
        assert "process_data" in symbols

    def test_go_functions(self):
        """Test extraction of Go function names."""
        code = """
func main() {
    fmt.Println("Hello")
}

func (s *Server) HandleRequest(w http.ResponseWriter, r *http.Request) {
    // ...
}
"""
        symbols = extract_code_symbols(code)
        assert "main" in symbols
        assert "HandleRequest" in symbols

    def test_no_duplicates(self):
        """Test that duplicate symbols are removed."""
        code = """
def process():
    pass

def process():
    pass
"""
        symbols = extract_code_symbols(code)
        assert symbols.count("process") == 1


class TestTranscriptExtractor:
    """Tests for the TranscriptExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a TranscriptExtractor instance."""
        return TranscriptExtractor(context_lines=3)

    @pytest.fixture
    def temp_jsonl(self):
        """Create a temporary JSONL file."""
        def create(entries):
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.jsonl', delete=False
            ) as f:
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
                return Path(f.name)
        return create

    def test_extract_write_tool_format1(self, extractor, temp_jsonl):
        """Test extraction of Write tool in format 1."""
        entries = [
            {"role": "user", "content": "Create a hello function"},
            {
                "type": "tool_use",
                "name": "Write",
                "input": {
                    "file_path": "/test/hello.py",
                    "content": "def hello():\n    print('Hello')"
                }
            },
            {"role": "assistant", "content": "Done!"}
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 1
        change = changes[0]
        assert change.file_path == "/test/hello.py"
        assert change.tool_type == "Write"
        assert "def hello" in change.new_code
        assert change.old_code is None

        # Cleanup
        file_path.unlink()

    def test_extract_edit_tool(self, extractor, temp_jsonl):
        """Test extraction of Edit tool."""
        entries = [
            {"role": "user", "content": "Fix the typo"},
            {
                "type": "tool_use",
                "name": "Edit",
                "input": {
                    "file_path": "/test/file.py",
                    "old_string": "def helo():",
                    "new_string": "def hello():"
                }
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 1
        change = changes[0]
        assert change.file_path == "/test/file.py"
        assert change.tool_type == "Edit"
        assert change.old_code == "def helo():"
        assert change.new_code == "def hello():"

        file_path.unlink()

    def test_extract_multiple_changes(self, extractor, temp_jsonl):
        """Test extraction of multiple changes."""
        entries = [
            {
                "type": "tool_use",
                "name": "Write",
                "input": {"file_path": "/a.py", "content": "# a"}
            },
            {
                "type": "tool_use",
                "name": "Write",
                "input": {"file_path": "/b.py", "content": "# b"}
            },
            {
                "type": "tool_use",
                "name": "Edit",
                "input": {
                    "file_path": "/c.py",
                    "old_string": "old",
                    "new_string": "new"
                }
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 3
        assert changes[0].file_path == "/a.py"
        assert changes[1].file_path == "/b.py"
        assert changes[2].file_path == "/c.py"

        file_path.unlink()

    def test_context_extraction(self, extractor, temp_jsonl):
        """Test that context is extracted from surrounding messages."""
        entries = [
            {"role": "user", "content": "Add authentication middleware"},
            {"role": "assistant", "content": "I'll create the auth middleware..."},
            {
                "type": "tool_use",
                "name": "Write",
                "input": {"file_path": "/auth.py", "content": "def auth(): pass"}
            },
            {"role": "assistant", "content": "Created the authentication."}
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 1
        context = changes[0].context

        assert "authentication" in context.lower()

        file_path.unlink()

    def test_skip_non_write_edit_tools(self, extractor, temp_jsonl):
        """Test that non-Write/Edit tools are skipped."""
        entries = [
            {
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/test.py"}
            },
            {
                "type": "tool_use",
                "name": "Bash",
                "input": {"command": "ls -la"}
            },
            {
                "type": "tool_use",
                "name": "Write",
                "input": {"file_path": "/test.py", "content": "code"}
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 1
        assert changes[0].tool_type == "Write"

        file_path.unlink()

    def test_format2_tool_input(self, extractor, temp_jsonl):
        """Test extraction from format 2: {"tool": "...", "input": {...}}."""
        entries = [
            {
                "tool": "Write",
                "input": {"file_path": "/test.py", "content": "code"}
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 1
        assert changes[0].file_path == "/test.py"

        file_path.unlink()

    def test_format3_nested_content(self, extractor, temp_jsonl):
        """Test extraction from format 3: nested in content array."""
        entries = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Creating file..."},
                    {
                        "type": "tool_use",
                        "name": "Write",
                        "input": {"file_path": "/test.py", "content": "code"}
                    }
                ]
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert len(changes) == 1

        file_path.unlink()

    def test_handles_malformed_jsonl(self, extractor, temp_jsonl):
        """Test that malformed JSON lines are skipped."""
        # Create file manually to include malformed line
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.jsonl', delete=False
        ) as f:
            f.write('{"type": "tool_use", "name": "Write", "input": {"file_path": "/a.py", "content": "a"}}\n')
            f.write('not valid json\n')
            f.write('{"type": "tool_use", "name": "Write", "input": {"file_path": "/b.py", "content": "b"}}\n')
            file_path = Path(f.name)

        changes = extractor.extract_from_file(file_path)

        # Should still extract the valid entries
        assert len(changes) == 2

        file_path.unlink()

    def test_session_id_from_entry(self, extractor, temp_jsonl):
        """Test that session ID is extracted from entries."""
        entries = [
            {"session_id": "test_session_123"},
            {
                "type": "tool_use",
                "name": "Write",
                "input": {"file_path": "/test.py", "content": "code"}
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        assert changes[0].session_id == "test_session_123"

        file_path.unlink()

    def test_session_id_from_filename(self, extractor, temp_jsonl):
        """Test that session ID falls back to filename."""
        entries = [
            {
                "type": "tool_use",
                "name": "Write",
                "input": {"file_path": "/test.py", "content": "code"}
            }
        ]

        file_path = temp_jsonl(entries)
        changes = extractor.extract_from_file(file_path)

        # Session ID should be derived from filename
        assert changes[0].session_id == file_path.stem

        file_path.unlink()

    def test_empty_file(self, extractor, temp_jsonl):
        """Test handling of empty transcript file."""
        file_path = temp_jsonl([])
        changes = extractor.extract_from_file(file_path)

        assert changes == []

        file_path.unlink()
