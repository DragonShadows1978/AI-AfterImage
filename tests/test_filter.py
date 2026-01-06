"""Tests for the Code Filter module."""

import pytest
from afterimage.filter import CodeFilter, DEFAULT_CODE_EXTENSIONS, DEFAULT_SKIP_EXTENSIONS


class TestCodeFilter:
    """Tests for the CodeFilter class."""

    @pytest.fixture
    def filter(self):
        """Create a CodeFilter instance without loading config file."""
        return CodeFilter(load_config_file=False)

    def test_python_files_are_code(self, filter):
        """Test that Python files are recognized as code."""
        assert filter.is_code("/path/to/file.py")
        assert filter.is_code("script.py")
        assert filter.is_code("/project/src/module/app.py")

    def test_javascript_files_are_code(self, filter):
        """Test that JavaScript/TypeScript files are recognized as code."""
        assert filter.is_code("app.js")
        assert filter.is_code("component.jsx")
        assert filter.is_code("types.ts")
        assert filter.is_code("Button.tsx")

    def test_other_languages_are_code(self, filter):
        """Test recognition of various programming languages."""
        assert filter.is_code("main.rs")
        assert filter.is_code("app.go")
        assert filter.is_code("Main.java")
        assert filter.is_code("program.c")
        assert filter.is_code("module.cpp")
        assert filter.is_code("class.cs")
        assert filter.is_code("script.rb")
        assert filter.is_code("page.php")

    def test_markdown_is_not_code(self, filter):
        """Test that markdown files are not recognized as code."""
        assert not filter.is_code("README.md")
        assert not filter.is_code("/docs/guide.md")
        assert not filter.is_code("CHANGELOG.md")

    def test_json_yaml_are_not_code(self, filter):
        """Test that config files are not recognized as code."""
        assert not filter.is_code("package.json")
        assert not filter.is_code("config.yaml")
        assert not filter.is_code("settings.yml")
        assert not filter.is_code("pyproject.toml")

    def test_skip_paths_filter(self, filter):
        """Test that skip paths are respected."""
        assert not filter.is_code("/project/artifacts/report.py")
        assert not filter.is_code("/project/docs/api.py")
        assert not filter.is_code("/project/node_modules/lodash/index.js")
        assert not filter.is_code("/project/__pycache__/module.cpython-310.pyc")

    def test_unknown_extension_without_content(self, filter):
        """Test that unknown extensions without content are skipped."""
        assert not filter.is_code("file.xyz")
        assert not filter.is_code("script.unknown")

    def test_unknown_extension_with_code_content(self, filter):
        """Test that unknown extensions with code content are detected."""
        code_content = """
def hello_world():
    print("Hello, World!")

if __name__ == "__main__":
    hello_world()
"""
        assert filter.is_code("script.unknown", content=code_content)

    def test_unknown_extension_with_text_content(self, filter):
        """Test that unknown extensions with non-code content are skipped."""
        text_content = """
This is a simple text file.
It contains no code, just prose.
Nothing to see here.
"""
        assert not filter.is_code("document.unknown", content=text_content)

    def test_content_heuristics_function_defs(self, filter):
        """Test content heuristics detect function definitions."""
        python_code = "def process_data(items):\n    return [x * 2 for x in items]"
        assert filter.is_code("file.txt", content=python_code)

        js_code = "function processData(items) {\n    return items.map(x => x * 2);\n}"
        assert filter.is_code("file.txt", content=js_code)

    def test_content_heuristics_class_defs(self, filter):
        """Test content heuristics detect class definitions."""
        code = """
class DataProcessor:
    def __init__(self):
        self.data = []

    def process(self):
        return self.data
"""
        assert filter.is_code("file.txt", content=code)

    def test_content_heuristics_imports(self, filter):
        """Test content heuristics detect import statements."""
        code = """
import os
from pathlib import Path
require('lodash')
use std::collections::HashMap;
"""
        assert filter.is_code("file.txt", content=code)

    def test_compound_extensions(self, filter):
        """Test handling of compound extensions like .test.js."""
        assert filter.is_code("component.test.js")
        assert filter.is_code("utils.spec.ts")

    def test_add_code_extension(self, filter):
        """Test adding a new code extension."""
        assert not filter.is_code("script.xyz")

        filter.add_code_extension(".xyz")
        assert filter.is_code("script.xyz")

        # Test without leading dot
        filter.add_code_extension("abc")
        assert filter.is_code("file.abc")

    def test_add_skip_extension(self, filter):
        """Test adding a new skip extension."""
        assert filter.is_code("file.py")

        filter.add_skip_extension(".py")
        assert not filter.is_code("file.py")

    def test_add_skip_path(self, filter):
        """Test adding a new skip path."""
        assert filter.is_code("/project/src/app.py")

        filter.add_skip_path("src/")
        assert not filter.is_code("/project/src/app.py")

    def test_get_config(self, filter):
        """Test getting current configuration."""
        config = filter.get_config()

        assert "code_extensions" in config
        assert "skip_extensions" in config
        assert "skip_paths" in config

        assert ".py" in config["code_extensions"]
        assert ".md" in config["skip_extensions"]

    def test_custom_extensions(self):
        """Test creating filter with custom extensions."""
        custom = CodeFilter(
            code_extensions={".custom"},
            skip_extensions={".ignore"},
            skip_paths=["custom_skip/"],
            load_config_file=False
        )

        assert custom.is_code("file.custom")
        assert not custom.is_code("file.py")  # Not in custom list
        assert not custom.is_code("file.ignore")
        assert not custom.is_code("custom_skip/file.custom")

    def test_dotfiles(self, filter):
        """Test handling of dotfiles."""
        # Most dotfiles are config, not code
        assert not filter.is_code(".gitignore")
        assert not filter.is_code(".env")

    def test_empty_content(self, filter):
        """Test that empty content is not considered code."""
        assert not filter.is_code("file.unknown", content="")
        assert not filter.is_code("file.unknown", content="   ")

    def test_minified_files(self, filter):
        """Test that minified files are skipped."""
        assert not filter.is_code("bundle.min.js")
        assert not filter.is_code("styles.min.css")
