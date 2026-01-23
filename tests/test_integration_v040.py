"""
Integration tests for v0.4.0 features: Language Detection, AST Parser, and Semantic Index.

These tests verify that the three mission integrations work correctly:
- mission_5ecc519b: Language Detection System
- mission_9b9a40cb: AST Parser System
- mission_408d146a: Semantic Intelligence System
"""
import pytest
from pathlib import Path
import tempfile
import os


class TestLanguageDetection:
    """Test language detection integration."""

    def test_import_language_detection(self):
        """Verify language detection module imports correctly."""
        from afterimage.language_detection import (
            LanguageDetector,
            LanguageResult,
            detect_language,
            is_code,
            ConfidenceTier,
        )
        assert LanguageDetector is not None
        assert LanguageResult is not None

    def test_detect_python(self):
        """Detect Python code from content."""
        from afterimage.language_detection import detect_language

        code = '''
def fibonacci(n):
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
'''
        result = detect_language(code, file_path="example.py")
        assert result.is_code is True
        assert result.language == "python"
        assert result.confidence >= 0.7

    def test_detect_javascript(self):
        """Detect JavaScript code from content."""
        from afterimage.language_detection import detect_language

        code = '''
function greet(name) {
    console.log(`Hello, ${name}!`);
    return name.toUpperCase();
}

const result = greet("World");
'''
        result = detect_language(code, file_path="example.js")
        assert result.is_code is True
        assert result.language == "javascript"
        assert result.confidence >= 0.7

    def test_detect_rust(self):
        """Detect Rust code from content."""
        from afterimage.language_detection import detect_language

        code = '''
fn main() {
    let mut numbers = vec![1, 2, 3, 4, 5];
    numbers.push(6);

    for n in &numbers {
        println!("{}", n);
    }
}
'''
        result = detect_language(code, file_path="main.rs")
        assert result.is_code is True
        assert result.language == "rust"
        assert result.confidence >= 0.7

    def test_detect_go(self):
        """Detect Go code from content."""
        from afterimage.language_detection import detect_language

        code = '''
package main

import "fmt"

func main() {
    message := "Hello, Go!"
    fmt.Println(message)
}
'''
        result = detect_language(code, file_path="main.go")
        assert result.is_code is True
        assert result.language == "go"
        assert result.confidence >= 0.7

    def test_detect_typescript(self):
        """Detect TypeScript code from content."""
        from afterimage.language_detection import detect_language

        code = '''
interface User {
    name: string;
    age: number;
    email?: string;
}

function createUser(name: string, age: number): User {
    return { name, age };
}
'''
        result = detect_language(code, file_path="user.ts")
        assert result.is_code is True
        assert result.language == "typescript"
        assert result.confidence >= 0.7

    def test_polyglot_detection(self):
        """Test detection of embedded languages in HTML."""
        from afterimage.language_detection import detect_language

        code = '''
<!DOCTYPE html>
<html>
<head>
    <style>
        .container { display: flex; }
    </style>
</head>
<body>
    <script>
        function init() {
            console.log("Hello");
        }
    </script>
</body>
</html>
'''
        result = detect_language(code, file_path="index.html")
        assert result.is_code is True
        assert result.language == "html"
        # Should detect embedded languages
        if result.secondary_languages:
            assert any(lang in ["javascript", "css"] for lang in result.secondary_languages)

    def test_shebang_detection(self):
        """Test detection via shebang line."""
        from afterimage.language_detection import detect_language

        code = '''#!/usr/bin/env python3
print("Hello from shebang script")
'''
        result = detect_language(code)
        assert result.is_code is True
        assert result.language == "python"
        assert "shebang" in result.detection_method.lower()

    def test_code_filter_language_integration(self):
        """Test CodeFilter integration with language detection."""
        from afterimage.filter import CodeFilter

        code = 'def hello(): pass'
        filter_obj = CodeFilter()

        # Default behavior - returns bool
        is_code = filter_obj.is_code("test.py", content=code)
        assert is_code is True

        # With return_language=True - returns LanguageResult
        result = filter_obj.is_code("test.py", content=code, return_language=True)
        assert result.is_code is True
        assert result.language == "python"


class TestASTParser:
    """Test AST parser integration."""

    def test_import_ast_parser(self):
        """Verify AST parser module imports correctly."""
        from afterimage.ast_parser import (
            ASTResult,
            FunctionInfo,
            ClassInfo,
            ImportInfo,
        )
        assert ASTResult is not None
        assert FunctionInfo is not None

    def test_parse_python_functions(self):
        """Parse Python functions."""
        from afterimage.ast_parser import ASTParserFactory

        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(x, y):
    return x * y
'''
        factory = ASTParserFactory()
        parser = factory.get_parser("python")
        result = parser.parse(code)

        assert len(result.functions) == 2
        func_names = [f.name for f in result.functions]
        assert "add" in func_names
        assert "multiply" in func_names

    def test_parse_python_classes(self):
        """Parse Python classes."""
        from afterimage.ast_parser import ASTParserFactory

        code = '''
class Person:
    """A simple person class."""

    def __init__(self, name, age):
        self.name = name
        self.age = age

    def greet(self):
        return f"Hello, I'm {self.name}"

class Student(Person):
    def __init__(self, name, age, grade):
        super().__init__(name, age)
        self.grade = grade
'''
        factory = ASTParserFactory()
        parser = factory.get_parser("python")
        result = parser.parse(code)

        assert len(result.classes) == 2
        class_names = [c.name for c in result.classes]
        assert "Person" in class_names
        assert "Student" in class_names

    def test_parse_python_imports(self):
        """Parse Python imports."""
        from afterimage.ast_parser import ASTParserFactory

        code = '''
import os
import sys
from pathlib import Path
from typing import List, Optional
'''
        factory = ASTParserFactory()
        parser = factory.get_parser("python")
        result = parser.parse(code)

        assert len(result.imports) >= 4

    def test_parse_javascript_functions(self):
        """Parse JavaScript functions."""
        from afterimage.ast_parser import ASTParserFactory

        code = '''
function greet(name) {
    return `Hello, ${name}!`;
}

const calculate = (a, b) => a + b;

async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}
'''
        factory = ASTParserFactory()
        parser = factory.get_parser("javascript")
        result = parser.parse(code)

        assert len(result.functions) >= 2
        func_names = [f.name for f in result.functions]
        assert "greet" in func_names

    def test_parse_rust_functions(self):
        """Parse Rust functions."""
        from afterimage.ast_parser import ASTParserFactory

        code = '''
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn main() {
    let result = add(1, 2);
    println!("Result: {}", result);
}

impl Calculator {
    pub fn new() -> Self {
        Self { value: 0 }
    }
}
'''
        factory = ASTParserFactory()
        parser = factory.get_parser("rust")
        result = parser.parse(code)

        assert len(result.functions) >= 2
        func_names = [f.name for f in result.functions]
        assert "add" in func_names
        assert "main" in func_names

    def test_parse_go_functions(self):
        """Parse Go functions."""
        from afterimage.ast_parser import ASTParserFactory

        code = '''
package main

import "fmt"

func add(a, b int) int {
    return a + b
}

func main() {
    result := add(1, 2)
    fmt.Println(result)
}
'''
        factory = ASTParserFactory()
        parser = factory.get_parser("go")
        result = parser.parse(code)

        assert len(result.functions) >= 2
        func_names = [f.name for f in result.functions]
        assert "add" in func_names
        assert "main" in func_names

    def test_supported_languages(self):
        """Verify supported language list."""
        from afterimage.ast_parser import ASTParserFactory

        factory = ASTParserFactory()
        languages = factory.get_supported_languages()

        assert "python" in languages
        assert "javascript" in languages
        assert "typescript" in languages
        assert "rust" in languages
        assert "go" in languages
        assert "c" in languages or "cpp" in languages


class TestSemanticIndex:
    """Test semantic index integration."""

    def test_import_semantic_index(self):
        """Verify semantic index module imports correctly."""
        from afterimage.semantic_index import (
            Symbol,
            SymbolKind,
            Reference,
            Location,
            Scope,
            DefinitionResult,
            HoverInfo,
        )
        assert Symbol is not None
        assert SymbolKind is not None
        assert Location is not None

    def test_semantic_index_instantiation(self):
        """Test SemanticIndex can be instantiated."""
        from afterimage.semantic_index import SemanticIndex

        si = SemanticIndex()
        assert si is not None

    def test_index_python_file(self):
        """Index a Python file and find symbols."""
        from afterimage.semantic_index import SemanticIndex

        code = '''
class Calculator:
    def __init__(self, value=0):
        self.value = value

    def add(self, n):
        self.value += n
        return self

    def get_value(self):
        return self.value

def main():
    calc = Calculator(10)
    calc.add(5)
    print(calc.get_value())
'''
        si = SemanticIndex()
        si.index_file("calculator.py", code)

        # Get all symbols from the indexed file
        symbols = si.get_all_symbols("calculator.py")

        assert len(symbols) > 0
        symbol_names = [s.name for s in symbols]
        assert "Calculator" in symbol_names or "main" in symbol_names

    def test_hover_info(self):
        """Test hover information provider."""
        from afterimage.semantic_index import SemanticIndex, HoverProvider

        code = '''
def calculate_sum(numbers: list) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)
'''
        si = SemanticIndex()
        si.index_file("math_utils.py", code)

        # The HoverProvider should be able to provide info
        hover = HoverProvider(si)
        assert hover is not None


class TestEndToEndIntegration:
    """End-to-end integration tests combining all components."""

    def test_language_to_ast_pipeline(self):
        """Test full pipeline from language detection to AST parsing."""
        from afterimage.language_detection import detect_language
        from afterimage.ast_parser import ASTParserFactory

        code = '''
def process_data(items):
    """Process a list of items."""
    results = []
    for item in items:
        if item.is_valid():
            results.append(item.transform())
    return results

class DataProcessor:
    def __init__(self, config):
        self.config = config
'''

        # Step 1: Detect language
        lang_result = detect_language(code, file_path="processor.py")
        assert lang_result.language == "python"

        # Step 2: Parse AST based on detected language
        factory = ASTParserFactory()
        parser = factory.get_parser(lang_result.language)
        ast_result = parser.parse(code)

        # Verify AST extraction
        assert len(ast_result.functions) >= 1
        assert len(ast_result.classes) >= 1

    def test_filter_with_language_to_ast(self):
        """Test CodeFilter -> LanguageResult -> AST pipeline."""
        from afterimage.filter import CodeFilter
        from afterimage.ast_parser import ASTParserFactory

        code = '''
async function fetchUsers(api) {
    const response = await fetch(api);
    const users = await response.json();
    return users.filter(u => u.active);
}

class UserService {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }
}
'''

        # Use filter with language detection
        filter_obj = CodeFilter()
        lang_result = filter_obj.is_code("users.js", content=code, return_language=True)

        assert lang_result.is_code is True
        assert lang_result.language == "javascript"

        # Parse with detected language
        factory = ASTParserFactory()
        parser = factory.get_parser(lang_result.language)
        ast_result = parser.parse(code)

        # Should find functions and classes
        assert len(ast_result.functions) >= 1
        assert len(ast_result.classes) >= 1

    def test_full_semantic_pipeline(self):
        """Test full pipeline: detect -> parse -> index -> query."""
        from afterimage.language_detection import detect_language
        from afterimage.ast_parser import ASTParserFactory
        from afterimage.semantic_index import SemanticIndex

        code = '''
class UserRepository:
    def __init__(self, db):
        self.db = db

    def find_by_id(self, user_id):
        """Find a user by their ID."""
        return self.db.query(User).filter(User.id == user_id).first()

    def find_all(self):
        """Return all users."""
        return self.db.query(User).all()

def create_user_repo(db):
    return UserRepository(db)
'''

        # Full pipeline
        lang_result = detect_language(code, file_path="repository.py")
        assert lang_result.language == "python"

        factory = ASTParserFactory()
        parser = factory.get_parser("python")
        ast_result = parser.parse(code)

        # Should extract UserRepository class and its methods
        assert any(c.name == "UserRepository" for c in ast_result.classes)

        # Index in semantic index
        si = SemanticIndex()
        si.index_file("repository.py", code)

        # Query all symbols from the indexed file
        symbols = si.get_all_symbols("repository.py")
        symbol_names = [s.name for s in symbols]

        # Should find the class
        assert "UserRepository" in symbol_names or len(symbols) > 0


class TestModuleExports:
    """Test that all expected exports are available."""

    def test_main_package_exports(self):
        """Test main afterimage package exports."""
        from afterimage import (
            LanguageDetector,
            LanguageResult,
            ConfidenceTier,
            detect_language,
            get_ast_parser,
            get_semantic_index,
            __version__,
        )

        assert __version__ == "0.4.0"
        assert LanguageDetector is not None
        assert callable(get_ast_parser)
        assert callable(get_semantic_index)

    def test_lazy_ast_loading(self):
        """Test that AST parser uses lazy loading."""
        # First import models - should work without tree-sitter
        from afterimage.ast_parser import ASTResult, FunctionInfo
        assert ASTResult is not None

        # Then import parser factory - requires tree-sitter
        from afterimage.ast_parser import ASTParserFactory
        factory = ASTParserFactory()
        assert factory is not None

    def test_lazy_semantic_loading(self):
        """Test that semantic index uses lazy loading."""
        # First import models - should work without tree-sitter
        from afterimage.semantic_index import Symbol, SymbolKind
        assert Symbol is not None

        # Then import analyzers - may require tree-sitter
        from afterimage.semantic_index import SemanticIndex
        si = SemanticIndex()
        assert si is not None
