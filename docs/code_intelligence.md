# Code Intelligence Features

AI-AfterImage v0.4.0 introduces powerful code intelligence capabilities integrated from AtlasForge R&D missions:

- **Language Detection** - Content-based language identification
- **AST Parsing** - Tree-sitter based syntax analysis
- **Semantic Index** - IDE-like go-to-definition and find-references

## Installation

The code intelligence features require optional dependencies:

```bash
# Install with AST/semantic features
pip install ai-afterimage[ast]

# Install everything
pip install ai-afterimage[all]
```

## Language Detection

Detects programming languages from source code content with confidence scoring.

### Features

- 20+ supported languages
- Pattern-based detection with confidence scoring
- Polyglot file support (HTML with embedded JS/CSS)
- Multiple detection methods (extension, shebang, content patterns)

### Usage

```python
from afterimage import LanguageDetector, LanguageResult

detector = LanguageDetector()

# Detect from content
result = detector.detect('''
def hello():
    print("Hello, World!")
''')

print(result.language)      # "python"
print(result.confidence)    # 0.95
print(result.method)        # DetectionMethod.CONTENT
```

### Supported Languages

| Language | Extensions | Confidence Features |
|----------|------------|---------------------|
| Python | .py, .pyw, .pyi | imports, def/class, decorators |
| JavaScript | .js, .mjs, .cjs | const/let, arrow functions, require |
| TypeScript | .ts, .tsx | type annotations, interfaces |
| Rust | .rs | fn, impl, use, macro_rules |
| Go | .go | func, package, import |
| C | .c, .h | #include, main(), pointers |
| C++ | .cpp, .hpp, .cc | class, template, namespace |
| Java | .java | public class, import |
| And more... | | |

## AST Parser

Tree-sitter based AST parsing with incremental support.

### Features

- Multi-language support (Python, JavaScript, TypeScript, Rust, Go, C/C++)
- Incremental parsing for performance
- Semantic information extraction (functions, classes, imports)
- Language-specific node visitors

### Usage

```python
from afterimage import get_ast_parser

ASTParser = get_ast_parser()
parser = ASTParser()

# Parse source code
result = parser.parse('''
class Calculator:
    def add(self, a, b):
        return a + b
''', language="python")

# Access AST
print(result.root_node.type)  # "module"

# Get semantic information
for func in result.semantic.functions:
    print(f"Function: {func.name} at line {func.line}")
```

### Supported Languages

| Language | Parser | Features |
|----------|--------|----------|
| Python | tree-sitter-python | Functions, classes, imports, decorators |
| JavaScript | tree-sitter-javascript | Functions, classes, imports, exports |
| TypeScript | tree-sitter-typescript | Types, interfaces, generics |
| Rust | tree-sitter-rust | Functions, structs, traits, macros |
| Go | tree-sitter-go | Functions, types, interfaces |
| C/C++ | tree-sitter-c/cpp | Functions, structs, macros |

## Semantic Index

IDE-like semantic intelligence for code navigation.

### Features

- **Go to Definition** - Jump to symbol definitions
- **Find References** - Find all usages of a symbol
- **Hover Information** - Type and documentation info
- **Call Graph** - Function call relationships
- **Type Inference** - Basic type propagation

### Usage

```python
from afterimage import get_semantic_index

SemanticIndex = get_semantic_index()
index = SemanticIndex()

# Index a project
index.index_directory("/path/to/project")

# Go to definition
definition = index.goto_definition(
    file_path="/path/to/file.py",
    line=10,
    column=5
)
if definition:
    print(f"Defined at {definition.file_path}:{definition.line}")

# Find references
references = index.find_references(
    file_path="/path/to/file.py",
    line=10,
    column=5
)
for ref in references:
    print(f"Used at {ref.file_path}:{ref.line}")

# Get hover info
hover = index.hover(
    file_path="/path/to/file.py",
    line=10,
    column=5
)
if hover:
    print(hover.documentation)
    print(hover.type_info)
```

### Symbol Kinds

The semantic index tracks these symbol types:

- `FUNCTION` - Function definitions
- `CLASS` - Class definitions
- `METHOD` - Method definitions
- `VARIABLE` - Variable assignments
- `PARAMETER` - Function parameters
- `IMPORT` - Import statements
- `MODULE` - Module definitions
- `CONSTANT` - Constant values
- `TYPE_ALIAS` - Type aliases
- `INTERFACE` - Interface definitions (TypeScript)
- `ENUM` - Enum definitions

## Integration with AfterImage

These features integrate with AfterImage's episodic memory:

```python
from afterimage import KnowledgeBase, LanguageDetector

kb = KnowledgeBase()
detector = LanguageDetector()

# When storing code snippets, detect language
code = "def example(): pass"
result = detector.detect(code)

# Store with language metadata
kb.store(
    content=code,
    metadata={
        "language": result.language,
        "confidence": result.confidence
    }
)
```

## Performance Considerations

### AST Parsing

- Parser instances are cached per language
- Incremental parsing reuses previous parse trees
- File hashing prevents re-parsing unchanged files

### Semantic Index

- Symbol tables are per-file and incrementally updated
- Definition cache with LRU eviction
- Parallel file indexing support

## Architecture

```
afterimage/
├── language_detection/     # Language detection
│   ├── detector.py         # Main detector class
│   ├── patterns.py         # Language patterns
│   └── polyglot.py         # Multi-language file support
├── ast_parser/             # AST parsing
│   ├── parser.py           # Parser factory
│   ├── base_parser.py      # Base parser class
│   ├── python_parser.py    # Python-specific
│   ├── javascript_parser.py # JS/TS-specific
│   └── rust_parser.py      # Rust-specific
└── semantic_index/         # Semantic intelligence
    ├── semantic_index.py   # Main coordinator
    ├── definition_resolver.py # Go-to-definition
    ├── references_finder.py   # Find references
    ├── hover_provider.py      # Hover info
    ├── symbol_table.py        # Per-file symbols
    ├── call_graph.py          # Call relationships
    └── visitors/              # Language visitors
        ├── python_visitor.py
        ├── javascript_visitor.py
        └── rust_visitor.py
```

## Credits

These features were developed as part of AtlasForge R&D missions:

- **mission_5ecc519b** - Language Detection
- **mission_9b9a40cb** - AST Parser
- **mission_408d146a** - Semantic Intelligence (GoToDefinition)
