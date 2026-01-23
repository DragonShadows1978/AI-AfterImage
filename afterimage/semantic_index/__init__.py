"""
Semantic analyzers for IDE-like intelligence.

Provides:
- DefinitionResolver: Go to definition
- ReferencesFinder: Find all references
- HoverProvider: Hover information
- SemanticIndex: Project-level coordination
- TypeInferencer: Type inference for Python

Note: This module requires tree-sitter dependencies. Install with:
    pip install ai-afterimage[ast]
"""

# Models are available without tree-sitter
from .models import (
    Symbol,
    SymbolKind,
    Reference,
    Location,
    LocationRange,
    Scope,
    ScopeKind,
    TypeInfo,
    CallSite,
    DefinitionResult,
    ReferenceResult,
    HoverInfo,
)

__all__ = [
    # Models (always available)
    "Symbol",
    "SymbolKind",
    "Reference",
    "Location",
    "LocationRange",
    "Scope",
    "ScopeKind",
    "TypeInfo",
    "CallSite",
    "DefinitionResult",
    "ReferenceResult",
    "HoverInfo",
    # Analyzers (lazy loaded, require tree-sitter)
    "DefinitionResolver",
    "ReferencesFinder",
    "HoverProvider",
    "SemanticIndex",
    "TypeInferencer",
    "TypePropagator",
    "InferenceResult",
]


def __getattr__(name):
    """Lazy loading for tree-sitter dependent classes."""
    if name == "DefinitionResolver":
        from .definition_resolver import DefinitionResolver
        return DefinitionResolver
    elif name == "ReferencesFinder":
        from .references_finder import ReferencesFinder
        return ReferencesFinder
    elif name == "HoverProvider":
        from .hover_provider import HoverProvider
        return HoverProvider
    elif name == "SemanticIndex":
        from .semantic_index import SemanticIndex
        return SemanticIndex
    elif name in ("TypeInferencer", "TypePropagator", "InferenceResult"):
        from . import type_inference
        return getattr(type_inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
