"""
Language Detection Module - Intelligent programming language detection from file content.

This module provides content-based language detection with:
- Multi-pass detection: extension → shebang → modeline → content patterns
- Tiered confidence scoring
- Polyglot file support (HTML with embedded JS/CSS)
- Fallback to generic 'code' when language cannot be determined

Usage:
    from afterimage.language_detection import detect_language, LanguageResult, LanguageDetector

    # Full detection with metadata
    result = detect_language(code_content, file_path="example.py")
    print(result.language)      # "python"
    print(result.confidence)    # 0.95
    print(result.is_code)       # True

    # Or use the detector class directly
    detector = LanguageDetector()
    result = detector.detect(code_content, file_path="example.py")
"""

from .detector import (
    LanguageDetector,
    LanguageResult,
    detect_language,
    is_code,
    get_detector,
)
from .signatures import (
    LanguageSignature,
    LanguagePattern,
    ConfidenceTier,
    get_signature,
    get_all_extensions,
    get_language_names,
    LANGUAGE_SIGNATURES,
    SPECIAL_FILES,
    SHEBANG_PATTERNS,
)

__all__ = [
    # Main detector
    "LanguageDetector",
    "LanguageResult",
    "detect_language",
    "is_code",
    "get_detector",
    # Signatures
    "LanguageSignature",
    "LanguagePattern",
    "ConfidenceTier",
    "get_signature",
    "get_all_extensions",
    "get_language_names",
    "LANGUAGE_SIGNATURES",
    "SPECIAL_FILES",
    "SHEBANG_PATTERNS",
]
