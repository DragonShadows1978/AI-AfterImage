"""
Language Detector - Intelligent programming language detection from file content.

This module provides content-based language detection with:
- Multi-pass detection: extension → shebang → modeline → content patterns
- Tiered confidence scoring
- Polyglot file support (HTML with embedded JS/CSS)
- Fallback to generic 'code' when language cannot be determined

DETECTION HIERARCHY (in order of confidence):
============================================
1. Special filenames (Makefile, Dockerfile) - 95% confidence
2. Shebang lines (#!/usr/bin/env python) - 90% confidence
3. Vim/Emacs modelines - 90% confidence
4. File extensions (.py, .rs, .go) - 85% confidence
5. Content pattern matching - 50-95% confidence (based on matches)

EARLY-EXIT OPTIMIZATION:
=======================
For performance, the detector implements early-exit when a language scores > 0.90.
Languages are checked in order of popularity (Python, JS, Java first) to maximize
early-exit effectiveness on common code.

USAGE:
=====
    from afterimage.language_detection import detect_language, is_code

    # Full detection with metadata
    result = detect_language(code_content, file_path="example.py")
    print(result.language)      # "python"
    print(result.confidence)    # 0.95
    print(result.is_code)       # True

    # Simple boolean check
    if is_code(content):
        process_as_code(content)

POLYGLOT DETECTION:
==================
For files like HTML that embed other languages:
    result = detect_language(html_with_js)
    result.language            # "html"
    result.secondary_languages # ["javascript", "css"]
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from .signatures import (
    LANGUAGE_SIGNATURES,
    SPECIAL_FILES,
    SHEBANG_PATTERNS,
    MODELINE_PATTERNS,
    POLYGLOT_PATTERNS,
    SHARED_PATTERNS,
    get_all_extensions,
)


@dataclass
class LanguageResult:
    """Result of language detection with metadata."""
    is_code: bool
    language: Optional[str]  # e.g., "python", "rust", None
    confidence: float        # 0.0 to 1.0
    secondary_languages: List[str] = field(default_factory=list)  # for polyglot files
    detection_method: str = "unknown"  # "extension", "shebang", "modeline", "content", "special_file"

    # Additional details for debugging/analysis
    matched_patterns: List[str] = field(default_factory=list)
    pattern_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "is_code": self.is_code,
            "language": self.language,
            "confidence": self.confidence,
            "secondary_languages": self.secondary_languages,
            "detection_method": self.detection_method,
            "matched_patterns": self.matched_patterns,
            "pattern_scores": self.pattern_scores,
        }

    def __bool__(self) -> bool:
        """Allow using result in boolean context (for is_code check)."""
        return self.is_code


class LanguageDetector:
    """
    Detects programming language from file content using hierarchical detection.

    Detection hierarchy (in order of confidence):
    1. Special filenames (Makefile, Dockerfile, etc.) - highest confidence
    2. Shebang lines (#!/usr/bin/env python) - high confidence
    3. Vim/Emacs modelines - high confidence
    4. File extensions - high confidence for known extensions
    5. Content pattern matching - variable confidence based on matches

    For unknown extensions or content-only detection, uses pattern matching
    with confidence scoring based on pattern specificity and match count.
    """

    # Minimum confidence threshold to report a language (vs "unknown")
    MIN_CONFIDENCE_THRESHOLD = 0.3

    # Confidence level for detection methods
    DETECTION_CONFIDENCE = {
        "special_file": 0.95,
        "shebang": 0.90,
        "modeline": 0.90,
        "extension": 0.85,
        "content": 0.50,  # base, modified by pattern matches
    }

    def __init__(
        self,
        min_confidence: float = 0.3,
        detect_polyglot: bool = True,
        code_threshold: float = 0.4,  # min confidence to consider something "code"
    ):
        """
        Initialize the language detector.

        Args:
            min_confidence: Minimum confidence to return a language (vs None)
            detect_polyglot: Whether to detect embedded languages (JS in HTML, etc.)
            code_threshold: Minimum confidence to consider content as code
        """
        self.min_confidence = min_confidence
        self.detect_polyglot = detect_polyglot
        self.code_threshold = code_threshold

        # Build extension lookup
        self._ext_to_lang = get_all_extensions()

        # Compile shebang pattern
        self._shebang_pattern = re.compile(r'^#!\s*(?:/usr/bin/env\s+)?(?:/\w+/)*(\w+)', re.MULTILINE)

        # Compile modeline patterns
        self._modeline_patterns = []
        for pattern, method in MODELINE_PATTERNS.items():
            self._modeline_patterns.append((re.compile(pattern, re.IGNORECASE), method))

    def detect(
        self,
        content: str,
        file_path: Optional[str] = None,
        extension_hint: Optional[str] = None,
    ) -> LanguageResult:
        """
        Detect the programming language of content.

        Args:
            content: The file content to analyze
            file_path: Optional file path for extension/filename hints
            extension_hint: Optional explicit extension hint (e.g., ".py")

        Returns:
            LanguageResult with detection details
        """
        # Extract path info if provided
        filename = None
        extension = extension_hint

        if file_path:
            path = Path(file_path)
            filename = path.name
            if not extension:
                extension = path.suffix.lower() if path.suffix else None

        # Try detection methods in order of confidence

        # 1. Check special filenames
        if filename and filename in SPECIAL_FILES:
            lang = SPECIAL_FILES[filename]
            return LanguageResult(
                is_code=True,
                language=lang,
                confidence=self.DETECTION_CONFIDENCE["special_file"],
                detection_method="special_file",
            )

        # 2. Check shebang
        shebang_result = self._detect_from_shebang(content)
        if shebang_result:
            return shebang_result

        # 3. Check modelines (vim/emacs)
        modeline_result = self._detect_from_modeline(content)
        if modeline_result:
            return modeline_result

        # 4. Check extension
        if extension and extension in self._ext_to_lang:
            lang = self._ext_to_lang[extension]
            # Still analyze content to get confidence and polyglot detection
            content_result = self._detect_from_content(content, hint_language=lang)

            # Merge extension confidence with content analysis
            return LanguageResult(
                is_code=True,
                language=lang,
                confidence=max(self.DETECTION_CONFIDENCE["extension"], content_result.confidence),
                secondary_languages=content_result.secondary_languages,
                detection_method="extension",
                matched_patterns=content_result.matched_patterns,
                pattern_scores=content_result.pattern_scores,
            )

        # 5. Fall back to content-based detection
        return self._detect_from_content(content)

    def _detect_from_shebang(self, content: str) -> Optional[LanguageResult]:
        """Detect language from shebang line."""
        first_lines = content[:500]

        match = self._shebang_pattern.search(first_lines)
        if match:
            interpreter = match.group(1).lower()

            for pattern, lang in SHEBANG_PATTERNS.items():
                if re.match(pattern, interpreter):
                    return LanguageResult(
                        is_code=True,
                        language=lang,
                        confidence=self.DETECTION_CONFIDENCE["shebang"],
                        detection_method="shebang",
                        matched_patterns=[f"shebang: {interpreter}"],
                    )

        return None

    def _detect_from_modeline(self, content: str) -> Optional[LanguageResult]:
        """Detect language from vim/emacs modeline."""
        lines = content.split('\n')
        check_lines = lines[:5] + lines[-5:]
        check_text = '\n'.join(check_lines)

        for pattern, method in self._modeline_patterns:
            match = pattern.search(check_text)
            if match:
                lang = match.group(1).lower()
                lang_map = {
                    "python": "python", "python3": "python",
                    "javascript": "javascript", "js": "javascript",
                    "typescript": "typescript", "ts": "typescript",
                    "rust": "rust", "go": "go", "golang": "go",
                    "ruby": "ruby", "sh": "bash", "bash": "bash", "zsh": "bash",
                }
                normalized_lang = lang_map.get(lang, lang)

                if normalized_lang in LANGUAGE_SIGNATURES:
                    return LanguageResult(
                        is_code=True,
                        language=normalized_lang,
                        confidence=self.DETECTION_CONFIDENCE["modeline"],
                        detection_method="modeline",
                        matched_patterns=[f"modeline ({method}): {lang}"],
                    )

        return None

    EARLY_EXIT_THRESHOLD = 0.90

    LANGUAGE_CHECK_ORDER = [
        "python", "javascript", "typescript", "java", "go", "rust",
        "cpp", "c", "csharp", "php", "ruby", "bash", "sql",
        "html", "css", "yaml", "json", "markdown", "kotlin", "swift"
    ]

    def _detect_from_content(self, content: str, hint_language: Optional[str] = None) -> LanguageResult:
        """Detect language from content patterns with early-exit optimization."""
        if not content or not content.strip():
            return LanguageResult(is_code=False, language=None, confidence=0.0, detection_method="content")

        language_scores: Dict[str, float] = {}
        language_patterns: Dict[str, List[str]] = {}
        pattern_details: Dict[str, float] = {}

        check_order = self.LANGUAGE_CHECK_ORDER.copy()
        if hint_language and hint_language in check_order:
            check_order.remove(hint_language)
            check_order.insert(0, hint_language)

        for lang_name in LANGUAGE_SIGNATURES.keys():
            if lang_name not in check_order:
                check_order.append(lang_name)

        for lang_name in check_order:
            if lang_name not in LANGUAGE_SIGNATURES:
                continue
            signature = LANGUAGE_SIGNATURES[lang_name]
            score = 0.0
            matched = []

            for pattern in signature.tier1_patterns:
                if pattern.pattern.search(content):
                    score += pattern.weight
                    matched.append(pattern.description)
                    pattern_details[f"{lang_name}:{pattern.description}"] = pattern.weight

            if score > 0:
                language_scores[lang_name] = score
                language_patterns[lang_name] = matched

                if score >= self.EARLY_EXIT_THRESHOLD:
                    return self._build_content_result(
                        lang_name, score, language_patterns, pattern_details, hint_language, content
                    )

        shared_score = 0.0
        for name, pattern in SHARED_PATTERNS.items():
            if pattern.pattern.search(content):
                shared_score += pattern.weight

        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            best_score = language_scores[best_lang]
            return self._build_content_result(
                best_lang, best_score, language_patterns, pattern_details, hint_language, content
            )

        secondary_languages = self._detect_polyglot_languages(content) if self.detect_polyglot else []
        is_code = shared_score > 0.15

        return LanguageResult(
            is_code=is_code,
            language=None if not is_code else "code",
            confidence=min(shared_score + 0.1, 0.4) if is_code else 0.1,
            secondary_languages=secondary_languages,
            detection_method="content",
            pattern_scores=pattern_details,
        )

    def _build_content_result(self, best_lang: str, best_score: float, language_patterns: Dict[str, List[str]],
                              pattern_details: Dict[str, float], hint_language: Optional[str], content: str) -> LanguageResult:
        """Build a LanguageResult from content analysis data."""
        secondary_languages = self._detect_polyglot_languages(content) if self.detect_polyglot else []

        base_confidence = self.DETECTION_CONFIDENCE["content"]
        match_count = len(language_patterns.get(best_lang, []))

        confidence = base_confidence + min(best_score, 0.45)
        if match_count >= 3:
            confidence += 0.05
        if match_count >= 5:
            confidence += 0.05

        confidence = min(confidence, 0.95)

        if hint_language and hint_language == best_lang:
            confidence = max(confidence, 0.80)

        is_code = confidence >= self.code_threshold or best_score > 0.3

        if confidence < self.min_confidence:
            return LanguageResult(
                is_code=is_code, language=None, confidence=confidence,
                secondary_languages=secondary_languages, detection_method="content",
                matched_patterns=language_patterns.get(best_lang, []), pattern_scores=pattern_details,
            )

        return LanguageResult(
            is_code=is_code, language=best_lang, confidence=confidence,
            secondary_languages=secondary_languages, detection_method="content",
            matched_patterns=language_patterns.get(best_lang, []), pattern_scores=pattern_details,
        )

    def _detect_polyglot_languages(self, content: str) -> List[str]:
        """Detect embedded languages in polyglot files."""
        embedded = []
        for key, patterns in [("html_with_js", "javascript"), ("html_with_css", "css"), ("html_with_php", "php")]:
            for pattern in POLYGLOT_PATTERNS.get(key, []):
                if pattern.pattern.search(content):
                    embedded.append(patterns)
                    break

        for key, lang in [("template_jinja", "jinja"), ("template_erb", "erb")]:
            for pattern in POLYGLOT_PATTERNS.get(key, []):
                if pattern.pattern.search(content):
                    embedded.append(lang)
                    break

        for pattern in POLYGLOT_PATTERNS.get("vue_sfc", []):
            if pattern.pattern.search(content):
                if "javascript" not in embedded:
                    embedded.append("javascript")
                if "css" not in embedded:
                    embedded.append("css")
                break

        return embedded

    def detect_language(self, file_path: str, content: Optional[str] = None) -> LanguageResult:
        """Convenience method to detect language from file path and optional content."""
        if content is None:
            path = Path(file_path)
            filename = path.name
            extension = path.suffix.lower() if path.suffix else None

            if filename in SPECIAL_FILES:
                return LanguageResult(is_code=True, language=SPECIAL_FILES[filename],
                                      confidence=self.DETECTION_CONFIDENCE["special_file"], detection_method="special_file")

            if extension and extension in self._ext_to_lang:
                return LanguageResult(is_code=True, language=self._ext_to_lang[extension],
                                      confidence=self.DETECTION_CONFIDENCE["extension"], detection_method="extension")

            return LanguageResult(is_code=False, language=None, confidence=0.0, detection_method="unknown")

        return self.detect(content, file_path=file_path)

    def is_code(self, content: str, file_path: Optional[str] = None) -> bool:
        """Simple boolean check if content is code."""
        return self.detect(content, file_path=file_path).is_code

    def get_supported_languages(self) -> List[str]:
        """Get list of all supported language names."""
        return list(LANGUAGE_SIGNATURES.keys())

    def get_supported_extensions(self) -> Dict[str, str]:
        """Get mapping of supported extensions to languages."""
        return self._ext_to_lang.copy()


# Module-level convenience functions

_default_detector: Optional[LanguageDetector] = None

def get_detector() -> LanguageDetector:
    """Get or create the default detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = LanguageDetector()
    return _default_detector


def detect_language(content: str, file_path: Optional[str] = None) -> LanguageResult:
    """Detect programming language from content."""
    return get_detector().detect(content, file_path=file_path)


def is_code(content: str, file_path: Optional[str] = None) -> bool:
    """Check if content appears to be code."""
    return get_detector().is_code(content, file_path=file_path)
