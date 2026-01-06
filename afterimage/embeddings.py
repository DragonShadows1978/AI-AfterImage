"""
Embeddings: Generate vector embeddings for semantic search.

Uses sentence-transformers with the all-MiniLM-L6-v2 model,
which runs locally without API calls.
"""

import os
from pathlib import Path
from typing import List, Optional, Union
import hashlib
import json

# Lazy imports for optional dependencies
_model = None
_tokenizer = None


def get_cache_dir() -> Path:
    """Get cache directory for embeddings model."""
    cache_dir = Path.home() / ".afterimage" / "models"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _load_model():
    """Lazy load the sentence-transformers model."""
    global _model

    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer

        # Use CUDA if available, otherwise CPU
        device = "cuda" if _has_cuda() else "cpu"

        # Load model with cache directory
        cache_dir = get_cache_dir()
        _model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=str(cache_dir),
            device=device
        )

        return _model

    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embeddings. "
            "Install with: pip install sentence-transformers"
        )


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


class EmbeddingGenerator:
    """
    Generate embeddings for code snippets and queries.

    Uses all-MiniLM-L6-v2 which produces 384-dimensional vectors.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the embedding generator.

        Args:
            device: "cuda" or "cpu" (auto-detected if None)
        """
        self._device = device
        self._model = None

    @property
    def model(self):
        """Lazy load model on first use."""
        if self._model is None:
            self._model = _load_model()
            if self._device:
                self._model = self._model.to(self._device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension (384 for all-MiniLM-L6-v2)."""
        return 384

    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed (code, query, etc.)

        Returns:
            384-dimensional embedding vector
        """
        # Preprocess text for better code embedding
        text = self._preprocess(text)

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            List of embedding vectors
        """
        # Preprocess all texts
        processed = [self._preprocess(t) for t in texts]

        # Batch encode
        embeddings = self.model.encode(
            processed,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )

        return [e.tolist() for e in embeddings]

    def embed_code(
        self,
        code: str,
        file_path: Optional[str] = None,
        context: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding optimized for code.

        Combines code content with optional metadata for richer embeddings.

        Args:
            code: The code to embed
            file_path: Optional file path (adds semantic info)
            context: Optional conversation context

        Returns:
            384-dimensional embedding vector
        """
        parts = []

        # Add file path information (helps with semantic matching)
        if file_path:
            # Extract meaningful parts of path
            path_parts = Path(file_path).parts[-3:]  # Last 3 parts
            parts.append(f"File: {'/'.join(path_parts)}")

        # Add the code itself
        parts.append(code)

        # Add context summary if provided
        if context:
            # Truncate context to avoid overwhelming the code
            context_summary = context[:200] if len(context) > 200 else context
            parts.append(f"Context: {context_summary}")

        combined = "\n".join(parts)
        return self.embed(combined)

    def _preprocess(self, text: str) -> str:
        """
        Preprocess text for embedding.

        Handles code-specific preprocessing like:
        - Normalizing whitespace
        - Handling long texts
        - Preserving important code structure
        """
        # Normalize whitespace
        text = text.replace("\t", "    ")

        # Truncate very long texts (model max is ~256 tokens)
        # Aim for ~1500 chars which usually fits in context
        max_chars = 1500
        if len(text) > max_chars:
            # For code, prefer keeping the beginning (imports, declarations)
            # and end (main logic), truncating middle
            head = text[:max_chars // 2]
            tail = text[-(max_chars // 2):]
            text = head + "\n...\n" + tail

        return text.strip()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score between -1 and 1
    """
    import math

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def cosine_similarity_batch(
    query_vec: List[float],
    vectors: List[List[float]]
) -> List[float]:
    """
    Calculate cosine similarity between query and multiple vectors.

    Args:
        query_vec: Query embedding vector
        vectors: List of vectors to compare against

    Returns:
        List of similarity scores
    """
    return [cosine_similarity(query_vec, v) for v in vectors]


# Simple embedding cache for repeated queries
_embedding_cache: dict = {}
_cache_max_size = 1000


def cached_embed(text: str, generator: Optional[EmbeddingGenerator] = None) -> List[float]:
    """
    Get embedding with caching.

    Useful for repeated searches on the same query.
    """
    global _embedding_cache

    # Create hash key
    key = hashlib.md5(text.encode()).hexdigest()

    if key in _embedding_cache:
        return _embedding_cache[key]

    if generator is None:
        generator = EmbeddingGenerator()

    embedding = generator.embed(text)

    # Cache management
    if len(_embedding_cache) >= _cache_max_size:
        # Remove oldest entries (simple FIFO)
        keys_to_remove = list(_embedding_cache.keys())[:_cache_max_size // 4]
        for k in keys_to_remove:
            del _embedding_cache[k]

    _embedding_cache[key] = embedding
    return embedding


def clear_embedding_cache():
    """Clear the embedding cache."""
    global _embedding_cache
    _embedding_cache.clear()
