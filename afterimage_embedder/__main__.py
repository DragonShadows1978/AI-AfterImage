"""
Entry point for running afterimage_embedder as a module.

Usage:
    python -m afterimage_embedder [options]
    python -m afterimage_embedder --once  # Single cycle
    python -m afterimage_embedder --status  # Show status
"""

from .daemon import main

if __name__ == "__main__":
    main()
