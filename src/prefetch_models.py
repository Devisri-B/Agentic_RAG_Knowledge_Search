"""
Download the embedding + NLI models at image-build time so the running
container starts fast and never needs network access for models at runtime.
"""

from src.embeddings import get_sentence_transformer
from src.evaluator import _nli

if __name__ == "__main__":
    print("Prefetching embedding model (all-MiniLM-L6-v2)...")
    get_sentence_transformer()
    print("Prefetching NLI model (cross-encoder/nli-deberta-v3-small)...")
    _nli()
    print("Models cached.")
