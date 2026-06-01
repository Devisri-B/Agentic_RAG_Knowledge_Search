"""Download the embedding and NLI models at build time so the container runs offline."""

from src.embeddings import get_sentence_transformer
from src.evaluator import _nli

if __name__ == "__main__":
    print("Prefetching embedding and NLI models...")
    get_sentence_transformer()
    _nli()
    print("Done.")
