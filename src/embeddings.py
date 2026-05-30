"""
Shared embedding model — loaded once and reused everywhere.

The same all-MiniLM-L6-v2 model is needed by:
  - the RAG vector stores (via langchain's HuggingFaceEmbeddings)
  - the evaluator's cosine-similarity metrics (via the raw SentenceTransformer)

Loading it a single time keeps memory and cold-start cost low on HF Spaces.
"""

from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_NAME = "all-MiniLM-L6-v2"
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Shared langchain embeddings object for FAISS vector stores."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=_MODEL_NAME)
    return _embeddings


def get_sentence_transformer():
    """The underlying SentenceTransformer, reused by the evaluator.
    Falls back to a direct load if langchain's internal attribute changes."""
    embeddings = get_embeddings()
    model = getattr(embeddings, "client", None)
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_MODEL_NAME)
    return model
