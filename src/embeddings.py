from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_NAME = "all-MiniLM-L6-v2"
_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Shared embeddings instance, loaded once and reused by RAG and the evaluator."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=_MODEL_NAME)
    return _embeddings


def get_sentence_transformer():
    """The raw SentenceTransformer behind the shared embeddings (used by the evaluator)."""
    model = getattr(get_embeddings(), "client", None)
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(_MODEL_NAME)
    return model
