"""
Local evaluation metrics — no LLM API calls, CPU-friendly for HF Spaces.

faithfulness   : NLI entailment of each answer sentence against the best-matching
                 source passage (docs or web results it actually used).
                 Detects contradictions / unsupported claims, not just topic overlap.
                 Falls back to cosine similarity if the NLI model can't load.
answer_relevance: cosine similarity between question and answer — does the answer
                 address the question? Needs no reference.
accuracy       : ROUGE-L F1 vs a user-supplied reference answer. Only when provided.
"""

import re
import logging
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import util
from src.embeddings import get_sentence_transformer

logger = logging.getLogger(__name__)

_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-small"
# Label index for "entailment" in this model's output (0=contradiction, 1=entailment, 2=neutral)
_ENTAILMENT_IDX = 1

_nli_model = None
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# --- helpers ---------------------------------------------------------------

def _nli():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
    return _nli_model


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.atleast_2d(logits)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _cosine(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    m = get_sentence_transformer()
    return round(float(util.cos_sim(
        m.encode(text_a, convert_to_tensor=True),
        m.encode(text_b, convert_to_tensor=True),
    )), 3)


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _split_passages(context: str) -> list[str]:
    # retrieve() joins passages with blank lines
    parts = [p.strip() for p in context.split("\n\n") if p.strip()]
    return parts or [context.strip()]


# --- metrics ---------------------------------------------------------------

def faithfulness_score(answer: str, source_context: str) -> float:
    """Mean NLI entailment of answer claims given the source they were drawn from.

    For each answer sentence we pick the most similar source passage (so the NLI
    input stays short and on-point), then ask whether that passage entails the
    sentence. Returns the mean entailment probability across sentences (0–1)."""
    if not answer.strip() or not source_context.strip():
        return 0.0

    sentences = _split_sentences(answer) or [answer.strip()]
    passages = _split_passages(source_context)

    try:
        model = get_sentence_transformer()
        pas_emb = model.encode(passages, convert_to_tensor=True)
        sen_emb = model.encode(sentences, convert_to_tensor=True)
        best_idx = util.cos_sim(sen_emb, pas_emb).argmax(dim=1).tolist()
        pairs = [(passages[best_idx[i]], sentences[i]) for i in range(len(sentences))]

        logits = _nli().predict(pairs)
        entail = _softmax(np.asarray(logits))[:, _ENTAILMENT_IDX]
        return round(float(entail.mean()), 3)
    except Exception as e:
        logger.warning(f"NLI faithfulness unavailable ({e}); falling back to cosine.")
        return _cosine(answer, source_context)


def answer_relevance_score(question: str, answer: str) -> float:
    """Does the answer address the question? Cosine similarity (0–1)."""
    return _cosine(question, answer)


def accuracy_score(answer: str, reference: str) -> float:
    """ROUGE-L F1 vs a user-supplied reference answer (0–1)."""
    if not answer.strip() or not reference.strip():
        return 0.0
    return round(_rouge.score(reference, answer)["rougeL"].fmeasure, 3)
