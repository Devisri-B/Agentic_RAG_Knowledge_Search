"""Local answer-quality metrics: faithfulness (NLI), relevance (cosine), accuracy (ROUGE).
No LLM/API calls, so it stays cheap and CPU-friendly. Model libraries are imported lazily
inside the functions that need them, keeping module import (and CI) fast."""

import re
import logging
import numpy as np
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)

# FEVER/ANLI-trained NLI model: reliable for fact verification, unlike smaller NLI models
# that mislabel subset/superset and compound claims.
_NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
_TOP_EVIDENCE = 4
# Leading proper-noun subject of 2-4 capitalized words, e.g. "Devi Sri Bandaru ".
_SUBJECT_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s+")

_nli_model = None
_entail_idx = None
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _nli():
    """Load the NLI cross-encoder once and resolve its entailment label index
    (label order differs between models, so read it from the config)."""
    global _nli_model, _entail_idx
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder(_NLI_MODEL_NAME)
        id2label = _nli_model.model.config.id2label
        _entail_idx = next(i for i, lbl in id2label.items() if "entail" in lbl.lower())
    return _nli_model


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.atleast_2d(logits)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def _cosine(text_a: str, text_b: str) -> float:
    if not text_a.strip() or not text_b.strip():
        return 0.0
    from sentence_transformers import util
    from src.embeddings import get_sentence_transformer
    m = get_sentence_transformer()
    return round(float(util.cos_sim(
        m.encode(text_a, convert_to_tensor=True),
        m.encode(text_b, convert_to_tensor=True),
    )), 3)


def _is_claim(text: str) -> bool:
    """Keep verifiable statements; drop list lead-ins and "(Source: file.pdf)" lines."""
    if len(text) <= 15 or text.endswith(":"):
        return False
    return not re.match(r"^\(?\s*sources?\s*:", text, re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    """Break an answer into individual claims, handling bullet lists and markdown."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cleaned = []
    for p in parts:
        p = p.lstrip("*-•· \t")
        p = re.sub(r"[*_`]+", "", p).strip()
        cleaned.append(p)
    return [p for p in cleaned if _is_claim(p)]


def _split_evidence(context: str) -> list[str]:
    """Source text split into single sentences (NLI is far more reliable per-sentence
    than against a whole multi-sentence chunk). Source markers are stripped first."""
    cleaned = re.sub(r"\[(?:File|Source)[^\]]*\]", " ", context)
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


def _strip_subject(claim: str) -> str:
    """Drop a leading proper-noun subject so a subjectless source sentence can still
    entail the fact. Only ever used as an extra variant (we keep the max), so a wrong
    strip never lowers the score."""
    return _SUBJECT_RE.sub("", claim)


def faithfulness_score(answer: str, source_context: str) -> float:
    """Mean entailment of each answer claim against the source it was drawn from (0-1).

    Each claim is tested against its most similar source sentences and their
    concatenation, keeping the best match. A claim is faithful if some evidence entails
    it, so the score is robust to irrelevant passages while contradicted or unsupported
    claims fall toward 0."""
    if not answer.strip() or not source_context.strip():
        return 0.0

    claims = _split_sentences(answer) or [answer.strip()]
    evidence = _split_evidence(source_context)
    if not evidence:
        return 0.0

    try:
        from sentence_transformers import util
        from src.embeddings import get_sentence_transformer
        model = get_sentence_transformer()
        ev_emb = model.encode(evidence, convert_to_tensor=True)
        cl_emb = model.encode(claims, convert_to_tensor=True)
        sims = util.cos_sim(cl_emb, ev_emb)
        topk = min(_TOP_EVIDENCE, len(evidence))

        pairs, owners = [], []
        for i in range(len(claims)):
            idxs = sims[i].topk(topk).indices.tolist()
            best = evidence[idxs[0]]
            concat = " ".join(evidence[j] for j in idxs)
            premises = [best] if best == concat else [best, concat]
            variants = [claims[i]]
            stripped = _strip_subject(claims[i])
            if stripped != claims[i] and len(stripped) > 10:
                variants.append(stripped)
            for premise in premises:
                for variant in variants:
                    pairs.append((premise, variant))
                    owners.append(i)

        entail = _softmax(np.asarray(_nli().predict(pairs)))[:, _entail_idx]
        per_claim = [0.0] * len(claims)
        for owner, e in zip(owners, entail):
            per_claim[owner] = max(per_claim[owner], float(e))
        return round(float(np.mean(per_claim)), 3)
    except Exception as e:
        logger.warning(f"NLI faithfulness unavailable ({e}); falling back to cosine.")
        return _cosine(answer, source_context)


def answer_relevance_score(question: str, answer: str) -> float:
    """Cosine similarity between the question and the answer (0-1)."""
    return _cosine(question, answer)


def accuracy_score(answer: str, reference: str) -> float:
    """ROUGE-L F1 between the answer and a user-supplied reference (0-1)."""
    if not answer.strip() or not reference.strip():
        return 0.0
    return round(_rouge.score(reference, answer)["rougeL"].fmeasure, 3)
