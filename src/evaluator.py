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

# Heavy model libraries (sentence_transformers, torch) are imported lazily inside
# the functions that need them, so importing this module — and running the
# ROUGE/text-helper unit tests — stays fast and dependency-light in CI.

logger = logging.getLogger(__name__)

# FEVER/ANLI-trained NLI model — reliable for fact verification (handles
# subset/superset and compound claims, which the smaller NLI models miss).
_NLI_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

_nli_model = None
_entail_idx = None
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


# --- helpers ---------------------------------------------------------------

def _nli():
    """Lazily load the NLI cross-encoder and resolve its 'entailment' label index
    (label order varies between models, so we read it from the config)."""
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
    """Keep verifiable factual statements; drop framing and meta lines."""
    if len(text) <= 15:
        return False
    if text.endswith(":"):                                  # lead-in to a list
        return False
    if re.match(r"^\(?\s*sources?\s*:", text, re.IGNORECASE):  # "(Source: file.pdf)"
        return False
    return True


def _split_sentences(text: str) -> list[str]:
    # Split on sentence punctuation and newlines so bulleted/list answers
    # become individual claims; strip leading bullet markers, then drop
    # framing/meta lines that aren't verifiable factual statements.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cleaned = []
    for p in parts:
        p = p.lstrip("*-•· \t")
        p = re.sub(r"[*_`]+", "", p).strip()   # strip markdown emphasis (**bold**, *italic*, `code`)
        cleaned.append(p)
    return [p for p in cleaned if _is_claim(p)]


def _split_evidence(context: str) -> list[str]:
    """Break the retrieved context into individual evidence sentences.

    Source markers like "[File: x]" / "[Source: Page 3]" are stripped, then the
    text is split on sentence boundaries and newlines. NLI is far more reliable
    with single-sentence premises than with whole multi-sentence chunks."""
    cleaned = re.sub(r"\[(?:File|Source)[^\]]*\]", " ", context)
    parts = re.split(r"(?<=[.!?])\s+|\n+", cleaned.strip())
    return [p.strip() for p in parts if len(p.strip()) > 15]


# --- metrics ---------------------------------------------------------------

_TOP_EVIDENCE = 4   # source sentences considered per claim

# Leading proper-noun subject of 2-4 capitalized words, e.g. "Devi Sri Bandaru ".
_SUBJECT_RE = re.compile(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\s+")


def _strip_subject(claim: str) -> str:
    """Drop a leading proper-noun subject so a subjectless source sentence can
    still entail the fact. Used only as an extra variant (we keep the max), so a
    wrong strip never lowers the score."""
    return _SUBJECT_RE.sub("", claim)


def faithfulness_score(answer: str, source_context: str) -> float:
    """RAGAS-style faithfulness: is each claim in the answer supported by the source?

    For every answer sentence (claim) we take its most similar source sentences and
    test entailment against each one *and* their concatenation, keeping the best
    (max). Single sentences support simple/verbatim claims; the concatenation
    supports compound claims that combine facts from several source sentences. The
    score is the mean over claims (0-1) — robust to irrelevant passages (e.g. extra
    web results), while contradicted or unsupported claims correctly fall toward 0."""
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
            best = evidence[idxs[0]]                       # supports simple claims
            concat = " ".join(evidence[j] for j in idxs)   # supports compound claims
            premises = [best] if best == concat else [best, concat]
            # Document bullets are often subjectless ("Architected a RAG pipeline"),
            # while answers prepend the person's name ("Devi Sri Bandaru architected
            # ..."), which NLI reads as unsupported. Also test a subject-stripped
            # variant and keep the best.
            variants = [claims[i]]
            stripped = _strip_subject(claims[i])
            if stripped != claims[i] and len(stripped) > 10:
                variants.append(stripped)
            for premise in premises:
                for variant in variants:
                    pairs.append((premise, variant))
                    owners.append(i)

        model_nli = _nli()
        entail = _softmax(np.asarray(model_nli.predict(pairs)))[:, _entail_idx]

        per_claim = [0.0] * len(claims)
        for owner, e in zip(owners, entail):
            per_claim[owner] = max(per_claim[owner], float(e))
        return round(float(np.mean(per_claim)), 3)
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
