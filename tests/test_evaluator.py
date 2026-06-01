"""Unit tests for the lightweight evaluator pieces (ROUGE accuracy and text helpers).
The embedding/NLI metrics need model downloads, so they are validated manually, not in CI."""

import numpy as np
from src.evaluator import accuracy_score, _split_sentences, _softmax


class TestAccuracyScore:
    def test_identical_text_scores_high(self):
        assert accuracy_score("the cat sat on the mat", "the cat sat on the mat") > 0.9

    def test_unrelated_text_scores_low(self):
        assert accuracy_score("quantum physics equations", "a recipe for pasta") < 0.3

    def test_empty_inputs_return_zero(self):
        assert accuracy_score("", "something") == 0.0
        assert accuracy_score("something", "") == 0.0

    def test_score_is_bounded(self):
        s = accuracy_score("partial overlap here", "some partial overlap")
        assert 0.0 <= s <= 1.0


class TestSplitSentences:
    def test_splits_on_punctuation(self):
        sents = _split_sentences(
            "This is the first sentence. Here is the second one! And this is the third question?"
        )
        assert len(sents) == 3

    def test_drops_tiny_fragments(self):
        # fragments <= 15 chars are dropped
        assert _split_sentences("Ok. This is a long enough sentence to keep.") == [
            "This is a long enough sentence to keep."
        ]


class TestSoftmax:
    def test_rows_sum_to_one(self):
        probs = _softmax(np.array([[1.0, 2.0, 3.0]]))
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_handles_1d_input(self):
        probs = _softmax(np.array([0.0, 0.0]))
        assert probs.shape == (1, 2)
        assert abs(probs.sum() - 1.0) < 1e-6
