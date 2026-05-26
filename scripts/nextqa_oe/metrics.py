"""WUPS metrics for NExT-QA Open-Ended evaluation.

Originally authored by Xiao Junbin (NUS) as part of NExT-OE.
Vendored from https://github.com/doc-doc/NExT-OE (MIT licence).
"""
import numpy as np
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


def wup(word1: str, word2: str, alpha: float) -> float:
    """Wu-Palmer similarity between two individual words."""
    if word1 == word2:
        return 1.0
    w1 = wordnet.synsets(word1)
    w2 = wordnet.synsets(word2)
    if not w1 or not w2:
        return 0.0
    word_sim = w1[0].wup_similarity(w2[0])
    if word_sim is None:
        word_sim = 0.0
    if word_sim < alpha:
        word_sim = 0.1 * word_sim
    return word_sim


def wups(words1: list, words2: list, alpha: float) -> float:
    """WUPS similarity between two sequences of words."""
    sim = 1.0
    flag = False
    for word1 in words1:
        max_sim = 0.0
        for word2 in words2:
            s = wup(word1, word2, alpha)
            if s > max_sim:
                max_sim = s
        if max_sim == 0.0:
            continue
        sim *= max_sim
        flag = True
    return sim if flag else 0.0


def get_wups(pred: str, truth: str, alpha: float) -> float:
    """Compute WUPS score between a prediction string and a ground-truth string.

    Tokenises both sides, computes WUPS in both directions, and returns the
    minimum (same as NExT-OE eval_oe.py).
    """
    pred_words  = word_tokenize(pred)
    truth_words = word_tokenize(truth)
    item1 = wups(pred_words,  truth_words, alpha)
    item2 = wups(truth_words, pred_words,  alpha)
    return min(item1, item2)
