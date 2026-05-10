"""Dynamic MAD threshold filtering utilities for RAGDefender."""

from typing import List

import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.metrics.pairwise import cosine_similarity


def compute_suspicion_scores(embeddings: np.ndarray) -> np.ndarray:
    """Compute per-document suspicion from average cosine similarity.

    Args:
        embeddings: Document embedding matrix of shape (n_docs, dim).

    Returns:
        Array of suspicion scores where each value is the average cosine
        similarity of one document to all other documents.
    """
    if embeddings.size == 0:
        return np.array([])
    if embeddings.shape[0] == 1:
        return np.array([0.0])

    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, 0.0)
    return sims.sum(axis=1) / (embeddings.shape[0] - 1)


def dynamic_threshold(scores: np.ndarray, sensitivity: float = 2.5) -> float:
    """Compute a robust dynamic threshold from median and MAD.

    Args:
        scores: Suspicion scores for each document.
        sensitivity: MAD multiplier controlling aggressiveness.

    Returns:
        Threshold clipped to guard rails [0.30, 0.85].
    """
    if scores.size == 0:
        return 0.85
    med = float(np.median(scores))
    mad = float(median_abs_deviation(scores, scale=1.0))
    thr = med + sensitivity * mad
    return float(np.clip(thr, 0.30, 0.85))


def mad_filter(docs: List[str], embeddings: np.ndarray, sensitivity: float = 2.5) -> List[str]:
    """Filter suspicious documents using a dynamic MAD-based threshold.

    Args:
        docs: Candidate documents aligned with embeddings.
        embeddings: Embedding matrix corresponding to docs.
        sensitivity: MAD sensitivity multiplier.

    Returns:
        Documents with suspicion scores strictly below dynamic threshold.
    """
    if not docs:
        return []
    if len(docs) == 1:
        return docs

    scores = compute_suspicion_scores(embeddings)
    thr = dynamic_threshold(scores, sensitivity=sensitivity)
    kept = [doc for doc, score in zip(docs, scores) if score < thr]
    return kept if kept else docs
