"""NLI-based contradiction filtering for RAGDefender."""

from itertools import combinations
from typing import Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def nli_contradiction_filter(
    query: str,
    docs: List[str],
    encoder: Any,
    nli_model: Any,
    contradiction_threshold: float = 0.65,
) -> List[str]:
    """Remove contradictory documents while protecting a query anchor.

    Args:
        query: User query.
        docs: Candidate documents.
        encoder: Encoder exposing `encode` for query/doc embeddings.
        nli_model: NLI cross-encoder exposing `predict` returning 3-label scores.
        contradiction_threshold: Probability threshold for contradiction label.

    Returns:
        Filtered list of documents after contradiction voting.
    """
    if not docs:
        return []
    if len(docs) == 1 or nli_model is None:
        return docs

    doc_emb = np.array(encoder.encode(docs))
    query_emb = np.array(encoder.encode([query]))
    relevance = cosine_similarity(doc_emb, query_emb).reshape(-1)
    anchor_idx = int(np.argmax(relevance))

    votes = np.zeros(len(docs), dtype=float)
    pairs = list(combinations(range(len(docs)), 2))
    if not pairs:
        return docs

    nli_inputs = [(docs[i], docs[j]) for i, j in pairs]
    outputs = np.array(nli_model.predict(nli_inputs))
    if outputs.ndim == 1:
        outputs = np.expand_dims(outputs, 0)

    for (i, j), probs in zip(pairs, outputs):
        contradiction_prob = float(probs[0])
        if contradiction_prob > contradiction_threshold:
            votes[i] += 1
            votes[j] += 1

    avg_votes = float(votes.mean())
    survivors = [
        d for idx, d in enumerate(docs)
        if idx == anchor_idx or votes[idx] <= avg_votes
    ]
    return survivors if survivors else [docs[anchor_idx]]
