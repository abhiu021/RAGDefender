"""RAGDefender: Main defense interface against knowledge corruption attacks on RAG systems."""

import warnings

warnings.filterwarnings('ignore')

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.feature_extraction.text as text
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from sklearn.cluster import AgglomerativeClustering

from ragdefender.filtering.mad_threshold import mad_filter
from ragdefender.filtering.nli_filter import nli_contradiction_filter


def hdbscan_group(embeddings: np.ndarray, threshold: float = 0.4) -> tuple[list[int], list[int]]:
    """Group embeddings with HDBSCAN and return clean/adversarial indices.

    Args:
        embeddings: Dense embedding matrix of shape (n_docs, dim).
        threshold: Minimum membership probability to flag a clustered point.

    Returns:
        Tuple of (clean_indices, adversarial_indices).
    """
    if embeddings.size == 0:
        return [], []
    if embeddings.shape[0] == 1:
        return [0], []

    try:
        import hdbscan  # type: ignore
    except ImportError:
        warnings.warn('hdbscan is not installed; falling back to KMeans-style clustering behavior.')
        model = AgglomerativeClustering(n_clusters=2)
        labels = model.fit_predict(embeddings)
        counts = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
        adv_label = max(counts, key=counts.get)
        adv = [i for i, label in enumerate(labels) if label == adv_label]
        clean = [i for i in range(len(labels)) if i not in adv]
        return clean, adv

    model = hdbscan.HDBSCAN(min_cluster_size=2, metric='cosine', prediction_data=True)
    labels = model.fit_predict(embeddings)
    probs = getattr(model, 'probabilities_', np.zeros(len(labels)))

    cluster_sizes = {label: int(np.sum(labels == label)) for label in np.unique(labels) if label != -1}
    clean_indices: list[int] = []
    adversarial_indices: list[int] = []
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        if label == -1:
            clean_indices.append(idx)
            continue
        if cluster_sizes.get(label, 0) >= 2 and prob > threshold:
            adversarial_indices.append(idx)
        else:
            clean_indices.append(idx)
    return clean_indices, adversarial_indices


class RAGDefender:
    def __init__(self, device: str = 'cuda', similarity_model: str = 'sentence-transformers/all-MiniLM-L6-v2', gpu_id: int = 0,
                 use_hdbscan: bool = True, use_dynamic_threshold: bool = True, use_nli: bool = False):
        self.device = device
        self.use_hdbscan = use_hdbscan
        self.use_dynamic_threshold = use_dynamic_threshold
        self.use_nli = use_nli
        self.nli_model = None

        if device == 'cuda':
            torch.cuda.set_device(gpu_id)
        self.s_model = SentenceTransformer(similarity_model)
        if device == 'cuda':
            self.s_model = self.s_model.to(device)

        if self.use_nli:
            try:
                self.nli_model = CrossEncoder('cross-encoder/nli-deberta-v3-small', num_labels=3)
            except Exception as exc:
                warnings.warn(f'Failed to load NLI model; skipping NLI stage. Reason: {exc}')
                self.nli_model = None

    def defend(self, query: str, retrieved_docs: List[str], mode: str = 'multihop', top_k: Optional[int] = None) -> List[str]:
        if not retrieved_docs:
            return []
        if mode not in ['singlehop', 'multihop']:
            raise ValueError(f"Unknown mode: {mode}. Use 'singlehop' or 'multihop'")

        if mode == 'singlehop' and self.use_hdbscan:
            embeddings = np.array(self.s_model.encode(retrieved_docs, convert_to_tensor=False))
            clean_indices, _ = hdbscan_group(embeddings)
            clean_docs = [retrieved_docs[i] for i in clean_indices]
        else:
            num_poisoned = self._find_num_adversarial_agg(retrieved_docs) if mode == 'singlehop' else self._find_num_adversarial(retrieved_docs)
            if num_poisoned == 0:
                clean_docs = retrieved_docs
            else:
                clean_docs = retrieved_docs[: len(retrieved_docs) - num_poisoned]

        if self.use_dynamic_threshold and clean_docs:
            embeddings = np.array(self.s_model.encode(clean_docs, convert_to_tensor=False))
            clean_docs = mad_filter(clean_docs, embeddings)

        if self.use_nli and clean_docs:
            clean_docs = nli_contradiction_filter(query, clean_docs, self.s_model, self.nli_model)

        return clean_docs[:top_k] if top_k else clean_docs

    def _find_num_adversarial(self, text_list: List[str]) -> int:
        embeddings = self.s_model.encode(text_list, convert_to_tensor=True)
        cos_sim_matrix = util.cos_sim(embeddings, embeddings)
        avg = torch.mean(cos_sim_matrix, dim=0)
        median = torch.median(cos_sim_matrix, dim=0)
        avg_avg = avg.mean()
        avg_median = median.values.median()
        above_avg = [1 if score > avg_avg else 0 for score in avg]
        above_median = [1 if score > (avg_median + avg_avg) / 2 else 0 for score in median.values]
        final = [1 if above_avg[i] == 1 or above_median[i] == 1 else 0 for i in range(len(above_avg))]
        result = sum(final) if sum(final) > 0 and avg_avg < avg_median else len(text_list) - sum(final)
        return result

    def _find_num_adversarial_tfidf(self, text_list: List[str]) -> int:
        stop_words = list(text.ENGLISH_STOP_WORDS)
        tfidf = text.TfidfVectorizer(stop_words=stop_words)
        X = tfidf.fit_transform(text_list)
        all_data = tfidf.get_feature_names_out()
        dense = X.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=all_data)
        dict_tfidf = df.T.sum(axis=1).sort_values(ascending=False)
        top_3 = dict_tfidf[:5]
        indices = [[1 if word in sentence else 0 for sentence in text_list] for word in top_3.index]
        final = [1 if sum([index[i] for index in indices]) > math.floor(len(indices) / 2) else 0 for i in range(len(text_list))]
        return sum(final)

    def _find_num_adversarial_agg(self, text_list: List[str]) -> int:
        embeddings = self.s_model.encode(text_list, convert_to_tensor=True)
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(embeddings.cpu().detach().numpy())
        labels = list(model.labels_)
        num_labels = sum(labels)
        num_tfidf = self._find_num_adversarial_tfidf(text_list)
        return min(num_labels, len(text_list) - num_labels) if num_labels > 0 and num_tfidf <= int(len(text_list) / 2) else max(num_labels, len(text_list) - num_labels)


class ImprovedRAGDefender(RAGDefender):
    """Convenience defender with all optional improvements enabled."""

    def __init__(self, **kwargs: object):
        kwargs.setdefault('use_hdbscan', True)
        kwargs.setdefault('use_dynamic_threshold', True)
        kwargs.setdefault('use_nli', True)
        super().__init__(**kwargs)
