"""
RAGDefender: Main defense interface against knowledge corruption attacks on RAG systems.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer, util
import sklearn.feature_extraction.text as text
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import math


class RAGDefender:
    """
    Main interface for RAG defense mechanisms.

    RAGDefender provides efficient defense against knowledge corruption attacks
    on Retrieval-Augmented Generation (RAG) systems by detecting and isolating
    poisoned documents from retrieved context.

    Example:
        >>> from ragdefender import RAGDefender
        >>> defender = RAGDefender(device='cuda')
        >>> clean_docs = defender.defend(
        ...     query="What is the capital of France?",
        ...     retrieved_docs=["Paris is...", "Lyon is...", "Poisoned doc..."],
        ...     method='isolation'
        ... )
    """

    def __init__(
        self,
        device: str = 'cuda',
        similarity_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        gpu_id: int = 0
    ):
        """
        Initialize RAGDefender.

        Args:
            device: Device to use ('cuda' or 'cpu')
            similarity_model: Sentence transformer model for similarity computation
            gpu_id: GPU device ID to use
        """
        self.device = device
        if device == 'cuda':
            torch.cuda.set_device(gpu_id)

        # Initialize similarity model for defense
        self.s_model = SentenceTransformer(similarity_model)
        if device == 'cuda':
            self.s_model = self.s_model.to(device)

    def defend(
        self,
        query: str,
        retrieved_docs: List[str],
        mode: str = 'multihop',
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Defend against knowledge corruption attacks by filtering retrieved documents.

        RAGDefender uses different adversarial document detection algorithms based on
        the query type (single-hop vs multi-hop QA).

        Args:
            query: User query string
            retrieved_docs: List of retrieved documents (may contain poisoned docs)
            mode: Query type - 'singlehop' or 'multihop' (default: 'multihop')
                 - 'singlehop': For NQ, MSMARCO (uses aggregation-based detection)
                 - 'multihop': For HotpotQA (uses similarity-based detection)
            top_k: Number of documents to return (default: return all clean docs)

        Returns:
            List of clean documents with poisoned ones removed/isolated

        Example:
            >>> defender = RAGDefender()
            >>> # For single-hop questions (NQ, MSMARCO)
            >>> clean = defender.defend(query, docs, mode='singlehop')
            >>> # For multi-hop questions (HotpotQA)
            >>> clean = defender.defend(query, docs, mode='multihop')
        """
        if not retrieved_docs:
            return []

        if mode not in ['singlehop', 'multihop']:
            raise ValueError(f"Unknown mode: {mode}. Use 'singlehop' or 'multihop'")

        # Detect number of poisoned documents
        if mode == 'singlehop':
            num_poisoned = self._find_num_adversarial_agg(retrieved_docs)
        else:  # multihop
            num_poisoned = self._find_num_adversarial(retrieved_docs)

        if num_poisoned == 0:
            return retrieved_docs[:top_k] if top_k else retrieved_docs

        # Remove suspected poisoned documents
        num_clean = len(retrieved_docs) - num_poisoned
        clean_docs = retrieved_docs[:num_clean]

        return clean_docs[:top_k] if top_k else clean_docs

    def _find_num_adversarial(self, text_list: List[str]) -> int:
        """
        Detect number of adversarial documents using similarity analysis.

        This is the core detection algorithm from the RAGDefender paper.
        """
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

        # Clean up
        del embeddings, cos_sim_matrix, avg, median
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return result

    def _find_num_adversarial_tfidf(self, text_list: List[str]) -> int:
        """Detect adversarial documents using TF-IDF analysis."""
        stop_words = list(text.ENGLISH_STOP_WORDS)
        tfidf = text.TfidfVectorizer(stop_words=stop_words)
        X = tfidf.fit_transform(text_list)
        all_data = tfidf.get_feature_names_out()
        dense = X.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=all_data)
        dict_tfidf = df.T.sum(axis=1)
        dict_tfidf = dict_tfidf.sort_values(ascending=False)
        top_3 = dict_tfidf[:5]
        indices = []
        for word in top_3.index:
            indices.append([1 if word in sentence else 0 for sentence in text_list])
        final = [
            1 if sum([index[i] for index in indices]) > math.floor(len(indices) / 2) else 0
            for i in range(len(text_list))
        ]
        return sum(final)

    def _find_num_adversarial_agg(self, text_list: List[str]) -> int:
        """Detect adversarial documents using aggregation clustering."""
        embeddings = self.s_model.encode(text_list, convert_to_tensor=True)
        model = AgglomerativeClustering(n_clusters=2)
        model.fit(embeddings.cpu().detach().numpy())
        labels = model.labels_
        labels = list(labels)
        num_labels = sum(labels)
        num_tfidf = self._find_num_adversarial_tfidf(text_list)

        result = (
            min(num_labels, len(text_list) - num_labels)
            if num_labels > 0 and num_tfidf <= int(len(text_list) / 2)
            else max(num_labels, len(text_list) - num_labels)
        )

        # Clean up
        del embeddings
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return result

    def get_metrics(
        self,
        original_docs: List[str],
        defended_docs: List[str],
        poisoned_indices: List[int]
    ) -> Dict[str, float]:
        """
        Calculate defense performance metrics.

        Args:
            original_docs: Original retrieved documents (with poison)
            defended_docs: Documents after defense
            poisoned_indices: Indices of poisoned documents in original_docs

        Returns:
            Dictionary with metrics (precision, recall, f1)
        """
        # Determine which docs were kept
        kept_indices = [i for i, doc in enumerate(original_docs) if doc in defended_docs]
        removed_indices = [i for i in range(len(original_docs)) if i not in kept_indices]

        # True positives: correctly removed poisoned docs
        tp = len(set(removed_indices) & set(poisoned_indices))
        # False positives: incorrectly removed clean docs
        fp = len(set(removed_indices) - set(poisoned_indices))
        # False negatives: failed to remove poisoned docs
        fn = len(set(kept_indices) & set(poisoned_indices))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
