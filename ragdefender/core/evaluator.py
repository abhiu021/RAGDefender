"""
Evaluator module for RAGDefender performance assessment.
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Optional, Any
import json
import numpy as np
from tqdm import tqdm


class Evaluator:
    """
    Evaluator for assessing RAGDefender performance.

    Provides methods to evaluate defense effectiveness against various
    knowledge corruption attacks on RAG systems.

    Example:
        >>> from ragdefender import RAGDefender, Evaluator
        >>> defender = RAGDefender()
        >>> evaluator = Evaluator(defender)
        >>> results = evaluator.evaluate(test_data, attack_method='poisonedrag')
    """

    def __init__(self, defender=None):
        """
        Initialize evaluator.

        Args:
            defender: RAGDefender instance to evaluate (optional)
        """
        self.defender = defender

    def evaluate(
        self,
        test_data: List[Dict[str, Any]],
        attack_method: str = 'poisonedrag',
        defense_mode: str = 'multihop',
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate defense performance on test data.

        Args:
            test_data: List of test examples, each containing:
                - 'query': str - The query
                - 'retrieved_docs': List[str] - Retrieved documents
                - 'poisoned_indices': List[int] - Indices of poisoned docs
                - 'ground_truth': str - Expected answer (optional)
            attack_method: Attack method being defended against
            defense_mode: Defense mode - 'singlehop' or 'multihop'
            verbose: Show progress bar

        Returns:
            Dictionary containing evaluation metrics
        """
        if self.defender is None:
            from ragdefender.core.defender import RAGDefender
            self.defender = RAGDefender()

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0

        iterator = tqdm(test_data) if verbose else test_data

        for example in iterator:
            query = example['query']
            retrieved_docs = example['retrieved_docs']
            poisoned_indices = example.get('poisoned_indices', [])

            # Apply defense
            defended_docs = self.defender.defend(
                query=query,
                retrieved_docs=retrieved_docs,
                mode=defense_mode
            )

            # Calculate metrics
            metrics = self.defender.get_metrics(
                original_docs=retrieved_docs,
                defended_docs=defended_docs,
                poisoned_indices=poisoned_indices
            )

            total_tp += metrics['true_positives']
            total_fp += metrics['false_positives']
            total_fn += metrics['false_negatives']

        # Calculate overall metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results = {
            'attack_method': attack_method,
            'defense_mode': defense_mode,
            'num_examples': len(test_data),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        }

        return results

    def evaluate_asr(
        self,
        test_data: List[Dict[str, Any]],
        llm_response_fn,
        defense_mode: str = 'multihop',
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate Attack Success Rate (ASR).

        Args:
            test_data: Test examples with 'target_answer' field
            llm_response_fn: Function that takes (query, docs) and returns LLM response
            defense_mode: Defense mode - 'singlehop' or 'multihop'
            verbose: Show progress bar

        Returns:
            Dictionary with ASR metrics
        """
        if self.defender is None:
            from ragdefender.core.defender import RAGDefender
            self.defender = RAGDefender()

        successful_attacks_before = 0
        successful_attacks_after = 0
        total_examples = len(test_data)

        iterator = tqdm(test_data) if verbose else test_data

        for example in iterator:
            query = example['query']
            retrieved_docs = example['retrieved_docs']
            target_answer = example.get('target_answer', '')

            # Evaluate before defense
            response_before = llm_response_fn(query, retrieved_docs)
            if target_answer.lower() in response_before.lower():
                successful_attacks_before += 1

            # Apply defense
            defended_docs = self.defender.defend(
                query=query,
                retrieved_docs=retrieved_docs,
                mode=defense_mode
            )

            # Evaluate after defense
            response_after = llm_response_fn(query, defended_docs)
            if target_answer.lower() in response_after.lower():
                successful_attacks_after += 1

        asr_before = successful_attacks_before / total_examples if total_examples > 0 else 0.0
        asr_after = successful_attacks_after / total_examples if total_examples > 0 else 0.0
        asr_reduction = ((asr_before - asr_after) / asr_before * 100) if asr_before > 0 else 0.0

        return {
            'asr_before_defense': asr_before,
            'asr_after_defense': asr_after,
            'asr_reduction_percent': asr_reduction,
            'attacks_successful_before': successful_attacks_before,
            'attacks_successful_after': successful_attacks_after,
            'total_examples': total_examples
        }

    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file.

        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    def load_results(self, input_path: str) -> Dict[str, Any]:
        """
        Load evaluation results from JSON file.

        Args:
            input_path: Path to load results from

        Returns:
            Evaluation results dictionary
        """
        with open(input_path, 'r') as f:
            return json.load(f)


def main():
    """CLI entry point for evaluator."""
    import argparse

    parser = argparse.ArgumentParser(description='RAGDefender Evaluator')
    parser.add_argument('--test_data', type=str, required=True, help='Path to test data JSON')
    parser.add_argument('--attack_method', type=str, default='poisonedrag',
                        choices=['poisonedrag', 'blind', 'garag'])
    parser.add_argument('--defense_mode', type=str, default='multihop',
                        choices=['singlehop', 'multihop'])
    parser.add_argument('--output', type=str, default='results.json', help='Output file path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Initialize defender and evaluator
    from ragdefender.core.defender import RAGDefender
    defender = RAGDefender(device=args.device)
    evaluator = Evaluator(defender)

    # Run evaluation
    results = evaluator.evaluate(
        test_data=test_data,
        attack_method=args.attack_method,
        defense_mode=args.defense_mode
    )

    # Save results
    evaluator.save_results(results, args.output)
    print(f"\nResults saved to {args.output}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")


if __name__ == '__main__':
    main()
