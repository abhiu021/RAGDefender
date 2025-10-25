"""
Command-line interface for RAGDefender.
"""

import argparse
import sys
import json
from typing import List, Dict, Any

from ragdefender import RAGDefender, Evaluator, __version__


def defend_command(args):
    """Execute defense on query and documents."""
    # Initialize defender
    defender = RAGDefender(
        device=args.device,
        gpu_id=args.gpu_id
    )

    # Load documents
    if args.corpus.endswith('.json'):
        with open(args.corpus, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                documents = data
            elif isinstance(data, dict) and 'documents' in data:
                documents = data['documents']
            else:
                print("Error: JSON must be a list or dict with 'documents' key")
                return 1
    else:
        # Assume text file with one document per line
        with open(args.corpus, 'r') as f:
            documents = [line.strip() for line in f if line.strip()]

    # Apply defense
    clean_docs = defender.defend(
        query=args.query,
        retrieved_docs=documents,
        mode=args.mode,
        top_k=args.top_k
    )

    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'query': args.query,
                'original_docs': documents,
                'defended_docs': clean_docs,
                'num_removed': len(documents) - len(clean_docs)
            }, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(f"\n=== Query ===")
        print(args.query)
        print(f"\n=== Clean Documents ({len(clean_docs)}/{len(documents)}) ===")
        for i, doc in enumerate(clean_docs, 1):
            print(f"\n{i}. {doc[:200]}..." if len(doc) > 200 else f"\n{i}. {doc}")

    return 0


def evaluate_command(args):
    """Execute evaluation on test dataset."""
    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Initialize defender and evaluator
    defender = RAGDefender(device=args.device, gpu_id=args.gpu_id)
    evaluator = Evaluator(defender)

    # Run evaluation
    print(f"Evaluating {len(test_data)} examples...")
    results = evaluator.evaluate(
        test_data=test_data,
        attack_method=args.attack,
        defense_mode=args.mode,
        verbose=True
    )

    # Save results
    output_file = args.output or f"eval_results_{args.attack}_{args.mode}.json"
    evaluator.save_results(results, output_file)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"Attack Method:     {results['attack_method']}")
    print(f"Defense Mode:      {results['defense_mode']}")
    print(f"Examples:          {results['num_examples']}")
    print(f"Precision:         {results['precision']:.4f}")
    print(f"Recall:            {results['recall']:.4f}")
    print(f"F1 Score:          {results['f1_score']:.4f}")
    print(f"\nResults saved to {output_file}")

    return 0


def info_command(args):
    """Display package information."""
    print(f"RAGDefender v{__version__}")
    print("\nEfficient defense against knowledge corruption attacks on RAG systems")
    print("\nPaper: 'Rescuing the Unpoisoned: Efficient Defense against")
    print("       Knowledge Corruption Attacks on RAG Systems' (ACSAC 2025)")
    print("\nAuthors: SecAI Lab")
    print("License: MIT")
    print(f"\nRepository: https://github.com/SecAI-Lab/RAGDefender")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='RAGDefender - Defense against knowledge corruption attacks on RAG systems',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Defend against poisoned documents
  ragdefender defend --query "What is the capital?" --corpus docs.json

  # Evaluate on test dataset
  ragdefender evaluate --test-data test.json --attack poisonedrag

  # Show package information
  ragdefender info
        """
    )

    parser.add_argument('--version', action='version', version=f'RAGDefender {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Defend command
    defend_parser = subparsers.add_parser(
        'defend',
        help='Apply defense to query and documents'
    )
    defend_parser.add_argument(
        '--query',
        type=str,
        required=True,
        help='Query string'
    )
    defend_parser.add_argument(
        '--corpus',
        type=str,
        required=True,
        help='Path to corpus file (JSON or text file)'
    )
    defend_parser.add_argument(
        '--mode',
        type=str,
        default='multihop',
        choices=['singlehop', 'multihop'],
        help='Query mode: singlehop (NQ, MSMARCO) or multihop (HotpotQA) - default: multihop'
    )
    defend_parser.add_argument(
        '--top-k',
        type=int,
        default=None,
        help='Number of documents to return (default: all clean docs)'
    )
    defend_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    defend_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    defend_parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate defense on test dataset'
    )
    eval_parser.add_argument(
        '--test-data',
        type=str,
        required=True,
        help='Path to test data JSON file'
    )
    eval_parser.add_argument(
        '--attack',
        type=str,
        default='poisonedrag',
        choices=['poisonedrag', 'blind', 'garag'],
        help='Attack method (default: poisonedrag)'
    )
    eval_parser.add_argument(
        '--mode',
        type=str,
        default='multihop',
        choices=['singlehop', 'multihop'],
        help='Query mode: singlehop or multihop (default: multihop)'
    )
    eval_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )
    eval_parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    eval_parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='GPU device ID (default: 0)'
    )

    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show package information'
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    if args.command == 'defend':
        return defend_command(args)
    elif args.command == 'evaluate':
        return evaluate_command(args)
    elif args.command == 'info':
        return info_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
