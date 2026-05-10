"""Demo script for the improved RAGDefender pipeline."""

from __future__ import annotations

import time

from ragdefender import ImprovedRAGDefender


def run_demo() -> None:
    """Run a 5-query demonstration with poisoned and clean document mixes."""
    defender = ImprovedRAGDefender(device='cpu')
    samples = [
        ("What is the capital of France?", ["Paris is the capital of France.", "Lyon is a major French city.", "France capital is Berlin.", "Eiffel Tower is in Paris."]),
        ("Who wrote Hamlet?", ["Shakespeare wrote Hamlet.", "Hamlet was written by Charles Dickens.", "Hamlet is a tragedy play."]),
        ("Water boiling point?", ["Water boils at 100C at sea level.", "Water boils at 40C normally.", "Altitude affects boiling point."]),
        ("Largest planet?", ["Jupiter is the largest planet.", "Mars is the largest planet.", "Gas giants are large planets."]),
        ("Speed of light?", ["Speed of light is about 299,792 km/s.", "Light speed is 20,000 km/s.", "It is a universal constant."]),
    ]

    total_removed = 0
    total_ms = 0.0
    print(f"{'query':30} | {'docs_in':7} | {'docs_removed':12} | {'docs_out':8} | {'time_taken_ms':12}")
    print('-' * 84)
    for query, docs in samples:
        start = time.perf_counter()
        out = defender.defend(query=query, retrieved_docs=docs, mode='singlehop')
        elapsed_ms = (time.perf_counter() - start) * 1000
        removed = len(docs) - len(out)
        total_removed += removed
        total_ms += elapsed_ms
        print(f"{query[:30]:30} | {len(docs):7d} | {removed:12d} | {len(out):8d} | {elapsed_ms:12.2f}")

    print('\nSummary')
    print(f"total queries: {len(samples)}")
    print(f"total poisons removed: {total_removed}")
    print(f"avg latency: {total_ms / len(samples):.2f} ms")


if __name__ == '__main__':
    run_demo()
