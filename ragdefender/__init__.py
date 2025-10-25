"""
RAGDefender: Efficient defense against knowledge corruption attacks on RAG systems

This package provides implementation of defense methods against various
knowledge poisoning attacks on Retrieval-Augmented Generation systems.

Paper: "Rescuing the Unpoisoned: Efficient Defense against Knowledge
Corruption Attacks on RAG Systems" (ACSAC 2025)
"""

__version__ = "0.1.1"
__author__ = "SecAI Lab"
__license__ = "MIT"

from ragdefender.core.defender import RAGDefender
from ragdefender.core.evaluator import Evaluator

__all__ = [
    "RAGDefender",
    "Evaluator",
    "__version__",
]
