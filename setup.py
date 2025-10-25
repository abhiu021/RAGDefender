"""
Setup configuration for RAGDefender package
"""

from setuptools import setup, find_packages
import os

# Read the README for long description
def read_long_description():
    with open("README_PYPI.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ragdefender",
    version="0.1.1",
    author="SecAI Lab",
    author_email="for8821@g.skku.edu",
    description="Efficient defense against knowledge corruption attacks on RAG systems",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/SecAI-Lab/RAGDefender",
    project_urls={
        "Bug Tracker": "https://github.com/SecAI-Lab/RAGDefender/issues",
        "Documentation": "https://github.com/SecAI-Lab/RAGDefender",
        "Source Code": "https://github.com/SecAI-Lab/RAGDefender",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*", "claims*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
        "tqdm>=4.60.0",
        "scikit-learn>=0.24.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "flake8>=3.9",
            "isort>=5.9",
            "mypy>=0.900",
            "pre-commit>=2.15",
        ],
        "cuda": [
            "faiss-gpu>=1.7.0",  # GPU version of faiss
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "sphinx-autodoc-typehints>=1.12",
        ],
    },
    entry_points={
        "console_scripts": [
            "ragdefender=ragdefender.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
