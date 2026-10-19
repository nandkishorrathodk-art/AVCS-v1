"""Deterministic, parameter-free text query embeddings for cortical grounding.

The cortical grounding head (`LanguageModulation`) needs a query embedding that
is *identical* between training and inference. An earlier version of the
training pipeline fed it `torch.randn(...)`, so the head was conditioned on
noise and could only learn the dataset's mean coordinate. This module provides
a stable hashed bag-of-character-n-grams representation instead: no learned
parameters, no cloud dependency, and the same function is used at train and
inference time.
"""
from __future__ import annotations

import zlib
from typing import Iterable, List

import torch


def ngram_query_embedding(query: str, dim: int) -> torch.Tensor:
    """Hashes character bigrams and trigrams of `query` into a normalized vector.

    The signed hashing trick keeps the representation dense and stable: the same
    string always maps to the same vector on any machine.
    """
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive")

    q = (query or "").lower().strip()
    ngrams: List[str] = [q[i : i + 2] for i in range(len(q) - 1)]
    ngrams += [q[i : i + 3] for i in range(len(q) - 2)]
    if not ngrams:
        ngrams = [q or "<empty>"]

    vec = torch.zeros(dim, dtype=torch.float32)
    for ng in ngrams:
        h = zlib.crc32(ng.encode("utf-8"))
        vec[h % dim] += 1.0 if (h >> 16) & 1 else -1.0

    norm = vec.norm()
    return vec / norm if norm > 0 else vec


def ngram_query_batch(queries: Iterable[str], dim: int) -> torch.Tensor:
    """Stacked embeddings for a batch of query strings."""
    return torch.stack([ngram_query_embedding(q, dim) for q in queries], dim=0)
