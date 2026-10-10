"""Shared NLP resources: the Sentence-BERT encoder and the WordNet lexicon."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from nltk.corpus.reader.wordnet import WordNetCorpusReader
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def encoder() -> SentenceTransformer:
    """Load the Sentence-BERT model once per process."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str]) -> np.ndarray:
    """Return L2-normalised embeddings (rows), so dot products are cosine similarities."""
    if not texts:
        raise ValueError("embed() needs at least one text")
    return np.asarray(encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


@lru_cache(maxsize=1)
def wordnet() -> WordNetCorpusReader:
    """Return the WordNet corpus, failing fast with the install command if it is missing."""
    from nltk.corpus import wordnet as wn

    try:
        wn.ensure_loaded()
    except LookupError as error:
        raise RuntimeError("WordNet corpus missing: run `python -m nltk.downloader wordnet`") from error
    return wn


@lru_cache(maxsize=4)
def dictionary_words(min_length: int = 3) -> frozenset[str]:
    """All single-word, purely alphabetic WordNet lemmas, upper-cased."""
    return frozenset(
        name.upper()
        for name in wordnet().all_lemma_names()
        if name.isalpha() and name.isascii() and len(name) >= min_length
    )
