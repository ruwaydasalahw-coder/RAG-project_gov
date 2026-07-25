from importlib import import_module

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import time

start = time.time()
preprocessing = import_module("02_preprocessing")
chunks = import_module("03_chunking").build_chunks()
print("Chunking time:", time.time() - start)

ALPHA = 0.6
MODEL_NAME = "all-MiniLM-L6-v2"

# BM25 is cheap to build, so we still build it eagerly.
tokenized_chunks = [chunk["search_text"].split() for chunk in chunks]
bm25 = BM25Okapi(tokenized_chunks)
print("BM25 ready:", time.time() - start)

# The embedding model is the expensive part. Load it lazily and only once
# per process, instead of loading it (and re-encoding every chunk) on every
# import. This is what made retrieval slow to start up.
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)


# 1. عرف المتغير في أصل الملف (Global Variable)
_model = None


def get_model():
    global _model  # 2. استدعي المتغير العام داخل الدالة

    if _model is None:
        from sentence_transformers import SentenceTransformer

        # استبدل باسم النموذج الذي تستخدمه
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model
print("Model loaded:", time.time() - start)


def encode(texts):
    return get_model().encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def compute_chunk_embeddings():
    """Encode every chunk. Only called when (re)building the Chroma store,
    not on every app/query start."""
    return encode([chunk["search_text"] for chunk in chunks])


def min_max_normalize(scores):
    scores = np.array(scores, dtype=float)
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())
