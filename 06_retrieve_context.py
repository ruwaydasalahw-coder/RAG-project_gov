from importlib import import_module

import chromadb
import numpy as np
from chromadb.config import Settings

vectors = import_module("04_vector_representation")
store = import_module("05_create_chroma_store")

_client = chromadb.PersistentClient(
    path=str(store.DB_PATH),
    settings=Settings(anonymized_telemetry=False),
)

try:
    _collection = _client.get_collection(store.COLLECTION_NAME)
except Exception as error:
    raise RuntimeError(
        "Chroma collection not found. Run 05_create_chroma_store.py first."
    ) from error

if _collection.count() == 0:
    raise RuntimeError(
        "Chroma collection is empty. Run 05_create_chroma_store.py first."
    )

chunk_lookup = {chunk["chunk_id"]: chunk for chunk in vectors.chunks}

# Countries are discovered from the actual dataset, not hardcoded, so this
# works for whatever country folders exist under data/.
KNOWN_COUNTRIES = sorted({chunk["country"] for chunk in vectors.chunks}, key=len, reverse=True)
COUNTRY_BOOST = 0.15


def detect_country_mentions(query):
    query_lower = query.lower()
    return {country for country in KNOWN_COUNTRIES if country.lower() in query_lower}


def semantic_search(clean_query, pool_size):
    query_embedding = vectors.encode([clean_query])
    results = _collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(pool_size, _collection.count()),
    )
    ids = results["ids"][0]
    # Chroma's default distance is cosine distance (0 = identical), so we
    # convert it back into a similarity score.
    distances = results["distances"][0]
    return {chunk_id: 1 - distance for chunk_id, distance in zip(ids, distances)}


def bm25_search(clean_query, pool_size):
    scores = vectors.bm25.get_scores(clean_query.split())
    ranking = np.argsort(scores)[::-1][:pool_size]
    return {vectors.chunks[i]["chunk_id"]: scores[i] for i in ranking}


def hybrid_search(query, k=4, pool_size=15):
    clean_query = vectors.preprocessing.preprocess_text(query)
    mentioned_countries = detect_country_mentions(query)

    bm25_candidates = bm25_search(clean_query, pool_size)
    semantic_candidates = semantic_search(clean_query, pool_size)

    candidate_ids = list(set(bm25_candidates) | set(semantic_candidates))
    bm25_values = np.array([bm25_candidates.get(cid, 0.0) for cid in candidate_ids])
    semantic_values = np.array([semantic_candidates.get(cid, 0.0) for cid in candidate_ids])

    hybrid_values = (1 - vectors.ALPHA) * vectors.min_max_normalize(bm25_values) + (
        vectors.ALPHA * vectors.min_max_normalize(semantic_values)
    )

    # If the question names a country explicitly, nudge that country's
    # chunks up. This is a small boost, not a hard filter, so a genuinely
    # more relevant cross-country (e.g. UN) chunk can still win.
    if mentioned_countries:
        for index, chunk_id in enumerate(candidate_ids):
            if chunk_lookup[chunk_id]["country"] in mentioned_countries:
                hybrid_values[index] += COUNTRY_BOOST

    ranking = np.argsort(hybrid_values)[::-1][:k]
    results = []
    for index in ranking:
        chunk_id = candidate_ids[index]
        chunk = dict(chunk_lookup[chunk_id])
        chunk["score"] = float(hybrid_values[index])
        results.append(chunk)

    return results


def build_context(question, k=4, max_sources=3):
    rows = hybrid_search(question, k=k)
    rows = sorted(rows, key=lambda row: row["score"], reverse=True)

    selected = []
    seen_documents = set()

    for row in rows:
        if row["score"] <= 0:
            continue
        if row["document_id"] in seen_documents:
            continue
        selected.append(row)
        seen_documents.add(row["document_id"])
        if len(selected) == max_sources:
            break

    context = ""
    for source_number, row in enumerate(selected, start=1):
        context += (
            f"[Source {source_number}] {row['country']} \u2014 {row['title']}\n"
            f"{row['chunk_text']}\n\n"
        )

    return context.strip(), selected
