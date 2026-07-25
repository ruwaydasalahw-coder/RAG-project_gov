from importlib import import_module
from pathlib import Path

import chromadb
from chromadb.config import Settings

vectors = import_module("04_vector_representation")

# Portable path (was a hardcoded Windows path before, which broke on any
# other machine, including the grading machine).
DB_PATH = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "course_support_docs"


def create_vector_store():
    client = chromadb.PersistentClient(
        path=str(DB_PATH),
        settings=Settings(anonymized_telemetry=False),
    )

    # Rebuild from scratch every time so chunks from an older version of the
    # documents can never linger and get mixed into retrieval results.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    chunk_embeddings = vectors.compute_chunk_embeddings()

    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in vectors.chunks],
        documents=[chunk["chunk_text"] for chunk in vectors.chunks],
        metadatas=[
            {
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "country": chunk["country"],
            }
            for chunk in vectors.chunks
        ],
        embeddings=chunk_embeddings.tolist(),
    )

    return collection


if __name__ == "__main__":
    store = create_vector_store()
    print(f"Chroma vector store created with {store.count()} chunks at {DB_PATH}")
