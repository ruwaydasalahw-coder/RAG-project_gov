from importlib import import_module
import pickle
from pathlib import Path


preprocess_text = import_module("02_preprocessing").preprocess_text

# Long government reports need bigger windows than short FAQ-style text so a
# chunk holds a complete thought instead of a sentence fragment. 250 words
# with a 50-word overlap keeps context continuous across chunk boundaries.
CHUNK_SIZE = 250
CHUNK_OVERLAP = 50
CHUNKS_CACHE = Path(__file__).resolve().parent / "chunks_cache.pkl"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap

    return chunks


def build_chunks():
    import time

    # لو الـ chunks محفوظة بالفعل، اقرأها مباشرة
    if CHUNKS_CACHE.exists():
        print("Loading chunks from cache...")
        with open(CHUNKS_CACHE, "rb") as f:
            return pickle.load(f)

    start = time.time()
    print("Building chunks...")

    documents = import_module("01_documents").get_documents()

    rows = []

    for document in documents:
        for chunk_number, chunk in enumerate(chunk_text(document["text"])):
            rows.append(
                {
                    "chunk_id": f"{document['id']}_{chunk_number}",
                    "document_id": document["id"],
                    "title": document["title"],
                    "country": document["country"],
                    "chunk_text": chunk,
                    "search_text": preprocess_text(
                        f"{document['country']} {document['title']} {chunk}"
                    ),
                }
            )

    print(f"Chunks built in {time.time()-start:.2f} sec")

    # نحفظ الـ chunks بعد أول مرة
    with open(CHUNKS_CACHE, "wb") as f:
        pickle.dump(rows, f)

    return rows
