import re
import pickle
from collections import Counter #تحسب كل عنصر اتكرر كم مرة (هنستخدمها في حذف الهيدر والفوتر).
from pathlib import Path #أفضل من os.path للتعامل مع مسارات الملفات بطريقة منظمة

from pypdf import PdfReader #pdfplumber أدق في استخراج النصوص وبيحتفظ بتنسيق السطور أفضل، خصوصاً للتقرير والمستندات الرسمية

DATA_DIR = Path(__file__).resolve().parent / "digital government data" # Define absolute path to the data directory relative to this file
CACHE_FILE = Path(__file__).resolve().parent / "documents_cache.pkl"
PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?\s*$", re.IGNORECASE # Regex pattern to match various page number formats (e.g., "1", "Page 2", "3 of 10")
)

# Extract raw text from each page of a PDF using pdfplumber
def extract_pdf_pages(pdf_path):
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]

# Remove dynamic headers/footers appearing in >=60% of document pages
def strip_repeated_lines(pages):
    """Drop lines that repeat across most pages of the same PDF (running
    headers/footers). This generalizes across any report regardless of its
    specific header/footer wording, instead of hardcoding phrases."""
    if len(pages) < 3:
        return pages

    page_lines = [[line.strip() for line in page.splitlines()] for page in pages]

    line_counts = Counter()
    for lines in page_lines:
        for line in set(lines):
            if line:
                line_counts[line] += 1

    threshold = max(3, int(len(pages) * 0.6))
    repeated = {line for line, count in line_counts.items() if count >= threshold}

    return ["\n".join(line for line in lines if line not in repeated) for lines in page_lines]


def strip_page_numbers(text):
    lines = text.splitlines()
    return "\n".join(line for line in lines if not PAGE_NUMBER_PATTERN.match(line))

# Full pipeline: Extract, clean, and combine PDF text into a single string
def load_pdf_text(pdf_path):
    pages = extract_pdf_pages(pdf_path)
    pages = strip_repeated_lines(pages)
    pages = [strip_page_numbers(page) for page in pages]
    return "\n".join(page for page in pages if page.strip())


def load_documents():
    if not DATA_DIR.exists():
        raise RuntimeError(
            f"Data folder not found at {DATA_DIR}. Expected data/<country>/*.pdf "
            "(e.g. data/Estonia/report.pdf)."
        )

    documents = []

    for country_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for pdf_path in sorted(country_dir.glob("*.pdf")):
            try:
                text = load_pdf_text(pdf_path)
            except Exception as error:
                print(f"WARNING: could not read {pdf_path}: {error}")
                continue

            if not text.strip():
                print(f"WARNING: no extractable text in {pdf_path} (skipped)")
                continue

            documents.append(
                {
                    "id": f"{country_dir.name}_{pdf_path.stem}".lower().replace(" ", "_"),
                    "title": pdf_path.stem,
                    "country": country_dir.name,
                    "text": text,
                }
            )

    if not documents:
        raise RuntimeError(f"No readable PDF documents found under {DATA_DIR}.")

    return documents


_documents = None

def get_documents():
    global _documents

    if _documents is not None:
        return _documents

    # لو الكاش موجود اقرأه مباشرة
    if CACHE_FILE.exists():
        print("Loading documents from cache...")
        with open(CACHE_FILE, "rb") as f:
            _documents = pickle.load(f)
        return _documents

    # أول مرة فقط
    print("Reading PDF files...")
    _documents = load_documents()

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(_documents, f)

    return _documents

