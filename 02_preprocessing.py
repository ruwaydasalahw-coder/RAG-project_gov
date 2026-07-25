import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
protected_negation_words = {"no", "not", "nor", "never"}
# Define fallback dictionary and load English stopwords with offline safety net
fallback_lemma_map = {
    ("dropping", "v"): "drop",
    ("dropped", "v"): "drop",
    ("withdrawing", "v"): "withdraw",
    ("withdrawn", "v"): "withdraw",
    ("running", "v"): "run",
    ("classes", "v"): "class",
    ("studies", "v"): "study",
}

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    stop_words = {"the", "is", "and", "a", "an", "of", "to", "in", "for", "with", "on"}

# Friendly tokenizer: keeps hyphenated words and IDs together (Smart-ID,
# e-Government, e-Identity), and keeps numbers/percentages intact, instead of
# stripping punctuation and gluing/splitting entities incorrectly.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*%?")


def clean_pdf_artifacts(text):
    """Strip PDF-extraction noise (urls, page numbers, broken linebreaks)
    without touching real content. Safe no-op on non-PDF text."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", " ", text)#لو فيه أكثر من السطرين ورا بعض فاضيين (\n مكرر مرتين أو أكتر) بيستبدلهم بمسافة واحدة.
    text = re.sub(r"\s+", " ", text).strip() #سافات مكررة متتالية
    return text


def safe_word_tokenize(text):
    return TOKEN_PATTERN.findall(text)

# Protect proper nouns, hyphenated terms, and numbers from being lemmatized
def is_protected_token(token):
    # Hyphenated entities, IDs, numbers and percentages must never be
    # stemmed/lemmatized away, e.g. "Smart-ID", "e-Government", "35%", "2024".
    if re.search(r"[-0-9%]", token):
        return True
    # Capitalized tokens are treated as likely proper nouns (country names,
    # organizations: "Estonia", "OECD", "Singapore") and are left as-is
    # rather than lemmatized. Must be checked before the text is lowered.
    if token[:1].isupper():
        return True
    return False


def safe_lemmatize(token, pos="v"):
    lowered = token.lower()

    if is_protected_token(token):
        return lowered

    try:
        return lemmatizer.lemmatize(lowered, pos=pos)
    except LookupError:
        pass

    if (lowered, pos) in fallback_lemma_map:
        return fallback_lemma_map[(lowered, pos)]

    #Rule-based Stemming
    if lowered.endswith("ing") and len(lowered) > 4:
        base = lowered[:-3]
        if len(base) >= 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if lowered.endswith("ed") and len(lowered) > 3:
        return lowered[:-2]
    if lowered.endswith("s") and not lowered.endswith("ss") and len(lowered) > 3:
        return lowered[:-1]
    return lowered


def preprocess_text(text):
    text = clean_pdf_artifacts(text)
    tokens = safe_word_tokenize(text)

    tokens = [
        token
        for token in tokens
        if token.lower() not in stop_words or token.lower() in protected_negation_words
    ]
    tokens = [safe_lemmatize(token) for token in tokens]
    return " ".join(tokens)
