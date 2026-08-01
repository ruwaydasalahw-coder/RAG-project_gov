from importlib import import_module
import os

from dotenv import load_dotenv
from openai import OpenAI

build_context = import_module("06_retrieve_context").build_context

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")


def build_prompt(question, context):
    return f"""You are a careful grounded assistant answering questions about government reports from multiple countries and organizations.
Use ONLY the provided context.
If the context is not enough, say you do not know.

CRITICAL INSTRUCTIONS FOR CITATIONS & SOURCES:
1. Pay attention to which country or organization each source belongs to, and do not mix up facts between countries.
2. Review ALL provided sources thoroughly before answering.
3. If the answer synthesizes information from multiple sources, you MUST cite EVERY source used (e.g., [Source 1], [Source 2]).
4. Do NOT rely on a single source if other provided sources contain relevant details for the question.

Question:
{question}

Context:
{context}
"""


def ask_openrouter(prompt):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content


def answer_question(question):
    context, sources = build_context(question)
    prompt = build_prompt(question, context)

    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY.", sources

    return ask_openrouter(prompt), sources


# ==================================================
# Ground Truth evaluation
# ==================================================
# Small, simple check that the pipeline retrieves and answers correctly.
# This is a sanity check, not a benchmark: passing these questions does not
# mean retrieval generalizes to unseen questions.
#
# IMPORTANT: these questions are PLACEHOLDERS. They were not written from
# your actual PDFs (this environment does not have your data/ folder), so
# they will not mean anything until you replace them. Pick 4-6 real
# questions you already know the answer to from your Estonia/OECD/
# Singapore/South Korea/UN reports, and list a couple of exact keywords
# you'd expect in a correct answer for each one.

GROUND_TRUTH = [
    {
        "question": "What is the primary digital identity carrier in Estonia, and what percentage of the population holds it?",
        "expected_keywords": ["id-card", "99%"],
    },
    {
        "question": "How does Smart-ID differ from other digital identity carriers regarding functionality?",
        "expected_keywords": ["smart-id", "i-voting"],
    },
    {
        "question": "What is the economic and operational impact of digital signatures in Estonia?",
        "expected_keywords": ["2%", "gdp", "5 days"],
    },
    {
        "question": "What is X-Road, and what type of technical architecture does it use?",
        "expected_keywords": ["x-road", "distributed"],
    },
    {
        "question": "How does the Estonian Health Information System handle data security and patient privacy?",
        "expected_keywords": ["blockchain", "patient"],
    },
    {
        "question": "How does Estonia score on the Digital Government Index (DGI) compared to the OECD average?",
        "expected_keywords": ["0.83", "0.70"],
    },
]


def evaluate_ground_truth():
    if any(case["question"].startswith("TODO") for case in GROUND_TRUTH):
        print(
            "GROUND_TRUTH still contains placeholder questions. "
            "Replace them with real questions/keywords from your PDFs."
        )
        return None

    passed = 0

    for case in GROUND_TRUTH:
        answer, _ = answer_question(case["question"])
        answer_lower = answer.lower()
        ok = all(keyword.lower() in answer_lower for keyword in case["expected_keywords"])

        status = "PASS" if ok else "FAIL"

        # ------------------ طباعة النتائج التفصيلية ------------------
        print(f"[{status}] Question: {case['question']}")
        print(f"👉 Expected Keywords: {case['expected_keywords']}")
        print(f"🤖 Generated Answer: {answer}")
        print("-" * 50)
        # -------------------------------------------------------------
        if ok:
            passed += 1

    accuracy = passed / len(GROUND_TRUTH)
    print(f"\nAccuracy: {accuracy:.0%} ({passed}/{len(GROUND_TRUTH)})")
    return accuracy


if __name__ == "__main__":
    evaluate_ground_truth()
