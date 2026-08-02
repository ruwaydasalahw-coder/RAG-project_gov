import math
import os
from importlib import import_module
from dotenv import load_dotenv
import pandas as pd
from openai import OpenAI

# استيراد دالة الاسترجاع من الملف السادس
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


def _extract_ids(doc_list):
    """دالة مساعدة لاستخراج المعرفات النصية إذا كانت المصادر القادمة عبارة عن Dictionaries"""
    clean_ids = []
    for item in doc_list:
        if isinstance(item, dict):
            # البحث عن مفتاح المعرف المناسب
            doc_id = item.get("doc_id") or item.get("source") or item.get("id")
            if not doc_id and len(item) > 0:
                doc_id = list(item.values())[0]
            clean_ids.append(str(doc_id))
        else:
            clean_ids.append(str(item))
    return clean_ids


def answer_question(question):
    context, sources = build_context(question)
    
    # تنظيف وتجهيز أسماء/معرفات المصادر لضمان سلامة التقييم
    formatted_sources = _extract_ids(sources)

    prompt = build_prompt(question, context)

    if not OPENROUTER_API_KEY:
        return "Missing OPENROUTER_API_KEY.", formatted_sources

    return ask_openrouter(prompt), formatted_sources


# ==================================================
# Ground Truth Benchmark (Updated with Real Source IDs)
# ==================================================
# ==============================================================================
# Complete Ground Truth Benchmark for Estonia (Covering all 8 PDF Factsheets)
# ==============================================================================
GROUND_TRUTH = [
    # --------------------------------------------------------------------------
    # File 1: Estonia_Digital_Identity.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 0,
        "question": "What is the primary digital identity carrier in Estonia, and what percentage of the population holds it?",
        "expected_keywords": ["id-card", "99%"],
        "relevant_docs": ["estonia_digital_identity_0"],
    },
    {
        "question_id": 1,
        "question": "How does Smart-ID differ from Mobile-ID regarding functionality in Estonia?",
        "expected_keywords": ["smart-id", "ivoting"],
        "relevant_docs": ["estonia_digital_identity_0"],
    },
    {
        "question_id": 2,
        "question": "What is the economic impact of using digital signatures in Estonia in terms of GDP?",
        "expected_keywords": ["2%", "gdp"],
        "relevant_docs": ["estonia_digital_identity_0"],
    },
    {
        "question_id": 3,
        "question": "How many working days per year does digital signature help save every citizen in Estonia?",
        "expected_keywords": ["5", "working days"],
        "relevant_docs": ["estonia_digital_identity_0"],
    },

    # --------------------------------------------------------------------------
    # File 2: Estonia_e-governance.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 4,
        "question": "By how much was the average length of weekly Estonian cabinet meetings reduced after adopting e-Cabinet?",
        "expected_keywords": ["30 minutes", "4-5 hours"],
        "relevant_docs": ["estonia_e-governance_0"],
    },
    {
        "question_id": 5,
        "question": "What principle in Estonian e-Governance ensures that citizens only have to submit their data once to the government?",
        "expected_keywords": ["once-only"],
        "relevant_docs": ["estonia_e-governance_0"],
    },
    {
        "question_id": 6,
        "question": "What rank did Estonia's e-Government achieve on the UN e-Government Survey 2024?",
        "expected_keywords": ["2nd", "un"],
        "relevant_docs": ["estonia_e-governance_1"],
    },

    # --------------------------------------------------------------------------
    # File 3: Estonia_e-health.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 7,
        "question": "How is blockchain technology utilized within the Estonian e-Health system?",
        "expected_keywords": ["blockchain", "integrity"],
        "relevant_docs": ["estonia_e-health_0", "estonia_e-health_1"],
    },
    {
        "question_id": 8,
        "question": "What percentage of medical prescriptions are issued digitally in Estonia?",
        "expected_keywords": ["100%"],
        "relevant_docs": ["estonia_e-health_0", "estonia_e-health_1"],
    },
    {
        "question_id": 9,
        "question": "How fast can the e-Ambulance solution detect and position an emergency phone call in Estonia?",
        "expected_keywords": ["30 seconds"],
        "relevant_docs": ["estonia_e-health_0"],
    },

    # --------------------------------------------------------------------------
    # File 4: Estonia_e-justice.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 10,
        "question": "What is the central information system that acts as the heart of the Estonian judicial system?",
        "expected_keywords": ["e-file"],
        "relevant_docs": ["estonia_e-justice_0"],
    },
    {
        "question_id": 11,
        "question": "What is the name of the speech recognition assistant launched in 2022 for Estonian courts?",
        "expected_keywords": ["salme"],
        "relevant_docs": ["estonia_e-justice_0"],
    },
    {
        "question_id": 12,
        "question": "How many days was the average length of Estonian court civil proceedings reduced to in 5 years?",
        "expected_keywords": ["99 days"],
        "relevant_docs": ["estonia_e-justice_0"],
    },

    # --------------------------------------------------------------------------
    # File 5: Estonia_e-residency_15-01-26.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 13,
        "question": "How long does it typically take to complete the process of becoming an e-resident of Estonia?",
        "expected_keywords": ["3-8 weeks"],
        "relevant_docs": ["estonia_e-residency_15-01-26_0"],
    },
    {
        "question_id": 14,
        "question": "How many e-residents and Estonian companies registered by e-residents were recorded by the end of 2025?",
        "expected_keywords": ["134,000", "39,000"],
        "relevant_docs": ["estonia_e-residency_15-01-26_0"],
    },

    # --------------------------------------------------------------------------
    # File 6: Estonia_proactive_government.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 15,
        "question": "What is 'Bürokratt' in Estonia, and what mythological creature is its metaphor based on?",
        "expected_keywords": ["kratt", "ai"],
        "relevant_docs": ["estonia_proactive_government_1"],
    },
    {
        "question_id": 16,
        "question": "Give an example of a live proactive service currently active in Estonia.",
        "expected_keywords": ["childbirth", "retirement"],
        "relevant_docs": ["estonia_proactive_government_0"],
    },

    # --------------------------------------------------------------------------
    # File 7: Estonia_XRoad.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 17,
        "question": "What architecture type does the X-Road platform rely on, and which non-Estonian country was first natively connected to it?",
        "expected_keywords": ["distributed", "finland"],
        "relevant_docs": ["estonia_XRoad_0"],
    },
    {
        "question_id": 18,
        "question": "How much working time was saved by X-Road last year in Estonia?",
        "expected_keywords": ["2589", "2,589", "years"],
        "relevant_docs": ["estonia_XRoad_0"],
    },

    # --------------------------------------------------------------------------
    # File 8: Estoniacyber-security.pdf
    # --------------------------------------------------------------------------
    {
        "question_id": 19,
        "question": "Which major incident in 2007 led to the founding of the NATO CCD COE in Tallinn?",
        "expected_keywords": ["cyber attacks", "2007"],
        "relevant_docs": ["estoniacyber-security_0"],
    },
    {
        "question_id": 20,
        "question": "Which national authority in Estonia monitors compliance with cyber security requirements 24/7?",
        "expected_keywords": ["ria", "information system authority"],
        "relevant_docs": ["estoniacyber-security_0"],
           "relevant_docs": ["estoniacyber-security_0"],
    },    
   # =========================================================
   # Complete Ground Truth Benchmark for Singapore

     {
        "question_id": 0,
        "question_id": 0,
        "question": "What is the primary vision and slogan of the Digital Government Blueprint (DGB) in Singapore?",
        "expected_keywords": ["digital to the core", "serves with heart"],
        "relevant_docs": ["singapore_dgb_0"],
    },
    {
        "question_id": 1,
        "question": "What percentage of public services in Singapore are required to provide end-to-end digital options, pre-filled data, and e-payment options?",
        "expected_keywords": ["100%"],
        "relevant_docs": ["singapore_dgb_0"],
    },
    {
        "question_id": 2,
        "question": "What proportion of non-sensitive government systems in Singapore is targeted to be migrated to the commercial cloud?",
        "expected_keywords": ["70%"],
        "relevant_docs": ["singapore_dgb_0"],
    },
    {
        "question_id": 3,
        "question": "What is the targeted maximum timeframe for sharing data between government agencies for joint projects?",
        "expected_keywords": ["7 working days"],
        "relevant_docs": ["singapore_dgb_0"],
    },
    {
        "question_id": 4,
        "question": "What is CODEX in Singapore's digital government architecture, and what component forms its core tech stack?",
        "expected_keywords": ["codex", "sgts", "singapore government technology stack"],
        "relevant_docs": ["singapore_dgb_0"],
    },
    {
        "question_id": 5,
        "question": "Which platforms in Singapore provide citizens and businesses with single-entry digital services for life events and business licenses respectively?",
        "expected_keywords": ["lifesg", "gobusiness"],
        "relevant_docs": ["singapore_dgb_0"],
    },

    # --------------------------------------------------------------------------
    # File 2: National_AI_Strategy_2.0.pdf (NAIS 2.0)
    # --------------------------------------------------------------------------
    {
        "question_id": 6,
        "question": "What is the central vision and dual goals of Singapore's National AI Strategy 2.0 (NAIS 2.0)?",
        "expected_keywords": ["ai for the public good", "excellence", "empowerment"],
        "relevant_docs": ["singapore_nais2_0"],
    },
    {
        "question_id": 7,
        "question": "What are the three core strategic shifts introduced in NAIS 2.0 compared to NAIS 1.0?",
        "expected_keywords": ["necessity", "global", "systems"],
        "relevant_docs": ["singapore_nais2_0"],
    },
    {
        "question_id": 8,
        "question": "How many AI practitioners does Singapore target to train and grow under NAIS 2.0?",
        "expected_keywords": ["15,000", "practitioners"],
        "relevant_docs": ["singapore_nais2_0"],
    },
    {
        "question_id": 9,
        "question": "What internal AI tool was introduced to enhance public sector productivity in Singapore?",
        "expected_keywords": ["pair"],
        "relevant_docs": ["singapore_nais2_0"],
    },
    {
        "question_id": 10,
        "question": "How many key actions and core systems are defined in NAIS 2.0 to build the AI ecosystem?",
        "expected_keywords": ["3 systems", "15 actions"],
        "relevant_docs": ["singapore_nais2_0"],
    },

    # --------------------------------------------------------------------------
    # File 3: Smart_Nation_2.0.pdf (Smart Nation 2.0)
    # --------------------------------------------------------------------------
    {
        "question_id": 11,
        "question": "What is the overall vision and the three main strategic pillars of Smart Nation 2.0?",
        "expected_keywords": ["thriving digital future", "growth", "community", "trust"],
        "relevant_docs": ["singapore_smartnation2_0"],
    },
    {
        "question_id": 12,
        "question": "How many lines of effort (implementation pillars) guide the deployment of Smart Nation 2.0?",
        "expected_keywords": ["5 lines of effort", "infrastructure", "capability", "regulations", "inclusion", "international"],
        "relevant_docs": ["singapore_smartnation2_0"],
    },
    {
        "question_id": 13,
        "question": "Which national initiative under Smart Nation 2.0 drives digital inclusion to ensure no citizen is left behind?",
        "expected_keywords": ["digital for life"],
        "relevant_docs": ["singapore_smartnation2_0"],
    },
# ==============================================================================
# Complete Ground Truth Benchmark for South Korea Digital Government Frameworks
# ==============================================================================

    # --------------------------------------------------------------------------
    # File 1: OECD_Digital_Government_Review_Korea.pdf (OECD Review)
    # --------------------------------------------------------------------------
    {
        "question_id": 0,
        "question": "Which government ministry is primarily responsible for steering and coordinating digital government initiatives in South Korea?",
        "expected_keywords": ["mois", "ministry of the interior and safety"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },
    {
        "question_id": 1,
        "question": "What is South Korea's position in the OECD Digital Government Index (DGI) and Open Useful Re-usable Data (OURdata) Index?",
        "expected_keywords": ["1st", "top", "leader"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },
    {
        "question_id": 2,
        "question": "What are the four historical evolutionary phases of South Korea's digital government development?",
        "expected_keywords": ["computerization", "backbone network", "governance", "data-driven"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },
    {
        "question_id": 3,
        "question": "What key legislation enacted in 2001 established the legal foundation for electronic government processes in Korea?",
        "expected_keywords": ["electronic government act"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },

    # --------------------------------------------------------------------------
    # File 2: Digital_Platform_Government_Strategy.pdf (DPG Strategy)
    # --------------------------------------------------------------------------
    {
        "question_id": 4,
        "question": "What is the primary vision of the Digital Platform Government (DPG) initiative in South Korea?",
        "expected_keywords": ["digital platform government", "one-government", "data-driven platform"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },
    {
        "question_id": 5,
        "question": "Which flagship portal serves as the primary centralized portal for Korean citizens to access thousands of administrative services?",
        "expected_keywords": ["government24"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },
    {
        "question_id": 6,
        "question": "What initiative allows Korean citizens to manage, control, and transfer their personal administrative data across public and private platforms?",
        "expected_keywords": ["mydata"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },
    {
        "question_id": 7,
        "question": "What is the standard design framework used to unify UI/UX design across South Korean public services?",
        "expected_keywords": ["krds", "korean government-wide ui/ux design system"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },

    # --------------------------------------------------------------------------
    # File 3: Korea_Public_Sector_AI_and_Data_Policy.pdf (Data & AI Policy)
    # --------------------------------------------------------------------------
    {
        "question_id": 8,
        "question": "Which key law in South Korea promotes the sharing, management, and usage of data across public administration for evidence-based policy making?",
        "expected_keywords": ["act on data-based administration"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
    {
        "question_id": 9,
        "question": "What foundational framework governs ethical deployment and trustworthy AI standards in the Korean public sector?",
        "expected_keywords": ["national ai ethics standards", "ai safety institute"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
    {
        "question_id": 10,
        "question": "What key recommendation did the OECD issue regarding civil servant human resources in Korea's technical sectors?",
        "expected_keywords": ["limit mandatory role rotation", "preserve expertise"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
    {
        "question_id": 11,
        "question": "What mechanism does the OECD recommend Korea adopt to ensure public transparency and governance of AI algorithms in government?",
        "expected_keywords": ["public registry", "ai registry"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
# ==============================================================================
# Complete Ground Truth Benchmark for South Korea Digital Government Frameworks
# ==============================================================================

    # --------------------------------------------------------------------------
    # File 1: OECD_Digital_Government_Review_Korea.pdf (OECD Review)
    # --------------------------------------------------------------------------
    {
        "question_id": 0,
        "question": "Which government ministry is primarily responsible for steering and coordinating digital government initiatives in South Korea?",
        "expected_keywords": ["mois", "ministry of the interior and safety"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },
    {
        "question_id": 1,
        "question": "What is South Korea's position in the OECD Digital Government Index (DGI) and Open Useful Re-usable Data (OURdata) Index?",
        "expected_keywords": ["1st", "top", "leader"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },
    {
        "question_id": 2,
        "question": "What are the four historical evolutionary phases of South Korea's digital government development?",
        "expected_keywords": ["computerization", "backbone network", "governance", "data-driven"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },
    {
        "question_id": 3,
        "question": "What key legislation enacted in 2001 established the legal foundation for electronic government processes in Korea?",
        "expected_keywords": ["electronic government act"],
        "relevant_docs": ["korea_oecd_dgr_0"],
    },

    # --------------------------------------------------------------------------
    # File 2: Digital_Platform_Government_Strategy.pdf (DPG Strategy)
    # --------------------------------------------------------------------------
    {
        "question_id": 4,
        "question": "What is the primary vision of the Digital Platform Government (DPG) initiative in South Korea?",
        "expected_keywords": ["digital platform government", "one-government", "data-driven platform"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },
    {
        "question_id": 5,
        "question": "Which flagship portal serves as the primary centralized portal for Korean citizens to access thousands of administrative services?",
        "expected_keywords": ["government24"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },
    {
        "question_id": 6,
        "question": "What initiative allows Korean citizens to manage, control, and transfer their personal administrative data across public and private platforms?",
        "expected_keywords": ["mydata"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },
    {
        "question_id": 7,
        "question": "What is the standard design framework used to unify UI/UX design across South Korean public services?",
        "expected_keywords": ["krds", "korean government-wide ui/ux design system"],
        "relevant_docs": ["korea_dpg_strategy_0"],
    },

    # --------------------------------------------------------------------------
    # File 3: Korea_Public_Sector_AI_and_Data_Policy.pdf (Data & AI Policy)
    # --------------------------------------------------------------------------
    {
        "question_id": 8,
        "question": "Which key law in South Korea promotes the sharing, management, and usage of data across public administration for evidence-based policy making?",
        "expected_keywords": ["act on data-based administration"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
    {
        "question_id": 9,
        "question": "What foundational framework governs ethical deployment and trustworthy AI standards in the Korean public sector?",
        "expected_keywords": ["national ai ethics standards", "ai safety institute"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
    {
        "question_id": 10,
        "question": "What key recommendation did the OECD issue regarding civil servant human resources in Korea's technical sectors?",
        "expected_keywords": ["limit mandatory role rotation", "preserve expertise"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
    {
        "question_id": 11,
        "question": "What mechanism does the OECD recommend Korea adopt to ensure public transparency and governance of AI algorithms in government?",
        "expected_keywords": ["public registry", "ai registry"],
        "relevant_docs": ["korea_ai_data_policy_0"],
    },
# ==============================================================================
# Complete Ground Truth Benchmark for UN E-Government Development Index (EGDI)
# ==============================================================================

    # --------------------------------------------------------------------------
    # Section 1: Core Formula & Overall EGDI Architecture
    # --------------------------------------------------------------------------
    {
        "question_id": 0,
        "question": "What is the primary formula used to calculate the UN E-Government Development Index (EGDI)?",
        "expected_keywords": ["weighted average", "1/3 osi", "1/3 tii", "1/3 hci"],
        "relevant_docs": ["un_egdi_2024_report_ch1"],
    },
    {
        "question_id": 1,
        "question": "What are the three composite component indices that make up the EGDI?",
        "expected_keywords": ["online services index", "telecommunications infrastructure index", "human capital index", "osi", "tii", "hci"],
        "relevant_docs": ["un_egdi_2024_report_ch1"],
    },
    {
        "question_id": 2,
        "question": "What are the four primary rating tiers used by the UN to categorize countries' EGDI performance?",
        "expected_keywords": ["very high", "high", "middle", "low", "vh", "h", "m", "l"],
        "relevant_docs": ["un_egdi_2024_report_ch1"],
    },

    # --------------------------------------------------------------------------
    # Section 2: Online Services Index (OSI) & E-Participation (EPI)
    # --------------------------------------------------------------------------
    {
        "question_id": 3,
        "question": "Which five criteria form the assessment framework of the Online Services Index (OSI) in the 2024 Survey?",
        "expected_keywords": ["institutional framework", "services provision", "content provision", "technology", "e-participation index"],
        "relevant_docs": ["un_egdi_2024_report_ch2"],
    },
    {
        "question_id": 4,
        "question": "What are the three levels of public engagement measured under the E-Participation Index (EPI)?",
        "expected_keywords": ["e-information", "e-consultation", "e-decision-making"],
        "relevant_docs": ["un_egdi_2024_report_ch2"],
    },

    # --------------------------------------------------------------------------
    # Section 3: Telecommunications Infrastructure & Human Capital (TII & HCI)
    # --------------------------------------------------------------------------
    {
        "question_id": 5,
        "question": "Which new subindex was introduced into the Telecommunications Infrastructure Index (TII) in 2024 to replace fixed broadband subscriptions?",
        "expected_keywords": ["broadband affordability"],
        "relevant_docs": ["un_egdi_2024_report_ch3"],
    },
    {
        "question_id": 6,
        "question": "What five sub-components constitute the Human Capital Index (HCI) in the latest assessment?",
        "expected_keywords": ["adult literacy rate", "gross enrolment ratio", "expected years of schooling", "mean years of schooling", "e-government literacy"],
        "relevant_docs": ["un_egdi_2024_report_ch3"],
    },
    {
        "question_id": 7,
        "question": "Which entity provides the primary raw data for evaluating the Telecommunications Infrastructure Index (TII)?",
        "expected_keywords": ["itu", "international telecommunication union"],
        "relevant_docs": ["un_egdi_2024_report_ch3"],
    },

    # --------------------------------------------------------------------------
    # Section 4: Subnational Evaluation (LOSI) & Methodology Principles
    # --------------------------------------------------------------------------
    {
        "question_id": 8,
        "question": "What extension of the EGDI methodology is used to evaluate digital government services at the local/city level?",
        "expected_keywords": ["losi", "local online services index"],
        "relevant_docs": ["un_egdi_2024_report_ch4"],
    },
    {
        "question_id": 9,
        "question": "What scoring system is predominantly used in evaluating online portals to maintain objective assessments?",
        "expected_keywords": ["binary scoring", "0 or 1", "binary system"],
        "relevant_docs": ["un_egdi_2024_report_ch5"],
    },
]

# ==================================================
# Retrieval Evaluation Metrics (Lab 6 Compliant)
# ==================================================
def compute_precision_at_k(retrieved, relevant, k):
    retrieved_k = _extract_ids(retrieved[:k])
    relevant_ids = _extract_ids(relevant)
    if not retrieved_k:
        return 0.0
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / k


def compute_recall_at_k(retrieved, relevant, k):
    retrieved_k = _extract_ids(retrieved[:k])
    relevant_ids = _extract_ids(relevant)
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_k) & set(relevant_ids))
    return hits / len(relevant_ids)


def compute_hit_rate_at_k(retrieved, relevant, k):
    retrieved_k = _extract_ids(retrieved[:k])
    relevant_ids = _extract_ids(relevant)
    hits = len(set(retrieved_k) & set(relevant_ids))
    return 1.0 if hits > 0 else 0.0


def compute_mrr(retrieved, relevant):
    retrieved_ids = _extract_ids(retrieved)
    relevant_ids = _extract_ids(relevant)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# ==================================================
# Full Pipeline Evaluation
# ==================================================
def evaluate_pipeline(k=3):
    if any(case["question"].startswith("TODO") for case in GROUND_TRUTH):
        print("GROUND_TRUTH contains placeholders. Update before evaluating.")
        return

    metrics_list = []
    passed_answers = 0

    print("=" * 60)
    print(f"🚀 RUNNING PIPELINE EVALUATION (Retrieval Metrics @ K={k} + Answer Quality)")
    print("=" * 60)

    for case in GROUND_TRUTH:
        q = case["question"]
        expected_kw = case["expected_keywords"]
        relevant_docs = case.get("relevant_docs", [])

        # 1. تشغيل الـ Pipeline
        answer, retrieved_sources = answer_question(q)

        # 2. تقييم دقة الإجابة (Keyword Match)
        answer_lower = answer.lower()
        kw_passed = all(kw.lower() in answer_lower for kw in expected_kw)
        if kw_passed:
            passed_answers += 1

        # 3. تقييم كفاءة الاسترجاع
        prec = compute_precision_at_k(retrieved_sources, relevant_docs, k)
        rec = compute_recall_at_k(retrieved_sources, relevant_docs, k)
        hit = compute_hit_rate_at_k(retrieved_sources, relevant_docs, k)
        mrr = compute_mrr(retrieved_sources, relevant_docs)

        metrics_list.append(
            {
                "Question ID": case["question_id"],
                "KW Pass": "PASS" if kw_passed else "FAIL",
                f"Precision@{k}": prec,
                f"Recall@{k}": rec,
                f"Hit Rate@{k}": hit,
                "MRR": mrr,
            }
        )

        status_icon = "✅" if kw_passed else "❌"
        print(f"\n[{status_icon}] Question: {q}")
        print(f"👉 Expected Keywords: {expected_kw}")
        print(f"📚 Retrieved Sources: {retrieved_sources}")
        print(f"🤖 Answer Output: {answer[:120]}... (truncated)")
        print(f"📊 Retrieval Score: Precision@{k}: {prec:.2f} | Recall@{k}: {rec:.2f} | MRR: {mrr:.2f}")
        print("-" * 60)

    # عرض الجدول الملخص
    df_results = pd.DataFrame(metrics_list)
    accuracy = passed_answers / len(GROUND_TRUTH)

    print("\n" + "=" * 60)
    print("📈 AGGREGATED EVALUATION SUMMARY")
    print("=" * 60)
    print(df_results.to_string(index=False))
    print("-" * 60)
    print(f"🎯 Final Keyword Answer Accuracy: {accuracy:.0%} ({passed_answers}/{len(GROUND_TRUTH)})")
    print(f"📌 Mean Precision@{k}: {df_results[f'Precision@{k}'].mean():.2f}")
    print(f"📌 Mean Recall@{k}: {df_results[f'Recall@{k}'].mean():.2f}")
    print(f"📌 Hit Rate@{k}: {df_results[f'Hit Rate@{k}'].mean():.0%}")
    print(f"📌 MRR (Mean Reciprocal Rank): {df_results['MRR'].mean():.2f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_pipeline(k=3)