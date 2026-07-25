from importlib import import_module

import streamlit as st

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🏛️",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def load_rag():
    return import_module("07_prompting")


st.title("🌍 Knowledge Transfer Between Nations")
st.markdown("""
### Cross-Lingual RAG Engine

Connecting knowledge across countries through AI-powered semantic retrieval.

---
**🌟 Current Domain**
- 🏛️ Digital Government

**🚀 Future Expansion**
- 💧 Water Management
- 🎓 Education
- 🌾 Agriculture
- ⚡ Energy
- 🏥 Healthcare

---
Ask a question below to explore knowledge from government reports.
""")

try:
    rag = load_rag()
except RuntimeError as error:
    st.error(str(error))
    st.stop()

try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass

question = st.text_area(
    "Question",
    placeholder="Type your question here...",
    height=120,
)

if st.button("🔍 Get Answer", use_container_width=True) and question.strip():

    with st.spinner("Searching documents..."):
        answer, sources = rag.answer_question(question)

    st.subheader("Answer")
    st.success(answer)

    with st.expander("📚 Retrieved Sources"):
        for i, source in enumerate(sources, start=1):
            st.markdown(f"### Source {i}")
            st.write(f"**Country:** {source['country']}")
            st.write(f"**Document:** {source['title']}")
            st.info(source["chunk_text"])