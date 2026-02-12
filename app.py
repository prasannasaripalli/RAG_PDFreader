import os
import io
import re
import hashlib
import streamlit as st
import pandas as pd
from pypdf import PdfReader
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 0) Page config MUST be first
# -----------------------------
st.set_page_config(page_title="RAG Test Case Generator", layout="wide")
st.title("RAG Test Case Generator from PDFs")
st.caption("Upload PDF(s), paste acceptance criteria and generate test cases.")


# -----------------------------
# 0.1) Unicode cleanup to avoid surrogate errors
# -----------------------------
_SURROGATES = re.compile(r"[\ud800-\udfff]")

def clean_text(s: str) -> str:
    s = s or ""
    s = _SURROGATES.sub("", s)
    return s.encode("utf-8", "replace").decode("utf-8")


# -----------------------------
# 1) OpenAI key handling
# -----------------------------
def get_openai_key() -> str:
    key = st.secrets.get("OPENAI_API_KEY", None) or os.getenv("OPENAI_API_KEY", None)
    if not key:
        st.error("OPENAI_API_KEY is missing. Set it in Streamlit Secrets or as an environment variable.")
        st.stop()
    if not isinstance(key, str):
        st.error("OPENAI_API_KEY must be a string (not a callable/coroutine).")
        st.stop()
    key = key.strip()
    if not key:
        st.error("OPENAI_API_KEY is empty. Fix your Streamlit Secrets or environment variable.")
        st.stop()
    return key

OPENAI_API_KEY = get_openai_key()
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", None) or os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# -----------------------------
# 2) Schema (strict for OpenAI structured output)
# -----------------------------
class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    expected_result: str

class TestDataItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    value: str

class TestCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    type: str = Field(..., description="functional|negative|boundary|regression|api|ui")
    priority: str = Field(..., description="P0|P1|P2")
    preconditions: List[str] = []
    test_data: List[TestDataItem] = []
    steps: List[Step] = []
    tags: List[str] = []
    notes: Optional[str] = None

class TestSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_name: str
    test_cases: List[TestCase]


# -----------------------------
# 3) PDF utilities
# -----------------------------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        t = clean_text(page.extract_text() or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        start = max(end - overlap, 0)
        if end == n:
            break
    return [clean_text(c) for c in chunks if c.strip()]

def hash_pdfs(pdf_files) -> str:
    h = hashlib.sha256()
    for f in pdf_files:
        h.update(f.getvalue())
    return h.hexdigest()


# -----------------------------
# 4) Build FAISS index (cached)
# -----------------------------
@st.cache_resource(show_spinner=False)
def build_faiss_index(pdf_hash: str, pdf_bytes_list: List[bytes]) -> FAISS:
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

    all_chunks = []
    for b in pdf_bytes_list:
        text = pdf_bytes_to_text(b)
        all_chunks.extend(chunk_text(text))

    all_chunks = [clean_text(c) for c in all_chunks if c.strip()]
    if not all_chunks:
        raise RuntimeError(
            "No text could be extracted from the PDF(s). If scanned images, OCR is required."
        )

    return FAISS.from_texts(all_chunks, embeddings)

def retrieve_context(vs: FAISS, query: str, top_k: int) -> str:
    docs = vs.similarity_search(clean_text(query), k=top_k)
    return clean_text("\n\n".join(d.page_content for d in docs))


# -----------------------------
# 5) LLM generator
# -----------------------------
PROMPT = """
You are a QA Engineer. Generate {num_cases} test cases based on acceptance criteria and PDF context.

Requirement:
{requirement}

Acceptance Criteria:
{acceptance_criteria}

Relevant PDF Context:
{context}

Rules:
- Include happy path, negative, boundary/edge, and at least 2 regression cases.
- Steps must be atomic and each step must include expected_result.
- Avoid duplicates (unique titles).
- If something is unclear, add a short note in notes (do not invent behavior).
- test_data MUST be a list of objects like: [{{"key":"...", "value":"..."}}]
Return ONLY the JSON object matching the schema.
"""

TOP_K = 6  # fixed value; removed from sidebar

def generate_test_suite(vs: FAISS, requirement: str, acceptance: str, num_cases: int) -> TestSuite:
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.1, api_key=OPENAI_API_KEY)
    chain = ChatPromptTemplate.from_template(PROMPT) | llm.with_structured_output(TestSuite)

    requirement = clean_text(requirement).strip()
    acceptance = clean_text(acceptance).strip()

    context = retrieve_context(vs, f"{requirement}\n{acceptance}", top_k=TOP_K)

    suite = chain.invoke({
        "requirement": requirement,
        "acceptance_criteria": acceptance,
        "context": context,
        "num_cases": int(num_cases),
    })

    # Dedupe by title
    seen = set()
    deduped = []
    for tc in suite.test_cases:
        title = clean_text(tc.title).strip()
        key = title.lower()
        if title and key not in seen:
            tc.title = title
            seen.add(key)
            deduped.append(tc)
    suite.test_cases = deduped

    suite.suite_name = clean_text((suite.suite_name or "").strip()) or "Generated Test Suite"
    return suite


# -----------------------------
# 6) Session state (persist results across reruns)
# -----------------------------
st.session_state.setdefault("suite", None)
st.session_state.setdefault("selected_tc_id", None)


# -----------------------------
# 7) UI
# -----------------------------
with st.sidebar:
    st.header("Upload PDFs")
    pdf_files = st.file_uploader("Select one or more PDFs", type=["pdf"], accept_multiple_files=True)

    st.header("Settings")
    num_cases = st.slider("Number of test cases", 6, 20, 12, 1)

    if st.button("Clear results"):
        st.session_state.suite = None
        st.session_state.selected_tc_id = None


c1, c2 = st.columns(2)
with c1:
    requirement = st.text_area("Requirement / User Story", height=100, value="Student API: Create a Student")
with c2:
    acceptance = st.text_area(
        "Acceptance Criteria",
        height=160,
        value="Validate input; age 0–120; grade allowed list; email must be unique; return 201 or 400/409 accordingly"
    )

generate_btn = st.button("Generate Test Cases", type="primary")

if not pdf_files:
    st.info("Upload PDF(s) in the sidebar to begin.")
    st.stop()

# Build index (cached) every run but fast due to cache
try:
    pdf_hash = hash_pdfs(pdf_files)
    pdf_bytes_list = [f.getvalue() for f in pdf_files]
    with st.spinner("Indexing PDFs (FAISS)..."):
        vs = build_faiss_index(pdf_hash, pdf_bytes_list)
except Exception as e:
    st.error(f"Failed to build index: {e}")
    st.stop()

# Generate ONLY when button is clicked
if generate_btn:
    if not requirement.strip() or not acceptance.strip():
        st.error("Please enter both requirement and acceptance criteria.")
        st.stop()

    try:
        with st.spinner("Generating test cases..."):
            st.session_state.suite = generate_test_suite(vs, requirement, acceptance, num_cases)

        if not st.session_state.suite.test_cases:
            st.warning("No test cases generated. Try increasing num_cases or improving acceptance criteria.")
            st.session_state.suite = None
            st.stop()

        # Set default selection
        st.session_state.selected_tc_id = st.session_state.suite.test_cases[0].id
        st.success(f"Generated {len(st.session_state.suite.test_cases)} test cases")

    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

# Render from session_state (so selection doesn't wipe results)
suite = st.session_state.get("suite", None)
if not suite:
    st.info("Click **Generate Test Cases** after uploading PDFs.")
    st.stop()

st.subheader(suite.suite_name)

df = pd.DataFrame([{
    "id": tc.id,
    "title": tc.title,
    "type": tc.type,
    "priority": tc.priority
} for tc in suite.test_cases])

st.dataframe(df, use_container_width=True, hide_index=True)

ids = df["id"].tolist()
if not ids:
    st.warning("No test cases available. Generate again.")
    st.stop()

# Keep selection stable across reruns
if st.session_state.selected_tc_id not in ids:
    st.session_state.selected_tc_id = ids[0]

tc_id = st.selectbox(
    "Select test case",
    ids,
    index=ids.index(st.session_state.selected_tc_id),
    key="tc_selectbox"
)
st.session_state.selected_tc_id = tc_id

tc = next(t for t in suite.test_cases if t.id == tc_id)

st.divider()
st.subheader("Test Case Details")
st.markdown(f"### {tc.id}: {tc.title}")
st.write(f"**Type:** {tc.type} | **Priority:** {tc.priority}")

if tc.preconditions:
    st.write("**Preconditions:**", "; ".join(clean_text(p) for p in tc.preconditions))

if tc.test_data:
    st.write("**Test Data:**")
    st.json({item.key: item.value for item in tc.test_data})

st.write("**Steps:**")
for i, s in enumerate(tc.steps, 1):
    st.write(f"{i}. {clean_text(s.action)}  \n   ✅ {clean_text(s.expected_result)}")

if tc.notes:
    st.write("**Notes:**", clean_text(tc.notes))
