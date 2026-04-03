import os
import io
import re
import json
import hashlib
from typing import List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

st.set_page_config(page_title="RAG Test Case Generator - V2", layout="wide")
st.title("RAG Test Case Generator - Version B")
st.caption("Simplified version with a stronger prompt and cleaner output.")

TOP_K = 8
_SURROGATES = re.compile(r"[\ud800-\udfff]")
VALID_TYPES = {"functional", "negative", "boundary", "regression", "api", "ui"}
VALID_PRIORITIES = {"P0", "P1", "P2"}


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value: str) -> str:
    value = value or ""
    value = _SURROGATES.sub("", value)
    return value.encode("utf-8", "replace").decode("utf-8")


def get_secret_or_env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = None
    try:
        value = st.secrets.get(name, None)
    except Exception:
        value = None
    if value is None:
        value = os.getenv(name, default)
    return value


def get_openai_key() -> str:
    key = get_secret_or_env("OPENAI_API_KEY")
    if not key or not isinstance(key, str) or not key.strip():
        st.error("OPENAI_API_KEY is missing or invalid.")
        st.stop()
    return key.strip()


OPENAI_API_KEY = get_openai_key()
OPENAI_MODEL = get_secret_or_env("OPENAI_MODEL", "gpt-4o-mini")


# -----------------------------
# Schema
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
    preconditions: List[str] = Field(default_factory=list)
    test_data: List[TestDataItem] = Field(default_factory=list)
    steps: List[Step] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None


class TestSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    suite_name: str
    test_cases: List[TestCase] = Field(default_factory=list)


# -----------------------------
# Prompt
# -----------------------------
PROMPT_B = """
You are a QA engineer creating test cases from the requirement, acceptance criteria, and PDF context.

Requirement:
{requirement}

Acceptance Criteria:
{acceptance_criteria}

Relevant PDF Context:
{context}

Instructions:
- Generate up to {num_cases} distinct test cases.
- Include functional, negative, boundary, and regression scenarios where applicable.
- Prioritize validations, field rules, uniqueness checks, allowed values, status codes, and failure paths.
- Keep every test case grounded in the requirement, acceptance criteria, and PDF context.
- Do not invent unsupported behavior. If required information is missing or unclear, mention it briefly in notes instead of guessing.
- Use unique titles with no repeated scenarios.
- Every test case must include:
  - id
  - title
  - type
  - priority
  - at least one step
- Every step must include both action and expected_result.
- test_data must be a list of objects like {{"key":"...", "value":"..."}}.
Return ONLY the JSON object that matches the schema.
"""
# -----------------------------
# PDF and vector store
# -----------------------------
def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        text = clean_text(page.extract_text() or "").strip()
        if text:
            pages.append(text)
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    text = clean_text(text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == text_length:
            break
        start = max(end - overlap, 0)

    return chunks


def hash_pdfs(pdf_files) -> str:
    h = hashlib.sha256()
    for file in pdf_files:
        h.update(file.getvalue())
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def build_faiss_index(pdf_hash: str, pdf_bytes_list: List[bytes]) -> FAISS:
    _ = pdf_hash
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    all_chunks: List[str] = []

    for pdf_bytes in pdf_bytes_list:
        text = pdf_bytes_to_text(pdf_bytes)
        all_chunks.extend(chunk_text(text))

    all_chunks = [chunk for chunk in all_chunks if chunk.strip()]
    if not all_chunks:
        raise RuntimeError("No text could be extracted from the PDF(s). If the files are scanned, OCR is needed.")

    return FAISS.from_texts(all_chunks, embeddings)


def retrieve_context(vs: FAISS, requirement: str, acceptance: str) -> str:
    query = clean_text(f"{requirement}\n{acceptance}")
    docs = vs.similarity_search(query, k=TOP_K)
    return "\n\n".join(clean_text(doc.page_content) for doc in docs if clean_text(doc.page_content).strip())


# -----------------------------
# Generation
# -----------------------------
def build_chain():
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(PROMPT_B)
    return prompt | llm.with_structured_output(TestSuite)


def clean_suite(suite: TestSuite) -> TestSuite:
    cleaned_cases: List[TestCase] = []
    seen_titles = set()

    for tc in suite.test_cases:
        title = clean_text(tc.title).strip()
        title_key = title.lower()
        if not title or title_key in seen_titles:
            continue

        cleaned_steps = []
        for step in tc.steps:
            action = clean_text(step.action).strip()
            expected = clean_text(step.expected_result).strip()
            if action and expected:
                cleaned_steps.append(Step(action=action, expected_result=expected))

        if not cleaned_steps:
            continue

        cleaned_test_data = []
        for item in tc.test_data:
            key = clean_text(item.key).strip()
            value = clean_text(item.value).strip()
            if key:
                cleaned_test_data.append(TestDataItem(key=key, value=value))

        tc_type = clean_text(tc.type).strip().lower()
        tc_priority = clean_text(tc.priority).strip().upper()

        cleaned_case = TestCase(
            id=clean_text(tc.id).strip() or f"TC-{len(cleaned_cases) + 1:03d}",
            title=title,
            type=tc_type if tc_type in VALID_TYPES else "functional",
            priority=tc_priority if tc_priority in VALID_PRIORITIES else "P1",
            preconditions=[clean_text(item).strip() for item in tc.preconditions if clean_text(item).strip()],
            test_data=cleaned_test_data,
            steps=cleaned_steps,
            tags=[clean_text(item).strip() for item in tc.tags if clean_text(item).strip()],
            notes=clean_text(tc.notes).strip() if tc.notes else None,
        )

        cleaned_cases.append(cleaned_case)
        seen_titles.add(title_key)

    suite_name = clean_text(suite.suite_name).strip() or "Generated Test Suite - Version B"
    return TestSuite(suite_name=suite_name, test_cases=cleaned_cases)


def generate_test_suite(vs: FAISS, requirement: str, acceptance: str, num_cases: int) -> TestSuite:
    context = retrieve_context(vs, requirement, acceptance)
    if not context:
        raise RuntimeError("No relevant context was retrieved from the PDF(s).")

    chain = build_chain()
    suite = chain.invoke(
        {
            "requirement": clean_text(requirement).strip(),
            "acceptance_criteria": clean_text(acceptance).strip(),
            "context": context,
            "num_cases": int(num_cases),
        }
    )
    return clean_suite(suite)


# -----------------------------
# Session state
# -----------------------------
st.session_state.setdefault("suite", None)
st.session_state.setdefault("selected_tc_id", None)


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Upload PDFs")
    pdf_files = st.file_uploader("Select one or more PDFs", type=["pdf"], accept_multiple_files=True)

    st.header("Settings")
    num_cases = st.slider("Number of test cases", 6, 20, 12, 1)

    if st.button("Clear results"):
        st.session_state.suite = None
        st.session_state.selected_tc_id = None
        st.session_state.pop("tc_selectbox", None)
        st.rerun()

col1, col2 = st.columns(2)
with col1:
    requirement = st.text_area("Requirement / User Story", height=100, value="Student API: Create a Student")
with col2:
    acceptance = st.text_area(
        "Acceptance Criteria",
        height=160,
        value="Validate input; age 0–120; grade allowed list; email must be unique; return 201 or 400/409 accordingly",
    )

generate_btn = st.button("Generate Test Cases", type="primary")

if not pdf_files:
    st.info("Upload PDF(s) in the sidebar to begin.")
    st.stop()

try:
    pdf_hash = hash_pdfs(pdf_files)
    pdf_bytes_list = [file.getvalue() for file in pdf_files]
    with st.spinner("Indexing PDFs (FAISS)..."):
        vs = build_faiss_index(pdf_hash, pdf_bytes_list)
except Exception as e:
    st.error(f"Failed to build index: {e}")
    st.stop()

if generate_btn:
    if not requirement.strip() or not acceptance.strip():
        st.error("Please enter both requirement and acceptance criteria.")
        st.stop()

    try:
        with st.spinner("Generating test cases..."):
            suite = generate_test_suite(vs, requirement, acceptance, num_cases)

        if not suite.test_cases:
            st.warning("No test cases generated. Try updating the requirement or acceptance criteria.")
            st.stop()

        st.session_state.suite = suite
        st.session_state.selected_tc_id = suite.test_cases[0].id
        st.session_state["tc_selectbox"] = suite.test_cases[0].id
        st.success(f"Generated {len(suite.test_cases)} test cases")
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

suite = st.session_state.get("suite")

if suite is None:
    st.info("Click **Generate Test Cases** after uploading PDFs.")
else:
    st.subheader(suite.suite_name)

    df = pd.DataFrame(
        [
            {
                "id": tc.id,
                "title": tc.title,
                "type": tc.type,
                "priority": tc.priority,
            }
            for tc in suite.test_cases
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    ids = df["id"].tolist()
    if ids:
        if st.session_state.get("selected_tc_id") not in ids:
            st.session_state.selected_tc_id = ids[0]
            st.session_state["tc_selectbox"] = ids[0]

        selected_id = st.selectbox(
            "Select test case",
            ids,
            index=ids.index(st.session_state.selected_tc_id),
            key="tc_selectbox",
        )
        st.session_state.selected_tc_id = selected_id

        selected_tc = next(tc for tc in suite.test_cases if tc.id == selected_id)

        st.divider()
        st.subheader("Test Case Details")
        st.markdown(f"### {selected_tc.id}: {selected_tc.title}")
        st.write(f"**Type:** {selected_tc.type} | **Priority:** {selected_tc.priority}")

        if selected_tc.preconditions:
            st.write("**Preconditions:**", "; ".join(selected_tc.preconditions))

        if selected_tc.test_data:
            st.write("**Test Data:**")
            st.json({item.key: item.value for item in selected_tc.test_data})

        st.write("**Steps:**")
        for i, step in enumerate(selected_tc.steps, 1):
            st.write(f"{i}. {step.action}  \n   ✅ {step.expected_result}")

        if selected_tc.notes:
            st.write("**Notes:**", selected_tc.notes)

    st.download_button(
        label="Download JSON",
        data=json.dumps(suite.model_dump(), indent=2),
        file_name="test_suite_v2.json",
        mime="application/json",
    )