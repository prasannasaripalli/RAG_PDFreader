import os
import io
import re
import json
import csv
import hashlib
from typing import List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict
from pypdf import PdfReader

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

st.set_page_config(page_title="RAG A/B Testing", layout="wide")
st.title("RAG Test Case Generator - A/B Comparison")
st.caption("Compare Version A and Version B using the same PDFs and inputs.")

TOP_K = 6
_SURROGATES = re.compile(r"[\ud800-\udfff]")
VALID_TYPES = {"functional", "negative", "boundary", "regression", "api", "ui"}
VALID_PRIORITIES = {"P0", "P1", "P2"}
RATINGS_LOG_FILE = "ratings_log.csv"


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value: str) -> str:
    value = value or ""
    value = _SURROGATES.sub("", value)
    return value.encode("utf-8", "replace").decode("utf-8")


def get_secret_or_env(name: str, default: Optional[str] = None) -> Optional[str]:
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


def log_rating(version: str, rating: int) -> None:
    file_exists = os.path.isfile(RATINGS_LOG_FILE)
    with open(RATINGS_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["version", "rating"])
        writer.writerow([version, rating])


def average_rating(values: List[int]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 2)


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
# Prompt versions
# -----------------------------
PROMPT_A = """
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
- Avoid duplicates.
- If something is unclear, add a short note in notes.
- test_data must be a list of objects like {{"key":"...", "value":"..."}}
Return ONLY the JSON object matching the schema.
"""

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
- Cover functional, negative, boundary, and regression scenarios where relevant.
- Keep each test case grounded in the requirement, acceptance criteria, and PDF context.
- Do not invent unsupported behavior. If something is unclear, mention it briefly in notes.
- Use unique titles.
- Every step must include action and expected_result.
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
    return "\n\n".join(
        clean_text(doc.page_content)
        for doc in docs
        if clean_text(doc.page_content).strip()
    )


# -----------------------------
# Generation
# -----------------------------
def build_chain(prompt_text: str):
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_template(prompt_text)
    return prompt | llm.with_structured_output(TestSuite)


def clean_suite(suite: TestSuite, label: str) -> TestSuite:
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

    return TestSuite(
        suite_name=f"Generated Test Suite - Version {label}",
        test_cases=cleaned_cases,
    )


def compute_metrics(suite: TestSuite) -> dict:
    test_cases = suite.test_cases
    total = len(test_cases)

    if total == 0:
        return {
            "generated_cases": 0,
            "coverage_score": 0,
            "avg_steps_per_case": 0.0,
            "cases_with_notes": 0,
        }

    coverage_types = {tc.type for tc in test_cases}
    coverage_score = sum(
        1 for name in ["functional", "negative", "boundary", "regression"]
        if name in coverage_types
    )
    avg_steps = round(sum(len(tc.steps) for tc in test_cases) / total, 2)
    cases_with_notes = sum(1 for tc in test_cases if tc.notes)

    return {
        "generated_cases": total,
        "coverage_score": coverage_score,
        "avg_steps_per_case": avg_steps,
        "cases_with_notes": cases_with_notes,
    }


def generate_suite(
    vs: FAISS,
    requirement: str,
    acceptance: str,
    num_cases: int,
    label: str,
) -> Tuple[TestSuite, dict]:
    context = retrieve_context(vs, requirement, acceptance)
    if not context:
        raise RuntimeError("No relevant context was retrieved from the PDF(s).")

    prompt_text = PROMPT_A if label == "A" else PROMPT_B
    chain = build_chain(prompt_text)

    suite = chain.invoke(
        {
            "requirement": clean_text(requirement).strip(),
            "acceptance_criteria": clean_text(acceptance).strip(),
            "context": context,
            "num_cases": int(num_cases),
        }
    )

    suite = clean_suite(suite, label)
    metrics = compute_metrics(suite)
    return suite, metrics


# -----------------------------
# Session state
# -----------------------------
st.session_state.setdefault("suite_a", None)
st.session_state.setdefault("suite_b", None)
st.session_state.setdefault("metrics_a", None)
st.session_state.setdefault("metrics_b", None)
st.session_state.setdefault("ratings_a", [])
st.session_state.setdefault("ratings_b", [])


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Upload PDFs")
    pdf_files = st.file_uploader("Select one or more PDFs", type=["pdf"], accept_multiple_files=True)

    st.header("Settings")
    num_cases = st.slider("Number of test cases", 6, 20, 12, 1)

    if st.button("Clear results"):
        st.session_state.suite_a = None
        st.session_state.suite_b = None
        st.session_state.metrics_a = None
        st.session_state.metrics_b = None
        st.session_state.ratings_a = []
        st.session_state.ratings_b = []
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

run_btn = st.button("Run A/B Comparison", type="primary")

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

if run_btn:
    if not requirement.strip() or not acceptance.strip():
        st.error("Please enter both requirement and acceptance criteria.")
        st.stop()

    try:
        with st.spinner("Running Version A..."):
            suite_a, metrics_a = generate_suite(vs, requirement, acceptance, num_cases, "A")

        with st.spinner("Running Version B..."):
            suite_b, metrics_b = generate_suite(vs, requirement, acceptance, num_cases, "B")

        st.session_state.suite_a = suite_a
        st.session_state.suite_b = suite_b
        st.session_state.metrics_a = metrics_a
        st.session_state.metrics_b = metrics_b
        st.success("A/B comparison completed")
    except Exception as e:
        st.error(f"Comparison failed: {e}")
        st.stop()

suite_a = st.session_state.get("suite_a")
suite_b = st.session_state.get("suite_b")
metrics_a = st.session_state.get("metrics_a")
metrics_b = st.session_state.get("metrics_b")

if suite_a and suite_b:
    # Handle rating submissions before summary is built
    col_submit_a, col_submit_b = st.columns(2)

    with col_submit_a:
        rating_a = st.slider("Rate Version A", 1, 5, 3, key="rating_a_slider")
        if st.button("Submit Rating A"):
            st.session_state.ratings_a.append(rating_a)
            log_rating("A", rating_a)
            st.success("Saved rating for Version A")

    with col_submit_b:
        rating_b = st.slider("Rate Version B", 1, 5, 3, key="rating_b_slider")
        if st.button("Submit Rating B"):
            st.session_state.ratings_b.append(rating_b)
            log_rating("B", rating_b)
            st.success("Saved rating for Version B")

    ratings_a = st.session_state.get("ratings_a", [])
    ratings_b = st.session_state.get("ratings_b", [])
    avg_rating_a = average_rating(ratings_a)
    avg_rating_b = average_rating(ratings_b)

    st.subheader("Comparison Summary")

    comparison_rows = [
        {"metric": "generated_cases", "version_a": metrics_a["generated_cases"], "version_b": metrics_b["generated_cases"]},
        {"metric": "coverage_score", "version_a": metrics_a["coverage_score"], "version_b": metrics_b["coverage_score"]},
        {"metric": "avg_steps_per_case", "version_a": metrics_a["avg_steps_per_case"], "version_b": metrics_b["avg_steps_per_case"]},
        {"metric": "cases_with_notes", "version_a": metrics_a["cases_with_notes"], "version_b": metrics_b["cases_with_notes"]},
        {"metric": "avg_user_rating", "version_a": avg_rating_a if avg_rating_a is not None else "N/A", "version_b": avg_rating_b if avg_rating_b is not None else "N/A"},
    ]

    comparison_df = pd.DataFrame(comparison_rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Version A")
        df_a = pd.DataFrame(
            [
                {
                    "id": tc.id,
                    "title": tc.title,
                    "type": tc.type,
                    "priority": tc.priority,
                }
                for tc in suite_a.test_cases
            ]
        )
        st.dataframe(df_a, use_container_width=True, hide_index=True)

        if avg_rating_a is not None:
            st.write(f"Avg Rating A: {avg_rating_a}")

        st.download_button(
            label="Download Version A JSON",
            data=json.dumps(suite_a.model_dump(), indent=2),
            file_name="test_suite_v1.json",
            mime="application/json",
        )

    with c2:
        st.markdown("### Version B")
        df_b = pd.DataFrame(
            [
                {
                    "id": tc.id,
                    "title": tc.title,
                    "type": tc.type,
                    "priority": tc.priority,
                }
                for tc in suite_b.test_cases
            ]
        )
        st.dataframe(df_b, use_container_width=True, hide_index=True)

        if avg_rating_b is not None:
            st.write(f"Avg Rating B: {avg_rating_b}")

        st.download_button(
            label="Download Version B JSON",
            data=json.dumps(suite_b.model_dump(), indent=2),
            file_name="test_suite_v2.json",
            mime="application/json",
        )

    st.download_button(
        label="Download Comparison CSV",
        data=comparison_df.to_csv(index=False),
        file_name="ab_comparison.csv",
        mime="text/csv",
    )