"""
DocMind — RAG-based Document Assistant (Streamlit)
Upload a PDF, ask questions, get answers grounded in your document.
"""

import os
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import pyttsx3
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Quiet Chroma/Hugging Face noise on Windows; disable broken Chroma telemetry
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Load OPENAI_API_KEY from .env locally, platform env vars, or Streamlit secrets
load_dotenv()


def _streamlit_secrets_available() -> bool:
    """Only read st.secrets when a secrets.toml exists (avoids local 'No secrets found' error)."""
    app_dir = Path(__file__).resolve().parent
    candidates = [
        app_dir / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]
    return any(p.is_file() for p in candidates)


def load_api_key_from_streamlit_secrets() -> None:
    """Optional: load OPENAI_API_KEY from Streamlit secrets (cloud or local secrets.toml)."""
    if os.getenv("OPENAI_API_KEY") or not _streamlit_secrets_available():
        return
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Page config & custom CSS (dark gradient, card chat, message bubbles)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind — Chat with your Documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0d29 0%, #302b63 50%, #24243e 100%);
    }
    .main-title {
        text-align: center;
        color: #f0f0f5;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        letter-spacing: -0.02em;
    }
    .main-subtitle {
        text-align: center;
        color: #a8a8b8;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .chat-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        max-width: 900px;
        margin: 0 auto 1rem auto;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .user-bubble {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0 0.5rem 15%;
        text-align: right;
    }
    .ai-bubble {
        background: rgba(255, 255, 255, 0.1);
        color: #e8e8f0;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 15% 0.5rem 0;
        text-align: left;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .outside-warning {
        background: #fff3cd;
        color: #856404;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-top: 0.5rem;
    }
    [data-testid="stSidebar"], [data-testid="collapsedControl"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Shortcut button labels → extra instruction appended to the user query
SHORTCUT_PROMPTS = {
    "comprehensive": (
        "📖 Comprehensive Explanation",
        "Give a detailed, thorough explanation covering all relevant points from the document.",
    ),
    "oneline": (
        "⚡ One Line Summary",
        "Summarize the answer in exactly one clear sentence.",
    ),
    "plain": (
        "🗣️ Plain English",
        "Explain as simply as possible, as if speaking to a 10-year-old.",
    ),
}

SYSTEM_INSTRUCTION = (
    "Answer only from the provided document context. "
    "If you use any knowledge outside the provided context, "
    "clearly start that part with: ⚠️ Outside Knowledge: "
)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are DocMind, a helpful document assistant.\n\n"
        + SYSTEM_INSTRUCTION
        + "\n\n--- Document context ---\n{context}\n--- End context ---\n\n"
        "User question: {question}\n\nAnswer:"
    ),
)


# ---------------------------------------------------------------------------
# Session state defaults (vector index lives in a temp folder for this session)
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize Streamlit session keys once per browser session."""
    defaults = {
        "messages": [],  # chat history: {"role": "user"|"assistant", "content": str}
        "vectorstore": None,
        "current_collection": None,
        "uploaded_filename": None,
        "chunk_count": 0,
        "embeddings": None,
        "chroma_persist_dir": None,
        "processed_file_signature": None,
        "pending_query": None,
        "auto_submit": False,
        "reading": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


load_api_key_from_streamlit_secrets()
init_session_state()


# ---------------------------------------------------------------------------
# Helpers: PDF parse, chunk, embed, store; TTS; collection naming
# ---------------------------------------------------------------------------
def safe_collection_name(filename: str) -> str:
    """Chroma collection names must be alphanumeric with _ or -."""
    base = os.path.splitext(os.path.basename(filename))[0]
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", base)
    if len(safe) < 3:
        safe = f"doc_{safe}"
    return safe[:63]


def get_embeddings() -> HuggingFaceEmbeddings:
    """Load sentence-transformers all-MiniLM-L6-v2 via langchain-huggingface."""
    if st.session_state.embeddings is None:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return st.session_state.embeddings


def reset_chroma_storage() -> str:
    """New temp folder per PDF — avoids Chroma sqlite 'no such table' errors on re-upload."""
    if st.session_state.chroma_persist_dir:
        shutil.rmtree(st.session_state.chroma_persist_dir, ignore_errors=True)
    path = tempfile.mkdtemp(prefix="docmind_chroma_")
    st.session_state.chroma_persist_dir = path
    return path


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Parse PDF bytes with PyMuPDF and return concatenated page text."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages).strip()


def chunk_text(text: str) -> List[str]:
    """Split document text into overlapping chunks via LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)


def process_pdf(uploaded_file) -> bool:
    """
    Full ingest pipeline: parse → chunk → embed → Chroma collection.
    Runs automatically when a new PDF is uploaded.
    """
    pdf_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    collection_name = safe_collection_name(filename)

    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        st.error("No extractable text found in this PDF.")
        return False

    chunks = chunk_text(text)
    if not chunks:
        st.error("Could not create text chunks from the PDF.")
        return False

    embeddings = get_embeddings()
    persist_dir = reset_chroma_storage()

    # LangChain manages Chroma in a session temp folder (stable with chromadb 0.5+)
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    st.session_state.vectorstore = vectorstore
    st.session_state.current_collection = collection_name
    st.session_state.uploaded_filename = filename
    st.session_state.chunk_count = len(chunks)
    st.session_state.messages = []  # fresh chat for new document
    return True


def file_signature(uploaded_file) -> str:
    """Unique id so we only re-index when the user picks a different file."""
    return f"{uploaded_file.name}:{uploaded_file.size}"


def get_openai_api_key() -> Optional[str]:
    return os.getenv("OPENAI_API_KEY", "").strip() or None


def run_rag_query(question: str) -> str:
    """Retrieve top-5 chunks, then answer with GPT-3.5-turbo via LLMChain."""
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        openai_api_key=get_openai_api_key(),
    )
    chain = LLMChain(llm=llm, prompt=RAG_PROMPT)
    result = chain.invoke({"context": context, "question": question})
    # LLMChain may return {"text": "..."} or a message object depending on LangChain version
    if isinstance(result, dict):
        return result.get("text") or result.get("output") or str(result)
    return str(result)


def speak_text(text: str) -> None:
    """Read response aloud with pyttsx3 (runs in a background thread)."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass
    finally:
        st.session_state.reading = False


def start_read_aloud(content: str) -> None:
    """Start TTS in a daemon thread so the UI can show 'Reading...' status."""
    if st.session_state.reading:
        return
    st.session_state.reading = True
    thread = threading.Thread(target=speak_text, args=(content,), daemon=True)
    thread.start()


def render_outside_knowledge_warning(text: str) -> None:
    """If the model used outside knowledge, show that section in a yellow box."""
    marker = "⚠️ Outside Knowledge:"
    if marker not in text:
        return
    idx = text.find(marker)
    before = text[:idx].strip()
    after = text[idx:].strip()
    if before:
        st.markdown(f'<div class="ai-bubble">{before}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="outside-warning"><strong>{after}</strong></div>',
        unsafe_allow_html=True,
    )


def display_ai_message(content: str, msg_index: int) -> None:
    """Render assistant bubble, outside-knowledge highlight, and Read Aloud button."""
    marker = "⚠️ Outside Knowledge:"
    if marker in content:
        render_outside_knowledge_warning(content)
    else:
        st.markdown(f'<div class="ai-bubble">{content}</div>', unsafe_allow_html=True)

    if st.session_state.reading:
        st.caption("🔊 Reading...")

    if st.button("🔊 Read Aloud", key=f"read_aloud_{msg_index}"):
        start_read_aloud(content)
        st.rerun()


def handle_user_question(question: str) -> None:
    """Validate inputs, run RAG, append messages to session chat."""
    question = question.strip()
    if not question:
        return

    if not st.session_state.vectorstore:
        st.warning("Please upload a PDF first.")
        return

    if not get_openai_api_key():
        st.error("OpenAI API key missing. Add OPENAI_API_KEY to your `.env` file.")
        return

    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        try:
            answer = run_rag_query(question)
        except Exception as exc:
            answer = f"Sorry, something went wrong: {exc}"

    st.session_state.messages.append({"role": "assistant", "content": answer})


def handle_pdf_upload(uploaded) -> None:
    """Auto-index PDF when user selects a new file."""
    if uploaded is None:
        st.session_state.processed_file_signature = None
        return

    signature = file_signature(uploaded)
    if st.session_state.processed_file_signature == signature:
        return

    with st.spinner("Indexing your PDF (first time may take a minute)..."):
        ok = process_pdf(uploaded)
    if ok:
        st.session_state.processed_file_signature = signature
        st.success(f"Ready: **{st.session_state.uploaded_filename}** ({st.session_state.chunk_count} chunks)")
    else:
        st.session_state.processed_file_signature = None


# ---------------------------------------------------------------------------
# Main UI: title → chat → shortcuts → prompt → upload
# ---------------------------------------------------------------------------
st.markdown('<p class="main-title">DocMind — Chat with your Documents</p>', unsafe_allow_html=True)

# Chat history (only when there are messages — no empty box above Quick prompts)
if st.session_state.messages:
    st.markdown('<div class="chat-card">', unsafe_allow_html=True)
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            display_ai_message(msg["content"], i)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("##### Quick prompts")
col1, col2, col3 = st.columns(3)
for col, (key, (label, instruction)) in zip([col1, col2, col3], SHORTCUT_PROMPTS.items()):
    with col:
        if st.button(label, use_container_width=True, key=f"shortcut_{key}"):
            st.session_state.pending_query = instruction
            st.session_state.auto_submit = True
            st.rerun()

# Question box + Search button (form so Enter also submits)
with st.form("query_form", clear_on_submit=True):
    input_col, btn_col = st.columns([6, 1])
    with input_col:
        user_query = st.text_input(
            "Your question",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.form_submit_button(
            "🔍 Search",
            type="primary",
            use_container_width=True,
        )

# PDF upload directly under the prompt
uploaded = st.file_uploader(
    "Browse and upload a PDF",
    type=["pdf"],
    help="Select a PDF — it indexes automatically so you can chat.",
)
handle_pdf_upload(uploaded)

# Process shortcut, Search button, or pending query
query_to_run = None
if st.session_state.auto_submit and st.session_state.pending_query:
    query_to_run = st.session_state.pending_query
    st.session_state.pending_query = None
    st.session_state.auto_submit = False
elif search_clicked and user_query and user_query.strip():
    query_to_run = user_query.strip()

if query_to_run:
    handle_user_question(query_to_run)
    st.rerun()
