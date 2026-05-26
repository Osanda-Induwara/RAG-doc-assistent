"""
DocMind — RAG-based Document Assistant (Streamlit)
Upload a PDF, ask questions, get answers grounded in your document.
"""

# Env + telemetry patches must run before chromadb / torch are imported
import os
import warnings

os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Examining the path of torch.classes.*")

import base64
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

APP_DIR = Path(__file__).resolve().parent
BG_IMAGE_PATH = APP_DIR / "assets" / "document-bg.jpg"
BG_IMAGE_FALLBACK = (
    "https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8?w=1920&q=80"
)

import chromadb
import fitz  # PyMuPDF
import pyttsx3
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default Gemini model (free tier: gemini-1.5-flash at https://aistudio.google.com)
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

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
    """Optional: load GEMINI_API_KEY from Streamlit secrets (cloud or local secrets.toml)."""
    if get_gemini_api_key() or not _streamlit_secrets_available():
        return
    try:
        key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
        if key:
            os.environ["GEMINI_API_KEY"] = key
    except Exception:
        pass


def get_background_image_url() -> str:
    """Local asset as data URL, or remote fallback for document-themed background."""
    if BG_IMAGE_PATH.is_file():
        encoded = base64.b64encode(BG_IMAGE_PATH.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    return BG_IMAGE_FALLBACK


def inject_theme_css() -> None:
    """Mockup-style UI: charcoal base, bottom document image with gradient fade."""
    bg_url = get_background_image_url()
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Playfair+Display:wght@500;600&display=swap" rel="stylesheet">
        <style>
        /* Hide chrome */
        [data-testid="stSidebar"], [data-testid="collapsedControl"],
        #MainMenu, footer, header {{ visibility: hidden; height: 0; }}
        .stApp {{
            background-color: #232528;
        }}
        /* Bottom document image + fade into solid charcoal above */
        .stApp::before {{
            content: "";
            position: fixed;
            left: 0; right: 0; bottom: 0;
            height: 62vh;
            z-index: 0;
            pointer-events: none;
            background-image:
                linear-gradient(
                    180deg,
                    #232528 0%,
                    #232528 18%,
                    rgba(35, 37, 40, 0.97) 38%,
                    rgba(35, 37, 40, 0.75) 58%,
                    rgba(35, 37, 40, 0.45) 78%,
                    rgba(35, 37, 40, 0.15) 100%
                ),
                url("{bg_url}");
            background-size: cover;
            background-position: center bottom;
            background-repeat: no-repeat;
        }}
        .main .block-container {{
            max-width: 720px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
            position: relative;
            z-index: 1;
        }}
        .dm-header {{
            text-align: center;
            margin-bottom: 2rem;
        }}
        .dm-title {{
            font-family: 'Playfair Display', Georgia, serif;
            font-size: clamp(2.8rem, 6vw, 3.75rem);
            font-weight: 500;
            color: #ffffff;
            margin: 0 0 0.35rem 0;
            letter-spacing: 0.02em;
        }}
        .dm-subtitle {{
            font-family: 'Inter', system-ui, sans-serif;
            font-size: 0.72rem;
            font-weight: 500;
            letter-spacing: 0.28em;
            text-transform: uppercase;
            color: rgba(255, 255, 255, 0.92);
            margin: 0;
        }}
        .dm-section {{
            margin-bottom: 1.25rem;
        }}
        /* Quick prompt pills */
        .st-key-shortcut_comprehensive button,
        .st-key-shortcut_oneline button,
        .st-key-shortcut_plain button {{
            background: transparent !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.85) !important;
            border-radius: 999px !important;
            padding: 0.55rem 0.35rem !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.78rem !important;
            font-weight: 400 !important;
            min-height: 2.6rem !important;
            transition: background 0.2s ease !important;
        }}
        .st-key-shortcut_comprehensive button:hover,
        .st-key-shortcut_oneline button:hover,
        .st-key-shortcut_plain button:hover {{
            background: rgba(255, 255, 255, 0.08) !important;
            border-color: #ffffff !important;
        }}
        /* Search bar container */
        [data-testid="stForm"] {{
            border: 1px solid rgba(255, 255, 255, 0.35);
            border-radius: 999px;
            padding: 0.35rem 0.35rem 0.35rem 1.1rem;
            background: rgba(20, 21, 23, 0.55);
            backdrop-filter: blur(6px);
        }}
        [data-testid="stForm"] [data-testid="stTextInput"] input {{
            background: transparent !important;
            color: #ffffff !important;
            border: none !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
        }}
        [data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {{
            color: rgba(255, 255, 255, 0.45) !important;
        }}
        [data-testid="stForm"] [data-testid="stTextInput"] label {{
            display: none !important;
        }}
        [data-testid="stFormSubmitButton"] > button {{
            background: transparent !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.85) !important;
            border-radius: 999px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
            min-height: 2.5rem !important;
            padding: 0 1.35rem !important;
            width: 100% !important;
        }}
        [data-testid="stFormSubmitButton"] > button:hover {{
            background: rgba(255, 255, 255, 0.1) !important;
        }}
        [data-testid="stFormSubmitButton"] > button p {{
            font-size: 0.9rem !important;
        }}
        /* File upload drop zone */
        [data-testid="stFileUploader"] {{
            background: rgba(20, 21, 23, 0.35);
            border: 1.5px dashed rgba(255, 255, 255, 0.35);
            border-radius: 20px;
            padding: 0.5rem 0.25rem 1rem;
        }}
        [data-testid="stFileUploader"] section {{
            padding: 1.5rem 1rem !important;
        }}
        [data-testid="stFileUploader"] label {{
            display: none !important;
        }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {{
            border: none !important;
            background: transparent !important;
        }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] div {{
            color: rgba(255, 255, 255, 0.9) !important;
            font-family: 'Inter', sans-serif !important;
        }}
        [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] svg {{
            stroke: #ffffff !important;
            fill: #ffffff !important;
        }}
        [data-testid="stFileUploader"] small {{
            color: rgba(255, 255, 255, 0.5) !important;
            font-family: 'Inter', sans-serif !important;
        }}
        [data-testid="stFileUploader"] button {{
            background: transparent !important;
            color: #ffffff !important;
            border: 1px solid rgba(255, 255, 255, 0.85) !important;
            border-radius: 999px !important;
            font-family: 'Inter', sans-serif !important;
        }}
        [data-testid="stFileUploader"] button:hover {{
            background: rgba(255, 255, 255, 0.08) !important;
        }}
        /* Chat area */
        .chat-card {{
            background: rgba(20, 21, 23, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(8px);
        }}
        .user-bubble {{
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.2);
            color: #ffffff;
            padding: 0.75rem 1rem;
            border-radius: 16px 16px 4px 16px;
            margin: 0.5rem 0 0.5rem 10%;
            text-align: right;
            font-family: 'Inter', sans-serif;
        }}
        .ai-bubble {{
            background: rgba(255, 255, 255, 0.06);
            color: #f0f0f0;
            padding: 0.75rem 1rem;
            border-radius: 16px 16px 16px 4px;
            margin: 0.5rem 10% 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.12);
            font-family: 'Inter', sans-serif;
        }}
        .outside-warning {{
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
        }}
        .dm-status {{
            text-align: center;
            color: rgba(255, 255, 255, 0.7);
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            margin-top: -0.5rem;
            margin-bottom: 1rem;
        }}
        div[data-testid="stAlert"] {{
            background: rgba(20, 21, 23, 0.85) !important;
            border-radius: 12px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DocMind",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)

inject_theme_css()

# Shortcut buttons (display label, RAG instruction)
SHORTCUT_PROMPTS = {
    "comprehensive": (
        "☰  Comprehensive explanation",
        "Give a detailed, thorough explanation covering all relevant points from the document.",
    ),
    "oneline": (
        "⚡  One line summary",
        "Summarize the answer in exactly one clear sentence.",
    ),
    "plain": (
        "🗣  Plain English",
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
        "chroma_client": None,
        "processed_file_signature": None,
        "pending_query": None,
        "auto_submit": False,
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
    """New temp folder per PDF — avoids Chroma sqlite errors on re-upload."""
    if st.session_state.chroma_persist_dir:
        shutil.rmtree(st.session_state.chroma_persist_dir, ignore_errors=True)
    path = tempfile.mkdtemp(prefix="docmind_chroma_")
    st.session_state.chroma_persist_dir = path
    st.session_state.chroma_client = None
    return path


def create_chroma_client(persist_dir: str):
    """PersistentClient only — do not use chromadb.Client() (tenant / settings conflicts)."""
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )


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
    chroma_client = create_chroma_client(persist_dir)
    st.session_state.chroma_client = chroma_client

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        client=chroma_client,
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


def get_gemini_api_key() -> Optional[str]:
    """GEMINI_API_KEY in .env, or GOOGLE_API_KEY as alias."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key.strip() if key else None


def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL


def run_rag_query(question: str) -> str:
    """Retrieve top-5 chunks, then answer with Google Gemini (LangChain LCEL chain)."""
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    llm = ChatGoogleGenerativeAI(
        model=get_gemini_model(),
        temperature=0.3,
        google_api_key=get_gemini_api_key(),
    )
    chain = RAG_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


def speak_text(text: str) -> None:
    """Read response aloud with pyttsx3 (background thread — no Streamlit context)."""
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception:
        pass


def start_read_aloud(content: str) -> None:
    """Play TTS in a daemon thread; avoid touching st.session_state from the thread."""
    threading.Thread(target=speak_text, args=(content,), daemon=True).start()
    st.toast("Reading aloud…", icon="🔊")


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

    if not get_gemini_api_key():
        st.error(
            "Gemini API key missing. Add GEMINI_API_KEY to your `.env` file. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
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

    with st.spinner("Indexing your PDF..."):
        ok = process_pdf(uploaded)
    if ok:
        st.session_state.processed_file_signature = signature
    else:
        st.session_state.processed_file_signature = None


# ---------------------------------------------------------------------------
# Main UI (matches mockup: header → chat → pills → search → upload)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dm-header">
        <h1 class="dm-title">DocMind</h1>
        <p class="dm-subtitle">Chat with your documents</p>
    </div>
    """,
    unsafe_allow_html=True,
)

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

col1, col2, col3 = st.columns(3, gap="small")
for col, (key, (label, instruction)) in zip([col1, col2, col3], SHORTCUT_PROMPTS.items()):
    with col:
        if st.button(label, use_container_width=True, key=f"shortcut_{key}"):
            st.session_state.pending_query = instruction
            st.session_state.auto_submit = True
            st.rerun()

with st.form("query_form", clear_on_submit=True):
    input_col, btn_col = st.columns([5, 1])
    with input_col:
        user_query = st.text_input(
            "Your question",
            placeholder="Ask anything about your document...",
            label_visibility="collapsed",
        )
    with btn_col:
        search_clicked = st.form_submit_button(
            "Search",
            use_container_width=True,
        )

uploaded = st.file_uploader(
    "Upload",
    type=["pdf"],
    label_visibility="collapsed",
    help="PDF only, max 200 MB",
)
handle_pdf_upload(uploaded)

if st.session_state.uploaded_filename and st.session_state.processed_file_signature:
    st.markdown(
        f'<p class="dm-status">✓ {st.session_state.uploaded_filename} — ready to chat</p>',
        unsafe_allow_html=True,
    )

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
