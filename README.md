# DocMind — Chat with your Documents

A beginner-friendly **RAG** (Retrieval-Augmented Generation) web app built with **Streamlit**. Upload a PDF, ask questions, and get answers grounded in your document—with optional read-aloud and clear warnings when the model uses outside knowledge.

## Features

- PDF upload and text extraction (PyMuPDF)
- Chunking + local embeddings (`all-MiniLM-L6-v2`) + in-memory ChromaDB
- GPT-3.5-turbo answers via LangChain RAG pipeline
- Chat UI with shortcut prompts (comprehensive / one-line / plain English)
- Read aloud (pyttsx3)
- Source transparency for outside-knowledge sections

## Tech stack

| Component | Library |
|-----------|---------|
| UI | Streamlit |
| PDF parsing | PyMuPDF (`fitz`) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers` via `langchain-huggingface` |
| Vector DB | ChromaDB (session temp folder) |
| LLM | OpenAI GPT-3.5-turbo (LangChain) |
| TTS | pyttsx3 |

## Local setup

### 1. Prerequisites

- Python 3.10 or 3.11 recommended
- An [OpenAI API key](https://platform.openai.com/api-keys)

### 2. Clone or download this folder

```bash
cd "RAG doc assistent"
```

### 3. Create a virtual environment (recommended)

**Windows (Command Prompt):**

```cmd
cd "C:\Users\ASUS\Desktop\RAG doc assistent"
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

**Windows (PowerShell):**

```powershell
cd "C:\Users\ASUS\Desktop\RAG doc assistent"
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

> Run `python -m venv venv` and activate the venv as **two separate steps**. Do not paste the activate path into the `venv` command.

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB). This can take a few minutes.

### 5. Configure OpenAI API key

Copy the example env file and add your key:

```bash
copy .env.example .env
```

Edit `.env`:

```
OPENAI_API_KEY=sk-your-actual-key
```

**You do not need** a `.streamlit/secrets.toml` file for local development — use `.env` only. For Streamlit Cloud, set `OPENAI_API_KEY` in the app's Secrets UI (see `.streamlit/secrets.toml.example`).

### 6. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Ensure `OPENAI_API_KEY` is set in `.env`.
2. Upload a PDF below the question box — it indexes automatically (wait for “Ready”).
3. Ask a question with **Search**, or use a shortcut button (Comprehensive / One Line / Plain English).
4. Click **Read Aloud** under any AI reply to hear it (desktop; requires system TTS).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (include `app.py`, `requirements.txt`, `README.md`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the repo.
3. Set **Main file path** to `app.py`.
4. Under **Secrets**, add:

   ```toml
   OPENAI_API_KEY = "sk-your-key"
   ```

5. Deploy. Chroma uses an **in-memory** client so no disk persistence is required on free tiers.

> **Note:** `pyttsx3` read-aloud works on your local machine; cloud hosts usually have no audio output. Other features work in the cloud.

## Deploy elsewhere (Render, Railway, etc.)

- Use the same `requirements.txt` and start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Set `OPENAI_API_KEY` as an environment variable.
- Expect cold starts and memory use from `sentence-transformers` + `torch`.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Please upload a PDF first" | Process a PDF in the sidebar first. |
| API key errors | Set `OPENAI_API_KEY` in `.env` (restart the app after editing). |
| Re-upload same file | The app replaces the Chroma collection for that filename automatically. |
| Slow first question | Embedding model loads on first use; wait for processing to finish. |
| Read aloud silent on cloud | Use locally; cloud VMs typically have no speakers. |

## Project layout

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Pinned dependencies
├── .env.example        # Example environment variables
└── README.md           # This file
```

## License

MIT — use and modify freely for learning and projects.
