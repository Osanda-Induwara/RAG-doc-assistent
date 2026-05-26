# DocMind — Chat with your Documents

A beginner-friendly **RAG** (Retrieval-Augmented Generation) web app built with **Streamlit**. Upload a PDF, ask questions, and get answers grounded in your document—with optional read-aloud and clear warnings when the model uses outside knowledge.

## Features

- PDF upload and text extraction (PyMuPDF)
- Chunking + local embeddings (`all-MiniLM-L6-v2`) + ChromaDB
- **Google Gemini** answers via LangChain RAG pipeline (free tier at AI Studio)
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
| LLM | Google Gemini (`langchain-google-genai`) |
| TTS | pyttsx3 |

## Local setup

### 1. Prerequisites

- Python 3.10 or 3.11 recommended
- A free [Google AI Studio API key](https://aistudio.google.com/apikey) (Gemini)

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

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads the embedding model (~90 MB). This can take a few minutes.

### 5. Configure Gemini API key

Copy the example env file and add your key:

```bash
copy .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your-actual-key-from-aistudio
```

Optional — change the model (default `gemini-2.5-flash`):

```env
GEMINI_MODEL=gemini-2.5-flash
```

**You do not need** a `.streamlit/secrets.toml` file for local development — use `.env` only. For Streamlit Cloud, set `GEMINI_API_KEY` in the app's Secrets UI.

### 6. Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

## Usage

1. Ensure `GEMINI_API_KEY` is set in `.env`.
2. Upload a PDF below the question box — it indexes automatically (wait for “Ready”).
3. Ask a question with **Search**, or use a shortcut button.
4. Click **Read Aloud** under any AI reply to hear it (desktop only).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Connect at [share.streamlit.io](https://share.streamlit.io).
3. Set **Main file path** to `app.py`.
4. Under **Secrets**, add:

   ```toml
   GEMINI_API_KEY = "your-gemini-api-key"
   ```

5. Deploy.

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Please upload a PDF first" | Upload and wait until the file shows as ready. |
| Gemini API key errors | Set `GEMINI_API_KEY` in `.env` and restart Streamlit. |
| Quota / 429 errors | Check usage at [AI Studio](https://aistudio.google.com/); wait or switch `GEMINI_MODEL`. |
| Slow first question | Embedding model loads on first use; wait for PDF indexing to finish. |
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
