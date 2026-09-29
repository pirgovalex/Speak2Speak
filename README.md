# Speak2Speak

Ask questions about an anatomy PDF by typing or speaking, and get the answer back as text and speech. The backend finds the relevant passages with hybrid search (vector + keyword), reranks them, and a local Mistral-7B model answers from that context only. There is a Tkinter desktop app with voice input and output, and a Solid.js web client that talks to a FastAPI backend.

<!-- screenshot -->

## How it works

- **Indexing** (`backend/load_pdf.py`): the PDF is split into 800-character chunks with 120 overlap, embedded with `all-MiniLM-L6-v2` and saved as a FAISS index. The raw chunks are pickled next to it for BM25. PDFs uploaded through the API are merged into the same index.
- **Retrieval** (`backend/hybrid_search.py`): FAISS similarity search and BM25 each return 20 chunks, LangChain's `EnsembleRetriever` merges them (weights 0.5 / 0.5), then the `BAAI/bge-reranker-base` cross-encoder rescores them and the top 5 go into the prompt.
- **Answering** (`backend/LLM.py`): `mistralai/Mistral-7B-Instruct-v0.3` through a transformers pipeline. With a CUDA GPU it loads in 4-bit NF4 with double quantization (bitsandbytes); without one it falls back to float32 on CPU. The prompt asks for a comma-separated list of muscle names only (a description only if the question contains `DESCRIPTION`), or "I do not have that information in the documents". Duplicate names are removed from the output.
- **API** (`backend/app.py`): `POST /query` with `{"question": "..."}` returns `{"answer": "...", "sources": []}`. `POST /upload_pdf` takes a PDF as multipart form data.
- **Desktop app** (`gui.py`): records 6 seconds from the mic with `sounddevice`, transcribes it with faster-whisper (`tiny.en`), shows the answer and reads it out with gTTS. Recording and model calls run on background threads so the window stays responsive. It imports the backend modules directly, no HTTP.
- **Web client** (`docgraph-ui/`): chat view with PDF upload and a light/dark theme, calling the API through axios.

## Stack

Python, FastAPI, LangChain (FAISS, BM25, EnsembleRetriever), sentence-transformers, Hugging Face transformers, bitsandbytes, faster-whisper, gTTS, Tkinter, pytest.
Web client: Solid.js, TypeScript, Vite, Tailwind CSS, Vitest.

## Running it

The `data/` folder is not in the repo (PDFs and the FAISS index are gitignored), so bring your own PDF and put it at `data/anatomy.pdf`. A CUDA GPU is strongly recommended. The Mistral model is gated on Hugging Face, so accept its terms and run `huggingface-cli login` first.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
pip install accelerate bitsandbytes           # GPU / 4-bit loading
pip install faster-whisper sounddevice gTTS   # desktop app only

cd backend
python -c "from load_pdf import load_and_store_pdf; load_and_store_pdf()"   # build the index once
python app.py                                 # API on http://localhost:8000
```

The index has to exist before the backend starts, because `hybrid_search.py` loads it on import.

Web client, in another terminal:

```bash
cd docgraph-ui
cp .env.example .env    # VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev             # http://localhost:3000, the origin the API allows in CORS
```

Desktop app, from the repo root: `PYTHONPATH=backend python gui.py`

Tests: `cd backend && pip install pytest pytest-mock && pytest`, and `cd docgraph-ui && npm test`. The backend tests mock the retrievers and the reranker, but importing `hybrid_search` still loads the index and the models.

## Notes / limitations

- The prompt is written for anatomy questions. For other documents, change `prime_text` in `backend/LLM.py`.
- After uploading a new PDF, restart the backend: the retrievers are built once at startup.
- `sources` in the `/query` response is always empty for now.
- The mic device index is hard-coded (`device = 12` in `gui.py`). `backend/get_mistral.py` just prints your audio devices so you can find the right one.
- Spoken replies only play on Windows for now (`os.startfile`), and gTTS needs an internet connection.
- The CPU fallback loads the full model in float32 (around 30 GB of RAM) and is very slow.
- The web client already has JWT token handling and refresh logic, but the backend has no auth endpoints yet.
