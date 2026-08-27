# DocQA — Ask Questions About Your Own PDF Documents (RAG Pipeline)

DocQA is a Retrieval-Augmented Generation (RAG) system. In simple terms: you give it a folder of PDF documents, and it lets you ask questions in plain English — and it answers **only using the content of those documents**, showing you exactly which file (and which part of it) the answer came from. If the answer isn't in your documents, it tells you honestly instead of making something up.

This project was built as a hands-on learning exercise to understand how RAG systems actually work — chunking, embeddings, vector search, prompt design, and proper error handling — not just how to call one through an API.

---

## Table of Contents
1. [What this project does](#what-this-project-does)
2. [How it works (architecture)](#how-it-works-architecture)
3. [Before you start](#before-you-start)
4. [Step-by-step setup](#step-by-step-setup)
5. [How to ask questions](#how-to-ask-questions)
6. [How to run the evaluation](#how-to-run-the-evaluation)
7. [Project structure](#project-structure)
8. [Known limitations](#known-limitations)
9. [Troubleshooting](#troubleshooting)

---

## What this project does

- Reads every PDF you put in the `Data/` folder
- Breaks each document into small, overlapping text chunks
- Converts each chunk into a numerical "embedding" (a way of representing meaning as numbers) — this runs **locally on your computer, for free**
- Stores those embeddings in a searchable index (FAISS)
- When you ask a question, it finds the most relevant chunks and sends them to an LLM (Groq) to generate an answer
- The answer is grounded strictly in your documents, with citations showing exactly which file it came from
- If your question isn't answerable from your documents, it says so honestly instead of guessing
- If something goes wrong (a network hiccup, an API timeout), it automatically retries instead of crashing

## How it works (architecture)

```
Your PDFs  →  ingest.py  →  chunk + embed  →  saved index (vector_store/)
                                                        │
Your question  →  rag_pipeline.py  →  embed the question
                                                │
                                    search the index for the closest matches
                                                │
                                    send question + matches to the LLM (Groq)
                                                │
                                    you get back a grounded, cited answer
```

`ingest.py` is something you run **once** (or whenever you change your documents). `rag_pipeline.py` is what you run **every time** you want to ask a question.

---

## Before you start

You'll need:
- **Python 3.10 or newer** installed on your computer ([python.org](https://www.python.org/downloads/) if you don't have it)
- A free **Groq API key** (you'll create this in Step 4 below — no credit card required)
- Some PDF documents you want to ask questions about (or use the sample ones already included in `Data/`, if any are provided in this repo)

No prior experience with AI/ML is required to run this — just follow the steps below in order.

---

## Step-by-step setup

### Step 1 — Get the code onto your computer

If you have `git` installed:
```bash
git clone <your-repo-url>
cd rag-project
```
If you don't have `git`, click the green "Code" button on the GitHub page → "Download ZIP" → extract it → open a terminal inside the extracted folder.

### Step 2 — Create a virtual environment

A virtual environment keeps this project's Python packages separate from everything else on your computer.

```bash
python -m venv venv
```

Then activate it:
- **Windows (PowerShell):**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

You'll know it worked when you see `(venv)` at the start of your terminal line. **You'll need to run the activate command again every time you open a new terminal window** — this is normal.

### Step 3 — Install the required packages

```bash
pip install -r requirements.txt
```

This installs everything the project needs: FAISS (vector search), sentence-transformers (embeddings), the Groq client, and a few smaller helper libraries. This step downloads a few hundred MB (mainly PyTorch), so it may take a few minutes depending on your internet speed.

### Step 4 — Get a free Groq API key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up (free, no credit card needed)
3. Find "API Keys" in the left sidebar
4. Click "Create API Key," name it whatever you like
5. Copy the key immediately — it's only shown once

### Step 5 — Add your key to the project securely

Create a new file in the project's root folder named exactly `.env` (yes, starting with a dot). Put this single line inside it, replacing the placeholder with your real key:
```
GROQ_API_KEY=your-actual-key-here
```
No quotes, no spaces around the `=`. Save the file.

This key is loaded automatically every time you run the project — you won't need to type it manually. It's also excluded from GitHub via `.gitignore`, so it stays private.

### Step 6 — Add your documents

Put your `.pdf` files into the `Data/` folder. If this repo already comes with sample PDFs, you can use those as-is, or replace them with your own.

**Notes on PDFs:**
- Text-based PDFs work best (if you can select/highlight text in the PDF normally, you're fine)
- Scanned/image-only PDFs are **not** supported without extra OCR tooling (not included here)
- Start with a few PDFs (2-6) rather than dozens, to keep things fast while you're testing

### Step 7 — Build the search index

```bash
python ingest.py
```

This reads your PDFs, breaks them into chunks, generates embeddings (downloading the embedding model automatically on first run — about 90MB), and saves everything into a `vector_store/` folder. You'll see progress printed as it runs. Re-run this command any time you add or change documents in `Data/`.

### Step 8 — Ask your first question

```bash
python rag_pipeline.py "your question here"
```

You should see a real, grounded answer along with the source document it came from. 🎉

---

## How to ask questions

**One-off question:**
```bash
python rag_pipeline.py "What is RPC?"
```

**Interactive mode** (ask multiple questions without restarting):
```bash
python rag_pipeline.py
```
Then type your questions one at a time, and type `quit` to exit.

---

## How to run the evaluation

This project includes a small evaluation harness that tests the system against a fixed set of questions and reports how well it performs.

```bash
python eval/eval.py
```

This runs every question in `eval/eval_set.json` through the pipeline, scores each answer using an LLM-as-judge approach (0–10), and prints a summary: average score, latency stats, and any failure cases (answers scoring below 6). Full results are also saved to `eval/eval_results.json`.

**Example results from development:**
- Average score: ~8.9–9.0 / 10 across multiple runs
- Average latency: ~10–14 seconds per question
- Cost: $0.00 (Groq free tier)

Note: scores vary slightly between runs even at a low temperature setting — this is normal LLM non-determinism (in both the answer-generation and judging steps), not a bug.

---

## Project structure

```
rag-project/
├── Data/                   # your source PDF documents
├── vector_store/           # generated FAISS index + metadata (not in git — regenerated by ingest.py)
├── eval/
│   ├── eval_set.json       # test questions and expected answers
│   ├── eval.py              # evaluation runner + LLM judge
│   └── eval_results.json   # generated results (not in git)
├── config.py                 # all settings and prompts, documented in one place
├── ingest.py                  # PDF → chunks → embeddings → FAISS index
├── rag_pipeline.py            # retrieval + generation + error handling
├── requirements.txt
├── .env.example                # template — copy to .env and add your key
├── .gitignore
└── README.md
```

---

## Known limitations

- **Slide-based PDFs** (e.g., exported from PowerPoint) can lose spacing between text elements during extraction — you might see things like "SystemCourse" instead of "System Course." A partial cleanup fix is applied in `ingest.py`, but it isn't perfect for lowercase-to-lowercase word boundaries. A more robust fix would use a layout-aware library like `pdfplumber` or `unstructured`.
- **Scanned/image-only PDFs are not supported** — the text must already be selectable in the PDF.
- **No conversation memory** — each question is treated independently; the system doesn't remember earlier questions in the same session.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pip'` or similar** → Your virtual environment isn't active. Run the activate command from Step 2 again.

**`python rag_pipeline.py` says vector store not found** → Run `python ingest.py` first to build the index.

**`GROQ_API_KEY environment variable not set`** → Make sure you created a `.env` file in the project root (not inside a subfolder) with your key, exactly as shown in Step 5.

**A specific PDF fails with decompression/header errors during `ingest.py`** → That file may have some internal corruption. Try re-saving/exporting it from its original source, or simply remove it from `Data/` — the pipeline will continue with the remaining documents.

**Answers seem to ignore a document that should be relevant** → Try increasing `TOP_K` in `config.py` (e.g., from 4 to 6) and re-run your question — this widens how many chunks get considered.

---

## License

This is a personal learning project. Feel free to fork and adapt it for your own learning.