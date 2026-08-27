"""
ingest.py
---------
Reads all PDF documents from DATA_DIR, extracts their text, splits it into
overlapping chunks, embeds each chunk locally with sentence-transformers,
and saves a FAISS index + metadata to disk.

Run this once whenever your documents change:
    python ingest.py
"""

import json
import os
import sys
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config




def extract_text_from_pdf(path: str) -> str:
    """Extracts and joins text from every page of a PDF, then cleans up
    common spacing issues that happen when extracting from slide-based PDFs
    (words getting glued together like 'SystemCourse')."""
    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    full_text = "\n".join(pages_text).strip()

    # Add a space where a lowercase letter is immediately followed by an
    # uppercase letter (e.g. "SystemCourse" -> "System Course")
    full_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', full_text)

    # Add a line break after bullet points for readability
    full_text = full_text.replace('•', '\n• ')

    return full_text


def load_documents(data_dir: str) -> list[dict]:
    """Load every .pdf file in data_dir. Returns list of {source, text}."""
    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory '{data_dir}' not found.")
        sys.exit(1)

    docs = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith(".pdf"):
            continue
        path = os.path.join(data_dir, filename)
        print(f"  Reading '{filename}'...")
        text = extract_text_from_pdf(path)
        if not text:
            print(f"  [WARN] No extractable text found in '{filename}' "
                  f"(it may be a scanned/image-only PDF). Skipping.")
            continue
        docs.append({"source": filename, "text": text})

    if not docs:
        print(f"[ERROR] No readable PDF files found in '{data_dir}'.")
        sys.exit(1)

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Sliding-window character chunker with overlap, snapped to whitespace."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        if end < text_len:
            snap = text.rfind(" ", start, end)
            if snap != -1 and snap > start:
                end = snap
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    return chunks


def build_index():
    print("Loading documents...")
    docs = load_documents(config.DATA_DIR)
    print(f"  Loaded {len(docs)} document(s).")

    print("Chunking documents...")
    all_chunks = []
    all_metadata = []
    for doc in docs:
        chunks = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": doc["source"],
                "chunk_index": i,
                "text": chunk,
            })
    print(f"  Produced {len(all_chunks)} chunks total.")

    if not all_chunks:
        print("[ERROR] No chunks were produced. Aborting.")
        sys.exit(1)

    print(f"Loading embedding model '{config.EMBEDDING_MODEL_NAME}' "
          f"(first run downloads it, may take a minute)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    print("Embedding chunks...")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Done. Index saved to '{config.FAISS_INDEX_PATH}', "
          f"metadata saved to '{config.METADATA_PATH}'.")
    print(f"Total vectors indexed: {index.ntotal}")


if __name__ == "__main__":
    build_index()