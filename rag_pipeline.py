"""
rag_pipeline.py
----------------
Given a user question, retrieve the most relevant chunks from the FAISS
index and use Groq's LLM to generate a grounded answer with citations.

Handles three required failure modes:
  1. Empty / weak retrieval  -> tell the user, don't hallucinate an answer
  2. API errors               -> caught, retried, clear message returned
  3. Timeouts                 -> caught, clear message returned

Usage:
    python rag_pipeline.py "What is RPC?"
    python rag_pipeline.py            # interactive mode
"""

import json
import os
import sys
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq, APITimeoutError, APIError, APIConnectionError
from dotenv import load_dotenv

import config

load_dotenv()


class RAGPipeline:
    def __init__(self):
        self._load_index()
        self._load_embedding_model()
        self._load_llm_client()

    # -- setup -------------------------------------------------------------

    def _load_index(self):
        if not os.path.exists(config.FAISS_INDEX_PATH) or not os.path.exists(config.METADATA_PATH):
            print("[ERROR] Vector store not found. Run `python ingest.py` first.")
            sys.exit(1)
        self.index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def _load_embedding_model(self):
        self.embed_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)

    def _load_llm_client(self):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("[ERROR] GROQ_API_KEY environment variable not set.")
            print("        Run: $env:GROQ_API_KEY='your-key-here'  (PowerShell)")
            sys.exit(1)
        self.llm_client = Groq(api_key=api_key, timeout=config.REQUEST_TIMEOUT_SECONDS)

    # -- retrieval -----------------------------------------------------------

    def retrieve(self, question: str):
        """Returns list of {source, chunk_index, text, score}, best first."""
        query_vec = self.embed_model.encode([question], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")

        scores, indices = self.index.search(query_vec, config.TOP_K)
        scores, indices = scores[0], indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append({
                "source": meta["source"],
                "chunk_index": meta["chunk_index"],
                "text": meta["text"],
                "score": float(score),
            })
        return results

    def _format_context(self, chunks: list) -> str:
        parts = []
        for c in chunks:
            parts.append(f"[source: {c['source']}, chunk {c['chunk_index']}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    # -- generation with error handling -------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Calls Groq with retries. Raises a clear error on failure."""
        last_error = None
        for attempt in range(1, config.MAX_RETRIES + 2):
            try:
                response = self.llm_client.chat.completions.create(
                    model=config.GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
                content = response.choices[0].message.content
                if not content or not content.strip():
                    raise RuntimeError("Model returned an empty response.")
                return content.strip()

            except APITimeoutError as e:
                last_error = e
                if attempt <= config.MAX_RETRIES:
                    print(f"  [warn] Timeout on attempt {attempt}, retrying...")
            except (APIError, APIConnectionError) as e:
                last_error = e
                if attempt <= config.MAX_RETRIES:
                    print(f"  [warn] API error on attempt {attempt}: {e}")
            except Exception as e:
                last_error = e
                print(f"  [warn] Unexpected error on attempt {attempt}: {e}")

            if attempt <= config.MAX_RETRIES:
                time.sleep(1.5 * attempt)

        if isinstance(last_error, APITimeoutError):
            raise TimeoutError(config.MSG_TIMEOUT)
        raise RuntimeError(config.MSG_API_ERROR)

    # -- main entrypoint -----------------------------------------------------

    def answer(self, question: str) -> dict:
        if not question or not question.strip():
            return {"answer": "Please provide a non-empty question.", "sources": [], "chunks_used": []}

        chunks = self.retrieve(question)

        strong_chunks = [c for c in chunks if c["score"] >= config.MIN_SIMILARITY_SCORE]
        if not strong_chunks:
            return {"answer": config.MSG_EMPTY_RETRIEVAL, "sources": [], "chunks_used": []}

        context = self._format_context(strong_chunks)
        user_prompt = config.USER_PROMPT_TEMPLATE.format(context=context, question=question)

        try:
            answer_text = self._call_llm(config.SYSTEM_PROMPT, user_prompt)
        except (TimeoutError, RuntimeError) as e:
            return {"answer": str(e), "sources": [], "chunks_used": strong_chunks}

        sources = sorted(set(c["source"] for c in strong_chunks))
        return {"answer": answer_text, "sources": sources, "chunks_used": strong_chunks}


def _print_result(question: str, result: dict):
    print(f"\nQ: {question}")
    print(f"A: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    print("-" * 70)


if __name__ == "__main__":
    pipeline = RAGPipeline()

    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        result = pipeline.answer(q)
        _print_result(q, result)
    else:
        print("RAG pipeline ready. Type a question (or 'quit' to exit).\n")
        while True:
            q = input("Question: ").strip()
            if q.lower() in ("quit", "exit"):
                break
            result = pipeline.answer(q)
            _print_result(q, result)