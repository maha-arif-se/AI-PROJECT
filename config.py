"""
config.py
---------
All settings and prompts for the RAG pipeline live here, documented,
so nothing is hardcoded inside ingest.py or rag_pipeline.py.
"""

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_DIR = "Data"                # folder containing your PDF documents
INDEX_DIR = "vector_store"       # where the FAISS index + metadata get saved
FAISS_INDEX_PATH = f"{INDEX_DIR}/index.faiss"
METADATA_PATH = f"{INDEX_DIR}/metadata.json"

# ---------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------
CHUNK_SIZE = 1200     # characters per chunk (~300 tokens)
CHUNK_OVERLAP = 200   # characters of overlap between consecutive chunks

# ---------------------------------------------------------------------
# Embedding model (runs locally, free, no API key needed)
# ---------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------
TOP_K = 6                     # how many chunks to retrieve per question
MIN_SIMILARITY_SCORE = 0.25   # below this, treat retrieval as "nothing relevant found"

# ---------------------------------------------------------------------
# Groq LLM settings
# ---------------------------------------------------------------------
GROQ_MODEL ="openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 600
REQUEST_TIMEOUT_SECONDS = 15
MAX_RETRIES = 2

# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant that answers questions ONLY using \
the provided context excerpts below. Follow these rules strictly:

1. Base your answer only on the given context. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say clearly: \
"I don't have enough information in the provided documents to answer that."
3. Keep answers concise and factual.
4. After your answer, list the source(s) you used, referencing them by the \
[source: filename] tags shown in the context.
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer the question using only the context above, and cite your sources.
"""

# ---------------------------------------------------------------------
# Failure-mode messages
# ---------------------------------------------------------------------
MSG_EMPTY_RETRIEVAL = (
    "I couldn't find any relevant information in the document set for that "
    "question. Try rephrasing, or this may be outside the scope of the "
    "loaded documents."
)
MSG_API_ERROR = (
    "The language model request failed after retries. Please check your "
    "API key and network connection, then try again."
)
MSG_TIMEOUT = (
    "The request to the language model timed out. Please try again in a "
    "moment."
)