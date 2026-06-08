"""
rag_pipeline.py  —  LangChain-powered RAG for NIST SP 800-53
=============================================================
Drop-in replacement for the original ChromaDB + sentence-transformers version.
The PUBLIC API is IDENTICAL — api_server.py imports these three functions
and does not need to change at all:

    build_nist_vectorstore(nist_csv_path)  -> vectorstore object
    get_nist_collection()                  -> vectorstore object
    retrieve_nist_control(query, vs)       -> {"control_id", "control_name", "text"}

What changed internally
------------------------
OLD (manual):  pandas → manual chunking → SentenceTransformer.encode()
               → chromadb.PersistentClient → collection.add() / collection.query()

NEW (LangChain): CSVLoader → RecursiveCharacterTextSplitter
                 → GoogleGenerativeAIEmbeddings
                 → Chroma (LangChain wrapper) → similarity_search()

Why LangChain here and nowhere else
-------------------------------------
The NIST document is unstructured prose — 1,000+ controls each with a
multi-paragraph description. That is exactly the problem LangChain's
document-loading + text-splitting pipeline is designed for.
The CSVs (assets, vulns, threat intel) are structured tabular data
answered by exact-match pandas filters, where LangChain adds no value.
"""

import os
import time
import logging
import pandas as pd

# ── LangChain imports ──────────────────────────────────────────────────────
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
_HERE             = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR        = os.path.join(_HERE, "chroma_db")
COLLECTION_NAME   = "nist_controls_lc"
CHUNK_SIZE        = 1000   # characters per chunk
CHUNK_OVERLAP     = 150    # overlap keeps context across boundaries
TOP_K             = 3     # candidates to retrieve before threshold filtering
SIMILARITY_THRESHOLD = 1.0  # L2 distance cutoff; above this = no confident match


# ── Internal helpers ───────────────────────────────────────────────────────

def _get_embeddings() -> FastEmbedEmbeddings:
    """
    Returns a local FastEmbed embedding object (BAAI/bge-small-en-v1.5).
    Runs entirely in-process — no API key, no quota, no cost.
    Model (~50MB) is downloaded once on first use and cached locally.
    """
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


def _load_nist_documents(nist_csv_path: str):
    """
    Step 1 — Load
    Reads nist_controls.csv (NIST SP 800-53 rev5 catalog) and builds one
    LangChain Document per row.
    Each document's page_content is a rich text blob:
        "SI-2 — Flaw Remediation\n<control_text>\n<discussion>"
    The control_id and control_name are stored as metadata so we can
    return them alongside the retrieved text without re-parsing.
    """
    df = pd.read_csv(nist_csv_path)

    # Build a single rich text column that will become page_content
    df["_content"] = (
        df["identifier"].fillna("").astype(str) + " — " +
        df["name"].fillna("").astype(str) + "\n" +
        df["control_text"].fillna("").astype(str) + "\n" +
        df["discussion"].fillna("").astype(str)
    ).str.strip()

    # Store the metadata columns separately so we can retrieve them later
    df["_control_id"]   = df["identifier"].fillna("").astype(str)
    df["_control_name"] = df["name"].fillna("").astype(str)

    # Drop rows with no useful content
    df = df[df["_content"].str.len() > 50].reset_index(drop=True)

    # DataFrameLoader turns each row into a LangChain Document
    # page_content_column = the column that becomes Document.page_content
    # All other columns become Document.metadata automatically
    loader = DataFrameLoader(df, page_content_column="_content")
    documents = loader.load()

    log.info(f"Loaded {len(documents)} NIST control documents from {nist_csv_path}")
    return documents


def _split_documents(documents):
    """
    Step 2 — Split
    NIST control descriptions can be long. We split them into overlapping
    chunks so a single large control doesn't get truncated at embedding time.
    RecursiveCharacterTextSplitter tries to split on paragraphs first,
    then sentences, then words — preserving semantic boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    log.info(f"Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


# ── Public API ─────────────────────────────────────────────────────────────

def build_nist_vectorstore(nist_csv_path: str):
    """
    Build the LangChain/Chroma vector store from scratch and persist it to disk.

    Pipeline:
        CSV  →  DataFrameLoader  →  Documents
             →  RecursiveCharacterTextSplitter  →  Chunks
             →  GoogleGenerativeAIEmbeddings  →  Vectors
             →  Chroma.from_documents()  →  persisted to CHROMA_DIR

    Call this ONCE. Subsequent runs call get_nist_collection() instead,
    which loads from disk in seconds without re-embedding.

    Returns: a LangChain Chroma vectorstore object.
    """
    log.info("Building NIST vector store with LangChain (first-time setup)...")

    # Steps 1 + 2: load and split
    documents = _load_nist_documents(nist_csv_path)
    chunks    = _split_documents(documents)

    # Step 3: embed + store — process in batches to stay within the free-tier
    # rate limit of 100 embed requests/minute (gemini-embedding-001).
    BATCH_SIZE      = 10   # chunks per add_documents call — stays inside 100 RPM free tier
    BATCH_DELAY     = 10   # seconds between batches → ~6 calls/min, well under limit
    RETRY_WAIT      = 120  # seconds to sleep when a 429 is returned

    embeddings  = _get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    total = len(chunks)
    i = 0
    while i < total:
        batch = chunks[i : i + BATCH_SIZE]
        try:
            vectorstore.add_documents(batch)
            print(f"  Embedded {min(i + BATCH_SIZE, total)}/{total} chunks...", flush=True)
            i += BATCH_SIZE
            if i < total:
                time.sleep(BATCH_DELAY)
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(f"  Rate limit hit — waiting {RETRY_WAIT}s before retry...", flush=True)
                time.sleep(RETRY_WAIT)
            else:
                raise

    log.info(f"✓ Vector store built and saved to {CHROMA_DIR}")
    return vectorstore


def get_nist_collection():
    """
    Load an existing LangChain/Chroma vector store from disk.
    Fast — no re-embedding, just loads the persisted index.

    Returns: a LangChain Chroma vectorstore object.
    Raises:  RuntimeError if the store hasn't been built yet.
    """
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"Vector store not found at {CHROMA_DIR}.\n"
            "Run build_nist_vectorstore() first, or call main.py which does this automatically."
        )

    embeddings  = _get_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    log.info(f"Loaded existing NIST vector store from {CHROMA_DIR}")
    return vectorstore


def retrieve_nist_control(query: str, vectorstore, top_k: int = TOP_K) -> dict:
    """
    Semantic search: given a plain-English query describing a risk,
    return the most relevant NIST SP 800-53 control above a confidence threshold.

    Retrieves top_k candidates with scores, filters by SIMILARITY_THRESHOLD
    (L2 distance — lower is better), and returns the best confident match.
    If no match clears the threshold, returns confidence="none" so the caller
    can skip the LLM rather than hallucinate a recommendation.

    Returns dict with keys:
        control_id, control_name, text, confidence ("high"|"low"|"none"), score
    """
    try:
        results = vectorstore.similarity_search_with_score(query, k=top_k)
    except Exception as e:
        log.error(f"Vector search failed: {e}")
        return {"control_id": "N/A", "control_name": "Search error", "text": str(e),
                "confidence": "none", "score": None}

    if not results:
        return {"control_id": "N/A", "control_name": "No match",
                "text": "No matching NIST control found.", "confidence": "none", "score": None}

    # Filter to candidates that clear the confidence threshold
    confident = [(doc, score) for doc, score in results if score <= SIMILARITY_THRESHOLD]

    if not confident:
        best_score = results[0][1]
        log.warning(
            f"No confident NIST match (best L2={best_score:.3f} > threshold={SIMILARITY_THRESHOLD}): "
            f"{query[:80]}"
        )
        return {
            "control_id": "N/A", "control_name": "No strong match",
            "text": "No NIST control met the confidence threshold for this risk.",
            "confidence": "none", "score": round(best_score, 4)
        }

    doc, score = confident[0]  # lowest distance = best match
    confidence = "high" if score < 0.5 else "low"

    metadata     = doc.metadata
    control_id   = metadata.get("_control_id",   "")
    control_name = metadata.get("_control_name", "")

    # Fallback: parse from text if metadata is missing
    if not control_id and doc.page_content:
        first_line = doc.page_content.split("\n")[0]
        if " — " in first_line:
            parts        = first_line.split(" — ", 1)
            control_id   = parts[0].strip()
            control_name = parts[1].strip() if len(parts) > 1 else ""

    log.info(f"NIST match: {control_id} (L2={score:.3f}, confidence={confidence})")

    return {
        "control_id"  : control_id,
        "control_name": control_name,
        "text"        : doc.page_content,
        "confidence"  : confidence,
        "score"       : round(score, 4)
    }
