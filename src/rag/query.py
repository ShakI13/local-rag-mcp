import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    LANE_K,
    RRF_K,
)
from rag.bm25_index import Bm25Index
from rag.expand import expand_query
from rag.rrf import rrf_fuse

# Lazy globals — avoid loading heavy models on import (tests inject lane doubles)
_model = None
index = None
chunks = []
_bm25_index = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _rebuild_bm25():
    global _bm25_index
    _bm25_index = Bm25Index(chunks) if chunks else None


def _ensure_index_exists():
    """Ensure FAISS index exists, build it if it doesn't."""
    global index, chunks
    import faiss

    # Resolve paths relative to src directory
    src_dir = Path(__file__).parent.parent
    index_path = src_dir / FAISS_INDEX_PATH
    chunks_path = src_dir / CHUNKS_PATH

    # Check if index exists
    if index_path.exists() and chunks_path.exists():
        try:
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            _rebuild_bm25()
            return True
        except Exception as e:
            print(f"⚠️  Warning: Error loading existing index: {e}")
            print("Rebuilding index...")

    # Index doesn't exist or failed to load, build it
    print("📦 Index not found. Building index from documents...")
    try:
        from rag.build_index import build_index
        build_index()

        # Load the newly created index
        if index_path.exists() and chunks_path.exists():
            index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            _rebuild_bm25()
            print("✅ Index built and loaded successfully")
            return True
        else:
            print("❌ Failed to build index. No documents found or error occurred.")
            from config import DOCUMENTS_DIR
            docs_path = src_dir / DOCUMENTS_DIR
            print(f"   Check that documents exist in: {docs_path}")
            return False
    except Exception as e:
        print(f"❌ Error building index: {e}")
        import traceback
        traceback.print_exc()
        return False


def _default_ensure_ready():
    if index is None or len(chunks) == 0:
        return _ensure_index_exists()
    return True


def _default_vector_search(query: str, k: int):
    import faiss

    if index is None or len(chunks) == 0 or k <= 0:
        return []
    q_emb = _get_model().encode([query])
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, min(k, len(chunks)))
    hits = []
    for i in ids[0]:
        if i < 0:
            continue
        hits.append(chunks[i])
    return hits


def _default_bm25_search(keywords: str, k: int):
    if _bm25_index is None:
        return []
    return _bm25_index.search(keywords, k)


def retrieve(
    query: str,
    *,
    _vector_search=None,
    _bm25_search=None,
    _expand=None,
    _ensure_ready=None,
    _lane_k=None,
    _top_k=None,
    _rrf_k=None,
):
    """Retrieve relevant chunks via Hybrid Search (Vector ∥ BM25 → RRF → Top-K)."""
    ensure_ready = _ensure_ready or _default_ensure_ready
    if not ensure_ready():
        return []

    vector_search = _vector_search or _default_vector_search
    bm25_search = _bm25_search or _default_bm25_search

    if index is None or len(chunks) == 0:
        # Injected lane doubles may still run without a loaded corpus
        if _vector_search is None and _bm25_search is None:
            return []

    lane_k = LANE_K if _lane_k is None else _lane_k
    top_k = TOP_K if _top_k is None else _top_k
    rrf_k = RRF_K if _rrf_k is None else _rrf_k

    expand = _expand or expand_query
    keywords = expand(query)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_vector = pool.submit(vector_search, query, lane_k)
        fut_bm25 = pool.submit(bm25_search, keywords, lane_k)
        vector_hits = fut_vector.result()
        bm25_hits = fut_bm25.result()

    return rrf_fuse([vector_hits, bm25_hits], rrf_k=rrf_k, top_k=top_k)


def build_prompt(query, contexts):
    """Build prompt with retrieved context."""
    if not contexts:
        return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question based on your general knowledge. If you don't know, say so.</instructions>

<query>
{query}
</query>

<assistant>
"""

    context_text = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in contexts
    )

    return f"""
<role>You are a helpful assistant that answers questions about company information.</role>
<instructions>Answer the question ONLY based on the context provided below. If the answer is not in the context, say "I don't have that information in the knowledge base."</instructions>

<context>
{context_text}
</context>

<query>
{query}
</query>

<assistant>
"""


def ask_llm(prompt):
    """Query Ollama LLM."""
    import requests

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


def ask(query: str):
    """Answer a question using RAG."""
    contexts = retrieve(query)
    prompt = build_prompt(query, contexts)
    return ask_llm(prompt), contexts


if __name__ == "__main__":
    while True:
        q = input("\n❓ Question: ")
        if q.lower() in {"exit", "quit"}:
            break
        print("\n🤖 Answer:\n")
        answer, sources = ask(q)
        print(answer)
        if sources:
            print("\n📚 Sources:")
            for src in sources:
                print(f"  - {src['source']}")
