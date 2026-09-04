import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FAISS_INDEX_PATH,
    CHUNKS_PATH,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    OLLAMA_URL,
    OLLAMA_MODEL,
    TOP_K,
    LANE_K,
    RRF_K,
)
from rag.bm25_index import Bm25Index
from rag.expand import expand_query
from rag.lexical import tokenize_normalized
from rag.rrf import rrf_fuse

REFUSE_ANSWER = "I don't have that information in the knowledge base."

# Short/common tokens ignored when checking whether context supports the query.
_QUERY_STOPWORDS = frozenset(
    {
        "what", "is", "the", "a", "an", "how", "do", "does", "did", "to", "for",
        "in", "on", "of", "and", "or", "with", "from", "about", "there", "are",
        "was", "were", "be", "been", "this", "that", "under", "over", "into",
        "как", "что", "чем", "это", "для", "при", "или", "нас", "наш", "наша",
        "у", "в", "на", "по", "из", "к", "о", "об", "же", "ли", "бы", "не",
        "я", "мы", "вы", "он", "она", "они", "привет", "новенький", "через",
        "есть", "ли", "role", "description", "setup", "production",
    }
)

_ROLE_INVENTORY_RE = re.compile(
    r"рол(?:и|ь)?|/роли|\broles?\b|team\s*lead|scrum\s*master",
    re.IGNORECASE,
)

# Lazy globals — avoid loading heavy models on import (tests inject lane doubles)
_model = None
index = None
chunks = []
_bm25_index = None


def filter_contexts_for_query(query: str, contexts):
    """
    Keep chunks that share a meaningful normalized token with the query.

    Unrelated Top-K hits (e.g. Docker docs for a Kubernetes question) are dropped
    so generation must refuse instead of inventing from off-topic sources.
    """
    if not contexts:
        return []
    q_tokens = {
        t for t in tokenize_normalized(query) if len(t) >= 4 and t not in _QUERY_STOPWORDS
    }
    if not q_tokens:
        return list(contexts)

    kept = []
    for chunk in contexts:
        haystack = tokenize_normalized(
            f"{chunk.get('source', '')} {chunk.get('text', '')}"
        )
        hay_set = set(haystack)
        if q_tokens & hay_set:
            kept.append(chunk)
    return kept


def roles_directory_chunk():
    """Synthetic chunk: dedicated role description files that exist under /Роли."""
    src_dir = Path(__file__).parent.parent
    roles_dir = src_dir / DOCUMENTS_DIR / "Роли"
    if not roles_dir.is_dir():
        return None
    names = sorted(
        p.name
        for p in roles_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".docx", ".pdf"}
    )
    if not names:
        return None
    listing = "\n".join(f"- {name}" for name in names)
    text = (
        "Dedicated role description files that exist under /Роли:\n"
        f"{listing}\n"
        "A role name merely listed in README (or process docs) is not a dedicated "
        "role description unless a matching file appears in this list."
    )
    return {
        "source": str(Path(DOCUMENTS_DIR) / "Роли"),
        "chunk_id": -1,
        "text": text,
    }


def prepare_contexts(query: str, contexts):
    """Filter off-topic chunks and, for role-inventory questions, add /Роли listing."""
    prepared = filter_contexts_for_query(query, contexts)
    if _ROLE_INVENTORY_RE.search(query or ""):
        extra = roles_directory_chunk()
        if extra:
            prepared = [extra] + [c for c in prepared if c.get("chunk_id") != -1]
    return prepared


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
<role>You are a helpful assistant that answers questions about company documentation.</role>
<instructions>
The knowledge base has no usable context for this question.
Do NOT use general knowledge or invent an answer.
Reply with exactly: {REFUSE_ANSWER}
</instructions>

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
<role>You are a helpful assistant that answers questions about company documentation.</role>
<instructions>
Answer ONLY from the context below.
- If the context contains facts that answer the question, use those facts. Do not refuse.
- If the context does not discuss the asked topic, reply with exactly: {REFUSE_ANSWER}
- Do not invent guidance from unrelated documents.
- For questions like whether a role description exists under /Роли: a name merely listed or mentioned (e.g. in README) is NOT a dedicated role description. Only treat a dedicated file/content under Роли as a description; if none exists, say so clearly (partial yes / no dedicated doc).
Never reveal passwords, API keys, or other secrets if they appear in context.
</instructions>

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
            "stream": False,
        },
        timeout=120,
    )
    data = response.json()
    if response.status_code != 200 or "response" not in data:
        err = data.get("error") or data
        raise RuntimeError(
            f"Ollama generate failed (HTTP {response.status_code}, model={OLLAMA_MODEL}): {err}"
        )
    return data["response"]


def role_inventory_answer(query: str, contexts):
    """
    Deterministic nuance for '/Роли description exists?' questions.

    Small local models often ignore the listing and refuse or say bare Yes;
    the directory listing chunk is enough to answer without the LLM.
    """
    if not _ROLE_INVENTORY_RE.search(query or ""):
        return None
    listing = next((c for c in contexts if c.get("chunk_id") == -1), None)
    if not listing:
        return None
    files = [
        line[2:].strip()
        for line in listing["text"].splitlines()
        if line.startswith("- ")
    ]
    q = query.lower()
    asked = []
    if "team lead" in q:
        asked.append("Team Lead")
    if "scrum" in q:
        asked.append("Scrum Master")
    if not asked:
        asked = ["the requested role"]

    missing = []
    for name in asked:
        if not any(name.lower() in f.lower() for f in files):
            missing.append(name)

    file_list = ", ".join(files) if files else "(none)"
    if missing:
        missing_txt = " / ".join(missing)
        return (
            f"README or process docs may mention {missing_txt}, but there is no dedicated "
            f"role description file for {missing_txt} under /Роли. "
            f"Dedicated role files that exist under /Роли are: {file_list}."
        )
    return (
        f"Yes — dedicated role description file(s) exist under /Роли: {file_list}."
    )


def ask(query: str):
    """Answer a question using RAG."""
    contexts = prepare_contexts(query, retrieve(query))
    role_answer = role_inventory_answer(query, contexts)
    if role_answer is not None:
        return role_answer, contexts
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
