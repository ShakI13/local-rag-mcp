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

_DIR_REF_RE = re.compile(
    r"(?:under|in|в)\s+/([^\s/?#,]+)|/([^\s/?#,]+)",
    re.IGNORECASE,
)
_EXISTENCE_ASK_RE = re.compile(
    r"is\s+there|есть\s+ли|does\s+(?:there\s+)?exist|описан",
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

    Unrelated Top-K hits are dropped so generation must refuse instead of
    inventing from off-topic sources.
    """
    if not contexts:
        return []
    q_tokens = {
        t for t in tokenize_normalized(query) if len(t) >= 4 and t not in _QUERY_STOPWORDS
    }
    if not q_tokens:
        # No meaningful tokens left → cannot judge support; treat as unsupported.
        return []

    kept = []
    for chunk in contexts:
        haystack = tokenize_normalized(
            f"{chunk.get('source', '')} {chunk.get('text', '')}"
        )
        hay_set = set(haystack)
        if q_tokens & hay_set:
            kept.append(chunk)
    return kept


def _subdir_named_in_query(query: str) -> str | None:
    """Return a docs subdirectory name if the query references /Name."""
    if not query:
        return None
    matches = []
    for m in _DIR_REF_RE.finditer(query):
        name = m.group(1) or m.group(2)
        if name:
            matches.append(name.strip().strip("/\\"))
    return matches[-1] if matches else None


def directory_listing_chunk(subdir: str):
    """
    Synthetic chunk: file listing for a docs subdirectory named in the question.

    Path comes from the user query (e.g. /Foo), never from a hardcoded trap folder.
    """
    if not subdir or subdir in {".", ".."} or "/" in subdir or "\\" in subdir:
        return None
    src_dir = Path(__file__).parent.parent
    docs_root = (src_dir / DOCUMENTS_DIR).resolve()
    target = (docs_root / subdir).resolve()
    try:
        target.relative_to(docs_root)
    except ValueError:
        return None
    if not target.is_dir():
        return None
    names = sorted(
        p.name
        for p in target.iterdir()
        if p.is_file() and p.suffix.lower() in {".md", ".txt", ".docx", ".pdf"}
    )
    if not names:
        return None
    listing = "\n".join(f"- {name}" for name in names)
    text = (
        f"Files that exist under /{subdir}:\n"
        f"{listing}\n"
        "A name merely listed or mentioned elsewhere (e.g. README) is not a dedicated "
        f"description under /{subdir} unless a matching file appears in this list."
    )
    return {
        "source": str(Path(DOCUMENTS_DIR) / subdir),
        "chunk_id": 0,
        "is_directory_listing": True,
        "text": text,
    }


def prepare_contexts(query: str, contexts):
    """Filter off-topic chunks; for existence asks naming /Dir, add that dir's listing."""
    prepared = filter_contexts_for_query(query, contexts)
    if _EXISTENCE_ASK_RE.search(query or ""):
        subdir = _subdir_named_in_query(query)
        if subdir:
            extra = directory_listing_chunk(subdir)
            if extra:
                prepared = [extra] + [
                    c for c in prepared if not c.get("is_directory_listing")
                ]
    return prepared


def _asked_names_from_existence_query(query: str) -> list[str]:
    """Pull candidate entity names from an existence question (path-agnostic)."""
    m = re.search(
        r"(?:is\s+there(?:\s+an?)?|есть\s+ли|does\s+(?:a\s+|an\s+|there\s+)?exist)\s+"
        r"(.+?)(?:\s+(?:role\s+)?description|\s+описан|\s+under\s+/|\s+в\s+/|\s+file\b)",
        query or "",
        re.IGNORECASE | re.DOTALL,
    )
    if not m:
        return []
    parts = re.split(r"\s*/\s*|\s*,\s*|\s+and\s+|\s+и\s+", m.group(1), flags=re.IGNORECASE)
    names = []
    for part in parts:
        cleaned = re.sub(
            r"\s+(?:role|description|file|описан\w*)\s*$",
            "",
            part.strip(),
            flags=re.IGNORECASE,
        ).strip(" ?.,;:")
        if cleaned and cleaned.lower() not in {"a", "an", "the", "role"}:
            names.append(cleaned)
    return names


def existence_listing_answer(query: str, contexts):
    """
    Deterministic answer when an existence ask already has a directory listing.

    Small models often refuse or say bare Yes despite the listing; answer from
    the listing instead. Folder path and names come from the query — not hardcoded.
    """
    if not _EXISTENCE_ASK_RE.search(query or ""):
        return None
    listing = next((c for c in contexts if c.get("is_directory_listing")), None)
    if not listing:
        return None

    subdir = _subdir_named_in_query(query) or "the named folder"
    files = [
        line[2:].strip()
        for line in listing["text"].splitlines()
        if line.startswith("- ")
    ]
    asked = _asked_names_from_existence_query(query) or ["the requested item"]

    missing = [
        name for name in asked if not any(name.lower() in f.lower() for f in files)
    ]
    file_list = ", ".join(files) if files else "(none)"
    if missing:
        missing_txt = " / ".join(missing)
        return (
            f"A README or process mention is not enough: there is no dedicated "
            f"description file for {missing_txt} under /{subdir}. "
            f"Files that exist under /{subdir} are: {file_list}."
        )
    present_txt = " / ".join(asked)
    return (
        f"Yes — dedicated file(s) for {present_txt} exist under /{subdir}: {file_list}."
    )


def bm25_query_from_expand(keywords: str, query: str) -> str:
    """
    BM25 input: expanded keywords plus the original question.

    Near-miss English expand (e.g. `migration`) must not drop original Cyrillic
    terms that would match Russian docs.
    """
    kw = (keywords or "").strip()
    q = (query or "").strip()
    if not kw:
        return q
    if not q or kw == q:
        return kw
    return f"{kw} {q}"


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
    bm25_query = bm25_query_from_expand(keywords, query)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_vector = pool.submit(vector_search, query, lane_k)
        fut_bm25 = pool.submit(bm25_search, bm25_query, lane_k)
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
- For questions about whether a dedicated description/file exists under a folder named in the query: a name merely listed or mentioned elsewhere (e.g. README) is NOT enough. Prefer the directory listing / files under that folder; if none match, say so clearly (partial yes / no dedicated doc).
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


def ask(query: str):
    """Answer a question using RAG."""
    contexts = prepare_contexts(query, retrieve(query))
    listing_answer = existence_listing_answer(query, contexts)
    if listing_answer is not None:
        return listing_answer, contexts
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
