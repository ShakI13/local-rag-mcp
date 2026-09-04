"""Hybrid RAG Retrieve seam tests (no live Ollama / FAISS)."""
import unittest

from rag.query import retrieve
from rag.rrf import chunk_identity


def _chunk(source: str, chunk_id: int, text: str = "") -> dict:
    return {"source": source, "chunk_id": chunk_id, "text": text or f"{source}:{chunk_id}"}


class TestRetrieveHybrid(unittest.TestCase):
    def test_returns_fused_top_k_by_chunk_identity(self):
        a = _chunk("asyncpg.md", 0)
        b = _chunk("migrations.md", 0)
        c = _chunk("vector-only.md", 0)
        d = _chunk("bm25-only.md", 0)

        def vector_search(query, k):
            self.assertEqual(query, "how does asyncpg work?")
            return [a, c, b][:k]

        def bm25_search(keywords, k):
            self.assertEqual(keywords, "how does asyncpg work?")
            return [b, a, d][:k]

        results = retrieve(
            "how does asyncpg work?",
            _vector_search=vector_search,
            _bm25_search=bm25_search,
            _expand=lambda q: q,
            _ensure_ready=lambda: True,
            _lane_k=10,
            _top_k=4,
            _rrf_k=60,
        )

        self.assertEqual(
            [chunk_identity(c) for c in results],
            [
                ("asyncpg.md", 0),
                ("migrations.md", 0),
                ("vector-only.md", 0),
                ("bm25-only.md", 0),
            ],
        )

    def test_single_lane_hits_remain_eligible(self):
        only_vector = _chunk("v.md", 0)
        only_bm25 = _chunk("b.md", 0)

        results = retrieve(
            "q",
            _vector_search=lambda q, k: [only_vector],
            _bm25_search=lambda kw, k: [only_bm25],
            _expand=lambda q: q,
            _ensure_ready=lambda: True,
            _top_k=5,
            _rrf_k=60,
        )
        ids = {chunk_identity(c) for c in results}
        self.assertEqual(ids, {("v.md", 0), ("b.md", 0)})

    def test_empty_index_returns_empty_list(self):
        results = retrieve(
            "q",
            _ensure_ready=lambda: False,
        )
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
