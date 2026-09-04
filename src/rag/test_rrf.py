"""RRF fusion — pure logic seam (SPEC Testing Decisions)."""
import unittest

from rag.rrf import chunk_identity, rrf_fuse


def _chunk(source: str, chunk_id: int, text: str = "") -> dict:
    return {"source": source, "chunk_id": chunk_id, "text": text or f"{source}:{chunk_id}"}


class TestRrfFuse(unittest.TestCase):
    def test_dual_lane_hit_outranks_single_lane_hits(self):
        # Toy ranks from docs/RAG-junior-guide.md §17 (k=60)
        a = _chunk("asyncpg.md", 0)
        b = _chunk("migrations.md", 0)
        c = _chunk("vector-only.md", 0)
        d = _chunk("bm25-only.md", 0)

        vector = [a, c, b]  # ranks 1, 2, 3
        bm25 = [b, a, d]  # ranks 1, 2, 3

        fused = rrf_fuse([vector, bm25], rrf_k=60, top_k=4)

        self.assertEqual(
            [chunk_identity(c) for c in fused],
            [
                ("asyncpg.md", 0),
                ("migrations.md", 0),
                ("vector-only.md", 0),
                ("bm25-only.md", 0),
            ],
        )

    def test_dedupes_by_source_and_chunk_id_not_bare_chunk_id(self):
        left = _chunk("a.md", 3)
        right = _chunk("b.md", 3)
        fused = rrf_fuse([[left], [right]], rrf_k=60, top_k=5)
        identities = {(c["source"], c["chunk_id"]) for c in fused}
        self.assertEqual(identities, {("a.md", 3), ("b.md", 3)})

    def test_respects_top_k(self):
        chunks = [_chunk("x.md", i) for i in range(5)]
        fused = rrf_fuse([chunks, []], rrf_k=60, top_k=2)
        self.assertEqual(len(fused), 2)


if __name__ == "__main__":
    unittest.main()
