"""BM25 index seam — in-memory corpus from chunk list."""
import unittest

from rag.bm25_index import Bm25Index


class TestBm25Index(unittest.TestCase):
    def test_ranks_chunk_with_literal_rare_term_first(self):
        chunks = [
            {"text": "general database notes about storage", "source": "a.md", "chunk_id": 0},
            {"text": "we use asyncpg for postgres access", "source": "b.md", "chunk_id": 0},
            {"text": "unrelated frontend styling guide", "source": "c.md", "chunk_id": 0},
        ]
        index = Bm25Index(chunks)
        hits = index.search("asyncpg postgres", k=2)
        self.assertEqual(hits[0]["source"], "b.md")
        self.assertIn("asyncpg", hits[0]["text"])

    def test_empty_corpus_returns_empty(self):
        self.assertEqual(Bm25Index([]).search("anything", k=5), [])


if __name__ == "__main__":
    unittest.main()
