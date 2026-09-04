"""Query Expansion — Keywords soft-parse and fallback via retrieve seam."""
import unittest

from rag.expand import expand_query, parse_keywords
from rag.query import bm25_query_from_expand, retrieve


def _chunk(source: str, chunk_id: int) -> dict:
    return {"source": source, "chunk_id": chunk_id, "text": source}


class TestParseKeywords(unittest.TestCase):
    def test_soft_parses_comma_separated_phrases(self):
        self.assertEqual(
            parse_keywords("asyncpg, Postgres, миграции"),
            "asyncpg, Postgres, миграции",
        )

    def test_strips_wrapping_noise_and_empty_parts(self):
        self.assertEqual(
            parse_keywords("  Keywords: asyncpg ,  , postgres  "),
            "asyncpg, postgres",
        )

    def test_empty_or_unusable_returns_none(self):
        self.assertIsNone(parse_keywords(""))
        self.assertIsNone(parse_keywords("   ,  , "))
        self.assertIsNone(parse_keywords(None))


class TestBm25QueryFromExpand(unittest.TestCase):
    def test_unions_english_near_miss_with_original_cyrillic(self):
        original = (
            "Привет, я новенький — как у нас ходить в Postgres через asyncpg "
            "и что с миграциями?"
        )
        bm25_q = bm25_query_from_expand(
            "asyncpg, Postgres, migration, async",
            original,
        )
        self.assertIn("migration", bm25_q.lower())
        self.assertIn("миграциями", bm25_q)

    def test_does_not_duplicate_when_expand_fell_back_to_original(self):
        q = "original question"
        self.assertEqual(bm25_query_from_expand(q, q), q)


class TestExpandViaRetrieve(unittest.TestCase):
    def test_bm25_gets_keywords_plus_original_vector_gets_original(self):
        seen = {}
        original = "привет, подскажи про asyncpg пожалуйста"

        def vector_search(query, k):
            seen["vector"] = query
            return [_chunk("v.md", 0)]

        def bm25_search(keywords, k):
            seen["bm25"] = keywords
            return [_chunk("b.md", 0)]

        def fake_complete(prompt, temperature):
            self.assertEqual(temperature, 0.1)
            return "asyncpg, postgres"

        results = retrieve(
            original,
            _vector_search=vector_search,
            _bm25_search=bm25_search,
            _expand=lambda q: expand_query(q, complete_fn=fake_complete),
            _ensure_ready=lambda: True,
            _top_k=2,
        )

        self.assertEqual(seen["vector"], original)
        self.assertIn("asyncpg, postgres", seen["bm25"])
        self.assertIn(original, seen["bm25"])
        self.assertEqual(len(results), 2)

    def test_falls_back_to_original_when_model_returns_empty(self):
        seen = {}

        def bm25_search(keywords, k):
            seen["bm25"] = keywords
            return [_chunk("b.md", 0)]

        retrieve(
            "original question",
            _vector_search=lambda q, k: [_chunk("v.md", 0)],
            _bm25_search=bm25_search,
            _expand=lambda q: expand_query(q, complete_fn=lambda p, t: "  , , "),
            _ensure_ready=lambda: True,
        )
        self.assertEqual(seen["bm25"], "original question")

    def test_falls_back_to_original_when_model_errors(self):
        seen = {}

        def boom(prompt, temperature):
            raise TimeoutError("keyword model timed out")

        def bm25_search(keywords, k):
            seen["bm25"] = keywords
            return [_chunk("b.md", 0)]

        retrieve(
            "keep working",
            _vector_search=lambda q, k: [_chunk("v.md", 0)],
            _bm25_search=bm25_search,
            _expand=lambda q: expand_query(q, complete_fn=boom),
            _ensure_ready=lambda: True,
        )
        self.assertEqual(seen["bm25"], "keep working")

    def test_english_near_miss_expand_still_sends_original_to_bm25(self):
        seen = {}
        original = (
            "Привет, я новенький — как у нас ходить в Postgres через asyncpg "
            "и что с миграциями?"
        )

        def bm25_search(keywords, k):
            seen["bm25"] = keywords
            return [_chunk("migration_policy.md", 0)]

        retrieve(
            original,
            _vector_search=lambda q, k: [_chunk("asyncpg.md", 0)],
            _bm25_search=bm25_search,
            _expand=lambda q: "asyncpg, Postgres, migration, async",
            _ensure_ready=lambda: True,
        )
        self.assertIn("миграциями", seen["bm25"])
        self.assertIn("migration", seen["bm25"].lower())


if __name__ == "__main__":
    unittest.main()
