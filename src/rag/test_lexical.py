"""Lexical Normalization for Russian BM25."""
import unittest

from rag.bm25_index import Bm25Index
from rag.lexical import tokenize_normalized


class TestLexicalNormalize(unittest.TestCase):
    def test_russian_inflected_forms_share_a_lemma(self):
        lemma = tokenize_normalized("миграция")[0]
        a = tokenize_normalized("миграции базы данных")
        b = tokenize_normalized("миграций базы данных")
        self.assertIn(lemma, a)
        self.assertIn(lemma, b)

    def test_latin_and_api_tokens_stay_lowercased_raw(self):
        tokens = tokenize_normalized("asyncpg API RateLimiting")
        self.assertEqual(tokens, ["asyncpg", "api", "ratelimiting"])

    def test_mixed_russian_and_latin_keeps_identifiers(self):
        tokens = tokenize_normalized("библиотека asyncpg для Postgres")
        self.assertIn("asyncpg", tokens)
        self.assertIn("postgres", tokens)


class TestBm25WithNormalization(unittest.TestCase):
    def test_inflected_query_matches_inflected_chunk(self):
        chunks = [
            {
                "text": "Документ описывает процесс миграций базы данных на проде.",
                "source": "migrations.md",
                "chunk_id": 0,
            },
            {
                "text": "Frontend uses CSS modules and styled components.",
                "source": "frontend.md",
                "chunk_id": 0,
            },
        ]
        index = Bm25Index(chunks)
        hits = index.search("миграция базы", k=2)
        self.assertEqual(hits[0]["source"], "migrations.md")

    def test_latin_identifier_still_matches_exactly(self):
        chunks = [
            {"text": "Wire the asyncpg pool in the worker.", "source": "db.md", "chunk_id": 0},
            {"text": "Общие заметки про базы данных.", "source": "notes.md", "chunk_id": 0},
        ]
        index = Bm25Index(chunks)
        hits = index.search("asyncpg", k=1)
        self.assertEqual(hits[0]["source"], "db.md")


if __name__ == "__main__":
    unittest.main()
