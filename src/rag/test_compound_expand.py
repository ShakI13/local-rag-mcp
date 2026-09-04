"""Query expansion should cover every distinct topic in a compound question."""
import unittest

from rag.expand import EXPAND_PROMPT, expand_query, parse_keywords


class TestCompoundExpansion(unittest.TestCase):
    def test_expand_prompt_asks_for_every_topic(self):
        lower = EXPAND_PROMPT.lower()
        self.assertTrue("every" in lower or "all" in lower or "each" in lower)
        self.assertIn("{query}", EXPAND_PROMPT)

    def test_parse_keeps_both_halves_of_compound_keywords(self):
        parsed = parse_keywords("asyncpg, postgres, migrations, alembic")
        self.assertIsNotNone(parsed)
        lower = parsed.lower()
        self.assertIn("asyncpg", lower)
        self.assertIn("migration", lower)

    def test_expand_fallback_still_returns_original_on_empty(self):
        q = "asyncpg и миграции"
        self.assertEqual(
            expand_query(q, complete_fn=lambda p, t: "  , , "),
            q,
        )


if __name__ == "__main__":
    unittest.main()
