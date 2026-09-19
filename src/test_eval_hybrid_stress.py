"""Parser tests for hybrid-stress gold (no live retrieve / Ollama)."""
import unittest
from pathlib import Path

from eval_hybrid_stress import GOLD_PATH, hits_must, parse_stress


class TestHybridStressGold(unittest.TestCase):
    def test_parses_twelve_items(self):
        items = parse_stress(GOLD_PATH)
        self.assertEqual([i.hid for i in items], [f"H{n}" for n in range(1, 13)])
        self.assertTrue(all(i.question and i.must_retrieve for i in items))

    def test_h1_mentions_asyncpg(self):
        by = {i.hid: i for i in parse_stress(GOLD_PATH)}
        self.assertIn("asyncpg", by["H1"].question)
        self.assertEqual(by["H1"].must_retrieve, "asyncpg")

    def test_h8_needle_is_specific_architecture_path(self):
        by = {i.hid: i for i in parse_stress(GOLD_PATH)}
        self.assertIn("Архитектура.md", by["H8"].must_retrieve)
        self.assertIn("01-Обзор проекта", by["H8"].must_retrieve)

    def test_hits_must_is_substring(self):
        self.assertTrue(
            hits_must(
                [r"docs\Документация\Бэкенд\База данных\Использование asyncpg.md"],
                "asyncpg",
            )
        )
        self.assertFalse(hits_must([r"docs\README.md"], "asyncpg"))


if __name__ == "__main__":
    unittest.main()
