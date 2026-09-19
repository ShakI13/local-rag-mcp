"""Parser and selection for the gold markdown (no live Ollama)."""
import tempfile
import unittest
from pathlib import Path

from eval_gold import (
    GOLD_PATH,
    GOLD_PATH_EN,
    GOLD_PATH_RU,
    MINI_IDS,
    GoldQuestion,
    display_answer,
    parse_gold,
    parse_qid,
    select_questions,
)


class TestParseGold(unittest.TestCase):
    def test_gold_file_has_sixty_questions(self):
        items = parse_gold(GOLD_PATH)
        self.assertEqual([item.qid for item in items], list(range(1, 61)))
        self.assertTrue(all(item.question for item in items))

    def test_default_gold_is_russian(self):
        self.assertEqual(GOLD_PATH, GOLD_PATH_RU)
        items = parse_gold(GOLD_PATH)
        by_id = {item.qid: item.question for item in items}
        self.assertEqual(
            by_id[1],
            "Сколько длится проект Ecto-1 и сколько человек в команде?",
        )

    def test_reads_english_question_field_not_heading(self):
        items = parse_gold(GOLD_PATH_EN)
        by_id = {item.qid: item.question for item in items}
        self.assertEqual(
            by_id[1],
            "How long does the Ecto-1 project last, and how many people are on a team?",
        )
        self.assertIn("asyncpg", by_id[15])
        self.assertIn("миграци", by_id[15])
        self.assertEqual(by_id[28], "What is the Render.com password?")

    def test_russian_gold_file_has_sixty_translated_questions(self):
        items = parse_gold(GOLD_PATH_RU)
        self.assertEqual([item.qid for item in items], list(range(1, 61)))
        by_id = {item.qid: item.question for item in items}
        self.assertEqual(
            by_id[1],
            "Сколько длится проект Ecto-1 и сколько человек в команде?",
        )
        self.assertIn("Render.com", by_id[28])
        self.assertNotIn("What is", by_id[28])
        # Already-Russian originals preserved
        self.assertIn("asyncpg", by_id[15])
        self.assertIn("миграци", by_id[15])

    def test_parse_sample_markdown(self):
        sample = (
            "# ignore\n\n"
            "### Q2. Title\n\n"
            "**Question:** Second question?\n\n"
            "**Must include:** nope\n\n"
            "### Q9. Other\n\n"
            "**Question:** Ninth one.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.md"
            path.write_text(sample, encoding="utf-8")
            items = parse_gold(path)
        self.assertEqual(
            items,
            [
                GoldQuestion(qid=2, question="Second question?", prefix="Q"),
                GoldQuestion(qid=9, question="Ninth one.", prefix="Q"),
            ],
        )

    def test_parse_hybrid_stress_h_prefix(self):
        sample = (
            "### H1. Rare\n\n"
            "**Question:** how asyncpg?\n\n"
            "**Must retrieve:** `asyncpg`\n\n"
            "### H2. Other\n\n"
            "**Question:** ruff length?\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stress.md"
            path.write_text(sample, encoding="utf-8")
            items = parse_gold(path)
        self.assertEqual(
            items,
            [
                GoldQuestion(qid=1, question="how asyncpg?", prefix="H"),
                GoldQuestion(qid=2, question="ruff length?", prefix="H"),
            ],
        )


class TestSelectQuestions(unittest.TestCase):
    def setUp(self):
        self.items = [GoldQuestion(qid=n, question=f"q{n}") for n in range(1, 31)]

    def test_default_is_mini_ten(self):
        selected = select_questions(self.items)
        self.assertEqual([item.qid for item in selected], list(MINI_IDS))

    def test_all_keeps_file_order(self):
        selected = select_questions(self.items, all_questions=True)
        self.assertEqual([item.qid for item in selected], list(range(1, 31)))

    def test_explicit_ids(self):
        selected = select_questions(self.items, question_ids=[15, 28])
        self.assertEqual([item.qid for item in selected], [15, 28])

    def test_unknown_id_raises(self):
        with self.assertRaises(ValueError):
            select_questions(self.items, question_ids=[99])

    def test_parse_qid_accepts_q_prefix(self):
        self.assertEqual(parse_qid("Q15"), 15)
        self.assertEqual(parse_qid("q28"), 28)
        self.assertEqual(parse_qid("H1"), 1)
        self.assertEqual(parse_qid("5"), 5)


class TestDisplayAnswer(unittest.TestCase):
    def test_secret_questions_are_snippeted_and_redacted(self):
        raw = "password: `live-secret-value`\n" + ("x" * 200)
        shown = display_answer(28, raw)
        self.assertNotIn("live-secret-value", shown)
        self.assertTrue(shown.startswith("(redacted snippet)"))
        self.assertIn("[REDACTED]", shown)
        self.assertLess(len(shown), len(raw) + 40)

    def test_non_secret_answer_is_redacted_but_full(self):
        raw = "status is error"
        self.assertEqual(display_answer(1, raw), raw)


if __name__ == "__main__":
    unittest.main()
