"""Prompt seam: refuse OOD, use grounded context, distinguish mention vs description."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.query import (
    REFUSE_ANSWER,
    build_prompt,
    existence_listing_answer,
    filter_contexts_for_query,
    inventory_overview_answer,
    prepare_contexts,
)


class TestBuildPromptQuality(unittest.TestCase):
    def test_empty_contexts_refuse_not_general_knowledge(self):
        prompt = build_prompt("What is an unsupported out-of-corpus topic?", [])
        self.assertIn(REFUSE_ANSWER, prompt)
        self.assertNotIn("based on your general knowledge", prompt.lower())
        self.assertIn("do not use general knowledge", prompt.lower())

    def test_unrelated_contexts_treated_as_unsupported(self):
        contexts = [
            {
                "source": "docs/local-setup.md",
                "chunk_id": 0,
                "text": "How to run the app locally with compose.",
            }
        ]
        filtered = filter_contexts_for_query(
            "What is an unsupported out-of-corpus topic XYZABC?", contexts
        )
        self.assertEqual(filtered, [])
        prompt = build_prompt("What is an unsupported out-of-corpus topic XYZABC?", filtered)
        self.assertIn(REFUSE_ANSWER, prompt)

    def test_relevant_context_text_reaches_prompt_and_must_be_used(self):
        contexts = [
            {
                "source": "docs/db-client.md",
                "chunk_id": 0,
                "text": "Use the raw SQL client for bulk work; keep ordinary CRUD on the ORM.",
            }
        ]
        prompt = build_prompt(
            "how do we talk to the DB via the raw SQL client?",
            contexts,
        )
        self.assertIn("raw SQL client for bulk work", prompt)
        self.assertIn("do not refuse", prompt.lower())

    def test_existence_prompt_distinguishes_mention_from_description(self):
        contexts = [
            {
                "source": "docs/README.md",
                "chunk_id": 0,
                "text": "- Widget Owner\n- Other Role",
            }
        ]
        prompt = build_prompt(
            "Is there a Widget Owner description under /Squad?",
            contexts,
        )
        lower = prompt.lower()
        self.assertTrue(
            "mention" in lower or "listed" in lower or "dedicated" in lower
        )
        self.assertIn("folder", lower)


class TestFilterContexts(unittest.TestCase):
    def test_keeps_chunk_sharing_normalized_query_term(self):
        contexts = [
            {
                "source": "a.md",
                "chunk_id": 0,
                "text": "Политика миграций через Alembic.",
            },
            {
                "source": "b.md",
                "chunk_id": 0,
                "text": "Настройка Docker Compose.",
            },
        ]
        kept = filter_contexts_for_query(
            "что с миграциями в проекте?", contexts
        )
        sources = {c["source"] for c in kept}
        self.assertIn("a.md", sources)
        self.assertNotIn("b.md", sources)


class TestPrepareContextsDirectoryListing(unittest.TestCase):
    def test_existence_question_injects_listing_for_path_named_in_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            squad = root / "Squad"
            squad.mkdir()
            (squad / "Alpha.md").write_text("alpha role", encoding="utf-8")
            (root / "README.md").write_text("- Beta\n- Alpha\n", encoding="utf-8")

            with patch("rag.query.DOCUMENTS_DIR", str(root)):
                prepared = prepare_contexts(
                    "Is there a Beta description under /Squad?",
                    [
                        {
                            "source": str(root / "README.md"),
                            "chunk_id": 0,
                            "text": "- Beta\n- Alpha",
                        }
                    ],
                )

            self.assertTrue(prepared)
            self.assertTrue(prepared[0].get("is_directory_listing"))
            self.assertIn("Squad", prepared[0]["source"])
            self.assertIn("Alpha.md", prepared[0]["text"])
            self.assertNotIn("Beta.md", prepared[0]["text"])

    def test_content_question_does_not_inject_directory_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            squad = root / "Squad"
            squad.mkdir()
            (squad / "Alpha.md").write_text("alpha", encoding="utf-8")

            with patch("rag.query.DOCUMENTS_DIR", str(root)):
                prepared = prepare_contexts(
                    "What does Alpha do in /Squad?",
                    [
                        {
                            "source": str(squad / "Alpha.md"),
                            "chunk_id": 0,
                            "text": "Alpha owns delivery.",
                        }
                    ],
                )

            self.assertFalse(any(c.get("is_directory_listing") for c in prepared))


class TestExistenceListingAnswer(unittest.TestCase):
    def test_uses_listing_not_refuse_when_asked_file_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            squad = root / "Squad"
            squad.mkdir()
            (squad / "Alpha.md").write_text("alpha", encoding="utf-8")
            (squad / "Gamma.md").write_text("gamma", encoding="utf-8")

            with patch("rag.query.DOCUMENTS_DIR", str(root)):
                contexts = prepare_contexts(
                    "Is there a Beta / Delta description under /Squad?",
                    [
                        {
                            "source": str(root / "README.md"),
                            "chunk_id": 0,
                            "text": "- Beta\n- Delta",
                        }
                    ],
                )
                answer = existence_listing_answer(
                    "Is there a Beta / Delta description under /Squad?",
                    contexts,
                )

            self.assertIsNotNone(answer)
            lower = answer.lower()
            self.assertNotEqual(answer.strip(), REFUSE_ANSWER)
            self.assertIn("mention", lower)
            self.assertIn("no dedicated", lower)
            self.assertIn("alpha.md", lower)
            self.assertIn("gamma.md", lower)
            self.assertIn("/squad", lower)

    def test_does_not_hijack_content_questions(self):
        contexts = [
            {
                "source": "docs/Squad/Alpha.md",
                "chunk_id": 0,
                "text": "Alpha owns delivery.",
            }
        ]
        self.assertIsNone(
            existence_listing_answer("What does Alpha do under /Squad?", contexts)
        )


class TestInventoryOverviewAnswer(unittest.TestCase):
    def test_general_coverage_question_lists_areas_not_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Backend").mkdir()
            (root / "Backend" / "asyncpg.md").write_text("db", encoding="utf-8")
            (root / "Frontend").mkdir()
            (root / "Frontend" / "ui.md").write_text("ui", encoding="utf-8")
            (root / "readme.md").write_text("root", encoding="utf-8")

            with patch("rag.query.DOCUMENTS_DIR", str(root)):
                answer, sources = inventory_overview_answer(
                    "About what you have info?"
                )

            self.assertIsNotNone(answer)
            lower = answer.lower()
            self.assertNotEqual(answer.strip(), REFUSE_ANSWER)
            self.assertIn("backend", lower)
            self.assertIn("frontend", lower)
            self.assertIn("readme.md", lower)
            self.assertTrue(sources)
            self.assertEqual(sources[0]["source"], str(root))

    def test_does_not_hijack_topical_questions(self):
        self.assertIsNone(
            inventory_overview_answer("What information do we have about asyncpg?")
        )
        self.assertIsNone(
            inventory_overview_answer("What do you know about migrations?")
        )

    def test_help_questions_explain_usage_and_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Backend").mkdir()
            (root / "Backend" / "asyncpg.md").write_text("db", encoding="utf-8")

            with patch("rag.query.DOCUMENTS_DIR", str(root)):
                for q in ("How to use you?", "What can I do with you?"):
                    result = inventory_overview_answer(q)
                    self.assertIsNotNone(result, q)
                    answer, _sources = result
                    lower = answer.lower()
                    self.assertNotEqual(answer.strip(), REFUSE_ANSWER, q)
                    self.assertIn("knowledge-base", lower, q)
                    self.assertIn("backend", lower, q)


if __name__ == "__main__":
    unittest.main()
