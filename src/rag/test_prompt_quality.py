"""Prompt seam: refuse OOD, use grounded context, distinguish mention vs description."""
import unittest

from rag.query import (
    REFUSE_ANSWER,
    build_prompt,
    filter_contexts_for_query,
    prepare_contexts,
)


class TestBuildPromptQuality(unittest.TestCase):
    def test_empty_contexts_refuse_not_general_knowledge(self):
        prompt = build_prompt("What is Kubernetes production HA setup?", [])
        self.assertIn(REFUSE_ANSWER, prompt)
        # Must not invite the model to answer from memory
        self.assertNotIn("based on your general knowledge", prompt.lower())
        self.assertIn("do not use general knowledge", prompt.lower())

    def test_unrelated_contexts_treated_as_unsupported(self):
        contexts = [
            {
                "source": "docs/docker.md",
                "chunk_id": 0,
                "text": "How to run docker compose locally for the app.",
            }
        ]
        filtered = filter_contexts_for_query(
            "What is Kubernetes production HA setup?", contexts
        )
        self.assertEqual(filtered, [])
        prompt = build_prompt("What is Kubernetes production HA setup?", filtered)
        self.assertIn(REFUSE_ANSWER, prompt)

    def test_relevant_context_text_reaches_prompt_and_must_be_used(self):
        contexts = [
            {
                "source": "docs/Использование asyncpg.md",
                "chunk_id": 0,
                "text": "Use asyncpg for bulk SQL; keep ordinary CRUD on the ORM.",
            }
        ]
        prompt = build_prompt(
            "как у нас ходить в Postgres через asyncpg?",
            contexts,
        )
        self.assertIn("Use asyncpg for bulk SQL", prompt)
        self.assertIn("asyncpg", prompt.lower())
        # Must not prefer refuse when facts are present
        self.assertIn("do not refuse", prompt.lower())

    def test_role_prompt_distinguishes_mention_from_description(self):
        contexts = [
            {
                "source": "docs/README.md",
                "chunk_id": 0,
                "text": "- Team Lead / Scrum Master\n- Product Owner",
            }
        ]
        prompt = build_prompt(
            "Is there a Team Lead / Scrum Master role description under /Роли?",
            contexts,
        )
        lower = prompt.lower()
        self.assertTrue(
            "mention" in lower or "listed" in lower or "dedicated" in lower
        )
        self.assertTrue("роли" in lower or "role" in lower)


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


class TestPrepareContextsRoles(unittest.TestCase):
    def test_role_question_prepends_roles_directory_listing(self):
        contexts = [
            {
                "source": "docs/README.md",
                "chunk_id": 0,
                "text": "- Team Lead / Scrum Master\n- Product Owner",
            }
        ]
        prepared = prepare_contexts(
            "Is there a Team Lead / Scrum Master role description under /Роли?",
            contexts,
        )
        self.assertTrue(prepared)
        self.assertIn("Роли", prepared[0]["source"])
        self.assertIn("Tech Lead", prepared[0]["text"])
        self.assertNotIn("Team Lead.md", prepared[0]["text"])
        self.assertTrue(any("README" in c.get("source", "") for c in prepared))

    def test_role_inventory_answer_is_nuanced_not_bare_yes(self):
        from rag.query import role_inventory_answer

        contexts = prepare_contexts(
            "Is there a Team Lead / Scrum Master role description under /Роли?",
            [
                {
                    "source": "docs/README.md",
                    "chunk_id": 0,
                    "text": "- Team Lead / Scrum Master",
                }
            ],
        )
        answer = role_inventory_answer(
            "Is there a Team Lead / Scrum Master role description under /Роли?",
            contexts,
        )
        self.assertIsNotNone(answer)
        lower = answer.lower()
        self.assertIn("no dedicated", lower)
        self.assertIn("tech lead", lower)
        self.assertFalse(lower.strip().startswith("yes"))


if __name__ == "__main__":
    unittest.main()
