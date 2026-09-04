"""Assistant MCP result unwrapping — bad tool payloads must not poison RAG answers."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from assistant import CompanyKBAssistant, mcp_tool_text
from rag.query import REFUSE_ANSWER



class TestMcpToolText(unittest.TestCase):
    def test_unwraps_fastmcp_structured_content(self):
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "_meta": {"fastmcp": {"wrap_result": True}},
                "content": [
                    {"text": "Error: File not found: docs/asyncpg.md", "type": "text"}
                ],
                "isError": False,
                "structuredContent": {
                    "result": "Error: File not found: docs/asyncpg.md"
                },
            },
        }
        text = mcp_tool_text(payload)
        self.assertEqual(text, "Error: File not found: docs/asyncpg.md")
        self.assertIsInstance(text, str)
        # Regression: old code used result.get('result') → dict with len==4
        self.assertNotEqual(len(text), 4)

    def test_unwraps_successful_text_content(self):
        payload = {
            "result": {
                "content": [{"type": "text", "text": "Use asyncpg for raw SQL."}],
                "structuredContent": {"result": "Use asyncpg for raw SQL."},
            }
        }
        self.assertEqual(mcp_tool_text(payload), "Use asyncpg for raw SQL.")


class TestAssistantDropsFailedMcpWhenContextsExist(unittest.TestCase):
    def test_bad_read_document_does_not_refuse_when_chunks_answer(self):
        contexts = [
            {
                "source": "docs/db/asyncpg.md",
                "chunk_id": 0,
                "text": "Use asyncpg for raw SQL against Postgres.",
            },
            {
                "source": "docs/db/migrations.md",
                "chunk_id": 0,
                "text": "Migrations run via Alembic.",
            },
        ]
        asst = CompanyKBAssistant.__new__(CompanyKBAssistant)
        asst.mcp = MagicMock()
        asst.llm_client = MagicMock()

        bad_payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [
                    {"text": "Error: File not found: docs/asyncpg.md", "type": "text"}
                ],
                "structuredContent": {
                    "result": "Error: File not found: docs/asyncpg.md"
                },
            },
        }
        asst.mcp.call_tool.return_value = bad_payload

        with patch("assistant.retrieve", return_value=contexts), patch(
            "assistant.prepare_contexts", return_value=contexts
        ), patch(
            "assistant.inventory_overview_answer", return_value=None
        ), patch(
            "assistant.existence_listing_answer", return_value=None
        ), patch.object(
            asst,
            "_llm_decide_mcp_usage",
            return_value=("read_document", {"file_path": "docs/asyncpg.md"}),
        ), patch(
            "assistant.ask_llm",
            return_value="Use asyncpg for raw SQL; migrations via Alembic.",
        ) as ask_mock:
            result = asst.query(
                "как у нас ходить в Postgres через asyncpg и что с миграциями?",
                verbose=False,
            )

        self.assertNotEqual(result["answer"].strip(), REFUSE_ANSWER)
        self.assertIn("asyncpg", result["answer"].lower())
        self.assertFalse(result["mcp_used"])
        # Prompt must not include the failed MCP payload / dict dump
        prompt = ask_mock.call_args[0][0]
        self.assertNotIn("additional_info_from_mcp_tool", prompt)
        self.assertNotIn("File not found", prompt)
        self.assertNotIn("structuredContent", prompt)


class TestMcpDecisionPromptHintsSources(unittest.TestCase):
    def test_prompt_lists_sources_and_forbids_invented_paths(self):
        asst = CompanyKBAssistant.__new__(CompanyKBAssistant)
        contexts = [
            {
                "source": r"docs\Документация\Бэкенд\База данных\Использование asyncpg.md",
                "chunk_id": 0,
                "text": "Use asyncpg for raw SQL.",
            }
        ]
        prompt = asst._mcp_decision_prompt("как через asyncpg?", contexts)
        self.assertIn("Available source paths", prompt)
        self.assertIn(contexts[0]["source"], prompt)
        self.assertIn("MUST be copied EXACTLY", prompt)
        self.assertIn("do not invent docs/asyncpg.md", prompt)
        # Example path is taken from retrieved sources, not a fake shortcut
        self.assertIn(
            f'"file_path": "{contexts[0]["source"]}"',
            prompt,
        )

    def test_invented_read_document_path_is_rejected(self):
        asst = CompanyKBAssistant.__new__(CompanyKBAssistant)
        asst.mcp = MagicMock()
        asst.llm_client = MagicMock()
        asst.llm_client.chat.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "use_mcp": True,
                        "tool": "read_document",
                        "args": {"file_path": "docs/asyncpg.md"},
                    }
                )
            }
        }
        contexts = [
            {
                "source": r"docs\db\Использование asyncpg.md",
                "chunk_id": 0,
                "text": "asyncpg",
            }
        ]
        tool, args = asst._llm_decide_mcp_usage("asyncpg?", contexts)
        self.assertIsNone(tool)
        self.assertIsNone(args)

    def test_read_document_path_from_sources_is_allowed(self):
        asst = CompanyKBAssistant.__new__(CompanyKBAssistant)
        asst.mcp = MagicMock()
        asst.llm_client = MagicMock()
        real = r"docs\db\Использование asyncpg.md"
        asst.llm_client.chat.return_value = {
            "message": {
                "content": json.dumps(
                    {
                        "use_mcp": True,
                        "tool": "read_document",
                        "args": {"file_path": real},
                    }
                )
            }
        }
        contexts = [{"source": real, "chunk_id": 0, "text": "asyncpg"}]
        tool, args = asst._llm_decide_mcp_usage("asyncpg?", contexts)
        self.assertEqual(tool, "read_document")
        self.assertEqual(args.get("file_path"), real)


class TestListDocumentsDoesNotRefuse(unittest.TestCase):
    def test_list_documents_with_empty_rag_returns_overview(self):
        asst = CompanyKBAssistant.__new__(CompanyKBAssistant)
        asst.mcp = MagicMock()
        asst.llm_client = MagicMock()
        asst.mcp.call_tool.return_value = {
            "result": {
                "content": [{"type": "text", "text": "- Backend/asyncpg.md"}],
            }
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Backend").mkdir()
            (root / "Backend" / "asyncpg.md").write_text("db", encoding="utf-8")

            with patch("assistant.retrieve", return_value=[]), patch(
                "assistant.prepare_contexts", return_value=[]
            ), patch(
                "assistant.inventory_overview_answer", return_value=None
            ), patch(
                "assistant.existence_listing_answer", return_value=None
            ), patch.object(
                asst,
                "_llm_decide_mcp_usage",
                return_value=("list_documents", {}),
            ), patch(
                "rag.query.DOCUMENTS_DIR", str(root)
            ), patch(
                "assistant.ask_llm"
            ) as ask_mock:
                result = asst.query("random meta that skipped detector", verbose=False)

        ask_mock.assert_not_called()
        self.assertNotEqual(result["answer"].strip(), REFUSE_ANSWER)
        self.assertIn("knowledge-base", result["answer"].lower())
        self.assertIn("backend", result["answer"].lower())
        self.assertTrue(result["mcp_used"])
        self.assertEqual(result["mcp_tool"], "list_documents")


if __name__ == "__main__":
    unittest.main()

