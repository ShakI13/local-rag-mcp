"""Credential docs are excluded from ingest and blocked for MCP reads."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.secrets_policy import (
    credential_read_denial,
    is_credential_document,
)


class TestSecretsPolicy(unittest.TestCase):
    def test_dostupy_and_english_credential_names_are_flagged(self):
        self.assertTrue(is_credential_document("docs/Документация/Доступы.md"))
        self.assertTrue(is_credential_document("docs/credentials.md"))
        self.assertTrue(is_credential_document("passwords.txt"))
        self.assertFalse(is_credential_document("docs/Роли/Tech Lead.md"))
        self.assertFalse(is_credential_document("migration_policy.md"))

    def test_ingest_skips_credential_documents(self):
        from rag.ingest import ingest_documents

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.md").write_text("safe content", encoding="utf-8")
            (root / "Доступы.md").write_text("email: a\npassword: b\n", encoding="utf-8")

            with patch("rag.ingest.DOCUMENTS_DIR", str(root)):
                docs = ingest_documents()

            paths = {Path(d["path"]).name for d in docs}
            self.assertIn("ok.md", paths)
            self.assertNotIn("Доступы.md", paths)

    def test_credential_paths_get_mcp_denial_without_file_body(self):
        denial = credential_read_denial("docs/Документация/Доступы.md")
        self.assertIsNotNone(denial)
        self.assertIn("Access denied", denial)
        self.assertIsNone(credential_read_denial("docs/Роли/Tech Lead.md"))


if __name__ == "__main__":
    unittest.main()
