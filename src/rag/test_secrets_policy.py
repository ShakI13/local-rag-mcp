"""Credential docs are ingested redacted; MCP returns redacted bodies, not omissions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.secrets_policy import (
    is_credential_document,
    prepare_document_text,
    redact_secrets,
)


class TestSecretsPolicy(unittest.TestCase):
    def test_credential_filename_stems_are_flagged(self):
        self.assertTrue(is_credential_document("docs/internal/Доступы.md"))
        self.assertTrue(is_credential_document("docs/credentials.md"))
        self.assertTrue(is_credential_document("passwords.txt"))
        self.assertFalse(is_credential_document("docs/handbook/onboarding.md"))
        self.assertFalse(is_credential_document("migration_policy.md"))

    def test_redact_secrets_keeps_labels_hides_values(self):
        raw = "### Render.com\nemail: `user@example.com`\npassword: `s3cret-value`\n"
        redacted = redact_secrets(raw)
        self.assertIn("password:", redacted)
        self.assertIn("[REDACTED]", redacted)
        self.assertNotIn("s3cret-value", redacted)
        self.assertIn("user@example.com", redacted)

    def test_ingest_keeps_credential_docs_but_redacts_bodies(self):
        from rag.ingest import ingest_documents

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ok.md").write_text("safe content", encoding="utf-8")
            (root / "credentials.md").write_text(
                "service login\npassword: `s3cret-value`\n",
                encoding="utf-8",
            )

            with patch("rag.ingest.DOCUMENTS_DIR", str(root)):
                docs = ingest_documents()

            by_name = {Path(d["path"]).name: d["text"] for d in docs}
            self.assertIn("ok.md", by_name)
            self.assertIn("credentials.md", by_name)
            self.assertNotIn("s3cret-value", by_name["credentials.md"])
            self.assertIn("[REDACTED]", by_name["credentials.md"])

    def test_prepare_document_text_only_redacts_credential_paths(self):
        plain = prepare_document_text("notes.md", "password: `should-stay`")
        self.assertIn("should-stay", plain)
        scrubbed = prepare_document_text(
            "credentials.md", "password: `should-go`"
        )
        self.assertNotIn("should-go", scrubbed)
        self.assertIn("[REDACTED]", scrubbed)


if __name__ == "__main__":
    unittest.main()
