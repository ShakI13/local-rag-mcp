"""Exclude credential / access docs from searchable corpus and MCP reads."""
from pathlib import Path

# Filename stems (case-insensitive) that hold secrets — not ordinary Q&A material.
_CREDENTIAL_STEMS = frozenset(
    {
        "доступы",
        "credentials",
        "passwords",
        "secrets",
    }
)

_CREDENTIAL_READ_DENIED = (
    "Error: Access denied. Credential/access documents are not available via MCP."
)


def is_credential_document(path) -> bool:
    """True if path looks like a credentials/access doc (by filename stem)."""
    stem = Path(path).stem.lower()
    if stem in _CREDENTIAL_STEMS:
        return True
    # e.g. "render-credentials.md", "db-passwords.txt"
    return any(marker in stem for marker in ("credential", "password", "secret", "доступ"))


def credential_read_denial(path) -> str | None:
    """Return an MCP denial message for credential paths, else None."""
    if is_credential_document(path):
        return _CREDENTIAL_READ_DENIED
    return None
