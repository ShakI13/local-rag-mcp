"""Redact secrets in credential/access docs (ingest + MCP), do not drop the files."""
import re
from pathlib import Path

# Filename stems (case-insensitive) that typically hold credentials.
_CREDENTIAL_STEMS = frozenset(
    {
        "доступы",
        "credentials",
        "passwords",
        "secrets",
    }
)

_REDACTED = "[REDACTED]"

# password: value  /  password: `value`  (and similar labels)
_SECRET_LINE_RE = re.compile(
    r"(?i)(\b(?:password|passwd|secret|token|api[_-]?key)\b\s*[:=]\s*)(`?)([^\s`\n]+)\2"
)


def is_credential_document(path) -> bool:
    """True if path looks like a credentials/access doc (by filename stem)."""
    stem = Path(path).stem.lower()
    if stem in _CREDENTIAL_STEMS:
        return True
    return any(marker in stem for marker in ("credential", "password", "secret", "доступ"))


def redact_secrets(text: str) -> str:
    """Replace secret values after common labels; leave structure/labels intact."""
    if not text:
        return text

    def _sub(match: re.Match) -> str:
        prefix, tick, _value = match.group(1), match.group(2), match.group(3)
        if tick:
            return f"{prefix}{tick}{_REDACTED}{tick}"
        return f"{prefix}{_REDACTED}"

    return _SECRET_LINE_RE.sub(_sub, text)


def prepare_document_text(path, text: str) -> str:
    """Return text safe for the searchable corpus / MCP reads."""
    if is_credential_document(path):
        return redact_secrets(text)
    return text
