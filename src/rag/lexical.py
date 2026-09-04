"""Lexical Normalization for BM25 (Russian morphology via pymorphy3)."""
import re

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

_morph = None


def _get_morph():
    global _morph
    if _morph is None:
        import pymorphy3
        _morph = pymorphy3.MorphAnalyzer()
    return _morph


def normalize_token(token: str) -> str:
    """
    Normalize one token for BM25.

    Cyrillic tokens go through pymorphy3 normal form; non-Cyrillic or morph
    failures stay as the raw lowercased token.
    """
    raw = token.lower()
    if not raw:
        return raw
    if not _CYRILLIC_RE.search(raw):
        return raw
    try:
        parsed = _get_morph().parse(raw)
        if not parsed:
            return raw
        return parsed[0].normal_form
    except Exception:
        return raw


def tokenize_normalized(text: str):
    """Tokenize text and apply Lexical Normalization to each token."""
    if not text:
        return []
    return [normalize_token(tok) for tok in _WORD_RE.findall(text.lower())]
