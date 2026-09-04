"""Query Expansion — turn a user question into Keywords for BM25 Search."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import EXPAND_TEMPERATURE, OLLAMA_MODEL, OLLAMA_URL

EXPAND_PROMPT = """Extract search keywords from the user question.
Return ONLY a comma-separated list of short keywords or phrases.
No greeting, no explanation, no complete sentences.

Question: {query}
Keywords:"""


def parse_keywords(raw) -> str | None:
    """Soft-parse model output into comma-separated Keywords, or None if unusable."""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    # Drop a leading label if the small model echoes the prompt
    lower = text.lower()
    for prefix in ("keywords:", "keyword:", "ключевые слова:"):
        if lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    parts = []
    for part in text.split(","):
        cleaned = part.strip().strip("-•*").strip()
        if cleaned:
            parts.append(cleaned)

    if not parts:
        return None
    return ", ".join(parts)


def _default_complete(prompt: str, temperature: float) -> str:
    import requests

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["response"]


def expand_query(query: str, *, complete_fn=None, temperature: float = None) -> str:
    """
    Expand a user question into Keywords for BM25 Search.

    On empty/unusable output or model errors/timeouts, fall back to the original question.
    """
    temp = EXPAND_TEMPERATURE if temperature is None else temperature
    complete = complete_fn or _default_complete
    prompt = EXPAND_PROMPT.format(query=query)

    try:
        raw = complete(prompt, temp)
        keywords = parse_keywords(raw)
        if keywords:
            return keywords
    except Exception:
        pass

    return query
