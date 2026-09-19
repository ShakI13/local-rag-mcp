"""Human-readable pipeline logs for hybrid retrieve and answer path."""
from __future__ import annotations

from pathlib import Path


def preview(text: str, n: int = 100) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= n:
        return flat
    return flat[: n - 1] + "…"


def chunk_label(chunk: dict) -> str:
    source = chunk.get("source") or "?"
    name = Path(str(source)).name
    cid = chunk.get("chunk_id", "?")
    return f"{name}#{cid}"


def section(title: str) -> None:
    print(f"\n── {title} ──")


def line(msg: str = "") -> None:
    print(msg)
