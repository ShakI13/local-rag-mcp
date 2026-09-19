#!/usr/bin/env python3
"""Compare vector-only vs hybrid retrieve on the hybrid-stress gold set.

Scores retrieval only (gold path substrings in chunk sources). No answer LLM.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import TOP_K
from rag import query as rag_query
from rag.query import retrieve

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = REPO_ROOT / "docs" / "RAG-eval-hybrid-stress.md"
LOG_PATH = REPO_ROOT / "tmp" / "rag-eval-hybrid-stress.md"

_QUESTION_RE = re.compile(
    r"^### (H\d+)\.[^\n]*\n+"
    r"\*\*Question:\*\*\s*(.+?)(?=\n\s*\n|\n\*\*|\Z)",
    re.MULTILINE | re.DOTALL,
)
_MUST_RETRIEVE_RE = re.compile(
    r"^### (H\d+)\.[^\n]*\n+"
    r".*?\*\*Must retrieve:\*\*\s*`([^`]+)`",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class StressQuestion:
    hid: str
    question: str
    must_retrieve: str


def parse_stress(path: Path = GOLD_PATH) -> list[StressQuestion]:
    text = path.read_text(encoding="utf-8")
    questions = {
        m.group(1): " ".join(m.group(2).split())
        for m in _QUESTION_RE.finditer(text)
    }
    musts = {m.group(1): m.group(2).strip() for m in _MUST_RETRIEVE_RE.finditer(text)}
    missing_q = sorted(set(musts) - set(questions), key=lambda h: int(h[1:]))
    missing_m = sorted(set(questions) - set(musts), key=lambda h: int(h[1:]))
    if missing_q or missing_m:
        raise ValueError(
            f"Gold parse mismatch: missing questions {missing_q}, "
            f"missing must-retrieve {missing_m}"
        )
    return [
        StressQuestion(
            hid=hid,
            question=questions[hid],
            must_retrieve=musts[hid],
        )
        for hid in sorted(questions, key=lambda h: int(h[1:]))
    ]


def sources_of(chunks) -> list[str]:
    out = []
    seen = set()
    for c in chunks or []:
        src = str(c.get("source") or "")
        if src and src not in seen:
            seen.add(src)
            out.append(src)
    return out


def hits_must(sources: list[str], needle: str) -> bool:
    n = needle.casefold()
    return any(n in s.casefold() for s in sources)


def retrieve_vector_only(question: str, *, top_k: int | None = None):
    """Same shape as pre-hybrid retrieve: FAISS Top-K on the raw question."""
    if not rag_query._default_ensure_ready():
        return []
    k = TOP_K if top_k is None else top_k
    return rag_query._default_vector_search(question, k)


def retrieve_hybrid(question: str, *, expand_identity: bool):
    if expand_identity:
        return retrieve(question, _expand=lambda q: q)
    return retrieve(question)


def _ensure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Vector-only vs hybrid retrieve on docs/RAG-eval-hybrid-stress.md"
    )
    p.add_argument("--gold", type=Path, default=GOLD_PATH)
    p.add_argument("--log", type=Path, default=LOG_PATH)
    p.add_argument(
        "--expand-identity",
        action="store_true",
        help="Skip expand LLM; BM25 gets the raw question (offline-friendly)",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"Override TOP_K for both modes (config default {TOP_K})",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    args = _parse_args(argv)
    gold_path = args.gold.resolve()
    log_path = args.log.resolve()
    try:
        items = parse_stress(gold_path)
    except (OSError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    top_k = args.top_k
    log_path.parent.mkdir(parents=True, exist_ok=True)

    hybrid_only = 0
    vector_only = 0
    both = 0
    neither = 0

    lines: list[str] = []
    def emit(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    emit("=" * 60)
    emit("Hybrid stress — retrieval compare")
    emit(f"Gold: {gold_path}")
    emit(f"When: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
    emit(f"TOP_K: {top_k if top_k is not None else TOP_K}")
    emit(
        "Expand: "
        + ("identity (no LLM)" if args.expand_identity else "expand_query (Ollama)")
    )
    emit("")
    emit(
        f"{'ID':<4} {'V':^6} {'H':^6} {'Verdict':<28} Must retrieve"
    )
    emit("-" * 72)

    for item in items:
        v_chunks = retrieve_vector_only(item.question, top_k=top_k)
        if top_k is not None:
            h_chunks = retrieve(
                item.question,
                _expand=(lambda q: q) if args.expand_identity else None,
                _top_k=top_k,
            )
        else:
            h_chunks = retrieve_hybrid(
                item.question, expand_identity=args.expand_identity
            )

        v_src = sources_of(v_chunks)
        h_src = sources_of(h_chunks)
        v_hit = hits_must(v_src, item.must_retrieve)
        h_hit = hits_must(h_src, item.must_retrieve)

        if h_hit and not v_hit:
            verdict = "Hybrid win"
            hybrid_only += 1
        elif v_hit and not h_hit:
            verdict = "Vector win (hybrid miss)"
            vector_only += 1
        elif v_hit and h_hit:
            verdict = "Both hit"
            both += 1
        else:
            verdict = "Both miss"
            neither += 1

        emit(
            f"{item.hid:<4} "
            f"{'hit' if v_hit else 'miss':^6} "
            f"{'hit' if h_hit else 'miss':^6} "
            f"{verdict:<28} `{item.must_retrieve}`"
        )
        emit(f"     Q: {item.question}")
        emit(f"     vector sources: {v_src[:5]}")
        emit(f"     hybrid sources: {h_src[:5]}")
        emit("")

    emit("=" * 60)
    emit("Summary")
    emit(f"  Hybrid win (V miss, H hit): {hybrid_only}")
    emit(f"  Both hit:                   {both}")
    emit(f"  Both miss:                  {neither}")
    emit(f"  Vector win (H miss):        {vector_only}")
    emit(f"  Total:                      {len(items)}")
    emit("=" * 60)

    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
