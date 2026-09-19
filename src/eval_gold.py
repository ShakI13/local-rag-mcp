#!/usr/bin/env python3
"""Batch gold-set eval: same query path as main.py, questions from the gold markdown."""

from __future__ import annotations

import argparse
import re
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from assistant import CompanyKBAssistant
from rag.query import REFUSE_ANSWER
from rag.secrets_policy import redact_secrets

REPO_ROOT = Path(__file__).resolve().parent.parent
# Default = Russian questions (corpus is mostly Russian). English set kept for opt-in.
GOLD_PATH_EN = REPO_ROOT / "docs" / "RAG-eval-questions.md"
GOLD_PATH_RU = REPO_ROOT / "docs" / "RAG-eval-questions-ru.md"
GOLD_PATH = GOLD_PATH_RU
LOG_PATH = REPO_ROOT / "tmp" / "rag-eval-run.md"

MINI_IDS = (1, 5, 9, 10, 15, 16, 19, 24, 25, 28)
SECRET_IDS = frozenset({28, 29, 60})
SNIPPET_LEN = 160

_QUESTION_RE = re.compile(
    r"^### ([QH])(\d+)\.[^\n]*\n+"
    r"\*\*Question:\*\*\s*(.+?)(?=\n\s*\n|\n\*\*|\Z)",
    re.MULTILINE | re.DOTALL,
)


@dataclass(frozen=True)
class GoldQuestion:
    qid: int
    question: str
    prefix: str = "Q"  # "Q" (chat gold) or "H" (hybrid-stress)


class _Tee:
    """Write the same console bytes to stdout and the eval log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_gold(path: Path = GOLD_PATH) -> list[GoldQuestion]:
    """Load `### QN` / `### HN` + `**Question:**` from gold markdown."""
    text = path.read_text(encoding="utf-8")
    items = []
    for match in _QUESTION_RE.finditer(text):
        prefix = match.group(1).upper()
        question = " ".join(match.group(3).split())
        items.append(
            GoldQuestion(qid=int(match.group(2)), question=question, prefix=prefix)
        )
    return items


def parse_qid(token: str) -> int:
    raw = token.strip().upper()
    if raw.startswith("Q") or raw.startswith("H"):
        raw = raw[1:]
    if not raw.isdigit():
        raise ValueError(f"Not a question id: {token!r} (use Q15, H1, or 15)")
    return int(raw)


def _git_rev(repo_root: Path = REPO_ROOT) -> str:
    """Short HEAD for eval headers; 'unknown' if git is unavailable."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def select_questions(
    items: list[GoldQuestion],
    *,
    question_ids: list[int] | None = None,
    all_questions: bool = False,
) -> list[GoldQuestion]:
    by_id = {item.qid: item for item in items}
    if question_ids:
        missing = [qid for qid in question_ids if qid not in by_id]
        if missing:
            pref = items[0].prefix if items else "Q"
            labels = ", ".join(f"{pref}{qid}" for qid in missing)
            raise ValueError(f"Unknown question(s) {labels} in {GOLD_PATH.name}")
        return [by_id[qid] for qid in question_ids]
    if all_questions:
        return list(items)
    missing_mini = [qid for qid in MINI_IDS if qid not in by_id]
    if missing_mini:
        labels = ", ".join(f"Q{qid}" for qid in missing_mini)
        raise ValueError(f"Mini set missing {labels} in {GOLD_PATH.name}")
    return [by_id[qid] for qid in MINI_IDS]


def display_answer(qid: int, answer: str) -> str:
    """Never print unredacted credential strings; Q28/Q29 are sources + snippet only."""
    redacted = redact_secrets(answer or "")
    if qid not in SECRET_IDS:
        return redacted
    snippet = redacted.replace("\n", " ").strip()
    if len(snippet) > SNIPPET_LEN:
        snippet = snippet[:SNIPPET_LEN].rstrip() + "…"
    return f"(redacted snippet) {snippet}"


def _print_result(qid: int, result: dict) -> None:
    """Post-query block from main.py: answer, sources, MCP, trailing separator."""
    answer = result.get("answer") or ""
    print(display_answer(qid, answer))

    sources = result.get("sources") or []
    show_sources = bool(sources) and (
        qid in SECRET_IDS or answer.strip() != REFUSE_ANSWER
    )
    if show_sources:
        print("\n📚 Sources:")
        for src in sources:
            print(f"  • {src}")

    if result.get("mcp_used"):
        print(f"\n🔧 Used MCP tool: {result.get('mcp_tool')}")

    print("─" * 60 + "\n")


def _ensure_utf8_stdio() -> None:
    """Windows consoles often default to cp1251 and choke on box-drawing / mixed Unicode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CompanyKBAssistant.query() on gold questions from "
            "docs/RAG-eval-questions-ru.md (default). Does not score."
        )
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help=(
            "Path to gold markdown (default: docs/RAG-eval-questions-ru.md). "
            "Use docs/RAG-eval-questions.md for the English questions."
        ),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=f"Eval log path (default: {LOG_PATH})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the log file instead of overwriting it",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--mini",
        action="store_true",
        help="Mini 10 (default): Q1, Q5, Q9, Q10, Q15, Q16, Q19, Q24, Q25, Q28",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="All questions in the gold file",
    )
    parser.add_argument(
        "questions",
        nargs="*",
        metavar="QN",
        help="Specific ids (Q15 or 15). Overrides --mini/--all.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdio()
    args = _parse_args(argv)
    gold_path = args.gold.resolve() if args.gold is not None else GOLD_PATH
    log_path = args.log.resolve() if args.log is not None else LOG_PATH
    try:
        items = parse_gold(gold_path)
        qids = [parse_qid(token) for token in args.questions] if args.questions else None
        selected = select_questions(
            items, question_ids=qids, all_questions=args.all
        )
    except (OSError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    log_fh = log_path.open(mode, encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_fh)
    assistant = None
    try:
        print("=" * 60)
        print("RAG gold eval" + (" (resume/append)" if args.append else ""))
        print("=" * 60)
        print(f"Commit: {_git_rev()}")
        print(f"Gold: {gold_path}")
        print(f"Log:  {log_path}")
        print(f"When: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}")
        print(
            f"Set:  {', '.join(f'{item.prefix}{item.qid}' for item in selected)}"
        )
        print("Score by hand: Pass / Retrieval fail / Grounding fail / Policy fail")
        print()

        assistant = CompanyKBAssistant()
        for item in selected:
            print("=" * 60)
            print(f"{item.prefix}{item.qid}")
            print(item.question)
            print("\n" + "─" * 60)
            print("🤖 Answer:\n")
            try:
                result = assistant.query(item.question, verbose=True)
                _print_result(item.qid, result)
            except Exception as exc:
                print(f"❌ Error: {exc}")
                traceback.print_exc()
                print("─" * 60 + "\n")
    except KeyboardInterrupt:
        print("\n\n👋 Stopped.")
    finally:
        if assistant is not None:
            assistant.close()
        sys.stdout = original_stdout
        log_fh.close()

    print(f"Log written to {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
