"""Live structural checks for rag-eval-quality tickets (no secret values printed)."""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.expand import expand_query
from rag.query import (
    REFUSE_ANSWER,
    ask,
    filter_contexts_for_query,
    prepare_contexts,
    retrieve,
)
from rag.secrets_policy import is_credential_document, prepare_document_text

SRC = Path(__file__).resolve().parent.parent
CHUNKS_PATH = SRC / "chunks.pkl"

ASYNC_Q = (
    "Привет, я новенький — как у нас ходить в Postgres через asyncpg и что с миграциями?"
)
K8S_Q = "What is Kubernetes production HA setup?"
ROLE_Q = "Is there a Team Lead / Scrum Master role description under /Роли?"
SECRET_Q = "What is the Render.com password?"

_UNREDACTED_PASSWORD = re.compile(
    r"(?i)password\s*[:=]\s*`(?!\[REDACTED\])[^`\n]+`"
)


def _sources(contexts):
    return [c["source"] for c in contexts]


def check_index_redacts_credentials():
    chunks = pickle.load(CHUNKS_PATH.open("rb"))
    cred = [c for c in chunks if is_credential_document(c["source"])]
    assert cred, "credential docs should remain in the index (with redaction)"
    for c in cred:
        assert not _UNREDACTED_PASSWORD.search(c["text"]), c["source"]
        assert "[REDACTED]" in c["text"], c["source"]
    print("OK 02a: credential docs indexed with redacted secret values")


def check_secret_question_retrieve():
    answer, contexts = ask(SECRET_Q)
    for c in contexts:
        assert not _UNREDACTED_PASSWORD.search(c.get("text", "")), c["source"]
    # MCP-style prepare must also redact
    sample = prepare_document_text(
        "docs/credentials.md", "password: `live-check-secret`\n"
    )
    assert "live-check-secret" not in sample
    assert "[REDACTED]" in sample
    print("OK 02b: secret question path does not surface raw password values")
    print(f"   sources={_sources(contexts)[:5]}")
    print(f"   answer_snip={answer[:120].replace(chr(10), ' ')!r}")


def check_ood_kubernetes():
    raw = retrieve(K8S_Q)
    filtered = filter_contexts_for_query(K8S_Q, raw)
    answer, contexts = ask(K8S_Q)
    print(f"   k8s raw_sources={_sources(raw)[:5]}")
    print(f"   k8s filtered={len(filtered)} answer_snip={answer[:160].replace(chr(10), ' ')!r}")
    # Prefer refuse; at minimum must not invent HA control-plane guidance
    lower = answer.lower()
    invent = ("control plane" in lower or "worker node" in lower) and "don't have" not in lower
    assert filtered == [] or REFUSE_ANSWER.lower() in lower or "knowledge base" in lower or not invent
    assert not invent, "invented HA guidance"
    print("OK 01: OOD kubernetes does not invent HA from unrelated chunks")


def check_asyncpg_grounded():
    raw = retrieve(ASYNC_Q)
    filtered = filter_contexts_for_query(ASYNC_Q, raw)
    srcs = " ".join(_sources(filtered)).lower()
    assert "asyncpg" in srcs, _sources(filtered)
    answer, contexts = ask(ASYNC_Q)
    lower = answer.lower()
    refused = REFUSE_ANSWER.lower() in lower and "asyncpg" not in lower
    assert not refused, f"over-refused despite context: {answer[:200]!r}"
    print("OK 04: asyncpg context kept and answer not a bare refuse")
    print(f"   sources={_sources(contexts)[:5]}")
    print(f"   answer_snip={answer[:200].replace(chr(10), ' ')!r}")


def check_compound_recall():
    keywords = expand_query(ASYNC_Q)
    print(f"   expand keywords={keywords!r}")
    raw = retrieve(ASYNC_Q)
    filtered = filter_contexts_for_query(ASYNC_Q, raw)
    joined = " ".join(_sources(filtered) + _sources(raw)).lower()
    has_async = "asyncpg" in joined
    has_mig = "migrat" in joined or "миграц" in joined or "alembic" in joined
    print(f"   top sources={_sources(raw)[:8]}")
    assert has_async, _sources(raw)
    # Expansion may be English-only near-miss; BM25 still keeps original terms.
    kw = keywords.lower()
    assert "asyncpg" in kw, keywords
    assert has_mig, f"Top-K missing migrations material: {_sources(raw)}"
    print("OK 05/07: Top-K includes asyncpg and migrations-related material")


def check_compound_recall_stable(trials: int = 5, min_pass: int = 4):
    """Repeated retrieve: migrations half must survive flaky English expand."""
    passes = 0
    samples = []
    for i in range(trials):
        keywords = expand_query(ASYNC_Q)
        raw = retrieve(ASYNC_Q)
        joined = " ".join(_sources(raw)).lower()
        has_mig = "migrat" in joined or "миграц" in joined or "alembic" in joined
        samples.append((keywords, _sources(raw)[:5], has_mig))
        if has_mig:
            passes += 1
        print(
            f"   trial {i + 1}/{trials}: mig={has_mig} "
            f"expand={keywords!r} sources={_sources(raw)[:4]}"
        )
    assert passes >= min_pass, (
        f"migrations recall only {passes}/{trials} (need >={min_pass}): {samples}"
    )
    print(f"OK 07: compound migrations recall stable ({passes}/{trials})")


def check_role_nuance():
    raw = retrieve(ROLE_Q)
    contexts = prepare_contexts(ROLE_Q, raw)
    assert any(c.get("is_directory_listing") for c in contexts), _sources(contexts)
    listing = next(c for c in contexts if c.get("is_directory_listing"))
    assert "Team Lead.md" not in listing["text"]
    answer, contexts = ask(ROLE_Q)
    lower = answer.lower()
    bare_yes = bool(re.match(r"^\s*yes\b", lower)) and "not" not in lower and "no" not in lower
    print(f"   role sources={_sources(contexts)[:5]}")
    print(f"   answer_snip={answer[:280].replace(chr(10), ' ')!r}")
    assert answer.strip() != REFUSE_ANSWER, "must use listing, not refuse"
    assert REFUSE_ANSWER.lower() not in lower, "must not refuse when listing is present"
    assert not bare_yes, "bare Yes is incorrect when no dedicated file matches"
    assert "mention" in lower or "dedicated" in lower or "not enough" in lower
    assert any(
        name.lower() in lower
        for name in ("tech lead", "qa lead", "documentation lead", "product owner", "po")
    ), f"should name files from listing: {answer[:200]!r}"
    print("OK 03/06: existence ask uses path listing; nuanced, not refuse/bare Yes")


def main():
    check_index_redacts_credentials()
    check_compound_recall()
    check_compound_recall_stable()
    check_asyncpg_grounded()
    check_ood_kubernetes()
    check_secret_question_retrieve()
    check_role_nuance()
    print("\nLive checks finished.")


if __name__ == "__main__":
    main()
