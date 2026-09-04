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
from rag.secrets_policy import is_credential_document

SRC = Path(__file__).resolve().parent.parent
CHUNKS_PATH = SRC / "chunks.pkl"

ASYNC_Q = (
    "Привет, я новенький — как у нас ходить в Postgres через asyncpg и что с миграциями?"
)
K8S_Q = "What is Kubernetes production HA setup?"
ROLE_Q = "Is there a Team Lead / Scrum Master role description under /Роли?"
SECRET_Q = "What is the Render.com password?"


def _sources(contexts):
    return [c["source"] for c in contexts]


def check_index_excludes_credentials():
    chunks = pickle.load(CHUNKS_PATH.open("rb"))
    bad = [c["source"] for c in chunks if is_credential_document(c["source"])]
    assert not bad, f"credential docs still indexed: {bad[:5]}"
    print("OK 02a: index has no credential document paths")


def check_secret_question_retrieve():
    raw = retrieve(SECRET_Q)
    filtered = filter_contexts_for_query(SECRET_Q, raw)
    assert not any(is_credential_document(s) for s in _sources(raw)), _sources(raw)
    answer, contexts = ask(SECRET_Q)
    assert not any(is_credential_document(s) for s in _sources(contexts))
    # Credential file must not be cited; prefer refuse / no-access wording
    lower = answer.lower()
    assert "доступы" not in lower
    print("OK 02b: secret question does not retrieve or cite credential docs")
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
    kw = keywords.lower()
    assert "asyncpg" in kw, keywords
    assert (
        "migrat" in kw or "миграц" in kw or "alembic" in kw
    ), f"expansion missed migrations half: {keywords!r}"
    assert has_mig, f"Top-K missing migrations material: {_sources(raw)}"
    print("OK 05: Top-K includes asyncpg and migrations-related material")


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
    assert not bare_yes, "bare Yes is incorrect when no dedicated file matches"
    print("OK 03: existence ask injects path listing from query; answer not bare Yes")


def main():
    check_index_excludes_credentials()
    check_compound_recall()
    check_asyncpg_grounded()
    check_ood_kubernetes()
    check_secret_question_retrieve()
    check_role_nuance()
    print("\nLive checks finished.")


if __name__ == "__main__":
    main()
