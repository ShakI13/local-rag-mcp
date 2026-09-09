# Hybrid RAG Retrieve (Query Expansion + BM25 + RRF)

Status: ready-for-agent

## Problem Statement

As a user of the local company knowledge-base assistant, I ask messy, chatty questions (including Russian) that mix small talk with rare exact terms (API names, abbreviations, process titles). Today RAG Retrieve only runs Vector Search, so those questions often miss the right Chunks or bury them under noise. I need Hybrid Search so answers stay grounded in the right documents without the app crashing when the small keyword model misbehaves.

## Solution

Before search, Query Expansion turns my question into Keywords. Vector Search runs on the original question and BM25 Search runs on the Keywords (with Lexical Normalization for Russian), in parallel. RRF merges the two ranked lists by Chunk Identity and keeps Top-K Chunks for the existing answer path. If Keywords cannot be produced, the system falls back to the original question and continues.

## User Stories

1. As a knowledge-base user, I want RAG Retrieve to find Chunks that match the meaning of my question, so that paraphrases still surface useful context.
2. As a knowledge-base user, I want RAG Retrieve to find Chunks that literally contain rare terms from my question, so that names like library APIs and abbreviations are not lost.
3. As a knowledge-base user, I want chatty preamble stripped into Keywords before lexical search, so that greetings and filler do not dominate BM25 Search.
4. As a knowledge-base user, I want the original question preserved for Vector Search, so that semantic intent is not over-narrowed by Keywords alone.
5. As a knowledge-base user, I want Hybrid Search results fused with RRF, so that Chunks strong in either lane (or both) rise without fragile score calibration.
6. As a knowledge-base user, I want only Top-K Chunks after fusion in the answer prompt, so that the small final model is not flooded with context.
7. As a knowledge-base user, I want duplicate Chunks from both lanes collapsed by Chunk Identity, so that the same passage is not pasted twice.
8. As a knowledge-base user, I want retrieval to keep working if Query Expansion returns empty or unusable output, so that a small model format failure does not break chat.
9. As a knowledge-base user, I want retrieval to keep working if the keyword model call errors or times out, so that search degrades gracefully to the original question.
10. As a knowledge-base user asking in Russian, I want Lexical Normalization on BM25 Search, so that inflected forms still match Chunk text.
11. As a knowledge-base user mixing Russian with Latin tokens, I want unknown or non-Cyrillic tokens kept as raw lowercased terms, so that exact identifiers still match.
12. As a knowledge-base user, I want Vector Search and BM25 Search to run in parallel, so that adding the lexical lane does not roughly double wait time.
13. As an operator, I want BM25 Search built from the existing chunk store at load or first retrieve, so that I do not manage a second on-disk FTS database for this homework.
14. As an operator, I want configurable lane depth and final Top-K, so that I can tune recall into fusion without changing code.
15. As an operator, I want a documented RRF constant, so that fusion behavior is explainable and stable.
16. As an operator, I want Query Expansion to use a low temperature, so that Keywords are boring and repeatable on a small local model.
17. As an operator, I want Keywords requested as comma-separated phrases with a short prompt, so that a 0.6B–3B model can comply.
18. As an implementer, I want RAG Retrieve upgraded behind the existing retrieve entry point, so that the assistant and pure-RAG ask path pick up Hybrid Search without a second public API.
19. As an implementer, I want Query Expansion isolated in its own helper, so that final-answer generation stays separate from keyword generation.
20. As an implementer, I want MCP tools left unchanged, so that filename search is not confused with BM25 Search over Chunk text.
21. As a grader/reviewer, I want the pipeline to match the homework shape (expand → parallel vector+lexical → RRF → Top-K), so that scored criteria are demonstrable.
22. As a grader/reviewer, I want fallback behavior when Keywords fail, so that the error-handling criterion is met.
23. As a knowledge-base user, I want sources on answers to still reflect the fused Chunks, so that citations remain trustworthy after Hybrid Search.
24. As an operator with an empty or missing index, I want existing index-build/load behavior preserved, so that Hybrid Search does not invent a new bootstrap story.
25. As a knowledge-base user, I want Chunks that rank well in both lanes to outrank single-lane hits when appropriate, so that agreement across lanes is rewarded by RRF.
26. As a knowledge-base user, I want Chunks that appear in only one lane to remain eligible, so that lexical-only or vector-only hits are not dropped before Top-K.
27. As an implementer, I want Lexical Normalization applied to both the BM25 corpus and the BM25 query, so that both sides of the lexical lane share one normalization rule.
28. As an operator, I want dependencies for BM25 and Russian morphology declared with the project, so that a fresh environment can run Hybrid Search.
29. As a knowledge-base user, I want always-on Query Expansion for retrieve, so that demos and grading exercise the full pipeline without a hidden flag.
30. As a maintainer, I want domain language (Chunk, Keywords, RRF, etc.) used consistently in this work, so that specs and code review stay aligned with the project glossary.

## Implementation Decisions

- **DoD:** Deliver the graded Hybrid Search homework path, plus Lexical Normalization for better Russian BM25 quality (intentional scope beyond the minimum checklist).
- **Primary seam:** Keep a single public RAG Retrieve entry point with the same input (user question string) and output (ordered list of Chunk dicts with text, source, and per-document chunk index). Callers (assistant and pure ask path) stay unchanged.
- **Query Expansion:** Add a dedicated expand helper. Prompt the small local chat model for comma-separated Keywords only. Use temperature 0.1. Soft-parse commas / light cleanup; if the result is empty or not usable as Keywords, or the call throws, fall back so Keywords become the original question. Do not change final-answer temperature requirements via this work.
- **Lane routing:** Vector Search embeds/searches with the original user question. BM25 Search queries with Keywords (already original on fallback).
- **Lexical engine:** BM25 over Chunk text (homework FTS lane). Do not add SQLite FTS5. Do not run a second lexical engine.
- **Lexical Normalization:** Use pymorphy3 with the Russian dictionary. Normalize both the in-memory BM25 corpus and the BM25 query the same way. On non-Cyrillic tokens or morph failure, keep the raw lowercased token.
- **BM25 lifecycle:** Build the BM25 index in memory from the loaded chunk list at index load / first retrieve. No new persisted FTS artifact required for this spec.
- **Parallelism:** Run Vector Search and BM25 Search concurrently via a thread pool so the lexical lane does not need an async rewrite of the stack.
- **Fusion:** Merge with RRF, formula sum of `1/(RRF_K + rank)` per lane, ranks 1-based best-first. Default `RRF_K = 60`. No weighted-sum fusion path.
- **Identity:** Score and dedupe by Chunk Identity `(source, chunk_id)`, never by bare chunk index alone.
- **Depths:** Config defaults `LANE_K = 10` (hits taken from each lane into fusion), `TOP_K = 5` (final Chunks after RRF). Both configurable alongside `RRF_K`.
- **Always-on:** Query Expansion and Hybrid Search run on every retrieve; no feature flag required by this spec.
- **MCP:** Out of the retrieve upgrade; leave list/read/filename-search behavior as-is.
- **Config:** Add knobs for lane depth, RRF constant, and any paths/settings strictly needed for the above; do not add unused engine switches.
- **No ADRs** for this homework pass; glossary lives in the domain context file.

## Testing Decisions

- **Good tests** assert observable RAG Retrieve behavior: given a question (and controlled doubles for the keyword model / search lanes where needed), the returned Chunk list is the fused Top-K with correct Chunk Identity dedupe and stable ordering rules—not internal thread scheduling, not private helper names.
- **Primary seam to test:** the RAG Retrieve entry point (ideal single seam). Prefer fakes/doubles at the LLM keyword boundary and, if needed, at lane boundaries injectable behind that entry point—rather than scattering new public seams across the codebase.
- **Pure logic worth covering through that seam or minimal direct tests only when the retrieve seam cannot express them cheaply:** Keywords soft-parse + fallback; RRF ordering on two small ranked lists; Lexical Normalization keeping Latin tokens unchanged.
- **Prior art:** no automated test suite in the repo today; introduce tests beside the RAG modules using the project’s existing Python tooling when added. Do not require live Ollama or a full FAISS corpus for the core fusion/fallback cases.
- **Seam check:** If this single-retrieve seam is too coarse for you, the next-best addition is one narrow pure RRF function—not a proliferation of search services.

## Out of Scope

- SQLite FTS5 or dual lexical engines
- Weighted-sum fusion, LLM reranker, GraphRAG, multi-agent loops
- Changing MCP tool semantics or replacing RAG Retrieve with MCP
- Before/after benchmark harness as a required deliverable (bonus only; not part of this spec’s DoD)
- Replacing the embedding model or chunking scheme
- Persisted BM25/FTS artifacts on disk
- Feature flags to disable Query Expansion
- Writing ADRs for BM25-vs-FTS5 or RRF-vs-weighted-sum

## Further Notes

- Homework brief and junior guide remain the grading narrative; this spec is the agreed implementation contract from the grill session.
- Domain terms must follow the project glossary: Chunk, Chunk Identity, Query Expansion, Keywords, Vector Search, BM25 Search, Lexical Normalization, Hybrid Search, RRF, Top-K, RAG Retrieve.
- ecto-style noisy Russian questions with rare exact tokens are the intended quality scenarios; Latin/code tokens must survive normalization.
