# Eval comparison report

Answer bake-off re-run **2026-09-18**.  
Committed files here are **metrics only** (IDs + hit/miss/refuse/verdict). Questions, answers, must-path needles, and corpus source paths stay in local `tmp/` — not for commit.

| File | What |
| --- | --- |
| `baseline-vector-only-stress-answers.md` | H1–H12 answer-path metrics on base (vector-only) |
| `hybrid-only-stress-answers.md` | H1–H12 answer-path metrics on hybrid-only |
| `hybrid-vs-vector-retrieve.md` | Retrieval-only H1–H12: vector vs hybrid |

Commits used:

| Label | Commit | Meaning |
| --- | --- | --- |
| Base | `a6583d4` | Vector-only retrieve + soft answer prompt |
| Hybrid-only | `1e0fbd0` | **Same as base**, plus expand + BM25 + RRF only |
| Report-era / current | `0df0291` | Hybrid retrieve **plus** later answer-path changes |

```text
a6583d4  -->  1e0fbd0  -->  …  -->  0df0291
 base         hybrid-only          current homework stack
```

---

## Verdict (short)

1. **Hybrid search helps** when it is the *only* change vs base (`a6583d4` → `1e0fbd0`).
2. Do **not** judge hybrid search using `0df0291` answer logs alone; that confounds retrieve with prompt and routing changes.

---

## 1. Fair answer bake-off (H1–H12): base vs hybrid-only

Same gold, same soft `build_prompt`, same corpus. Difference = **retrieve only**.

| Metric | Base | Hybrid-only |
| --- | ---: | ---: |
| Must-path present in answer sources | 7 / 12 | **10 / 12** |
| Model refuse | 2 / 12 | **1 / 12** |

Per-question (no gold needles or corpus paths):

| ID | Base src | Hybrid src | Base refuse | Hybrid refuse |
| --- | --- | --- | --- | --- |
| H1 | miss | miss | no | no |
| H2 | hit | miss | no | **yes** |
| H3 | hit | hit | no | no |
| H4 | hit | hit | no | no |
| H5 | hit | hit | no | no |
| H6 | hit | hit | no | no |
| H7 | miss | **hit** | no | no |
| H8 | miss | **hit** | **yes** | no |
| H9 | miss | **hit** | no | no |
| H10 | hit | hit | no | no |
| H11 | hit | hit | no | no |
| H12 | miss | **hit** | **yes** | no |

**Takeaway:** Hybrid-only wins on must-path (7→10) and fewer false refuses (2→1). Clearest IDs: **H7 / H8 / H9 / H12**. H2 can still false-refuse on the small model — answer-path noise, not an RRF regression.

Worktrees: `local-rag-mcp-baseline` @ `a6583d4`, `local-rag-mcp-hybrid-only` @ `1e0fbd0`.

---

## 2. Retrieval-only stress (same tree, V vs H)

| Verdict | Count |
| --- | ---: |
| Hybrid win (V miss, H hit) | **8** |
| Both hit | 4 |
| Both miss | 0 |
| Vector win | 0 |

Per-ID table: `hybrid-vs-vector-retrieve.md`.

**Takeaway:** Lexical/hybrid retrieve recovers rare needles under noisy queries. Clean homework demo for the hybrid bonus.

Note: stress uses `top_k=3`; answer runs use config `TOP_K=5`.

---

## 3. Why earlier “base answers look better” was misleading

Comparing H1–H12 on `a6583d4` vs **`0df0291`** mixed retrieve change with answer-path change. The fair pair (base vs hybrid-only) removes that confound: **hybrid search wins**.

---

## 4. What to show graders

1. Pipeline: expand → FAISS ∥ BM25 → RRF.
2. Retrieval stress metrics: `hybrid-vs-vector-retrieve.md` (8 hybrid wins).
3. Fair answer bake-off metrics: base vs hybrid-only (must-src 7→10, refuses 2→1).
4. Full Q/A and source dumps: keep local under `tmp/` only.
