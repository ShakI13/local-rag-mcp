# Hybrid vs vector — retrieval stress

When: 2026-09-14 · `TOP_K=3` · expand = identity (no LLM)  
Gold set: hybrid-stress H1–H12 (local gold file; not reproduced here)

Same IDs, two retrieve modes in one tree: **V** = vector-only FAISS, **H** = hybrid (expand + BM25 + RRF). Hit = gold must-path present in retrieved sources. Question text and source paths stay in local `tmp/` only.

| ID | V | H | Verdict |
| --- | --- | --- | --- |
| H1 | miss | hit | Hybrid win |
| H2 | hit | hit | Both hit |
| H3 | miss | hit | Hybrid win |
| H4 | miss | hit | Hybrid win |
| H5 | hit | hit | Both hit |
| H6 | hit | hit | Both hit |
| H7 | miss | hit | Hybrid win |
| H8 | miss | hit | Hybrid win |
| H9 | miss | hit | Hybrid win |
| H10 | hit | hit | Both hit |
| H11 | miss | hit | Hybrid win |
| H12 | miss | hit | Hybrid win |

| Verdict | Count |
| --- | ---: |
| Hybrid win (V miss, H hit) | **8** |
| Both hit | 4 |
| Both miss | 0 |
| Vector win | 0 |

```bash
cd src
python eval_hybrid_stress.py --expand-identity --top-k 3 --log ../tmp/rag-eval-hybrid-stress.md
```
