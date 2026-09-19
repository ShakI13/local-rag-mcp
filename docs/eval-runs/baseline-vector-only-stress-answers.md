# Baseline (vector-only) — stress answers H1–H12

Commit: `a6583d4` · When: 2026-09-18  
Gold set: hybrid-stress H1–H12 (local gold file; not reproduced here)

**Src** = gold must-path present in answer sources. **Refuse** = model returned the KB refuse string. Question/answer/source text stays in local `tmp/` only.

| ID | Src | Refuse | MCP |
| --- | --- | --- | --- |
| H1 | miss | no | list_documents |
| H2 | hit | no | — |
| H3 | hit | no | list_documents |
| H4 | hit | no | — |
| H5 | hit | no | — |
| H6 | hit | no | — |
| H7 | miss | no | — |
| H8 | miss | yes | — |
| H9 | miss | no | read_document |
| H10 | hit | no | — |
| H11 | hit | no | search_documents |
| H12 | miss | yes | — |

**Summary:** must-path **7/12**, refuse **2/12**.
