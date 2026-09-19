# Hybrid-only — stress answers H1–H12

Commit: `1e0fbd0` · When: 2026-09-18  
Gold set: hybrid-stress H1–H12 (local gold file; not reproduced here)

**Src** = gold must-path present in answer sources. **Refuse** = model returned the KB refuse string. Question/answer/source text stays in local `tmp/` only.

| ID | Src | Refuse | MCP |
| --- | --- | --- | --- |
| H1 | miss | no | — |
| H2 | miss | yes | read_document |
| H3 | hit | no | — |
| H4 | hit | no | read_document |
| H5 | hit | no | read_document |
| H6 | hit | no | read_document |
| H7 | hit | no | — |
| H8 | hit | no | — |
| H9 | hit | no | read_document |
| H10 | hit | no | read_document |
| H11 | hit | no | read_document |
| H12 | hit | no | — |

**Summary:** must-path **10/12**, refuse **1/12**.
