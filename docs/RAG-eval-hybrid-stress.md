# Hybrid stress set — retrieval-only (12 questions)

Corpus: `ecto-1-kb` under `src/docs`.  
**Goal:** show where **hybrid (expand + BM25 + RRF)** beats **vector-only FAISS**, not end-to-end answer Pass%.

Questions are intentionally **noisy**: onboarding / roles / ceremony / SSO chatter plus a **rare lexical needle**. Vector search often follows the chatter; BM25 follows the needle.

## How to score

For each question, check whether **any** retrieved chunk’s `source` path contains the **Must retrieve** string (case-insensitive substring).

Verdicts:

| Verdict | Meaning |
| --- | --- |
| **Vector miss / Hybrid hit** | Hybrid win (what we want to demonstrate) |
| **Both hit** | Too easy — not a hybrid differentiator |
| **Both miss** | Gold path wrong, or need stronger expand / index |
| **Vector hit / Hybrid miss** | Hybrid regression (investigate) |

Do **not** grade the LLM answer here. Use:

```bash
cd src
python eval_hybrid_stress.py --expand-identity --top-k 3
python eval_hybrid_stress.py --top-k 3
```

`--expand-identity` skips the expand LLM (BM25 still sees the raw question). Default uses real `expand_query` (needs Ollama).  
**`--top-k 3`** makes the scoreboard louder (fewer slots for a lucky vector hit).

Log default: `tmp/rag-eval-hybrid-stress.md`

---

### H1. Rare library name — asyncpg

**Question:** Привет, я новенький на онбординге по ролям и Sprint Planning — Documentation Lead сказал смотреть wiki; как у нас ходить в Postgres через asyncpg, не через ORM?

**Must retrieve:** `asyncpg`

**Why hybrid:** `asyncpg` is a BM25 needle; “онбординг / роли / Sprint Planning” pulls vector to `Роли/` and process docs.

### H2. Exact tooling doc — Ruff line length

**Question:** На ретро после Code Review Checkpoint QA Lead и Tech Lead спорили про стиль: не про FSD и не про SSO, а про Ruff line-length — какая максимальная длина строки?

**Must retrieve:** `Ruff, Pre-commit`

**Why hybrid:** Ceremony/role chatter distracts embeddings; `Ruff` / line-length stay lexical.

### H3. slowapi needle under process noise

**Question:** Product Owner на Backlog Refinement между оценками спросил мимоходом: какая библиотека slowapi у нас для rate limiting на FastAPI?

**Must retrieve:** `Rate Limiting`

**Why hybrid:** PO / refinement semantics vs exact `slowapi`.

### H4. Alembic command under Docker/onboarding noise

**Question:** Новичок после README про 3 месяца и 10 человек и после списка ролей спрашивает: какой точной командой `docker compose exec api alembic upgrade head` накатывают миграции?

**Must retrieve:** `Миграции.md`

**Why hybrid:** README/roles distractors; command string + Alembic favor migration files (require the `Миграции.md` path, not just any “миграц”).

### H5. Git `develop` under ceremony noise

**Question:** На Mid-Sprint Sync Scrum Master отвлёкся на Git: от какой ветки develop мы делаем feat/fix/chore, это в процессе или в отдельном гайде?

**Must retrieve:** `Работа с Git`

**Why hybrid:** Sync/SM ceremony language vs exact `develop` / `feat/` in the Git doc.

### H6. token_hash under SSO/frontend noise

**Question:** После возврата из SSO на фронте все обсуждают AuthPage и VITE_APP_ENV=local — а где на бэкенде хранится token_hash и что такое SESSION_TTL_SECONDS?

**Must retrieve:** `Архитектура бэкенда`

**Why hybrid:** Auth/SSO/frontend embeddings; opaque `token_hash` / `SESSION_TTL_SECONDS` are BM25 anchors to backend architecture.

### H7. English Rate Limiting under Russian process talk

**Question:** На Sprint Demo про роли и DoD кто-то с английским термином вставил: что такое Rate Limiting и какой HTTP-статус при превышении лимита?

**Must retrieve:** `Rate Limiting`

**Why hybrid:** Demo/roles/DoD distractors; English title token in a Russian filename.

### H8. userRoles FSD exception under /Роли confusion

**Question:** Не путать с файлами в /Роли: почему shared/mock импортирует userRoles из entities — это исключение FSD в архитектуре фронта?

**Must retrieve:** `01-Обзор проекта\Архитектура.md`

**Why hybrid:** `/Роли` and “роли” pull role markdown; `userRoles` / `shared/mock` should still lexical-hit frontend architecture. Needle is the specific path fragment.

### H9. Chatty onboarding + asyncpg

**Question:** Слушай, я с календаря сроков и из README про длительность проекта — без воды: async PostgreSQL клиент у нас это asyncpg? Куда в доке смотреть, не в Процессы?

**Must retrieve:** `asyncpg`

**Why hybrid:** Calendar/README/process distractors; keep `asyncpg` for BM25.

### H10. Filename spaces_endpoints under role onboarding noise

**Question:** На онбординге после README и папки /Роли тимлид сказал открыть именно файл spaces_endpoints.md — какие HTTP-ручки spaces там описаны на бэкенде?

**Must retrieve:** `spaces_endpoints`

**Why hybrid:** Role/README onboarding distractors; exact filename `spaces_endpoints.md` is a BM25 needle (avoid naming FSD layers that drown the query).

### H11. Alembic migrations under local-mode frontend noise

**Question:** Мы правили VITE_APP_ENV=local и моки spaces, а соседний вопрос: как через Alembic в docker compose применяются миграции БД (не фронт)?

**Must retrieve:** `Миграции.md`

**Why hybrid:** Local-mode / mocks / spaces embeddings; Alembic + миграции stay lexical toward `Миграции.md`.

### H12. get_remote_address under docs-role noise

**Question:** Documentation Lead на онбординге между календарями и тест-планом спросил: как slowapi через get_remote_address определяет клиента для лимитов?

**Must retrieve:** `Rate Limiting`

**Why hybrid:** Docs-role / calendar / testing chatter; rare symbol `get_remote_address` is the BM25 needle.

---

## Notes

- **Must retrieve** is a path substring — use a specific stem when “any миграц file” would be too loose.
- Prefer **Hybrid win** counts over answer quality.
- If the board is still quiet, re-run with `--top-k 3` (or even `2`).
