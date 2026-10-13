# ml-intern — Architecture Audit (Agent A)

Source: `github.com/huggingface/ml-intern` (cloned to `/tmp/ml-intern-audit`, ~12K LOC Python, top-level `agent/`, `backend/`, `frontend/`, `configs/`).

## 1. Agent loop shape

**Tool-calling ReAct loop over LiteLLM — not smolagents, not LangGraph, not actor-critic.** The only `smolagents` reference in the repo is the logo image (`frontend/public/smolagents.webp`, referenced in `README.md:2`, `frontend/index.html`, `WelcomeScreen.tsx`, `AppLayout.tsx`) — no code dependency.

Main control flow lives in `agent/core/agent_loop.py`:
- `submission_loop()` (`agent_loop.py:1152`) pulls Operations from a `submission_queue` and dispatches to `Handlers`.
- `Handlers.run_agent()` (`agent_loop.py:461-857`) is the inner while-loop: `max_iterations=300` (`agent/config.py:30`), each iteration does `litellm.acompletion(..., tool_choice="auto")` (`agent_loop.py:276` streaming / `:361` non-streaming), parses `tool_calls[]`, runs approval gate, executes via `ToolRouter`, appends results, loops until the model emits a turn with no tool calls.
- Embellishments: context auto-compaction before each LLM call (`_compact_and_notify`, `agent_loop.py:207`), doom-loop detection (`agent/core/doom_loop.py` — md5 hash of each tool-call's (name, args), fires if ≥3 identical consecutive OR a length-2-to-5 repeating sequence appears in the last 30 messages, injects a corrective user message at `doom_loop.py:113-133`), truncation retry (`agent_loop.py:536-575`), transient-error retry w/ 3 attempts and 5/15/30s backoff (`:162-180`), parallel tool execution via `asyncio.gather` (`:721-751`).
- Event-driven: every state change emits to `event_queue` (events enumerated in `README.md:167-184`), consumed by FastAPI+SSE in `backend/routes/agent.py` and a React SPA in `frontend/`.

## 2. Tool surface

Built-in tools registered in `agent/core/tools.py:282-376` (`create_builtin_tools`). Approval gating lives in `_needs_approval()` (`agent_loop.py:92-159`):

| Tool | File | Approval? |
|------|------|-----------|
| `research` (sub-agent) | `agent/tools/research_tool.py:457` | no |
| `explore_hf_docs`, `fetch_hf_docs`, `find_hf_api` | `agent/tools/docs_tools.py` | no |
| `hf_papers` | `agent/tools/papers_tool.py:1165-1271` | no |
| `hf_inspect_dataset` | `agent/tools/dataset_tools.py:387-423` | no |
| `plan_tool` | `agent/tools/plan_tool.py:85` | no |
| `hf_jobs` | `agent/tools/jobs_tool.py` | **yes** (GPU always; CPU if `confirm_cpu_jobs`) |
| `hf_repo_files`, `hf_repo_git` | `agent/tools/hf_repo_{files,git}_tool.py` | upload/delete/destructive gated |
| `github_find_examples`, `github_list_repos`, `github_read_file` | `agent/tools/github_*.py` | no |
| `sandbox_create` + `bash`, `read`, `write`, `edit`, `upload` | `agent/tools/sandbox_tool.py`, `sandbox_client.py` (250 LOC — duplicates a template HF Space, talks HTTPS) | **yes on create** |
| MCP tools from `https://huggingface.co/mcp?login` | wired via `fastmcp.Client` (`tools.py:139-172`) | per-op |
| Local-mode file tools (Read/Write/Edit/Bash on host) | `agent/tools/local_tools.py` (CLI only — swapped in for sandbox when `local_mode=True`) | no |

A hardcoded MCP blocklist `NOT_ALLOWED_TOOL_NAMES = ["hf_jobs", "hf_doc_search", "hf_doc_fetch", "hf_whoami"]` (`tools.py:65`) prevents MCP-exposed duplicates from shadowing built-ins.

## 3. Citation walking

Three backends blended in `agent/tools/papers_tool.py`:

1. **Semantic Scholar graph API** is the citation engine (`papers_tool.py:40-129`). Arxiv ID → `ARXIV:{id}` format → `GET /graph/v1/paper/{id}/references` and `/citations` (`_op_citation_graph`, `papers_tool.py:813-851`). Returns per-edge `isInfluential`, `intents`, and `contexts` snippets. In-process LRU cache (`_s2_cache`, 500 entries) survives across sessions. Auth-less 429/5xx retry in `_s2_request` (`:62-101`).
2. **Arxiv HTML parsing** for section-level reading (`_op_read_paper`, `papers_tool.py:692-749`). Fetches `https://arxiv.org/html/{id}` then `https://ar5iv.labs.arxiv.org/html/{id}` as fallback, parses with BeautifulSoup into `{title, abstract, sections[{id,title,level,text}]}` (`_parse_paper_html`, `:137-210`). Falls back to HF-API abstract if both fail.
3. **HF Hub paper APIs** for trending/search/linked-resource discovery: `/api/daily_papers`, `/api/papers/search`, `/api/papers/{id}`, and `/api/datasets?filter=arxiv:{id}` (`_op_find_datasets`, `:854`, and siblings).

Additionally: `snippet_search` hits S2's full-text snippet endpoint `/graph/v1/snippet/search` over 12M+ papers (`:1046-1085`); `recommend` uses S2 `/recommendations/v1/papers/forpaper/{id}` and a POST form that takes positive/negative examples (`:1093-1139`). No scraping of Google Scholar, no arxiv OAI-PMH.

## 4. Dataset quality judgment

**There is no programmatic quality scorer.** `hf_inspect_dataset` (`dataset_tools.py:35-145`) only fetches `/is-valid`, `/splits`, `/info`, `/first-rows`, `/parquet` from `datasets-server.huggingface.co` and formats them as markdown: status line, configs/splits table, column schema, 3 sample rows (truncated to 150 chars/value, `:18`), parquet file sizes. The messages-column analyzer (`_format_messages_structure`, `:250-350`) is a format-checker — it reports which ChatML keys are present (`role`, `content`, `tool_calls`, etc.) but doesn't score quality.

So the "healthcare dataset is too low-quality, synthesize 1100 points instead" conclusion is a pure **LLM judgment** made from reading the 3 sample rows the inspector returns. The code path has zero heuristics for duplication, length distribution, toxicity, label noise, or topical coverage. The tool description itself primes the LLM to think in format-compatibility terms, not quality terms (`dataset_tools.py:389-399`: "REQUIRED before any training job to verify dataset format matches training method"). Any quality claim the agent makes is a qualitative read of the sampled rows — with the obvious failure mode that 3 rows can't surface systematic issues.

## 5. Training-job launcher

**HF Jobs only, no local GPU fallback.** `jobs_tool.py` wraps `huggingface_hub.HfApi` — `HfJobsTool._run_job` (`:491-606`) calls `api.run_job(image, command, env, secrets, flavor, timeout, namespace)`. Two execution modes:
- Python mode: wraps script in `uv run` with optional `--with <dep>` deps and `-p <python>`, runs on `ghcr.io/astral-sh/uv:python3.12-bookworm` by default (`_build_uv_command`, `:163`; `_resolve_uv_command`, `:216` — URL stays as URL, multi-line goes base64→stdin via `_wrap_inline_script` at `:189`, else treated as file path).
- Docker mode: raw `command` on caller-specified `image`.

Hardware flavors hardcoded at `jobs_tool.py:31-61` (CPU: `cpu-basic`/`cpu-upgrade`; GPU: `t4-*`, `a10g-*`, `a100-*`, `l4x*`, `l40sx*`; specialized: `inf2x6`). `CPU_FLAVORS` drives the approval-gate heuristic in `_needs_approval`. Secrets auto-injected: `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are force-set from the session token (`_add_environment_variables`, `:143-160`), progress bars disabled via `_DEFAULT_ENV` (`:127-133`).

Monitoring: `_wait_for_job_completion` (`:382-489`) calls `api.fetch_job_logs` (blocking generator) on a worker thread, bridges to asyncio via a queue, streams each line back via `log_callback` → UI `tool_log` event. Retries up to **100 times** with 5s sleeps on connection drops, always re-checking `inspect_job` to confirm the job's still `RUNNING` before reconnecting. Final-status poll has its own 6-attempt/2.5s settle loop (`:480`) because the API can lag the log stream. UV install spam is filtered out of the returned logs (`_filter_uv_install_output`, `:82-117`). Failure diagnosis is limited to returning the full stderr log to the LLM — no structured parsing.

Dev/test loop is separate: the agent `sandbox_create`s a duplicated HF Space (template `burtenshaw/sandbox`, `sandbox_client.py:49`) running a FastAPI dev server in a Docker image it constructs on the fly (`_DOCKERFILE` at `:68-98`); the agent iterates with `bash`/`read`/`write`/`edit` until the script works, then ships to `hf_jobs` at scale. Interrupt cleanup at `_cleanup_on_cancel` (`agent_loop.py:236-259`) kills both sandbox processes and cancels any tracked `session._running_job_ids`.

## 6. Framework & LLM backend

**litellm + fastmcp, no smolagents.** `pyproject.toml:7-28` lists only: `litellm>=1.83`, `huggingface-hub>=1.0.1`, `fastmcp>=3.2.0`, `pydantic`, `fastapi`, `httpx`, `datasets`. LLM backend is whatever LiteLLM supports — default `anthropic/claude-opus-4-6` (`configs/main_agent_config.json:2`), with a custom router rewriter `_resolve_hf_router_params` (`agent_loop.py:25-67`) that maps `huggingface/<provider>/<org>/<model>` to the new `router.huggingface.co/<provider>/v3/openai` endpoint (the built-in LiteLLM `huggingface/` provider still points at the deprecated `api-inference.huggingface.co`). Context-window map is local (`session.py:20-33`) to avoid LiteLLM's `get_max_tokens` network hang. Auth chain for HF inference: `INFERENCE_TOKEN` env → session `hf_token` → `HF_TOKEN` env (`:57-61`), so the hosted Space can subsidize inference for free users.

## 7. Verdict

Genuinely novel: the **paper-first research sub-agent** and its deep Semantic Scholar integration. `research_tool.py:43-156` is a system prompt that forces a crawl pattern (anchor paper → downstream citation graph → methodology sections → dataset validation → code) before any implementation; it's spawned as an isolated LiteLLM call with its own 190K-token budget, a read-only tool subset (`RESEARCH_TOOL_NAMES` at `:29-41`), and the ability to attribute every recommendation to `(result, recipe, dataset, paper)`. Combined with `hf_papers`' S2 citation graph + intent labels + snippet search, this is a legitimately useful ML-research surface that most general-purpose coding agents don't have. The ChatML-format auto-check in `hf_inspect_dataset` is small but sharp — it catches the single most common training-script failure mode (KeyError on `messages` vs `prompt/completion`) before a GPU is spun up.

Everything else is **glue over existing HF primitives and 2025 agent-plumbing conventions**: the agent loop is a LiteLLM tool-calling ReAct loop with standard trimmings (doom detector, compaction, parallel execution, approval gates) that any competent team can assemble in ~2K LOC; `hf_jobs` is a thin wrapper around `HfApi.run_job`; sandboxes are duplicated HF Spaces; MCP integration is fastmcp defaults. Importantly, **no "quality judgment" for datasets exists in code** — the much-cited healthcare-synthesis story is the LLM reading 3 sample rows and deciding. That's an LLM capability claim, not an ml-intern architectural claim, and anyone reproducing that result should note the judgment is unaudited.
