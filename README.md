# Self-Corrective RAG (CRAG) Pipeline

A locally-hosted, LangChain-based Retrieval-Augmented Generation system that replaces
single-shot retrieval with an evaluation-driven control loop: every retrieved chunk is
graded by an LLM, weak context is progressively rejected at three tiers, and knowledge
gaps trigger automatic query rewriting and live web search fallback. Runs entirely on
consumer hardware (laptop CPU + 6 GB GPU), with no cloud LLM dependency.

Built on: LangChain, Ollama, BAAI/bge-m3, ChromaDB (HNSW), DuckDuckGo / Tavily.

---

## 1. Approach

Classical RAG is a fire-and-forget pipeline: embed documents, retrieve top-k chunks,
stuff them into a prompt, and generate. It has no notion of whether the retrieved
context actually answers the question. When retrieval fails, the generator either
hallucinates or refuses, and the user cannot tell which happened.

This project implements the Self-Corrective RAG (CRAG) paradigm: retrieval is treated
as fallible and is wrapped in an explicit control loop.

Three design principles:

1. **Evaluate before you generate.** An LLM evaluator scores every retrieved chunk on
   a 0.0-1.0 relevance scale before any context reaches the generator.
2. **Reject progressively.** Irrelevant material is discarded at three increasingly
   fine granularities: chunk level, sentence level, and source level (routing).
3. **Fail loudly, degrade gracefully.** Every fallback in the system prints what it
   is doing and why. No silent defaults: if the evaluator fails, the trace shows the
   raw model output that caused it. If the web search fails, the answer carries an
   explicit provenance disclaimer.

## 2. Architecture

Four modules, one responsibility each:

| File | Responsibility | Key components |
|---|---|---|
| `retrieval.py` | Ingestion and vector search | PyMuPDF/TextLoader, RecursiveCharacterTextSplitter (800/150), SentenceTransformer BGE-M3 (fp16, GPU), ChromaDB persistent HNSW (cosine), incremental sync manifest, process-pool parallel parsing |
| `evaluator_router.py` | Evaluation, routing, refinement, fallback | LLM chunk grader (0.0-1.0), threshold router, knowledge refiner (decompose-filter-recompose), query rewriter, DuckDuckGo + Tavily search chain |
| `generator.py` | Grounded synthesis | Hardened system prompt, context/question delimiters, token-streamed generation, source formatting |
| `main.py` | Orchestration and CLI | Pipeline state machine, chunk selection, circuit breakers, interactive loop, startup menu |

```
                          USER QUERY
                              |
                              v
              +-------------------------------+
              | [retrieval.py]                |
              | ChromaDB HNSW (cosine, ef)    |
              | top-k chunks (default k = 8)  |
              +-------------------------------+
                              |
                              v
              +-------------------------------+
              | [evaluator_router.py]         |
              | LLM retrieval evaluator       |
              | per-chunk score 0.0 - 1.0     |
              | + search_query (fused output) |
              +-------------------------------+
                              |
              TIER 1: chunk rejection
              drop score < 0.3, rank, cap at 8
                              |
                              v
                     +---------------+
                     | route_verdict |
                     +---------------+
                 /        |         \
          CORRECT    AMBIGUOUS     INCORRECT
          (max>=0.7) (mixed)      (all < 0.3)
                |         |             |
                |         |      discard local context
                |         |             |
                v         v             v
        +-------------+ +--------------------+
        | KNOWLEDGE   | | QUERY REWRITER     |
        | REFINEMENT  | | (search_query from |
        | decompose ->| |  grader, or LLM    |
        | filter ->   | |  rewrite fallback) |
        | recompose   | +--------------------+
        | (sentences) |          |
        +-------------+          v
                |       +--------------------+
                |       | WEB SEARCH CHAIN   |
                |       | DuckDuckGo ->      |
                |       | Tavily (optional)  |
                |       +--------------------+
                |          |
                |          v
                |       +--------------------+
                |       | WEB KNOWLEDGE      |
                |       | REFINEMENT         |
                |       +--------------------+
                |          |
                +----+-----+
                     |
          synthesized context blocks
          (+ explicit provenance notices)
                     |
                     v
              +-------------------------------+
              | [generator.py]                |
              | Ollama LLM, token-streamed    |
              | strict context grounding      |
              +-------------------------------+
                     |
                     v
          answer + source list + pipeline trace
```

## 3. Data Flow

### 3.1 Query path (per question)

1. **Retrieve.** The query is embedded (BGE-M3) and searched against the persistent
   HNSW index; the top-8 chunks are returned (tunable via `--k`, search depth via `--ef`).
2. **Grade.** One LLM call receives the question plus all chunks (truncated to 600
   characters each for the grader) and returns strict JSON: a 0.0-1.0 score per chunk,
   plus a `search_query` field (a keyword-optimized rewrite of the user question for
   web search, produced in the same call to save a round trip).
3. **Tier 1 rejection (chunk level).** Chunks scoring below 0.3 are discarded; the
   survivors are ranked by score and capped.
4. **Route.** `CORRECT` (any chunk >= 0.7): local only. `INCORRECT` (all < 0.3):
   local context discarded entirely, web fallback. `AMBIGUOUS` (everything else):
   local and web context are merged.
5. **Tier 2 rejection (sentence level).** Surviving chunks are decomposed into
   sentences; the LLM selects which sentences actually help answer the question;
   only those are recomposed into the final local context block. This strips the
   side-information that chunk overlap drags in.
6. **Web fallback.** On AMBIGUOUS/INCORRECT, the rewritten query goes to DuckDuckGo
   (8s timeout, one retry), then to Tavily if a `TAVILY_API_KEY` is set. Returned
   snippets pass through the same sentence-level refinement.
7. **Circuit breakers.** If both search providers return nothing: CORRECT-grade local
   context is used with an explicit caveat; otherwise the generator answers from
   parametric knowledge and is required to say so.
8. **Generate.** Context blocks, provenance notices, and the question are wrapped in
   explicit delimiters and passed to a hardened system prompt; the answer is streamed
   token-by-token, followed by the source list and pipeline trace.

### 3.2 Ingestion path (per directory)

```
directory scan (.txt .md .pdf, recursive)
        |
        v
manifest check (mtime + size per file) ---> unchanged: skip (zero re-embedding)
        |
        v
PROCESS POOL (spawn, x4):            MAIN THREAD (serial):
  PyMuPDF / TextLoader parse           GPU fp16 batch-64 embedding
  recursive split (800/150)            (per-batch progress ticks)
  content-hash chunk IDs               ChromaDB writes (batches of 500)
  per-file chunk dedup                 manifest checkpoint after every file
```

The parse pool and the embedding stream are deliberately decoupled: parsing is
CPU-bound and GIL-hostile (PDF decode), embedding is GPU-bound, so they run as an
overlapped pipeline with a bounded in-flight window (2 x workers) to cap memory.
Chunk IDs are `md5(abspath::chunk_text)`, which makes every write an idempotent
upsert: interrupted runs, re-runs, and re-ingests never duplicate data. Failed
writes roll back the partial batch before the manifest is touched.

## 4. Comparative Analysis: Traditional RAG vs. This Pipeline

| Dimension | Traditional (naive) RAG | This CRAG pipeline |
|---|---|---|
| Retrieval | single shot, top-k, unconditional | top-k, then LLM evaluation of every chunk |
| Trust in retrieval | assumed correct | assumed fallible; verified per chunk |
| Context construction | all k chunks forwarded | 3-tier rejection: chunk (score gate + ranking + cap), sentence (decompose-filter-recompose), source (verdict routing) |
| Failure handling | hallucination or silent refusal | explicit routing: query rewrite, web fallback, circuit breakers, parametric mode with forced disclaimer |
| Query transformation | none (raw user query) | keyword-optimized rewrite, fused into the grading call to eliminate a round trip |
| External knowledge | none | DuckDuckGo chain with optional Tavily keyed fallback |
| Ingestion | batch script, full re-run | incremental manifest (mtime/size), content-hash idempotency, deleted-file pruning, per-file checkpointing, partial-write rollback |
| Ingestion parallelism | sequential | process-pool parsing overlapped with GPU embedding pipeline |
| Observability | none | full trace: per-chunk scores, kept/dropped counts, verdict, rewritten query, search result counts, timing |
| Structured control output | n/a | dual-protocol with degradation (see Section 6) |
| Output discipline | prompt-dependent | delimiter-wrapped context, strict grounding rules, no fabricated sources |
| Security | none | untrusted-data framing at every LLM boundary, persona lock, injection-resistant prompts |
| GPU handling | default device placement | fp16 embeddings, tuned encode batch, VRAM budgeting for 6 GB GPUs |

Structurally, the change is from a **linear pipeline** to a **feedback control loop**:
the evaluator sits between retrieval and generation, its output drives branching
(routing), and the loop closes through the web fallback that repairs knowledge gaps
the local store could not cover.

## 5. New Approaches Introduced

Beyond the standard CRAG formulation (evaluator, thresholds, decompose-filter-recompose,
web fallback), this implementation adds:

1. **Fused evaluator + rewriter.** The grading call also emits the web search query.
   On fallback routes this removes one full LLM round trip (the standalone rewriter
   remains as a fallback if the fused field is unusable).
2. **A chunk-level rejection tier before refinement.** The CRAG paper refines inside
   chunks; this system additionally discards whole chunks below 0.3 and re-ranks the
   remainder, so the sentence-level refiner only processes chunks that already earned
   their place.
3. **Dual-protocol structured output with degradation.** Control calls (grading,
   refinement) must return machine-parseable output. The primary path uses the Ollama
   chat API in JSON mode with reasoning disabled; if that fails, a LangChain
   `ChatOllama` JSON call; if that fails, a final re-prompt asking for plain
   comma-separated values, parsed with regex. Every degradation prints the raw model
   output that triggered it.
4. **Interrupt-safe, idempotent ingestion.** Content-hash IDs + per-file manifest
   checkpoints + partial-write rollback mean Ctrl+C costs at most one file, and no
   re-run ever duplicates or orphans vectors.
5. **Pipelined parallel ingestion with bounded memory.** Process-pool parsing (spawn
   context, no forked CUDA state) overlapped with serial GPU embedding, windowed so
   at most `2 x workers` parsed files are resident in RAM.
6. **Loud-failure design.** The system never silently substitutes defaults without a
   trace line explaining what the model actually returned (this was a real failure
   mode found in testing: a broken grader defaulted every chunk to 0.5, which
   mechanically forced the AMBIGUOUS route on every query).
7. **Terminal-grade UX for a local pipeline.** Rich dual progress bars (files +
   per-chunk embedding) with a plain-renderer fallback, and token-streamed answers.

## 6. Hurdles Solved During Development

Each of these was hit in practice and is visible in the commit history of the code.

**CVE-2025-32434 / torch.load refusal.** Symptom: fatal startup error — transformers
refuses to load `pytorch_model.bin` weights on torch < 2.6 due to the `torch.load`
vulnerability. Resolution: torch >= 2.6 floor, pinned to an explicit CUDA build
(`torch==2.6.0+cu124` via the PyTorch extra index).

**Silent CPU-only torch wheel.** Symptom: embeddings ran on CPU (0% GPU utilization,
~50 min projected for a 600-chunk file). Root cause: a plain version specifier accepts
`+cpu` wheels, so a CPU-flavored torch satisfied the requirement. Resolution: the
local-version pin (`2.6.0+cu124`) makes the requirements file self-healing, plus a
CVE-specific error message at model load pointing at the exact fix.

**chromadb / posthog telemetry break.** Symptom: repeated
`Failed to send telemetry event ... capture() takes 1 positional argument` and a flood
of `Delete of nonexisting embedding ID` warnings burying the progress UI. Root cause:
posthog 3.13+ signature change vs chromadb 0.5.x, plus per-ID logging on the
idempotency pre-delete. Resolution: `posthog<3` pin, `anonymized_telemetry=False`
(env + client settings), and chromadb's logger clamped to ERROR.

**Duplicate chunk IDs on boilerplate-heavy PDFs.** Symptom: whole files rejected with
`Expected IDs to be unique` on upsert. Root cause: repeated identical page text
(licenses, headers) produced identical `md5(path::text)` IDs inside one batch.
Resolution: per-file ID dedup at parse time.

**Structured output vs. reasoning models.** Symptom: grading/refinement returned
unusable output while the plain-text rewriter worked — reasoning tokens emitted by
default conflicted with JSON-mode responses, and the failure was swallowed, defaulting
every score to a neutral 0.5 and freezing the router on AMBIGUOUS. Resolution: the
dual-protocol control path with explicit `think=False`, and failure diagnostics that
print the raw response (Section 5.3).

**Latency anatomy on a 6 GB laptop GPU.** A 196-second end-to-end query decomposed
into: hidden reasoning tokens on every control call, VRAM contention between the
embedding model (~1.2 GB fp16) and the 4B LLM forcing layer offload to CPU, four
sequential LLM round trips, and a failing web search burning its full timeout.
Resolution: reasoning disabled on control calls, tightened `num_predict`/`num_ctx`,
the fused grader+rewriter call (one fewer round trip on fallback routes),
`keep_alive=30m` to pin the model, fp16 embeddings with tuned batch size, and
streamed generation so the user sees output immediately.

**DuckDuckGo blocked at network level.** Symptom: 0 results on every query, not just
rate-limit spikes. Resolution: provider chain (DuckDuckGo, then optional keyed
Tavily), and the circuit-breaker notices described in Section 3.1 step 7.

**Windows-specific correctness.** ANSI escapes enabled explicitly for terminal
control; the process pool forced to `spawn` (no forked CUDA state); filenames with
`[` `]` sanitized before reaching rich's markup parser; quoted/expanded paths.

**Interrupt safety.** Symptom: a mid-ingest Ctrl+C previously lost all manifest
progress and forced full re-embedding. Resolution: per-file manifest checkpoints,
hash-addressed upserts, partial-write rollback, cooperative worker shutdown.

## 7. Performance Characteristics

Measured on the development machine (i5-13420H, RTX 4050 6 GB, 16 GB RAM, Windows):

- Ingestion: 42 PDF-heavy files -> 38,788 chunks in ~14.5 minutes wall clock
  (process pool x4 parse + fp16 GPU embedding, batch 64), roughly 45 chunks/sec
  sustained.
- Small-corpus sanity run: 1,624 chunks in ~54 seconds.
- Query latency: dominated by local LLM inference (4B model). An AMBIGUOUS-route
  query with a failed web-search attempt measured 85.9 s end-to-end before the
  control-call optimizations; CORRECT routes are strictly cheaper (no rewrite, no
  search, no web refinement). Control calls run at temperature 0 with capped output
  length; generation streams token-by-token.
- Index: HNSW, cosine space, M=16, construction_ef=200, default search_ef=64
  (recall/latency tunable at runtime).

## 8. Setup

### Prerequisites

- Python 3.10 - 3.12
- NVIDIA GPU + current driver (CUDA 12.4 build of torch is pinned in requirements)
- [Ollama](https://ollama.com) installed and running, with a model pulled, e.g.:

```
ollama pull gemma3n:e4b
```

Any Ollama chat model works; set the tag via `CRAG_MODEL`. A smaller model for the
control calls (grading/refinement) via `CRAG_CONTROL_MODEL` cuts query latency
significantly.

### Install

```
python -m venv crag
crag\Scripts\activate
pip install -r requirements.txt
```

Verify GPU torch:

```
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Run

```
python main.py
```

Startup menu: (1) chunk a directory, (2) continue with the existing store,
(3) incremental sync of the last directory (new/changed files only, deleted files
pruned). In-chat commands: `/menu`, `/stats`, `exit`.

Non-interactive:

```
python main.py --ingest-dir D:\corpus
python main.py --sync
python main.py --k 6 --ef 128
python main.py --reset
python main.py --quiet --no-menu
```

### Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CRAG_MODEL` | `gemma4:e4b` | Ollama model for answer generation |
| `CRAG_CONTROL_MODEL` | same as `CRAG_MODEL` | Ollama model for grading/refinement/rewrite (use a smaller model for speed) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama endpoint |
| `CRAG_SEARCH_EF` | `64` | HNSW search depth (16-32 fast, 128-256 max recall) |
| `CRAG_EMBED_BATCH` | `64` | embedding encode batch size (lower to 32 if VRAM is tight) |
| `CRAG_EMBED_DEVICE` | auto (cuda) | force `cpu` or `cuda` for embeddings |
| `TAVILY_API_KEY` | unset | optional keyed web-search fallback (`pip install tavily-python`) |

### CLI Flags

| Flag | Default | Purpose |
|---|---|---|
| `--ingest-dir DIR` | - | chunk a directory before chatting |
| `--sync` | off | incremental re-scan of the last directory |
| `--k N` | 8 | chunks retrieved and graded per query |
| `--ef N` | `CRAG_SEARCH_EF` | HNSW search depth override |
| `--reset` | off | wipe store + manifest and start fresh |
| `--quiet` | off | answer only, no pipeline trace |
| `--no-menu` | off | skip the startup menu |

## 9. Repository Layout

```
.
|-- main.py               orchestration, CLI, pipeline state machine
|-- retrieval.py          ingestion, embedding, vector store, parallel pipeline
|-- evaluator_router.py   grading, routing, refinement, rewriting, web search
|-- generator.py          grounded generation, prompt hardening, streaming
|-- requirements.txt      pinned, GPU-explicit dependency set
|-- README.md
`-- crag_vector_store/    (created at runtime; git-ignored)
```

## 10. Limitations and Future Work

- Thresholds (0.7 / 0.3) are empirical, not learned. The pipeline exposes them as
  constants; a natural next step in the course's direction is optimizing them (and
  the control prompts) with DSPy or TextGrad instead of hand tuning.
- Control calls execute serially (Ollama default `num_parallel`); on GPUs with
  headroom, parallelizing grader/refiner calls is a direct latency win.
- No cross-encoder reranking: ranking is embedding-similarity + LLM grade. Adding a
  small reranker between retrieval and grading is a drop-in extension.
- Web snippets are shallow (title + snippet); full-page fetch and extraction would
  improve INCORRECT-route answer quality.
- Retrieval evaluation is point-in-time: there is no automatic eval harness or recall
  benchmark against a labeled query set.
- ChromaDB is embedded and single-node; the `VectorEngine` boundary is deliberately
  narrow so Milvus/FAISS backends can be swapped via the LangChain vector store
  interface if scale demands it.

## 11. Acknowledgements

The CRAG control loop follows the Self-Corrective RAG formulation (Yan et al., 2024);
implementation choices, degradation protocols, ingestion pipeline, and hardening are
this project's own engineering.


## Notes

- The README describes your code as it stands right now (including the fused grader/rewriter, dual-protocol control calls, dedup fix, and telemetry hardening). If you re-add the thinking-display generator later, add a line to Section 3.1 step 8 — don't let the README drift from the code; graders check that.
- Section 6 is deliberately written as symptom → root cause → resolution. For a course project this is the strongest section you have — it demonstrates you debugged real systems problems (a CVE gate, a packaging trap, a library API break, an inference-configuration failure), not just wrote glue code. Keep it even if you trim elsewhere.
- The performance numbers are the ones actually measured in your sessions on your exact hardware — don't inflate them; "38,788 chunks / 14.5 min on a 6 GB laptop" is more credible than a made-up benchmark.
- If your model tag is genuinely `gemma4:e4b` in your local Ollama, keep it in the README default; if it's actually `gemma3n:e4b`, change the `CRAG_MODEL` default row to match so anyone cloning and pulling the README's model gets a working setup.
