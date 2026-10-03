import argparse
import os
import sys
import time

from evaluator_router import (
    LOWER_THRESHOLD,
    MAX_CONTEXT_CHUNKS,
    WEB_RESULT_LIMIT,
    grade_documents,
    refine_knowledge,
    rewrite_query,
    route_verdict,
    web_search_ddg,
)
from generator import format_sources, generate_answer
from retrieval import PERSIST_DIRECTORY, SEARCH_EF, VectorEngine

TOP_K = 8

BANNER = """
================================================================
  Self-Corrective RAG (CRAG)
  framework   : langchain
  embeddings  : BAAI/bge-m3, gpu fp16
  vector db   : chromadb persistent store, hnsw cosine
  fallback    : duckduckgo web search, optional tavily
  evaluator   : local ollama relevance grading
================================================================
"""

MENU = """----------------------------------------------------------------
  KNOWLEDGE BASE OPTIONS
  1. chunk a directory (recursive .txt .md .pdf)
  2. continue with the existing knowledge base
  3. chunk only new or changed documents from the last directory
----------------------------------------------------------------"""

WEB_FALLBACK_NOTICE = (
    "Live web search is unavailable or rate-limited right now. This answer relies on "
    "the local knowledge base, whose content was only partially relevant, so treat it with care."
)
PARAMETRIC_NOTICE = (
    "Neither the local knowledge base nor live web search returned usable information. "
    "The answer below comes from the model's general knowledge and is unverified."
)


def _log(message):
    print(f"[crag] {message}")


def _enable_ansi():
    if os.name == "nt":
        os.system("")


def _prompt_path(message):
    try:
        return input(message).strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        return ""


def _choose_ingestion(engine):
    print(MENU)
    try:
        choice = input("select an option (1/2/3): ").strip()
    except (EOFError, KeyboardInterrupt):
        return
    if choice == "1":
        path = _prompt_path("enter the directory path to chunk: ")
        if path:
            engine.ingest_directory(path)
        else:
            print("[crag] no directory given, continuing")
    elif choice == "3":
        last = engine.last_directory()
        if not last:
            print("[crag] no previous directory on record")
            last = _prompt_path("enter the directory path to chunk: ")
        if last:
            print(f"[crag] scanning {last}")
            engine.ingest_directory(last)
    else:
        print("[crag] continuing with the existing knowledge base")


def run_crag_pipeline(query, engine, top_k=TOP_K, ef=None, verbose=True):
    started = time.time()
    log = _log if verbose else (lambda message: None)

    chunks = []
    try:
        chunks = engine.retrieve(query, k=top_k)
    except Exception as error:
        log(f"local retrieval failed: {error}")
    documents = [chunk["text"] for chunk in chunks]
    log(f"retrieved {len(documents)} local chunk(s) for grading")

    try:
        grading = grade_documents(query, documents)
    except Exception:
        grading = {"scores": [0.5 for _ in documents], "search_query": None}
    scores = grading.get("scores") or []
    if scores:
        log("chunk scores: " + ", ".join(f"{value:.2f}" for value in scores))

    scored = []
    for chunk, score in zip(chunks, scores):
        if score >= LOWER_THRESHOLD:
            scored.append((score, chunk))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    scored = scored[:MAX_CONTEXT_CHUNKS]
    selected_chunks = [chunk for _score, chunk in scored]
    selected_documents = [chunk["text"] for chunk in selected_chunks]
    dropped = len(documents) - len(selected_chunks)
    log(
        f"chunk selection: kept {len(selected_chunks)}/{len(documents)} "
        f"(dropped {dropped} below {LOWER_THRESHOLD:.1f})"
    )

    verdict = route_verdict(scores)
    log(f"routing verdict: {verdict}")

    context_blocks = []
    sources = []
    notice = None
    local_refined = ""

    if verdict in ("CORRECT", "AMBIGUOUS") and selected_documents:
        local_refined = refine_knowledge(query, selected_documents)
        if local_refined:
            context_blocks.append({"source": "local documents", "text": local_refined})
            seen = set()
            for chunk in selected_chunks:
                key = (chunk["source"], chunk.get("path", ""))
                if key in seen:
                    continue
                seen.add(key)
                sources.append(
                    {"label": chunk["source"], "detail": chunk.get("path") or "local knowledge base"}
                )
        else:
            log("knowledge refinement discarded every local sentence")

    if verdict in ("AMBIGUOUS", "INCORRECT"):
        rewritten = grading.get("search_query") or rewrite_query(query)
        log(f"rewritten web query: {rewritten}")
        try:
            web_results = web_search_ddg(rewritten, max_results=WEB_RESULT_LIMIT)
        except Exception:
            web_results = []
        log(f"web search returned {len(web_results)} result(s)")
        if web_results:
            web_texts = [f"{item['title']}. {item['snippet']}" for item in web_results]
            refined_web = refine_knowledge(query, web_texts)
            if refined_web:
                context_blocks.append({"source": "live web results", "text": refined_web})
            else:
                context_blocks.append(
                    {"source": "live web results (unfiltered)", "text": " ".join(web_texts)}
                )
            for item in web_results:
                sources.append({"label": item["title"] or "web result", "detail": item["url"] or "web search"})
        else:
            log("web search unavailable or empty, engaging circuit breaker")
            if local_refined:
                notice = WEB_FALLBACK_NOTICE
            else:
                notice = PARAMETRIC_NOTICE

    if not context_blocks and not notice:
        notice = PARAMETRIC_NOTICE

    if notice:
        log(f"notice: {notice}")
        if context_blocks:
            context_blocks.append({"source": "system notice", "text": notice})

    log(f"generating answer from {len(context_blocks)} context block(s)")
    print("\nNova: ", end="", flush=True)
    for token in generate_answer(query, context_blocks):
        sys.stdout.write(token)
        sys.stdout.flush()
    print()

    elapsed = time.time() - started
    if verbose:
        print()
        print("[crag] sources:")
        print(format_sources(sources))
        print(f"[crag] finished in {elapsed:.1f}s")

    return {
        "verdict": verdict,
        "scores": scores,
        "selected": len(selected_chunks),
        "sources": sources,
        "notice": notice,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Self-Corrective RAG with intelligent web fallback")
    parser.add_argument("--ingest-dir", dest="ingest_dir", default=None, metavar="DIR",
                        help="chunk a directory recursively before chatting")
    parser.add_argument("--sync", action="store_true",
                        help="chunk only new or changed documents from the last directory")
    parser.add_argument("--k", type=int, default=TOP_K, help="number of chunks to retrieve and grade per query")
    parser.add_argument("--ef", type=int, default=None,
                        help=f"hnsw search depth, default {SEARCH_EF} (16-32 fast, 128-256 max recall)")
    parser.add_argument("--reset", action="store_true",
                        help="wipe the chroma store and the manifest before starting")
    parser.add_argument("--quiet", action="store_true", help="hide pipeline trace and source footer")
    parser.add_argument("--no-menu", action="store_true", help="skip the startup menu")
    args = parser.parse_args()

    _enable_ansi()

    if args.reset:
        manifest_path = os.path.join(PERSIST_DIRECTORY, "manifest.json")
        if os.path.isfile(manifest_path):
            os.remove(manifest_path)
            print("[crag] manifest removed")

    try:
        engine = VectorEngine(pool_kind="process", max_workers=4, drop_old=args.reset)
    except Exception as error:
        print(f"[fatal] could not initialize the vector engine: {error}")
        sys.exit(1)

    if args.ingest_dir:
        engine.ingest_directory(args.ingest_dir)
    elif args.sync:
        last = engine.last_directory()
        if last:
            print(f"[crag] scanning {last}")
            engine.ingest_directory(last)
        else:
            print("[crag] no previous directory on record, nothing to sync")

    print(BANNER)
    print(f"local chunks in store : {engine.chunk_count()}")
    print(f"retrieval             : top-{args.k} chunks, ef {args.ef or SEARCH_EF}")
    if engine.last_directory():
        print(f"last chunked directory: {engine.last_directory()}")
    print("controls              : /menu ingestion options, /stats store stats, exit to quit")

    if not (args.ingest_dir or args.sync or args.no_menu):
        _choose_ingestion(engine)
        print(f"local chunks in store : {engine.chunk_count()}")

    verbose = not args.quiet
    while True:
        try:
            print()
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNova: goodbye.")
            break
        if not query:
            continue
        lowered = query.lower()
        if lowered in {"exit", "quit", "/exit", "/quit", "bye"}:
            print("Nova: goodbye.")
            break
        if lowered == "/menu":
            _choose_ingestion(engine)
            print(f"local chunks in store : {engine.chunk_count()}")
            continue
        if lowered == "/stats":
            print(f"local chunks in store : {engine.chunk_count()}")
            print(f"last chunked directory: {engine.last_directory() or 'none'}")
            continue
        try:
            run_crag_pipeline(query, engine, top_k=args.k, ef=args.ef, verbose=verbose)
        except KeyboardInterrupt:
            print("\n[crag] interrupted")
        except Exception as error:
            print(f"\n[crag] pipeline error: {error}")
            print("Nova: something went wrong while processing that question. Please try rephrasing it.")


if __name__ == "__main__":
    main()