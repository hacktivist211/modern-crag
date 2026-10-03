import hashlib
import json
import os
import sys
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
import time
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)

from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
PERSIST_DIRECTORY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "crag_vector_store"
)
COLLECTION_NAME = "crag_knowledge"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf")
WRITE_BATCH_SIZE = 500
DELETE_BATCH_SIZE = 5000

try:
    SEARCH_EF = max(16, int(os.getenv("CRAG_SEARCH_EF", "64")))
except ValueError:
    SEARCH_EF = 64

try:
    EMBED_BATCH = max(8, int(os.getenv("CRAG_EMBED_BATCH", "64")))
except ValueError:
    EMBED_BATCH = 64

INDEX_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:M": 16,
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": SEARCH_EF,
}

DEFAULT_MAX_WORKERS = max(2, min(8, os.cpu_count() or 2))
PARALLEL_WINDOW_FACTOR = 2

PDF_HEAVY_RATIO = 0.6
PDF_HEAVY_MIN_FILES = 20

_embedding_function = None
_text_splitter = None


class _PlainProgress:
    _DRAW_THROTTLE = 0.2

    def __init__(self):
        self._tasks = {}
        self._interactive = sys.stdout.isatty()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def add_task(self, description, total=1.0, **kwargs):
        task_id = len(self._tasks)
        self._tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
            "last_draw": 0.0,
            "announced": False,
        }
        return task_id

    def update(self, task_id, *, advance=0, description=None, total=None,
               completed=None, visible=None, **kwargs):
        task = self._tasks.get(task_id)
        if task is None:
            return
        if description is not None:
            task["description"] = description
        if total is not None:
            task["total"] = total
        if completed is not None:
            task["completed"] = completed
        if advance:
            task["completed"] += advance
        self._draw(task)

    def advance(self, task_id, advance=1):
        task = self._tasks.get(task_id)
        if task is None:
            return
        task["completed"] += advance
        self._draw(task)

    def _draw(self, task, force=False):
        total_value = task.get("total") or 0
        if not total_value:
            return
        done = int(task["completed"])
        total = int(total_value)
        if not self._interactive:
            if done >= total and not task["announced"]:
                task["announced"] = True
                print(f"[ingest] {task['description']}: {done}/{total}")
            return
        now = time.time()
        finished = done >= total
        if not force and not finished and now - task["last_draw"] < self._DRAW_THROTTLE:
            return
        task["last_draw"] = now
        ratio = min(1.0, done / total)
        bar_width = 30
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(
            f"\r[ingest] {task['description']} [{bar}] {done}/{total}   "
        )
        sys.stdout.flush()
        if finished:
            sys.stdout.write("\n")
            sys.stdout.flush()

    def remove_task(self, task_id):
        task = self._tasks.pop(task_id, None)
        if task and self._interactive:
            total_value = task.get("total") or 0
            if total_value and task["completed"] < total_value:
                sys.stdout.write("\r" + " " * 100 + "\r")
                sys.stdout.flush()

    def start_task(self, task_id):
        pass

    def stop_task(self, task_id):
        pass

    def log(self, message):
        print(f"[ingest] {message}")


def _make_progress():
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
    return _PlainProgress()


def _progress_print(progress, message):
    if RICH_AVAILABLE:
        progress.console.print(message, markup=False, highlight=False)
    else:
        print(f"[ingest] {message}")


def _safe_short_name(basename, limit=48):
    safe = basename.replace("[", "(").replace("]", ")")
    return safe if len(safe) <= limit else safe[: limit - 3] + "..."


class GpuEmbeddings(Embeddings):
    def __init__(self, model_name=EMBEDDING_MODEL_NAME, batch_size=EMBED_BATCH):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self._tick = None
        self._model = None

    def set_progress(self, tick):
        self._tick = tick

    def warmup(self):
        self._load()

    def _load(self):
        if self._model is not None:
            return self._model
        import torch
        from sentence_transformers import SentenceTransformer

        requested = os.getenv("CRAG_EMBED_DEVICE")
        device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
        fp16 = device == "cuda"
        attempts = []
        if fp16:
            attempts.append({"model_kwargs": {"torch_dtype": torch.float16}})
        attempts.append({})
        errors = []
        for extra in attempts:
            for local_only in (True, False):
                kwargs = {"device": device, "local_files_only": local_only}
                kwargs.update(extra)
                try:
                    model = SentenceTransformer(self.model_name, **kwargs)
                except TypeError:
                    if extra:
                        break
                    raise
                except Exception as error:
                    text = str(error)
                    if "torch.load" in text or "32434" in text:
                        raise RuntimeError(
                            "bge-m3 weights are pytorch_model.bin and torch < 2.6 refuses "
                            "to load them (CVE-2025-32434). Fix: pip install --upgrade torch"
                        ) from error
                    errors.append(error)
                    continue
                if fp16 and not extra:
                    try:
                        model = model.half()
                    except Exception:
                        pass
                precision = "fp16" if fp16 else "fp32"
                print(f"[embeddings] {self.model_name} on {device} ({precision}, batch {self.batch_size})")
                self._model = model
                return model
        raise errors[-1] if errors else RuntimeError("embedding model failed to load")

    def embed_documents(self, texts):
        if not texts:
            return []
        model = self._load()
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = model.encode(
                batch,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            vectors.extend(encoded.tolist())
            if self._tick is not None:
                self._tick(len(batch))
        return vectors

    def embed_query(self, text):
        model = self._load()
        encoded = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return encoded.tolist()


def get_embeddings():
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = GpuEmbeddings()
    return _embedding_function


def get_splitter():
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _text_splitter


def _discover_documents(directory):
    found = []
    for root, _directory_names, file_names in os.walk(directory):
        for file_name in file_names:
            if file_name.lower().endswith(SUPPORTED_EXTENSIONS):
                found.append(os.path.abspath(os.path.join(root, file_name)))
    return sorted(found)


def _load_file(path):
    if path.lower().endswith(".pdf"):
        from langchain_community.document_loaders import PyMuPDFLoader
        loader = PyMuPDFLoader(path)
    else:
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(path, autodetect_encoding=True)
    return loader.load()


def _chunk_id(path, text):
    return hashlib.md5(f"{path}::{text}".encode("utf-8")).hexdigest()


def _check_text_encoding_support():
    try:
        import chardet  # noqa: F401
        return
    except ImportError:
        pass
    print(
        "[ingest] warning: chardet is not installed. Non-UTF-8 .txt/.md files "
        "may be misdecoded. Install with: pip install chardet"
    )


def _process_file_worker(path, manifest_snapshot, splitter):
    basename = os.path.basename(path)
    try:
        statistics = os.stat(path)
    except OSError as error:
        return {"path": path, "basename": basename, "status": "unreadable", "error": str(error)}

    entry = manifest_snapshot.get(path)
    if (
        entry
        and entry.get("mtime") == statistics.st_mtime
        and entry.get("size") == statistics.st_size
    ):
        return {"path": path, "basename": basename, "status": "unchanged"}

    try:
        pages = _load_file(path)
    except Exception as error:
        return {"path": path, "basename": basename, "status": "failed", "error": str(error)}

    chunks = splitter.split_documents(pages)
    texts, metadatas, ids = [], [], []
    seen_ids = set()
    for chunk in chunks:
        text = chunk.page_content.strip()
        if not text:
            continue
        chunk_id = _chunk_id(path, text)
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        page = chunk.metadata.get("page")
        texts.append(text)
        metadatas.append(
            {
                "chunk_id": chunk_id,
                "source": basename,
                "path": path,
                "page": page if isinstance(page, int) else -1,
            }
        )
        ids.append(chunk_id)
    if not texts:
        return {"path": path, "basename": basename, "status": "empty"}

    return {
        "path": path,
        "basename": basename,
        "status": "ready",
        "mtime": statistics.st_mtime,
        "size": statistics.st_size,
        "texts": texts,
        "metadatas": metadatas,
        "ids": ids,
        "had_previous": entry is not None,
    }


class VectorEngine:
    def __init__(
        self,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME,
        max_workers=None,
        pool_kind="thread",
        drop_old=False,
    ):
        os.makedirs(persist_directory, exist_ok=True)
        self.persist_directory = persist_directory
        self.manifest_path = os.path.join(persist_directory, "manifest.json")
        self.max_workers = max(1, int(max_workers or DEFAULT_MAX_WORKERS))
        self.pool_kind = pool_kind if pool_kind in ("thread", "process") else "thread"
        self.embeddings = get_embeddings()
        self.store = self._create_store(collection_name, persist_directory, drop_old)
        self.splitter = get_splitter()

    def _create_store(self, collection_name, persist_directory, drop_old):
        import logging
        import shutil
        from chromadb.config import Settings
        from langchain_chroma import Chroma

        logging.getLogger("chromadb").setLevel(logging.ERROR)

        if drop_old and os.path.isdir(persist_directory):
            shutil.rmtree(persist_directory, ignore_errors=True)
            os.makedirs(persist_directory, exist_ok=True)
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory,
            collection_metadata=dict(INDEX_METADATA),
            client_settings=Settings(anonymized_telemetry=False),
        )
    def _apply_search_ef(self, ef):
        try:
            ef = max(16, int(ef))
            collection = self.store._collection
            current = dict(collection.metadata or {})
            if current.get("hnsw:search_ef") == ef:
                return
            current["hnsw:search_ef"] = ef
            collection.modify(metadata=current)
        except Exception:
            pass

    def _load_manifest(self):
        if os.path.isfile(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and isinstance(data.get("files"), dict):
                    return data
            except Exception:
                pass
        return {"last_directory": None, "files": {}}

    def _save_manifest(self, manifest):
        temporary_path = self.manifest_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        os.replace(temporary_path, self.manifest_path)

    def _delete_ids(self, ids):
        ids = [identifier for identifier in ids if identifier]
        if not ids:
            return
        for start in range(0, len(ids), DELETE_BATCH_SIZE):
            try:
                self.store.delete(ids=ids[start:start + DELETE_BATCH_SIZE])
            except Exception:
                pass

    def last_directory(self):
        return self._load_manifest().get("last_directory")

    def chunk_count(self):
        try:
            return int(self.store._collection.count())
        except Exception:
            manifest = self._load_manifest()
            return sum(
                len(entry.get("ids", [])) for entry in manifest["files"].values()
            )

    @staticmethod
    def _looks_pdf_heavy(files):
        if len(files) < PDF_HEAVY_MIN_FILES:
            return False
        pdfs = sum(1 for p in files if p.lower().endswith(".pdf"))
        return pdfs / len(files) >= PDF_HEAVY_RATIO

    def ingest_directory(self, directory):
        directory = os.path.abspath(
            os.path.expanduser(str(directory).strip().strip('"').strip("'"))
        )
        if not os.path.isdir(directory):
            print(f"[ingest] not a directory: {directory}")
            return {"added": 0, "updated": 0, "skipped": 0, "removed": 0, "total": 0}

        _check_text_encoding_support()

        manifest = self._load_manifest()
        files = _discover_documents(directory)
        existing = manifest["files"]
        prefix = directory + os.sep

        removed = 0
        for path in list(existing.keys()):
            if path.startswith(prefix) and not os.path.isfile(path):
                self._delete_ids(existing[path].get("ids", []))
                existing.pop(path)
                removed += 1

        if not files:
            manifest["last_directory"] = directory
            self._save_manifest(manifest)
            print(f"[ingest] no supported documents found under {directory}")
            return {"added": 0, "updated": 0, "skipped": 0, "removed": removed, "total": 0}

        if self.pool_kind == "thread" and self._looks_pdf_heavy(files):
            print(
                "[ingest] note: this corpus looks PDF-heavy. ThreadPoolExecutor "
                "is still in use, but PDF parsing holds the GIL. Pass "
                'VectorEngine(pool_kind="process") to fully use multiple cores.'
            )

        manifest["last_directory"] = directory
        manifest_snapshot = {
            path: {"mtime": existing[path].get("mtime"), "size": existing[path].get("size")}
            for path in files
            if path in existing
        }

        added = updated = skipped = 0
        interrupted = False
        window = max(1, self.max_workers * PARALLEL_WINDOW_FACTOR)

        executor_kwargs = {"max_workers": self.max_workers}
        if self.pool_kind == "process":
            import multiprocessing
            executor_kwargs["mp_context"] = multiprocessing.get_context("spawn")
        else:
            executor_kwargs["thread_name_prefix"] = "crag-ingest"
        executor_cls = (
            ProcessPoolExecutor if self.pool_kind == "process" else ThreadPoolExecutor
        )

        print("[embeddings] warming up the model (first run also builds the CUDA kernels)...")
        try:
            self.embeddings.warmup()
        except Exception as error:
            print(f"[embeddings] warmup warning: {error}")

        try:
            with _make_progress() as progress:
                files_task = progress.add_task("files", total=len(files))

                with executor_cls(**executor_kwargs) as pool:
                    futures = {}
                    iterator = iter(files)

                    def pump():
                        while len(futures) < window:
                            try:
                                next_path = next(iterator)
                            except StopIteration:
                                return
                            future = pool.submit(
                                _process_file_worker,
                                next_path,
                                manifest_snapshot,
                                self.splitter,
                            )
                            futures[future] = next_path

                    pump()
                    while futures:
                        done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
                        for future in done:
                            path = futures.pop(future, None)
                            if path is None:
                                continue
                            try:
                                result = future.result()
                            except Exception as error:
                                progress.advance(files_task)
                                _progress_print(
                                    progress,
                                    f"worker crashed for {os.path.basename(path)}: {error}",
                                )
                                continue

                            status = result["status"]
                            basename = result["basename"]
                            short_name = _safe_short_name(basename)

                            if status == "unchanged":
                                skipped += 1
                                progress.advance(files_task)
                                progress.update(
                                    files_task, description=f"unchanged: {short_name}"
                                )
                                continue

                            if status == "unreadable":
                                _progress_print(
                                    progress, f"unreadable file skipped: {basename}"
                                )
                                progress.advance(files_task)
                                continue

                            if status == "failed":
                                _progress_print(
                                    progress,
                                    f"failed to read {basename}: {result['error']}",
                                )
                                progress.advance(files_task)
                                continue

                            if status == "empty":
                                entry = existing.get(path)
                                if entry:
                                    self._delete_ids(entry.get("ids", []))
                                    existing.pop(path, None)
                                    self._save_manifest(manifest)
                                progress.advance(files_task)
                                progress.update(
                                    files_task, description=f"empty: {short_name}"
                                )
                                continue

                            texts = result["texts"]
                            metadatas = result["metadatas"]
                            ids = result["ids"]
                            had_previous = result["had_previous"]

                            if had_previous:
                                self._delete_ids(existing[path].get("ids", []))
                            self._delete_ids(ids)

                            progress.update(
                                files_task,
                                description=f"embedding: {short_name} ({len(texts)} chunks)",
                            )
                            embed_task = progress.add_task(
                                f"embedding: {short_name} ({len(texts)} chunks)",
                                total=len(texts),
                            )

                            def tick(count, task=embed_task):
                                progress.advance(task, count)

                            self.embeddings.set_progress(tick)
                            failed = False
                            written = []
                            try:
                                for start in range(0, len(texts), WRITE_BATCH_SIZE):
                                    end = start + WRITE_BATCH_SIZE
                                    self.store.add_texts(
                                        texts=texts[start:end],
                                        metadatas=metadatas[start:end],
                                        ids=ids[start:end],
                                    )
                                    written.extend(ids[start:end])
                            except Exception as error:
                                failed = True
                                self._delete_ids(written)
                                _progress_print(
                                    progress,
                                    f"embedding failed for {basename}: {error}",
                                )
                            finally:
                                self.embeddings.set_progress(None)
                                progress.remove_task(embed_task)

                            if failed:
                                existing.pop(path, None)
                                progress.advance(files_task)
                                continue

                            existing[path] = {
                                "mtime": result["mtime"],
                                "size": result["size"],
                                "ids": ids,
                            }
                            self._save_manifest(manifest)
                            if had_previous:
                                updated += 1
                            else:
                                added += 1
                            progress.advance(files_task)
                            progress.update(
                                files_task,
                                description=f"stored: {short_name} ({len(texts)} chunks)",
                            )

                        pump()
        except KeyboardInterrupt:
            interrupted = True
            print("\n[ingest] interrupted, progress on completed files was saved")
        except Exception as error:
            interrupted = True
            print(f"\n[ingest] ingestion stopped by an error: {error}")

        self._save_manifest(manifest)

        status = " (interrupted)" if interrupted else ""
        print(
            f"[ingest] {directory}{status}\n"
            f"[ingest] pool          : {self.pool_kind} x {self.max_workers}\n"
            f"[ingest] files scanned : {len(files)}\n"
            f"[ingest] added         : {added}\n"
            f"[ingest] updated       : {updated}\n"
            f"[ingest] unchanged     : {skipped}\n"
            f"[ingest] removed       : {removed}\n"
            f"[ingest] store total   : {self.chunk_count()} chunks"
        )
        return {
            "added": added,
            "updated": updated,
            "skipped": skipped,
            "removed": removed,
            "total": len(files),
        }

    def retrieve(self, query, k=4, ef=None):
        if ef is not None:
            self._apply_search_ef(ef)
        results = self.store.similarity_search_with_score(query, k=k)
        chunks = []
        for document, distance in results:
            chunks.append(
                {
                    "text": document.page_content,
                    "source": document.metadata.get("source", "local document"),
                    "path": document.metadata.get("path", ""),
                    "distance": distance,
                }
            )
        return chunks