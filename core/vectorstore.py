import os
import platform
import logging
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Union, Sequence, Any

# system utils
try:
    import psutil
except Exception:
    psutil = None  # optional; we'll fallback gracefully

# LangChain and Core Imports
from langchain.schema import Document as LcDocument, BaseRetriever
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# External Libraries (assume installed in environment)
from pydantic import PrivateAttr, ConfigDict, BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# NOTE: Since flashrank requires torch, we put the import inside the function to avoid errors 
# if the environment does not have it, but for a standard RAG environment, it should be here.
from flashrank import Ranker 

import chromadb
from chromadb.config import Settings

# Configure chromadb telemetry if available
try:
    chromadb.configure(anonymized_telemetry=False)
except Exception:
    try:
        chromadb.settings = Settings(anonymized_telemetry=False)
    except Exception:
        pass

# -------------------- CONFIG --------------------
INITIAL_TOP_K = 15
FINAL_K_RERANKED = 7
VECTORSTORE_DIR = "vectorstore"

# Safety: don't spawn too many processes by default
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "2"))

# Env override to force mode: "thread" or "process"
ENV_FORCE_MODE = os.getenv("VECTOR_MODE", "").lower()  # "thread", "process", or ""

# Logging
logger = logging.getLogger(__name__)
# Assume logging is configured externally, if not, use basicConfig
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') 

# Global caches (per process)
_CACHED_RANKER = None
_CACHED_EMBEDDINGS = None
_EMBED_LOCK = threading.Lock()
_MPS_WARNING_SHOWN = False

# Flashrank cache dir
CUSTOM_CACHE_DIR = os.path.expanduser("~/.hf_cache_dir/flashrank_models")


# -------------------- Helper: detect environment & device --------------------
def _detect_system():
    """Return dict with cpu_count and total_ram_gb (may be None if psutil missing)."""
    cpu_count = os.cpu_count() or 4
    total_ram_gb = None
    if psutil:
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            total_ram_gb = None
    return {"cpu_count": cpu_count, "total_ram_gb": total_ram_gb, "platform": platform.system().lower()}


def _detect_torch_device():
    """Return best device string for HuggingFaceEmbeddings: 'cuda'|'mps'|'cpu' when available."""
    # avoid importing torch at top-level if not installed; check safely
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # mac mps support: torch.backends.mps.is_available() may exist
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# -------------------- Flashrank & Embedding preload --------------------
def preload_reranker_model(model_name: str = "ms-marco-MiniLM-L-12-v2"):
    """
    Preload Flashrank Ranker instance and keep it in module-level cache.
    This cache is per-process. Child processes will need to call this again.
    """
    global _CACHED_RANKER
    if _CACHED_RANKER is not None:
        return _CACHED_RANKER
    try:
        logger.info(f"📦 Preloading Flashrank model '{model_name}' (cache dir: {CUSTOM_CACHE_DIR})")
        _CACHED_RANKER = Ranker(model_name=model_name, cache_dir=CUSTOM_CACHE_DIR)
        logger.info(f"✅ Flashrank model '{model_name}' loaded")
        return _CACHED_RANKER
    except Exception as e:
        logger.warning(f"⚠️ Failed to preload Flashrank model '{model_name}': {e}")
        _CACHED_RANKER = None
        return None


def get_hf_embeddings(device_hint: Optional[str] = None):
    """
    Return a HuggingFaceEmbeddings instance (cached per process).
    device_hint can be 'cuda'|'mps'|'cpu' or None to auto-detect.
    Note: per-process caching — threads will share this instance within same process.
    """
    global _CACHED_EMBEDDINGS, _MPS_WARNING_SHOWN
    device = device_hint or _detect_torch_device()

    # Safety: if MPS is detected and we're in a multi-process environment, prefer CPU
    # because MPS + multiprocessing is fragile on macOS.
    # We choose CPU if VECTOR_MODE forced to 'process' or if platform is darwin and using processes
    sys_info = _detect_system()
    force_mode = ENV_FORCE_MODE
    using_process = (force_mode == "process") or (sys_info["cpu_count"] >= 8 and (sys_info["total_ram_gb"] or 0) >= 16)

    if device == "mps" and using_process and not _MPS_WARNING_SHOWN:
        logger.warning("⚠️ Detected MPS but running in process-parallel mode: forcing device -> cpu to avoid MPS multi-process failures")
        _MPS_WARNING_SHOWN = True
        device = "cpu"

    # allow env override to disable GPU/MPS: VECTOR_DISABLE_ACCEL=1
    if os.getenv("VECTOR_DISABLE_ACCEL", "").lower() in ("1", "true", "yes"):
        device = "cpu"

    if _CACHED_EMBEDDINGS is None:
        with _EMBED_LOCK:
            if _CACHED_EMBEDDINGS is None:
                try:
                    model_name = "sentence-transformers/all-MiniLM-L6-v2"
                    logger.info(f"📦 Creating HuggingFaceEmbeddings (model={model_name}, device={device})")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": device})
                except Exception as e:
                    # fallback to CPU if any issue
                    logger.warning(f"⚠️ Failed to create embeddings on device={device}: {e}. Falling back to CPU.")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return _CACHED_EMBEDDINGS


# -------------------- Flashrank Pydantic Request --------------------
class FlashrankRequest(BaseModel):
    query: str
    passages: list[dict[str, Any]]
    top_n: int


# -------------------- Custom Compressor using Flashrank --------------------
class CustomFlashrankCompressor(BaseDocumentCompressor):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ranker: Ranker
    top_n: int = FINAL_K_RERANKED

    def compress_documents(self, documents: Sequence[LcDocument], query: str, **kwargs) -> Sequence[LcDocument]:
        if not documents:
            return []

        # Prepare passages
        doc_list_for_rerank = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(documents)]
        run_input = FlashrankRequest(query=query, passages=doc_list_for_rerank, top_n=self.top_n)

        try:
            ranked_results = self.ranker.rerank(run_input)
        except Exception as e:
            logger.warning(f"⚠️ Flashrank.rerank failed: {e}. Returning original docs.")
            ranked_results = [{"id": i, "score": 0.0} for i in range(len(doc_list_for_rerank))]

        reranked_docs = []
        for res in ranked_results:
            idx = res.get("id", 0)
            score = res.get("score", 0.0)
            original_doc = documents[idx]
            reranked_docs.append(LcDocument(page_content=original_doc.page_content, metadata={**original_doc.metadata, "relevance_score": score}))
        return reranked_docs


# -------------------- Vectorstore helpers --------------------
def get_vectorstore_path(doc_type: Optional[str] = None):
    if doc_type:
        path = os.path.join(VECTORSTORE_DIR, doc_type)
        os.makedirs(path, exist_ok=True)
        return path
    return VECTORSTORE_DIR


def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> List[str]:
    target_path = os.path.join(base_path, doc_type) if doc_type else base_path
    if not os.path.exists(target_path):
        return []
    if doc_type:
        full_path = os.path.join(base_path, doc_type)
        if os.path.isdir(full_path):
            return [f for f in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, f))]
        return []
    return [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]


def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> bool:
    if doc_type:
        path = os.path.join(base_path, doc_type, doc_id)
    else:
        path = os.path.join(base_path, doc_id)
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "chroma.sqlite3")):
        return True
    return False


def save_to_vectorstore(doc_id: str, texts: List[str], metadatas: Optional[Union[dict, List[dict]]] = None, doc_type: str = "document", base_path: str = VECTORSTORE_DIR):
    """
    Save texts to Chroma vectorstore on disk.
    """
    docs = []
    if isinstance(metadatas, list):
        if len(texts) != len(metadatas):
            raise ValueError("When providing a list of metadatas, its length must match texts.")
        for i in range(len(texts)):
            docs.append(LcDocument(page_content=texts[i], metadata={**metadatas[i], "chunk_index": i + 1}))
    else:
        metadata_dict = metadatas or {}
        for i, t in enumerate(texts):
            docs.append(LcDocument(page_content=t, metadata={**metadata_dict, "source": doc_id, "chunk_index": i + 1}))

    embeddings = get_hf_embeddings()
    doc_dir = os.path.join(base_path, doc_type, doc_id)
    os.makedirs(doc_dir, exist_ok=True)
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=doc_dir)
    logger.info(f"📄 Saved {len(docs)} chunks for doc_id={doc_id} into {doc_dir}")
    return vectordb


# -------------------- Load single vectorstore retriever --------------------
def load_vectorstore(doc_id: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, doc_types: Union[list, str] = "document", base_path: str = VECTORSTORE_DIR):
    # ensure doc_types list
    if isinstance(doc_types, str):
        doc_types = [doc_types]

    embeddings = get_hf_embeddings()
    retriever = None

    # preload reranker model for main/threads
    reranker_instance = preload_reranker_model()

    for dtype in doc_types:
        path = os.path.join(base_path, dtype, doc_id)
        if os.path.isdir(path) and vectorstore_exists(doc_id, base_path, dtype):
            base_retriever = Chroma(persist_directory=path, embedding_function=embeddings).as_retriever(search_kwargs={"k": top_k})
            if reranker_instance:
                try:
                    compressor = CustomFlashrankCompressor(ranker=reranker_instance, top_n=final_k)
                    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
                    # print only in main process to avoid noisy child logs
                    # We check if the current process is the main process (not a child process)
                    if multiprocessing.current_process().name == 'MainProcess': 
                        logger.info(f"✅ Loaded Reranking Retriever for doc_id={doc_id} (doc_type={dtype}) with k={top_k}->{final_k}")
                except Exception as e:
                    logger.warning(f"⚠️ CustomFlashrankCompressor failed for {doc_id}: {e}. Falling back to base retriever.")
                    retriever = base_retriever
            else:
                logger.warning("⚙️ Reranker model not available. Using base retriever only.")
                retriever = base_retriever
            break

    if retriever is None:
        raise ValueError(f"❌ Vectorstore for doc_id '{doc_id}' not found in any of {doc_types}")
    return retriever


# -------------------- MultiDoc / Parallel Retriever --------------------
class NamedRetriever(BaseModel):
    """Picklable wrapper storing minimal params to load retriever inside child process."""
    doc_id: str
    doc_type: str
    top_k: int
    final_k: int

    def load_instance(self) -> BaseRetriever:
        """Load a retriever instance inside the current process using stored params."""
        return load_vectorstore(doc_id=self.doc_id, top_k=self.top_k, final_k=self.final_k, doc_types=[self.doc_type])


class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple NamedRetriever sources. Choose executor automatically (thread vs process).
    """

    _retrievers_list: list[NamedRetriever] = PrivateAttr()
    _k_per_doc: int = PrivateAttr()

    def __init__(self, retrievers_list: list[NamedRetriever], k_per_doc: int = INITIAL_TOP_K):
        super().__init__()
        self._retrievers_list = retrievers_list
        self._k_per_doc = k_per_doc

    @staticmethod
    def _static_retrieve_task(named_r: "NamedRetriever", query: str):
        """
        Static helper used in ProcessPoolExecutor. Executes inside child process.
        Must return list[Document] or None.
        """
        try:
            retriever_instance = named_r.load_instance()
            return retriever_instance.invoke(query)
        except Exception as e:
            # child process: print minimal info
            print(f"❌ Child retrieval error for {named_r.doc_id} ({named_r.doc_type}): {e}")
            return None

    def _thread_retrieve_task(self, named_r: "NamedRetriever", query: str):
        """
        Retrieval performed in a thread inside the same process (no pickling).
        We call named_r.load_instance() which will reuse cached embeddings/reranker in this process.
        """
        try:
            retriever_instance = named_r.load_instance()
            return retriever_instance.invoke(query)
        except Exception as e:
            logger.warning(f"⚠️ Thread retrieval error for {named_r.doc_id}: {e}")
            return None

    def _choose_executor(self):
        """
        Decide whether to use ProcessPoolExecutor or ThreadPoolExecutor.
        Logic:
          - If ENV_FORCE_MODE set -> obey it.
          - If platform is darwin (mac) and GPU device is mps -> prefer thread (MPS multi-process is fragile).
          - If total RAM is low (<12GB) -> prefer thread.
          - Else if CPU cores >= 8 and RAM >= 16GB -> prefer process.
        """
        sys_info = _detect_system()
        device = _detect_torch_device()
        force = ENV_FORCE_MODE

        # Force mode if user set
        if force in ("thread", "process"):
            mode = force
            logger.info(f"VECTOR_MODE override: forcing '{mode}' executor")
            return ("process" if mode == "process" else "thread")

        # prefer thread on macOS if MPS to avoid MPS multiprocessing issues
        if sys_info["platform"] == "darwin" and device == "mps":
            logger.info("Detected macOS + MPS -> choosing 'thread' executor to avoid multi-process MPS issues")
            return "thread"

        # prefer thread if RAM too small or explicit low-resource machine
        if sys_info["total_ram_gb"] is not None and sys_info["total_ram_gb"] < 12:
            logger.info(f"Low RAM ({sys_info['total_ram_gb']:.1f}GB) detected -> choosing 'thread' executor")
            return "thread"

        # otherwise prefer process on beefy machines
        if sys_info["cpu_count"] >= 8 and (sys_info["total_ram_gb"] or 0) >= 16:
            logger.info("High-resources machine detected -> choosing 'process' executor")
            return "process"

        # fallback to thread
        logger.info("Defaulting to 'thread' executor")
        return "thread"

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        """
        This is the LangChain compatible retrieval method.
        It will call each NamedRetriever either in parallel (process/thread) and then deduplicate results.
        """
        # Determine worker count
        max_workers = min(len(self._retrievers_list), MAX_PARALLEL_WORKERS)
        if max_workers <= 0:
            max_workers = 1

        chosen = self._choose_executor()
        ExecutorClass = ProcessPoolExecutor if chosen == "process" else ThreadPoolExecutor

        logger.info(f"⚙️ Running MultiDocRetriever with {chosen} executor ({max_workers} workers)")

        results = []
        if chosen == "process":
            # Use process pool: tasks must be picklable (we call static method)
            with ExecutorClass(max_workers=max_workers) as executor:
                futures = [executor.submit(MultiDocRetriever._static_retrieve_task, nr, query) for nr in self._retrievers_list]
                for f in futures:
                    try:
                        results.append(f.result())
                    except Exception as e:
                        logger.warning(f"Child process future failed: {e}")
                        results.append(None)
        else:
            # Use threads: we can run load_instance in-thread (reuses per-process caches)
            with ExecutorClass(max_workers=max_workers) as executor:
                futures = [executor.submit(self._thread_retrieve_task, nr, query) for nr in self._retrievers_list]
                for f in futures:
                    try:
                        results.append(f.result())
                    except Exception as e:
                        logger.warning(f"Thread future failed: {e}")
                        results.append(None)

        # Combine results and deduplicate
        seen = set()
        unique_docs = []
        for dlist, named_r in zip(results, self._retrievers_list):
            if not dlist:
                continue
            for d in dlist:
                # dedupe key: source + chunk + doc_id + a snippet
                src = d.metadata.get("source") or d.metadata.get("doc_source") or named_r.doc_id
                chunk = d.metadata.get("chunk_index") or d.metadata.get("chunk") or ""
                key = f"{src}_{chunk}_{named_r.doc_type}_{d.page_content[:120]}"
                if key not in seen:
                    seen.add(key)
                    d.metadata["doc_type"] = named_r.doc_type
                    d.metadata["doc_id"] = named_r.doc_id
                    d.metadata["doc_source"] = src
                    unique_docs.append(d)

        logger.info(f"📝 Query='{query[:80]}...' found {len(unique_docs)} unique docs across sources (Executor={chosen})")
        # debug: print reranked items
        for d in unique_docs:
            if "relevance_score" in d.metadata:
                score = d.metadata.get("relevance_score")
                logger.debug(f" - [Reranked] Source={d.metadata.get('doc_source')}, Score={score:.4f}, Type={d.metadata.get('doc_type')}, Content='{d.page_content[:80]}...'")
        return unique_docs


# -------------------- Load multiple vectorstores --------------------
def load_all_vectorstores(doc_ids: Optional[Union[List[str], str]] = None, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, doc_type: Optional[Union[str, List[str]]] = None, base_path: str = VECTORSTORE_DIR) -> MultiDocRetriever:
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]
    if doc_type is None:
        doc_types = ["evidence"]
    elif isinstance(doc_type, str):
        doc_types = [doc_type]
    else:
        doc_types = doc_type

    all_retrievers: List[NamedRetriever] = []
    for dt in doc_types:
        folders = list_vectorstore_folders(base_path=base_path, doc_type=dt)
        for folder in folders:
            if doc_ids and folder not in doc_ids:
                continue
            try:
                # Ensure vectorstore exists by attempting to load (this also preloads reranker in-process)
                load_vectorstore(folder, top_k=top_k, final_k=final_k, doc_types=[dt], base_path=base_path)
                nr = NamedRetriever(doc_id=folder, doc_type=dt, top_k=top_k, final_k=final_k)
                all_retrievers.append(nr)
            except ValueError:
                continue
            except Exception as e:
                logger.warning(f"⚠️ Skipping folder '{folder}' ({dt}): {e}")

    if not all_retrievers:
        raise ValueError(f"No vectorstores found for doc_ids={doc_ids} in doc_types={doc_types}")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)

# -------------------- VECTORSTORE EXECUTOR SINGLETON --------------------
# 🚨 REQUIRED by ingest_batch.py for shared resource management.
class VectorStoreExecutorSingleton:
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreExecutorSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not VectorStoreExecutorSingleton._is_initialized:
            # Use MAX_PARALLEL_WORKERS defined in the config section
            self.max_workers = MAX_PARALLEL_WORKERS
            # We use ThreadPoolExecutor here as it's safer for resource sharing in batch ingestion
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            logger.info(f"Initialized VectorStoreExecutorSingleton (ThreadPoolExecutor with {self.max_workers} workers) for batch tasks.")
            VectorStoreExecutorSingleton._is_initialized = True

    @property
    def executor(self) -> ThreadPoolExecutor:
        """Returns the shared ThreadPoolExecutor."""
        return self._executor

    def close(self):
        """Shutdown the executor when the application is done."""
        if self._is_initialized:
            logger.info("Shutting down VectorStoreExecutorSingleton ThreadPoolExecutor...")
            self._executor.shutdown(wait=True)
            VectorStoreExecutorSingleton._is_initialized = False 

def get_vectorstore() -> VectorStoreExecutorSingleton:
    """
    REQUIRED by ingest_batch.py. Returns the singleton instance managing the executor.
    """
    return VectorStoreExecutorSingleton()
