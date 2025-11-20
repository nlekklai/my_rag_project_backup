# core/vectorstore.py
import os
import platform
import logging
import threading
import multiprocessing
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Union, Sequence, Any, Dict, Set, Tuple

# system utils
try:
    import psutil
except ImportError:
    psutil = None

# LangChain-ish imports (adjust to your project's versions)
from langchain_core.documents import Document as LcDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import BaseDocumentCompressor

# Pydantic helpers
from pydantic import PrivateAttr, ConfigDict, BaseModel

# Chroma / HF embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Try import CrossEncoder (sentence-transformers)
try:
    from sentence_transformers import CrossEncoder
    _HAS_SENT_TRANS = True
except Exception:
    CrossEncoder = None
    _HAS_SENT_TRANS = False
    logger.warning("⚠️ sentence-transformers CrossEncoder not available. Reranker will be disabled.")

# Configure chromadb telemetry if available
try:
    chromadb.configure(anonymized_telemetry=False)
except Exception:
    try:
        chromadb.settings = Settings(anonymized_telemetry=False)
    except Exception:
        pass

# -------------------- Global Config --------------------
from config.global_vars import (
    VECTORSTORE_DIR,
    MAPPING_FILE_PATH,
    FINAL_K_RERANKED,
    INITIAL_TOP_K,
    EVIDENCE_DOC_TYPES,
    MAX_PARALLEL_WORKERS,
)

# -------------------- Vectorstore Constants --------------------
ENV_FORCE_MODE = os.getenv("VECTOR_MODE", "").lower()  # "thread", "process", or ""
ENV_DISABLE_ACCEL = os.getenv("VECTOR_DISABLE_ACCEL", "").lower() in ("1", "true", "yes")

# Global caches (per process)
_CACHED_EMBEDDINGS = None
_EMBED_LOCK = threading.Lock()
_MPS_WARNING_SHOWN = False

# -------------------- Helper: detect environment & device --------------------
def _detect_system():
    cpu_count = os.cpu_count() or 4
    total_ram_gb = None
    if psutil:
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            total_ram_gb = None
    return {"cpu_count": cpu_count, "total_ram_gb": total_ram_gb, "platform": platform.system().lower()}

def _detect_torch_device():
    try:
        import torch
        if ENV_DISABLE_ACCEL:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if platform.system().lower() == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS can be fragile for multiprocessing; we'll warn and possibly force cpu later
            return "mps"
    except Exception:
        pass
    return "cpu"

def get_hf_embeddings(device_hint: Optional[str] = None):
    """
    Return a HuggingFaceEmbeddings instance (cached per process).
    """
    global _CACHED_EMBEDDINGS, _MPS_WARNING_SHOWN
    device = device_hint or _detect_torch_device()

    sys_info = _detect_system()
    force_mode = ENV_FORCE_MODE

    using_process = (force_mode == "process") or (sys_info["cpu_count"] >= 8 and (sys_info["total_ram_gb"] or 0) >= 16)

    if device == "mps" and using_process and not _MPS_WARNING_SHOWN:
        logger.warning("⚠️ Detected MPS but running in process-parallel mode: forcing device -> cpu to avoid MPS multi-process failures")
        _MPS_WARNING_SHOWN = True
        device = "cpu"

    if ENV_DISABLE_ACCEL:
        device = "cpu"

    if _CACHED_EMBEDDINGS is None:
        with _EMBED_LOCK:
            if _CACHED_EMBEDDINGS is None:
                try:
                    model_name = "intfloat/multilingual-e5-large"
                    logger.info(f"📦 Creating HuggingFaceEmbeddings (model={model_name}, device={device})")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": device}
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed to create embeddings on device={device}: {e}. Falling back to all-MiniLM-L6-v2 on cpu.")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return _CACHED_EMBEDDINGS

# -------------------- Vectorstore helpers --------------------
def _get_collection_name(doc_type: str, enabler: Optional[str] = None) -> str:
    doc_type_norm = doc_type.strip().lower()
    if doc_type_norm == EVIDENCE_DOC_TYPES:
        enabler_norm = (enabler or "km").strip().lower()
        collection_name = f"{doc_type_norm}_{enabler_norm}"
    else:
        collection_name = doc_type_norm
    logger.critical(f"🧭 DEBUG: _get_collection_name(doc_type={doc_type}, enabler={enabler}) => {collection_name}")
    return collection_name

def get_vectorstore_path(doc_type: Optional[str] = None, enabler: Optional[str] = None) -> str:
    if not doc_type:
        return VECTORSTORE_DIR
    collection_name = _get_collection_name(doc_type, enabler)
    return os.path.join(VECTORSTORE_DIR, collection_name)

def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None, enabler: Optional[str] = None) -> List[str]:
    if not os.path.exists(base_path):
        return []
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    if doc_type:
        doc_type_norm = doc_type.lower().strip()
        if doc_type_norm == EVIDENCE_DOC_TYPES and not enabler:
            return [f for f in folders if f.startswith(f"{EVIDENCE_DOC_TYPES}_")]
        collection_name = _get_collection_name(doc_type_norm, enabler)
        return [collection_name] if collection_name in folders else []
    return folders

def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None, enabler: Optional[str] = None) -> bool:
    if not doc_type:
        return False
    collection_name = _get_collection_name(doc_type, enabler)
    path = os.path.join(base_path, collection_name)
    file_path = os.path.join(path, "chroma.sqlite3")
    if not os.path.isdir(path):
        logger.warning(f"❌ V-Exists Check 1: Directory not found for collection '{collection_name}' at {path}")
        return False
    if os.path.isfile(file_path):
        return True
    logger.error(f"❌ V-Exists Check 3: FAILED to find file chroma.sqlite3 at {file_path} for collection '{collection_name}'")
    return False

# =================================================================
# HuggingFace Cross-Encoder Reranker wrapper (singleton)
# =================================================================
class HuggingFaceCrossEncoderCompressor(BaseDocumentCompressor, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # default cross-encoder recommended model
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cpu"
    rerank_max_length: int = 512
    _cross_encoder: Any = PrivateAttr(None)

    def __init__(self, **data):
        super().__init__(**data)
        # determine device safely (force cpu for CrossEncoder on mac mps)
        try:
            device = _detect_torch_device()
            if device == "mps":
                # CrossEncoder on MPS can be unstable; force CPU
                device = "cpu"
            self.rerank_device = device
        except Exception:
            self.rerank_device = "cpu"

    def set_encoder_instance(self, encoder: Any):
        self._cross_encoder = encoder

    def compress_documents(self, documents: Sequence[LcDocument], query: str, top_n: int, callbacks: Optional[Any] = None) -> List[LcDocument]:
        if not documents:
            return []
        if self._cross_encoder is None or not hasattr(self._cross_encoder, "predict"):
            logger.error("HuggingFace Cross-Encoder is not initialized. Returning truncated documents.")
            return list(documents)[:top_n]

        # Prepare input pairs
        sentence_pairs = [[query, doc.page_content] for doc in documents]

        try:
            scores = self._cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        except TypeError:
            # Some CrossEncoder versions accept different args
            try:
                scores = self._cross_encoder.predict(sentence_pairs)
            except Exception as e:
                logger.error(f"❌ Cross-Encoder prediction failed: {e}. Returning truncated documents.")
                return list(documents)[:top_n]
        except Exception as e:
            logger.error(f"❌ Cross-Encoder prediction failed: {e}. Returning truncated documents.")
            return list(documents)[:top_n]

        # sort and return top_n
        doc_scores = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        final_docs = []
        for doc, score in doc_scores[:top_n]:
            # 🟢 FIX: Check and initialize metadata if it is None (prevents TypeError)
            if doc.metadata is None:
                doc.metadata = {}
                logger.warning(f"⚠️ Reranker: Found Document with None metadata. Initializing metadata to {{}}.")
                
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)
        return final_docs

_CACHED_RERANKER_INSTANCE: Optional[HuggingFaceCrossEncoderCompressor] = None
_CACHED_CROSS_ENCODER: Any = None

def get_global_reranker(final_k: int) -> Optional[HuggingFaceCrossEncoderCompressor]:
    """
    Returns a cached HuggingFaceCrossEncoderCompressor instance.
    If sentence-transformers is not installed or initialization fails, returns None.
    """
    global _CACHED_RERANKER_INSTANCE, _CACHED_CROSS_ENCODER

    if _CACHED_RERANKER_INSTANCE is None:
        try:
            if not _HAS_SENT_TRANS:
                logging.warning("⚠️ sentence-transformers not installed. Cross-Encoder reranker disabled.")
                return None

            # 1️⃣ สร้าง compressor instance
            instance = HuggingFaceCrossEncoderCompressor(
                rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1"
            )

            # 2️⃣ สร้าง CrossEncoder จริง
            from sentence_transformers import CrossEncoder
            cross_encoder_model = CrossEncoder(
                instance.rerank_model,
                device=instance.rerank_device
            )

            # 3️⃣ ตั้งค่าให้ compressor ใช้งานได้
            instance.set_encoder_instance(cross_encoder_model)

            # 4️⃣ เก็บ cache global
            _CACHED_RERANKER_INSTANCE = instance
            _CACHED_CROSS_ENCODER = cross_encoder_model

            logging.info(f"✅ Initialized global Cross-Encoder reranker: {instance.rerank_model} on {instance.rerank_device}")

        except Exception as e:
            logging.warning(f"Failed to initialize global reranker: {e}")
            return None

    return _CACHED_RERANKER_INSTANCE



# -------------------- VECTORSTORE MANAGER (SINGLETON) --------------------
class VectorStoreManager:
    _instance = None
    _is_initialized = False

    _chroma_cache: Dict[str, Chroma] = PrivateAttr({})
    _multi_doc_retriever: Optional['MultiDocRetriever'] = PrivateAttr(None)
    _lock = threading.Lock()

    _doc_id_mapping: Dict[str, Dict[str, Any]] = PrivateAttr({})
    _uuid_to_doc_id: Dict[str, str] = PrivateAttr({})

    _embeddings: Any = PrivateAttr(None)

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_path: str = VECTORSTORE_DIR):
        if not self._is_initialized:
            self._base_path = base_path
            self._chroma_cache = {}
            self._embeddings = get_hf_embeddings()
            self._load_doc_id_mapping()
            logger.info(f"Initialized VectorStoreManager. Loaded {len(self._doc_id_mapping)} stable doc IDs.")
            VectorStoreManager._is_initialized = True

    def close(self):
        with self._lock:
            if self._multi_doc_retriever and hasattr(self._multi_doc_retriever, "shutdown"):
                logger.info("Closing MultiDocRetriever executor via VSM.")
                self._multi_doc_retriever.shutdown()
                self._multi_doc_retriever = None
            self._chroma_cache = {}
            VectorStoreManager._is_initialized = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _load_doc_id_mapping(self):
        self._doc_id_mapping = {}
        self._uuid_to_doc_id = {}
        try:
            with open(MAPPING_FILE_PATH, "r", encoding="utf-8") as f:
                mapping_data: Dict[str, Dict[str, Any]] = json.load(f)
                cleaned_mapping = {k.strip(): v for k, v in mapping_data.items()}
                self._doc_id_mapping = cleaned_mapping
                for doc_id, doc_entry in cleaned_mapping.items():
                    if isinstance(doc_entry, dict) and "chunk_uuids" in doc_entry and isinstance(doc_entry.get("chunk_uuids"), list):
                        for uid in doc_entry["chunk_uuids"]:
                            self._uuid_to_doc_id[uid] = doc_id
            logger.info(f"✅ Loaded Doc ID Mapping: {len(self._doc_id_mapping)} original documents, {len(self._uuid_to_doc_id)} total chunks.")
        except FileNotFoundError:
            logger.warning(f"⚠️ Doc ID Mapping file not found at {MAPPING_FILE_PATH}.")
        except Exception as e:
            logger.error(f"❌ Failed to load Doc ID Mapping: {e}")

    def _re_parse_collection_name(self, collection_name: str) -> Tuple[str, Optional[str]]:
        collection_name_lower = collection_name.strip().lower()
        if collection_name_lower.startswith(f"{EVIDENCE_DOC_TYPES}_"):
            parts = collection_name_lower.split("_", 1)
            return EVIDENCE_DOC_TYPES, parts[1].upper() if len(parts) == 2 else None
        return collection_name_lower, None

    def _load_chroma_instance(self, collection_name: str) -> Optional[Chroma]:
        if collection_name in self._chroma_cache:
            return self._chroma_cache[collection_name]
        with self._lock:
            if collection_name in self._chroma_cache:
                return self._chroma_cache[collection_name]
            persist_directory = os.path.join(self._base_path, collection_name)
            doc_type, enabler = self._re_parse_collection_name(collection_name)
            if not vectorstore_exists(doc_id="N/A", base_path=self._base_path, doc_type=doc_type, enabler=enabler):
                logger.warning(f"⚠️ Chroma collection '{collection_name}' folder not found at {persist_directory}")
                return None
            try:
                vectordb = Chroma(persist_directory=persist_directory, embedding_function=self._embeddings, collection_name=collection_name)
                self._chroma_cache[collection_name] = vectordb
                logger.info(f"✅ Loaded Chroma instance for collection: {collection_name}")
                return vectordb
            except Exception as e:
                logger.error(f"❌ Failed to load Chroma collection '{collection_name}': {e}")
                return None

    def get_documents_by_id(self, stable_doc_ids: Union[str, List[str]], doc_type: str = "default_collection", enabler: Optional[str] = None) -> List[LcDocument]:
        """
        Retrieve documents from Chroma collection by stable_doc_ids.
        Compatible with latest Chroma where 'ids' is not a valid include field.
        """
        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
        stable_doc_ids = [uid.strip() for uid in stable_doc_ids if uid]
        if not stable_doc_ids:
            return []

        collection_name = _get_collection_name(doc_type, enabler)
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.warning(f"Cannot retrieve documents: Collection '{collection_name}' is not loaded.")
            return []

        try:
            collection = chroma_instance._collection
            # เอา 'ids' ออก เพราะ Chroma เวอร์ชันใหม่ไม่รองรับ
            result = collection.get(
                where={"stable_doc_uuid": {"$in": stable_doc_ids}},
                include=["documents", "metadatas"]
            )

            documents: List[LcDocument] = []
            docs = result.get("documents", [])
            metadatas = result.get("metadatas", [{}] * len(docs))

            for i, text in enumerate(docs):
                if not text:
                    continue
                metadata = metadatas[i]
                # map chunk_uuid จาก stable_doc_ids เอง
                metadata["chunk_uuid"] = stable_doc_ids[i] if i < len(stable_doc_ids) else f"unknown_{i}"
                metadata["doc_id"] = metadata.get("doc_id", "UNKNOWN")
                metadata["doc_type"] = doc_type
                documents.append(LcDocument(page_content=text, metadata=metadata))

            logger.info(f"✅ Retrieved {len(documents)} documents for {len(stable_doc_ids)} Stable IDs from '{collection_name}'.")
            return documents

        except Exception as e:
            logger.error(f"❌ Error retrieving documents by Stable IDs from collection '{collection_name}': {e}")
            return []


    def retrieve_by_chunk_ids(self, chunk_ids: List[str], collection_name: str) -> List[LcDocument]:
        if not chunk_ids:
            return []
        try:
            chroma_instance = self._load_chroma_instance(collection_name)
            if not chroma_instance:
                logger.error(f"VSM: Collection '{collection_name}' not loaded for chunk id retrieval.")
                return []
            collection = chroma_instance._collection
            retrieval_result = collection.get(ids=chunk_ids, include=["documents", "metadatas", "ids"])
            retrieved_docs: List[LcDocument] = []
            documents = retrieval_result.get("documents", [])
            metadatas = retrieval_result.get("metadatas", [])
            ids = retrieval_result.get("ids", [])
            num_results = len(documents)
            if num_results != len(chunk_ids):
                logger.warning(f"VSM: got {num_results} docs, requested {len(chunk_ids)} chunk ids.")
            for content, metadata, chunk_id in zip(documents, metadatas, ids):
                if content and isinstance(metadata, dict):
                    metadata["chunk_uuid"] = chunk_id
                    stable_doc_id = self._uuid_to_doc_id.get(chunk_id, metadata.get("stable_doc_uuid", "UNKNOWN"))
                    metadata["doc_id"] = stable_doc_id
                    metadata["doc_type"] = metadata.get("doc_type", self._re_parse_collection_name(collection_name)[0])
                    retrieved_docs.append(LcDocument(page_content=content, metadata=metadata))
            logger.info(f"VSM: Retrieved {len(retrieved_docs)} priority docs from persistent map for '{collection_name}'.")
            return retrieved_docs
        except Exception as e:
            logger.error(f"VSM: Error retrieving chunk ids for collection '{collection_name}': {e}")
            return []

    def get_limited_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], query: Union[str, List[str]], doc_type: str, enabler: Optional[str] = None, limit_per_doc: int = 5) -> List[LcDocument]:
        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
        stable_doc_ids = [uid for uid in stable_doc_ids if uid]
        if not stable_doc_ids:
            return []
        vector_search_query = query[0] if isinstance(query, list) and query else (query if isinstance(query, str) else "")
        if not vector_search_query:
            logger.warning("Limited chunk search skipped: Query is empty.")
            return []
        collection_name = _get_collection_name(doc_type, enabler)
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.error(f"Collection '{collection_name}' is not loaded.")
            return []
        all_limited_documents: List[LcDocument] = []
        total_chunks_retrieved = 0
        for stable_id in stable_doc_ids:
            stable_id_clean = stable_id.strip()
            doc_filter = {"stable_doc_uuid": stable_id_clean}
            try:
                custom_retriever = ChromaRetriever(vectorstore=chroma_instance, k=limit_per_doc, filter=doc_filter)
                limited_docs = custom_retriever.get_relevant_documents(query=vector_search_query)
                for doc in limited_docs:
                    doc.metadata["priority_search_type"] = "limited_vector_search"
                    doc.metadata["priority_limit"] = limit_per_doc
                    all_limited_documents.append(doc)
                total_chunks_retrieved += len(limited_docs)
            except Exception as e:
                logger.error(f"❌ Error performing limited vector search for Stable ID '{stable_id_clean}': {e}")
                continue
        logger.info(f"✅ Retrieved {total_chunks_retrieved} limited chunks (max {limit_per_doc}/doc) for {len(stable_doc_ids)} Stable IDs from '{collection_name}'.")
        return all_limited_documents

    # -------------------- Retriever Creation --------------------
    def get_retriever(self, collection_name: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, use_rerank: bool = True) -> Any:
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.warning(f"Retriever creation failed: Collection '{collection_name}' not loaded.")
            return None
        search_kwargs = {"k": top_k}
        try:
            base_retriever = chroma_instance.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        except Exception as e:
            logger.error(f"Failed to create base retriever for '{collection_name}': {e}")
            return None

        # Build wrapper that supports config with filter and reranking
        def make_invoke_with_rerank(base_retriever, top_k, final_k):
            def invoke_with_rerank(query: str, config: Optional[Dict] = None):
                docs = []
                chroma_filter = None
                if config and isinstance(config, dict):
                    chroma_filter = config.get("configurable", {}).get("search_kwargs", {}).get("filter")
                try:
                    if chroma_filter:
                        new_config = {"configurable": {"search_kwargs": {"k": top_k, "filter": chroma_filter}}}
                        docs = base_retriever.invoke(query, config=new_config)
                    else:
                        docs = base_retriever.invoke(query, config=config)
                except Exception as e:
                    logger.error(f"❌ Retrieval failed before rerank: {e}")
                    return []
                # Reranking
                try:
                    reranker = get_global_reranker(final_k)
                    if reranker and hasattr(reranker, "compress_documents"):
                        # return top_k after reranking
                        return reranker.compress_documents(docs, query, top_n=top_k)
                    logger.warning("⚠️ Reranker not available. Returning base docs truncated.")
                    return docs[:top_k]
                except Exception as e:
                    logger.warning(f"⚠️ Rerank failed: {e}. Returning base docs truncated to {top_k}")
                    return docs[:top_k]
            return invoke_with_rerank

        if use_rerank:
            invoke_with_rerank = make_invoke_with_rerank(base_retriever, top_k, final_k)
            class SimpleRetrieverWrapper(BaseRetriever):
                model_config = ConfigDict(arbitrary_types_allowed=True)
                def invoke(self, query: str, config: Optional[Dict] = None):
                    return invoke_with_rerank(query, config=config)
                def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LcDocument]:
                    config = None
                    return invoke_with_rerank(query, config=config)
            return SimpleRetrieverWrapper()
        else:
            class TruncatedRetrieverWrapper(BaseRetriever):
                model_config = ConfigDict(arbitrary_types_allowed=True)
                def invoke(self, query: str, config: Optional[Dict] = None):
                    chroma_filter = config.get("configurable", {}).get("search_kwargs", {}).get("filter") if config else None
                    if chroma_filter:
                        new_config = {"configurable": {"search_kwargs": {"k": top_k, "filter": chroma_filter}}}
                        docs = base_retriever.invoke(query, config=new_config)
                    else:
                        docs = base_retriever.invoke(query, config=config)
                    return docs[:top_k]
                def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LcDocument]:
                    config = run_manager.get_session_info() if run_manager else None
                    return self.invoke(query, config=config)
            return TruncatedRetrieverWrapper()

    def get_all_collection_names(self) -> List[str]:
        return list_vectorstore_folders(base_path=self._base_path)

    def get_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], doc_type: str, enabler: Optional[str] = None) -> List[LcDocument]:
        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
        stable_doc_ids = [uid for uid in stable_doc_ids if uid]
        if not stable_doc_ids:
            return []
        collection_name = _get_collection_name(doc_type, enabler)
        all_chunk_uuids = []
        skipped_docs = []
        found_stable_ids = []
        for stable_id in stable_doc_ids:
            stable_id_clean = stable_id.strip()
            if stable_id_clean in self._doc_id_mapping:
                doc_entry = self._doc_id_mapping[stable_id_clean]
                if isinstance(doc_entry, dict) and "chunk_uuids" in doc_entry and isinstance(doc_entry.get("chunk_uuids"), list):
                    chunk_uuids = doc_entry["chunk_uuids"]
                    if chunk_uuids:
                        all_chunk_uuids.extend(chunk_uuids)
                        found_stable_ids.append(stable_id_clean)
                    else:
                        logger.warning(f"Mapping found for Stable ID '{stable_id_clean}' but 'chunk_uuids' list is empty.")
                else:
                    logger.warning(f"Mapping entry for Stable ID '{stable_id_clean}' is malformed or missing 'chunk_uuids'.")
            else:
                skipped_docs.append(stable_id_clean)
        if skipped_docs:
            logger.warning(f"Skipping Stable IDs not found in mapping: {skipped_docs}")
        if not all_chunk_uuids:
            logger.warning(f"No valid chunk UUIDs found for provided Stable Document IDs: {skipped_docs}. Check doc_id_mapping.json.")
            return []
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.error(f"Collection '{collection_name}' is not loaded.")
            return []
        try:
            collection = chroma_instance._collection
            result = collection.get(ids=all_chunk_uuids, include=["documents", "metadatas", "ids"])
            documents: List[LcDocument] = []
            if not result.get("documents"):
                logger.warning(f"Chroma DB returned 0 documents for {len(all_chunk_uuids)} chunk UUIDs in collection '{collection_name}'.")
                return []
            for i, text in enumerate(result.get("documents", [])):
                if text:
                    metadata = result.get("metadatas", [{}])[i]
                    chunk_uuid_from_result = result.get("ids", [""])[i]
                    doc_id = self._uuid_to_doc_id.get(chunk_uuid_from_result, "UNKNOWN")
                    metadata["chunk_uuid"] = chunk_uuid_from_result
                    metadata["doc_id"] = doc_id
                    metadata["doc_type"] = doc_type
                    documents.append(LcDocument(page_content=text, metadata=metadata))
            logger.info(f"✅ Retrieved {len(documents)} chunks for {len(found_stable_ids)} Stable IDs from '{collection_name}'.")
            return documents
        except Exception as e:
            logger.error(f"❌ Error retrieving documents by Chunk UUIDs from collection '{collection_name}': {e}")
            return []

# Helper function
def get_vectorstore_manager() -> VectorStoreManager:
    return VectorStoreManager()

def load_vectorstore(doc_type: str, enabler: Optional[str] = None) -> Optional[Chroma]:
    collection_name = _get_collection_name(doc_type, enabler)
    return get_vectorstore_manager()._load_chroma_instance(collection_name)

class VectorStoreExecutorSingleton:
    _instance = None
    _is_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VectorStoreExecutorSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not VectorStoreExecutorSingleton._is_initialized:
            self.max_workers = MAX_PARALLEL_WORKERS
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            logger.info(f"Initialized VectorStoreExecutorSingleton (ThreadPoolExecutor with {self.max_workers} workers) for batch tasks.")
            VectorStoreExecutorSingleton._is_initialized = True

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def close(self):
        if self._is_initialized:
            logger.info("Shutting down VectorStoreExecutorSingleton ThreadPoolExecutor...")
            self._executor.shutdown(wait=True)
            VectorStoreExecutorSingleton._is_initialized = False

def get_vectorstore() -> VectorStoreExecutorSingleton:
    return VectorStoreExecutorSingleton()

# -------------------- Custom Retriever for Chroma --------------------
class ChromaRetriever(BaseRetriever):
    vectorstore: Any
    k: int
    filter: Optional[Dict] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
        try:
            return self.vectorstore.similarity_search(query=query, k=self.k, filter=self.filter)
        except Exception as e:
            logger.error(f"❌ Chroma similarity_search failed in custom retriever: {e}")
            return []

    def get_relevant_documents(self, query: str, **kwargs) -> List[LcDocument]:
        return self._get_relevant_documents(query, **kwargs)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
        return self._get_relevant_documents(query, run_manager=run_manager)

# -------------------- MultiDoc / Parallel Retriever --------------------
class NamedRetriever(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    doc_id: str
    doc_type: str
    top_k: int
    final_k: int
    base_path: str = VECTORSTORE_DIR
    enabler: Optional[str] = None

    def load_instance(self) -> Any:
        manager = VectorStoreManager(base_path=self.base_path)
        retriever = manager.get_retriever(collection_name=_get_collection_name(self.doc_type, self.enabler), top_k=self.top_k, final_k=self.final_k, use_rerank=True)
        if not retriever:
            raise ValueError(f"Retriever not found for collection '{_get_collection_name(self.doc_type, self.enabler)}' at path '{self.base_path}'")
        return retriever

class MultiDocRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _all_retrievers: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _retrievers_list: list[NamedRetriever] = PrivateAttr()
    _k_per_doc: int = PrivateAttr()
    _manager: VectorStoreManager = PrivateAttr()
    _doc_ids_filter: Optional[List[str]] = PrivateAttr()
    _chroma_filter: Optional[Dict[str, Any]] = PrivateAttr()
    _executor_type: str = PrivateAttr()
    _executor: Union[ThreadPoolExecutor, ProcessPoolExecutor, None] = PrivateAttr(None)

    def __init__(self, retrievers_list: list[NamedRetriever], k_per_doc: int = INITIAL_TOP_K, doc_ids_filter: Optional[List[str]] = None):
        super().__init__()
        self._retrievers_list = retrievers_list
        self._k_per_doc = k_per_doc
        self._manager = VectorStoreManager()
        self._all_retrievers = {}
        for named_r in retrievers_list:
            collection_name = _get_collection_name(named_r.doc_type, named_r.enabler)
            try:
                retriever_instance = named_r.load_instance()
                if retriever_instance:
                    self._all_retrievers[collection_name] = retriever_instance
                    logger.info(f"✅ MultiDocRetriever cached collection: {collection_name}")
                else:
                    logger.warning(f"⚠️ Failed to load instance for {collection_name} during MDR init.")
            except Exception as e:
                logger.error(f"❌ Error loading instance {collection_name} into MDR cache: {e}")

        self._doc_ids_filter = doc_ids_filter
        self._chroma_filter = None
        if doc_ids_filter:
            self._chroma_filter = {"doc_id": {"$in": doc_ids_filter}}
            logger.info(f"✅ MultiDocRetriever initialized with doc_ids filter for {len(doc_ids_filter)} Stable IDs.")

        self._executor_type = self._choose_executor()
        logger.info(f"MultiDocRetriever will use executor type: {self._executor_type} (workers={MAX_PARALLEL_WORKERS})")

    def _choose_executor(self) -> str:
        sys_info = _detect_system()
        device = _detect_torch_device()
        force = ENV_FORCE_MODE
        if force in ("thread", "process"):
            logger.info(f"VECTOR_MODE override: forcing '{force}' executor")
            return force
        if sys_info["platform"] == "darwin" and device == "mps":
            logger.warning("⚠️ Detected MPS on macOS: forcing executor -> thread to avoid multi-process failures.")
            return "thread"
        if sys_info["total_ram_gb"] and sys_info["total_ram_gb"] < 12:
            logger.warning(f"⚠️ Detected low RAM ({sys_info['total_ram_gb']:.1f}GB): forcing executor -> thread.")
            return "thread"
        if sys_info["cpu_count"] >= 8 and (sys_info["total_ram_gb"] or 0) >= 16:
            logger.info("High-resources machine detected -> choosing 'process' executor")
            return "process"
        logger.info("Defaulting to 'thread' executor")
        return "thread"

    def shutdown(self):
        if self._executor:
            executor_type_name = "ProcessPoolExecutor" if self._executor_type == "process" else "ThreadPoolExecutor"
            workers = self._executor._max_workers if hasattr(self._executor, "_max_workers") else "N/A"
            logger.info(f"Shutting down MultiDocRetriever's {executor_type_name} executor ({workers} workers).")
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _get_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        if self._executor is None:
            workers = MAX_PARALLEL_WORKERS
            if self._executor_type == "process":
                self._executor = ProcessPoolExecutor(max_workers=workers)
                logger.info(f"🛠️ Using ProcessPoolExecutor with {workers} workers.")
            else:
                self._executor = ThreadPoolExecutor(max_workers=workers)
                logger.info(f"🛠️ Using ThreadPoolExecutor with {workers} workers.")
        return self._executor

    @staticmethod
    def _static_retrieve_task(named_r: NamedRetriever, query: str, chroma_filter: Optional[Dict]):
        try:
            retriever_instance = named_r.load_instance()
            search_kwargs = {"k": named_r.top_k, "filter": chroma_filter} if chroma_filter else {"k": named_r.top_k}
            config = {"configurable": {"search_kwargs": search_kwargs}}
            docs = retriever_instance.invoke(query, config=config)
            for doc in docs:
                doc.metadata["retrieval_source"] = named_r.doc_type
                doc.metadata["collection_name"] = _get_collection_name(named_r.doc_type, named_r.enabler)
            return docs
        except Exception as e:
            print(f"❌ Child retrieval error for {named_r.doc_id} ({named_r.doc_type}): {e}")
            return []

    def _thread_retrieve_task(self, named_r: NamedRetriever, query: str, chroma_filter: Optional[Dict]):
        try:
            retriever_instance = named_r.load_instance()
            search_kwargs = {"k": named_r.top_k, "filter": chroma_filter} if chroma_filter else {"k": named_r.top_k}
            config = {"configurable": {"search_kwargs": search_kwargs}}
            docs = retriever_instance.invoke(query, config=config)
            for doc in docs:
                doc.metadata["retrieval_source"] = named_r.doc_type
                doc.metadata["collection_name"] = _get_collection_name(named_r.doc_type, named_r.enabler)
            return docs
        except Exception as e:
            logger.warning(f"⚠️ Thread retrieval error for {named_r.doc_id}: {e}")
            return []

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LcDocument]:
        max_workers = min(len(self._retrievers_list), MAX_PARALLEL_WORKERS)
        if max_workers <= 0:
            max_workers = 1
        chosen = self._executor_type
        logger.info(f"⚙️ Running MultiDocRetriever with {chosen} executor ({max_workers} workers) [Filter: {bool(self._chroma_filter)}]")
        all_docs: List[LcDocument] = []
        executor = self._get_executor()
        futures = []
        for named_r in self._retrievers_list:
            if chosen == "process":
                future = executor.submit(MultiDocRetriever._static_retrieve_task, named_r, query, self._chroma_filter)
            else:
                future = executor.submit(self._thread_retrieve_task, named_r, query, self._chroma_filter)
            futures.append(future)
        for f in futures:
            try:
                docs = f.result()
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Future failed: {e}")
        seen = set()
        unique_docs = []
        for d in all_docs:
            src = d.metadata.get("retrieval_source") or ""
            chunk_uuid = d.metadata.get("chunk_uuid") or d.metadata.get("ids") or ""
            key = f"{src}_{chunk_uuid}_{d.page_content[:120]}"
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)
        logger.info(f"📝 Query='{query[:80]}...' found {len(unique_docs)} unique docs across sources (Executor={chosen})")
        for d in unique_docs:
            score = d.metadata.get("relevance_score")
            if score is not None:
                logger.debug(f" - [Reranked] Source={d.metadata.get('doc_type')}, Score={score:.4f}, Content='{d.page_content[:80]}...'")
        return unique_docs

    def get_relevant_documents(self, query: str, **kwargs) -> List[LcDocument]:
        return self._get_relevant_documents(query, **kwargs)

# -------------------- load_all_vectorstores --------------------
def load_vectorstore_retriever(doc_id: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, doc_types: Union[list, str] = "default_collection", base_path: str = VECTORSTORE_DIR, enabler: Optional[str] = None):
    if isinstance(doc_types, str):
        target_doc_type = doc_types
    elif isinstance(doc_types, list) and doc_types:
        target_doc_type = doc_types[0]
    else:
        raise ValueError("doc_types must be a single string or a non-empty list containing the target doc_type.")
    collection_name = _get_collection_name(target_doc_type, enabler)
    manager = VectorStoreManager(base_path=base_path)
    retriever = None
    if vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=target_doc_type, enabler=enabler):
        retriever = manager.get_retriever(collection_name, top_k, final_k)
    if retriever is None:
        raise ValueError(f"❌ Vectorstore for collection '{collection_name}' not found.")
    return retriever

def load_all_vectorstores(doc_types: Optional[Union[str, List[str]]] = None, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, base_path: str = VECTORSTORE_DIR, evidence_enabler: Optional[str] = None, doc_ids: Optional[List[str]] = None) -> VectorStoreManager:
    doc_types = [doc_types] if isinstance(doc_types, str) else doc_types or []
    doc_type_filter = {dt.strip().lower() for dt in doc_types}
    manager = VectorStoreManager(base_path=base_path)
    all_retrievers: List[NamedRetriever] = []
    target_collection_names: Set[str] = set()
    if not doc_type_filter:
        target_collection_names.update(manager.get_all_collection_names())
    else:
        for dt_norm in doc_type_filter:
            if dt_norm == EVIDENCE_DOC_TYPES:
                if evidence_enabler:
                    collection_name = _get_collection_name(EVIDENCE_DOC_TYPES, evidence_enabler)
                    target_collection_names.add(collection_name)
                    logger.info(f"🔍 Added specific evidence collection: {collection_name}")
                else:
                    evidence_collections = list_vectorstore_folders(base_path=base_path, doc_type=EVIDENCE_DOC_TYPES)
                    target_collection_names.update(evidence_collections)
                    logger.info(f"🔍 Added all evidence collections found: {evidence_collections}")
            else:
                collection_name = _get_collection_name(dt_norm, None)
                target_collection_names.add(collection_name)
                logger.info(f"🔍 Added standard collection: {collection_name}")
    logger.info(f"🔍 DEBUG: Attempting to load {len(target_collection_names)} total target collections: {target_collection_names}")
    for collection_name in target_collection_names:
        doc_type_for_check, enabler_for_check = manager._re_parse_collection_name(collection_name)
        if not vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=doc_type_for_check, enabler=enabler_for_check):
            logger.warning(f"🔍 DEBUG: Skipping collection '{collection_name}' (vectorstore_exists failed).")
            continue
        nr = NamedRetriever(doc_id=collection_name, doc_type=doc_type_for_check, enabler=enabler_for_check, top_k=top_k, final_k=final_k, base_path=base_path)
        all_retrievers.append(nr)
        logger.info(f"🔍 DEBUG: Successfully added retriever for collection '{collection_name}'.")
    final_filter_ids = doc_ids
    if doc_ids:
        logger.info(f"✅ Hard Filter Enabled: Using {len(doc_ids)} original 64-char UUIDs for filtering.")
    logger.info(f"🔍 DEBUG: Final count of all_retrievers = {len(all_retrievers)}")
    if not all_retrievers:
        raise ValueError(f"No vectorstore collections found matching doc_types={doc_types} and evidence_enabler={evidence_enabler}")
    mdr = MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k, doc_ids_filter=final_filter_ids)
    manager._multi_doc_retriever = mdr
    logger.info(f"✅ MultiDocRetriever loaded with {len(mdr._all_retrievers)} collections and cached in VSM.")
    return manager
