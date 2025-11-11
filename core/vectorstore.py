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

# LangChain imports (รุ่นที่ปรับปรุงสำหรับ V1.x)
from langchain_core.documents import Document as LcDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import BaseDocumentCompressor 


from langchain_community.document_compressors import FlashrankRerank

# External libraries
from pydantic import PrivateAttr, ConfigDict, BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from flashrank import Ranker
import chromadb
from chromadb.config import Settings

from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

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
)

# -------------------- Vectorstore Constants --------------------
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "2"))
ENV_FORCE_MODE = os.getenv("VECTOR_MODE", "").lower()  # "thread", "process", or ""

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.handlers = logging.root.handlers

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


def get_hf_embeddings(device_hint: Optional[str] = None):
    """
    Return a HuggingFaceEmbeddings instance (cached per process).
    device_hint can be 'cuda'|'mps'|'cpu' or None to auto-detect.
    Note: per-process caching - threads will share this instance within same process.
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
                    # ใช้ E5-large เพื่อให้เข้ากับ ingest.py
                    model_name = "intfloat/multilingual-e5-large"
                    logger.info(f"📦 Creating HuggingFaceEmbeddings (model={model_name}, device={device})")
                    
                    # 🟢 การปรับปรุงสำหรับ E5: เพิ่ม query_instruction และ encode_kwargs
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                        model_name=model_name, 
                        model_kwargs={"device": device},
                        # 1. เพิ่มคำสั่งนำหน้า "query: " สำหรับข้อความค้นหา (Query) เพื่อประสิทธิภาพ E5
                        # query_instruction="query: ", #NOTE: Commented out as per original code template
                        # 2. แนะนำให้ normalize embeddings สำหรับ E5 เพื่อความแม่นยำ
                        # encode_kwargs={'normalize_embeddings': True} #NOTE: Commented out as per original code template
                    )

                except Exception as e:
                    # fallback to CPU if any issue
                    logger.warning(f"⚠️ Failed to create embeddings on device={device}: {e}. Falling back to CPU.")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return _CACHED_EMBEDDINGS


# -------------------- Vectorstore helpers (REVISED/CLEANED) --------------------

def _get_collection_name(doc_type: str, enabler: Optional[str] = None) -> str:
    """
    Calculates the Chroma collection name and directory name based on doc_type and enabler.
    For doc_type='evidence', it returns 'evidence_<ENABLER>'.
    Otherwise, it returns the doc_type string (e.g., 'document').
    """
    doc_type_norm = doc_type.strip().lower()

    if doc_type_norm == EVIDENCE_DOC_TYPES:
        # Apply default enabler if None, หรือใช้ enabler ที่ส่งมา
        # NOTE: การใช้ DEFAULT_ENABLER ต้องมั่นใจว่ามีการ import มาแล้ว
        enabler_norm = (enabler or "km").strip().lower() # 🟢 ใช้ "km" เป็นค่า default ชั่วคราว ถ้า DEFAULT_ENABLER ยังไม่ถูกกำหนด
        collection_name = f"{doc_type_norm}_{enabler_norm}"
    else:
        collection_name = doc_type_norm
        
    # ✅ เพิ่ม Log ที่ต้องการ
    logger.critical(f"🧭 DEBUG: _get_collection_name(doc_type={doc_type}, enabler={enabler}) => {collection_name}")
    
    return collection_name

def get_vectorstore_path(doc_type: Optional[str] = None, enabler: Optional[str] = None) -> str:
    """
    Returns the full path to the base dir or the specific collection directory.
    Uses _get_collection_name logic.
    """
    if not doc_type:
        return VECTORSTORE_DIR
    
    # 🟢 REVISED: ใช้ _get_collection_name
    collection_name = _get_collection_name(doc_type, enabler)
    return os.path.join(VECTORSTORE_DIR, collection_name)

def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None, enabler: Optional[str] = None) -> List[str]:
    """
    Lists available Chroma collections (which are folders inside VECTORSTORE_DIR).
    Returns collection names (e.g., 'document', 'evidence_km').
    """
    if not os.path.exists(base_path):
        return []
    
    # List folders inside VECTORSTORE_DIR
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    if doc_type:
        doc_type_norm = doc_type.lower().strip()
        
        if doc_type_norm == EVIDENCE_DOC_TYPES and not enabler:
            # 🟢 NEW LOGIC: Special case: 'evidence' without enabler means list ALL evidence_*
            return [f for f in folders if f.startswith(f"{EVIDENCE_DOC_TYPES}_")]
            
        # Specific collection requested (e.g., 'document' or 'evidence_km')
        collection_name = _get_collection_name(doc_type_norm, enabler)
        return [collection_name] if collection_name in folders else []
        
    return folders


def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None, enabler: Optional[str] = None) -> bool:
    """
    Checks if a Chroma collection exists on disk.
    doc_type and enabler define the collection name.
    """
    if not doc_type:
        return False
    
    # 🟢 REVISED: ใช้ _get_collection_name
    collection_name = _get_collection_name(doc_type, enabler)
    path = os.path.join(base_path, collection_name)
    file_path = os.path.join(path, "chroma.sqlite3")
    
    # 🟢 DEBUG: แสดงพาธที่กำลังจะตรวจสอบ (เพื่อยืนยันว่าถูกต้อง)
    # logger.info(f"🔍 V-Exists Check: Checking path: {file_path} (from CWD: {os.getcwd()})")
    
    if not os.path.isdir(path):
        logger.warning(f"❌ V-Exists Check 1: Directory not found for collection '{collection_name}' at {path}")
        return False
        
    # 2. ตรวจสอบไฟล์ฐานข้อมูลหลัก
    if os.path.isfile(file_path):
        # logger.info(f"✅ V-Exists Check 2: Found required file at {file_path}")
        return True
        
    # 🚨 DEBUG: หากถึงตรงนี้ แสดงว่าหาไฟล์ไม่เจอ (คือบั๊กที่เรากำลังหา)
    logger.error(f"❌ V-Exists Check 3: FAILED to find file chroma.sqlite3 at {file_path} for collection '{collection_name}'")
    return False


# -------------------- VECTORSTORE MANAGER (SINGLETON) --------------------
class VectorStoreManager:
    """
    Singleton class to manage and cache Chroma vectorstore instances (collections).
    Handles initialization of Embeddings, Reranker, and Doc ID Mapping.
    """
    _instance = None
    _is_initialized = False
    
    # Cache to store loaded Chroma instances
    _chroma_cache: Dict[str, Chroma] = PrivateAttr({}) 
    # Lock for thread-safe initialization and cache access
    _lock = threading.Lock()
    
    # Cache for Doc ID Mapping (doc_id -> DocInfo)
    # The structure is Dict[str, Dict[str, Any]] to match the DocInfo structure
    _doc_id_mapping: Dict[str, Dict[str, Any]] = PrivateAttr({}) 
    # Reverse cache: chunk_uuid -> doc_id
    _uuid_to_doc_id: Dict[str, str] = PrivateAttr({})
    
    # Embeddings (Shared across all instances)
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
            # Initialization of embeddings and reranker models
            self._embeddings = get_hf_embeddings()
            
            # Load mapping on startup
            self._load_doc_id_mapping()
            
            logger.info(f"Initialized VectorStoreManager. Loaded {len(self._doc_id_mapping)} stable doc IDs.")
            VectorStoreManager._is_initialized = True

    def _load_doc_id_mapping(self):
            """Loads doc_id_mapping.json into memory."""
            self._doc_id_mapping = {}
            self._uuid_to_doc_id = {}
            try:
                # Load mapping data (doc_id -> DocInfo structure)
                with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
                    mapping_data: Dict[str, Dict[str, Any]] = json.load(f)
                    
                    # 🎯 FIX: ทำความสะอาด (strip) คีย์ทั้งหมดของ Dictionary ทันทีที่โหลด
                    # เพื่อป้องกันอักขระที่ไม่คาดคิดที่อาจติดมาจากไฟล์ JSON
                    cleaned_mapping = {k.strip(): v for k, v in mapping_data.items()}
                    
                    self._doc_id_mapping = cleaned_mapping
                    
                    # Create reverse mapping for quick lookup UUID -> Doc ID
                    # ใชั cleaned_mapping ในการวนซ้ำ
                    for doc_id, doc_entry in cleaned_mapping.items(): 
                        # Ensure doc_entry is a dict and contains the 'chunk_uuids' key
                        if isinstance(doc_entry, dict) and 'chunk_uuids' in doc_entry and isinstance(doc_entry['chunk_uuids'], list):
                            for uid in doc_entry['chunk_uuids']:
                                self._uuid_to_doc_id[uid] = doc_id
                            
                logger.info(f"✅ Loaded Doc ID Mapping: {len(self._doc_id_mapping)} original documents, {len(self._uuid_to_doc_id)} total chunks.")
            except FileNotFoundError:
                logger.warning(f"⚠️ Doc ID Mapping file not found at {MAPPING_FILE_PATH}. This is expected if no documents have been ingested yet.")
            except Exception as e:
                logger.error(f"❌ Failed to load Doc ID Mapping: {e}")

    def _re_parse_collection_name(self, collection_name: str) -> Tuple[str, Optional[str]]:
        """Helper to safely re-parse collection name back to doc_type and enabler."""
        collection_name_lower = collection_name.strip().lower()
        if collection_name_lower.startswith(f"{EVIDENCE_DOC_TYPES}_"):
            parts = collection_name_lower.split("_", 1)
            # Return 'evidence' as doc_type, and the enabler part (uppercase)
            return EVIDENCE_DOC_TYPES, parts[1].upper() if len(parts) == 2 else None
            
        # For non-evidence types (document, faq, etc.)
        return collection_name_lower, None 

    def _load_chroma_instance(self, collection_name: str) -> Optional[Chroma]:
        """Loads a Chroma instance from disk or returns from cache."""
        if collection_name in self._chroma_cache:
            return self._chroma_cache[collection_name]

        with self._lock:
            # Re-check cache after acquiring lock
            if collection_name in self._chroma_cache:
                return self._chroma_cache[collection_name]
            
            persist_directory = os.path.join(self._base_path, collection_name)
            
            # NOTE: vectorstore_exists needs doc_type/enabler, parse collection_name.
            doc_type, enabler = self._re_parse_collection_name(collection_name)
            
            # Use the global helper to check existence on disk
            if not vectorstore_exists(doc_id="N/A", base_path=self._base_path, doc_type=doc_type, enabler=enabler):
                logger.warning(f"⚠️ Chroma collection '{collection_name}' folder not found at {persist_directory}")
                return None

            try:
                # Load Chroma DB
                vectordb = Chroma(
                    persist_directory=persist_directory, 
                    embedding_function=self._embeddings,
                    collection_name=collection_name
                )
                self._chroma_cache[collection_name] = vectordb
                logger.info(f"✅ Loaded Chroma instance for collection: {collection_name}")
                return vectordb
            except Exception as e:
                logger.error(f"❌ Failed to load Chroma collection '{collection_name}': {e}")
                return None

    def get_documents_by_id(self, doc_uuids: Union[str, List[str]], doc_type: str = "default_collection", enabler: Optional[str] = None) -> List[LcDocument]:
        """
        Retrieves chunks (Documents) from a specific Chroma collection 
        using their internal Chroma UUIDs (chunk_uuid).
        """
        if isinstance(doc_uuids, str):
            doc_uuids = [doc_uuids]
            
        doc_uuids = [uid for uid in doc_uuids if uid] # filter out None/empty strings
        if not doc_uuids:
            return []
            
        # Use the global helper to get the correct collection name
        collection_name = _get_collection_name(doc_type, enabler)
        chroma_instance = self._load_chroma_instance(collection_name)

        if not chroma_instance:
            logger.warning(f"Cannot retrieve documents: Collection '{collection_name}' is not loaded.")
            return []
        
        try:
            # 1. Get collection client
            # NOTE: We access the private attribute _collection as it's often the quickest way to the raw client
            collection = chroma_instance._collection
            
            # 2. Fetch data by IDs
            result = collection.get(
                ids=doc_uuids,
                include=['documents', 'metadatas'] 
            )
            
            # 3. Process results into LangChain Documents
            documents: List[LcDocument] = []
            for i, text in enumerate(result.get('documents', [])):
                if text:
                    metadata = result.get('metadatas', [{}])[i]
                    # Use ID from the Chroma result (which is the chunk UUID)
                    chunk_uuid_from_result = result.get('ids', [''])[i]
                    
                    # Map chunk UUID back to the stable doc_id
                    doc_id = self._uuid_to_doc_id.get(chunk_uuid_from_result, "UNKNOWN")
                    
                    # Ensure metadata contains necessary keys
                    metadata["chunk_uuid"] = chunk_uuid_from_result
                    metadata["doc_id"] = doc_id
                    metadata["doc_type"] = doc_type # Use input doc_type
                    
                    documents.append(LcDocument(page_content=text, metadata=metadata))
            
            logger.info(f"✅ Retrieved {len(documents)} documents for {len(doc_uuids)} UUIDs from '{collection_name}'.")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents by UUIDs from collection '{collection_name}': {e}")
            return []

# -------------------- Retriever Creation --------------------
    def get_retriever(base_retriever, final_k: int = 5, use_rerank: bool = True) -> Any:
        """
        base_retriever: retriever object ที่มี invoke(query) -> list docs
        """
        if use_rerank:
            def invoke_with_rerank(query: str):
                docs = base_retriever.invoke(query)
                try:
                    # ใช้ global reranker
                    return GLOBAL_RERANKER.rerank(query, docs)[:final_k]
                except Exception as e:
                    logger.warning(f"⚠️ FlashrankRerank failed: {e}. Returning base docs truncated to final_k")
                    return docs[:final_k]

            # Wrapper เพื่อให้เรียก invoke เหมือน retriever ปกติ
            class SimpleRetrieverWrapper:
                def invoke(self, query: str):
                    return invoke_with_rerank(query)

            return SimpleRetrieverWrapper()
        else:
            class TruncatedRetrieverWrapper:
                def invoke(self, query: str):
                    return base_retriever.invoke(query)[:final_k]

            return TruncatedRetrieverWrapper()

    def get_all_collection_names(self) -> List[str]:
        """Returns a list of all available collection names (folders in VECTORSTORE_DIR)."""
        # Use the global helper
        return list_vectorstore_folders(base_path=self._base_path)
    
    def get_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], doc_type: str, enabler: Optional[str] = None) -> List[LcDocument]:
            """
            Retrieves chunks (Documents) for a list of Stable Document IDs from a specific collection.
            This retrieves ALL chunks belonging to the specified Stable Document IDs.
            """
            if isinstance(stable_doc_ids, str):
                stable_doc_ids = [stable_doc_ids]
                
            stable_doc_ids = [uid for uid in stable_doc_ids if uid]
            if not stable_doc_ids:
                return []
                
            # Get the correct Collection Name
            collection_name = _get_collection_name(doc_type, enabler)

            all_chunk_uuids = []
            skipped_docs = []
            found_stable_ids = []

            # 1. ค้นหา Chunk UUIDs ทั้งหมดจาก Mapping
            for stable_id in stable_doc_ids:
                # 🟢 FIX: ทำความสะอาด ID โดยการ strip() ก่อนการเปรียบเทียบ
                stable_id_clean = stable_id.strip()

                if stable_id_clean in self._doc_id_mapping:
                    doc_entry = self._doc_id_mapping[stable_id_clean] 
                    
                    # ตรวจสอบโครงสร้าง: {"stable_id": {"chunk_uuids": ["uuid1", "uuid2", ...]}}
                    if isinstance(doc_entry, dict) and 'chunk_uuids' in doc_entry and isinstance(doc_entry.get('chunk_uuids'), list):
                        chunk_uuids = doc_entry['chunk_uuids']
                        if chunk_uuids:
                            all_chunk_uuids.extend(chunk_uuids)
                            found_stable_ids.append(stable_id_clean) # ใช้ ID ที่สะอาดในการบันทึก
                        else:
                            logger.warning(f"Mapping found for Stable ID '{stable_id_clean}' but 'chunk_uuids' list is empty.")
                    else:
                        logger.warning(f"Mapping entry for Stable ID '{stable_id_clean}' is malformed or missing 'chunk_uuids'.")
                else:
                    skipped_docs.append(stable_id_clean) # ใช้ ID ที่สะอาดในการบันทึกว่าถูก skip
                    
            if skipped_docs:
                logger.warning(f"Skipping Stable IDs not found in mapping: {skipped_docs}")

            if not all_chunk_uuids:
                logger.warning(f"No valid chunk UUIDs found for provided Stable Document IDs: {skipped_docs}. Check doc_id_mapping.json.")
                return []
                
            # 2. โหลด Chroma Instance
            chroma_instance = self._load_chroma_instance(collection_name) 

            if not chroma_instance:
                logger.error(f"Collection '{collection_name}' is not loaded.")
                return []

            # 3. Fetch data by Chunk IDs (Chroma UUIDs)
            try:
                collection = chroma_instance._collection
                result = collection.get(
                    ids=all_chunk_uuids,
                    include=['documents', 'metadatas'] 
                )
                
                # 4. Process results into LangChain Documents
                documents: List[LcDocument] = []
                
                if not result.get('documents'):
                    logger.warning(f"ChromaDB returned 0 documents for {len(all_chunk_uuids)} chunk UUIDs in collection '{collection_name}'.")
                    return []
                    
                for i, text in enumerate(result.get('documents', [])):
                    if text:
                        metadata = result.get('metadatas', [{}])[i]
                        chunk_uuid_from_result = result.get('ids', [''])[i]
                        doc_id = self._uuid_to_doc_id.get(chunk_uuid_from_result, "UNKNOWN")
                        
                        # Ensure metadata contains necessary keys
                        metadata["chunk_uuid"] = chunk_uuid_from_result
                        metadata["doc_id"] = doc_id
                        metadata["doc_type"] = doc_type # Use input doc_type
                        
                        documents.append(LcDocument(page_content=text, metadata=metadata))
                
                logger.info(f"✅ Retrieved {len(documents)} chunks for {len(found_stable_ids)} Stable IDs from '{collection_name}'.")
                return documents
                
            except Exception as e:
                logger.error(f"❌ Error retrieving documents by Chunk UUIDs from collection '{collection_name}': {e}")
                return []

# Helper function to get the manager instance
def get_vectorstore_manager() -> VectorStoreManager:
    """Returns the singleton instance of VectorStoreManager."""
    return VectorStoreManager()
    
# Backward compatibility function (if needed by other parts of the system)
def load_vectorstore(doc_type: str, enabler: Optional[str] = None) -> Optional[Chroma]:
    """Helper for other modules to load a Chroma instance directly."""
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
            # Placeholder implementation
            self.max_workers = MAX_PARALLEL_WORKERS
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            VectorStoreExecutorSingleton._is_initialized = True

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def close(self):
        if self._is_initialized:
            self._executor.shutdown(wait=True)
            VectorStoreExecutorSingleton._is_initialized = False 
            
def get_vectorstore() -> VectorStoreExecutorSingleton:
    """Returns the singleton instance managing the executor."""
    return VectorStoreExecutorSingleton()
    
def load_all_vectorstores():
    """Placeholder for loading all vectorstores if needed globally."""
    pass
    
def get_vectorstore_path(doc_type: str, enabler: Optional[str] = None) -> str:
    """Returns the path to the vectorstore folder."""
    collection_name = _get_collection_name(doc_type, enabler)
    return os.path.join(VECTORSTORE_DIR, collection_name)


# -------------------- MultiDoc / Parallel Retriever --------------------
class NamedRetriever(BaseModel):
    doc_id: str
    doc_type: str
    top_k: int
    final_k: int
    base_path: str = VECTORSTORE_DIR
    enabler: Optional[str] = None

    def load_instance(self) -> BaseRetriever:
        manager = VectorStoreManager(base_path=self.base_path)
        retriever = manager.get_retriever(
            collection_name=_get_collection_name(self.doc_type, self.enabler),
            top_k=self.top_k,
            final_k=self.final_k,
            use_rerank=True
        )
        if not retriever:
            raise ValueError(f"Retriever not found for collection '{self.doc_type}' at path '{self.base_path}'")
        return retriever


class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple NamedRetriever sources. Choose executor automatically (thread vs process).
    """

    _retrievers_list: list[NamedRetriever] = PrivateAttr()
    _k_per_doc: int = PrivateAttr()
    _manager: VectorStoreManager = PrivateAttr() 
    _doc_ids_filter: Optional[List[str]] = PrivateAttr()
    _chroma_filter: Optional[Dict[str, Any]] = PrivateAttr()

    def __init__(self, retrievers_list: list[NamedRetriever], k_per_doc: int = INITIAL_TOP_K, doc_ids_filter: Optional[List[str]] = None):
        super().__init__()
        self._retrievers_list = retrievers_list
        self._k_per_doc = k_per_doc
        self._manager = VectorStoreManager() 

        # NEW LOGIC: สร้าง Chroma Filter จาก Doc IDs
        self._doc_ids_filter = doc_ids_filter
        self._chroma_filter = None
        if doc_ids_filter:
            # สร้าง Chroma DB Metadata Filter: ค้นหาเฉพาะเอกสารที่มี 'doc_id' อยู่ในลิสต์
            self._chroma_filter = {
                # 🟢 FIX: เปลี่ยนคีย์เป็น "doc_id" ซึ่งเป็นคีย์ที่ใช้เก็บ Stable Document ID ใน Metadata
                "doc_id": {"$in": doc_ids_filter} 
            }

            logger.info(f"✅ MultiDocRetriever initialized with doc_ids filter for {len(doc_ids_filter)} Stable IDs.")

    # ... (omitted code - ส่วนที่เหลือของคลาส MultiDocRetriever ไม่ต้องแก้ไข) ...
    # 🟢 FIX: เมธอด _choose_executor ที่หายไป
    def _choose_executor(self):
        """
        Decide whether to use ProcessPoolExecutor or ThreadPoolExecutor.
        Logic:
          - If ENV_FORCE_MODE set -> obey it.
          - If platform is darwin (mac) and GPU device is mps -> prefer thread (MPS multi-process is fragile).
          - If total RAM is low (below 12GB) -> prefer thread.
          - Else if CPU cores 8 or more and RAM 16GB or more -> prefer process.
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

    # 🟢 FIX: แก้ไข _static_retrieve_task ให้รับ filter
    @staticmethod
    def _static_retrieve_task(named_r: "NamedRetriever", query: str, chroma_filter: Optional[Dict]):
        """
        Static helper used in ProcessPoolExecutor. Executes inside child process.
        """
        try:
            retriever_instance = named_r.load_instance()
            # 🟢 NEW: ส่ง filter ผ่าน search_kwargs
            search_kwargs = {"k": named_r.top_k, "filter": chroma_filter} if chroma_filter else {"k": named_r.top_k}
            # 🎯 เพิ่ม Log ตรงนี้:
            # logger.info(f"🎯 DEBUG: Task Filter Check for {named_r.doc_type} (Static): Filter is {bool(chroma_filter)}")
            print(f"🎯 DEBUG: Task Filter Check for {named_r.doc_type} (Static): Filter is {bool(chroma_filter)}")
            return retriever_instance.invoke(query, config={'configurable': {'search_kwargs': search_kwargs}})
        except Exception as e:
            print(f"❌ Child retrieval error for {named_r.doc_id} ({named_r.doc_type}): {e}")
            return None

    # 🟢 FIX: แก้ไข _thread_retrieve_task ให้รับ filter
    def _thread_retrieve_task(self, named_r: "NamedRetriever", query: str, chroma_filter: Optional[Dict]):
        """
        Retrieval performed in a thread inside the same process.
        """
        try:
            retriever_instance = named_r.load_instance()
            # 🟢 NEW: ส่ง filter ผ่าน search_kwargs
            search_kwargs = {"k": named_r.top_k, "filter": chroma_filter} if chroma_filter else {"k": named_r.top_k}
            # 🎯 เพิ่ม Log ตรงนี้:
            # logger.info(f"🎯 DEBUG: Task Filter Check for {named_r.doc_type} (Thread): Filter is {bool(chroma_filter)}")
            print(f"🎯 DEBUG: Task Filter Check for {named_r.doc_type} (Thread): Filter is {bool(chroma_filter)}")
            return retriever_instance.invoke(query, config={'configurable': {'search_kwargs': search_kwargs}})
        except Exception as e:
            logger.warning(f"⚠️ Thread retrieval error for {named_r.doc_id}: {e}")
            return None

    # 🟢 FIX: แก้ไข _get_relevant_documents เพื่อส่ง filter เข้าไป
    def _get_relevant_documents(self, query: str, *, run_manager=None):
        """
        This is the LangChain compatible retrieval method.
        """
        max_workers = min(len(self._retrievers_list), MAX_PARALLEL_WORKERS)
        if max_workers <= 0:
            max_workers = 1

        # 🟢 FIX: _choose_executor ถูกเรียกใช้ที่นี่
        chosen = self._choose_executor() 
        ExecutorClass = ProcessPoolExecutor if chosen == "process" else ThreadPoolExecutor

        logger.info(f"⚙️ Running MultiDocRetriever with {chosen} executor ({max_workers} workers) [Filter: {bool(self._chroma_filter)}]")

        results = []
        if chosen == "process":
            # Use process pool
            with ExecutorClass(max_workers=max_workers) as executor:
                # 🟢 NEW: ส่ง self._chroma_filter เข้าไปใน task
                futures = [executor.submit(MultiDocRetriever._static_retrieve_task, nr, query, self._chroma_filter) for nr in self._retrievers_list]
                for f in futures:
                    try:
                        results.append(f.result())
                    except Exception as e:
                        logger.warning(f"Child process future failed: {e}")
                        results.append(None)
        else:
            # Use threads
            with ExecutorClass(max_workers=max_workers) as executor:
                # 🟢 NEW: ส่ง self._chroma_filter เข้าไปใน task
                futures = [executor.submit(self._thread_retrieve_task, nr, query, self._chroma_filter) for nr in self._retrievers_list]
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
                # ใช้ doc_type ที่หมายถึง Collection Name เป็นส่วนหนึ่งของ key
                key = f"{src}_{chunk}_{named_r.doc_type}_{d.page_content[:120]}" 
                if key not in seen:
                    seen.add(key)
                    # NamedRetriever.doc_id และ doc_type ตอนนี้คือ Collection Name
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
     
# -------------------- Load single vectorstore retriever (REVISED) --------------------
def load_vectorstore(doc_id: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, doc_types: Union[list, str] = "default_collection", base_path: str = VECTORSTORE_DIR, enabler: Optional[str] = None):
    """
    Loads a retriever instance for a specific collection name (doc_type) and optional enabler. 
    """
    # 🟢 NEW: ตรรกะใหม่ใช้ doc_types/enabler เพื่อสร้าง collection_name ที่ถูกต้อง
    if isinstance(doc_types, str):
        target_doc_type = doc_types
    elif isinstance(doc_types, list) and target_doc_type:
         target_doc_type = doc_types[0] # Use the first one if list is provided
    else:
        raise ValueError("doc_types must be a single string or a list containing the target doc_type.")

    # 1. คำนวณชื่อ Collection ที่ถูกต้อง
    collection_name = _get_collection_name(target_doc_type, enabler)

    manager = VectorStoreManager(base_path=base_path)
    retriever = None

    # 2. ตรวจสอบว่า Collection นี้มีอยู่จริงหรือไม่
    # NOTE: vectorstore_exists จะใช้ collection_name ที่ถูกต้องในการตรวจสอบพาธ
    if vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=target_doc_type, enabler=enabler):
        # 3. ใช้ manager เพื่อโหลด retriever
        retriever = manager.get_retriever(collection_name, top_k, final_k)
    
    if retriever is None:
        raise ValueError(f"❌ Vectorstore for collection '{collection_name}' (derived from doc_type='{target_doc_type}' and enabler='{enabler}') not found.")
    return retriever


# -------------------- load_all_vectorstores (FINAL REVISED WITH CONDITIONAL ENABLER FIX) --------------------
def load_all_vectorstores(doc_types: Optional[Union[str, List[str]]] = None,
                          top_k: int = INITIAL_TOP_K,
                          final_k: int = FINAL_K_RERANKED,
                          base_path: str = VECTORSTORE_DIR,
                          evidence_enabler: Optional[str] = None,
                          doc_ids: Optional[List[str]] = None) -> MultiDocRetriever:
    
    """
    Load multiple vectorstore collections as MultiDocRetriever.
    ... (Docstring เดิม) ...
    """
    doc_types = [doc_types] if isinstance(doc_types, str) else doc_types or []
    doc_type_filter = {dt.strip().lower() for dt in doc_types}
    
    manager = VectorStoreManager(base_path=base_path)
    all_retrievers: List[NamedRetriever] = []
    
    # 1. กำหนดรายการ Collection ที่ต้องการโหลดจริง ๆ (Collection Name) - (เหมือนเดิม)
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
                    # โหลดทุก Evidence Collection (ถ้าไม่มี Enabler ระบุ)
                    evidence_collections = list_vectorstore_folders(base_path=base_path, doc_type=EVIDENCE_DOC_TYPES)
                    target_collection_names.update(evidence_collections)
                    logger.info(f"🔍 Added all evidence collections found: {evidence_collections}")
            else:
                # สำหรับ document, faq (ใช้ None เป็น Enabler)
                collection_name = _get_collection_name(dt_norm, None)
                target_collection_names.add(collection_name)
                logger.info(f"🔍 Added standard collection: {collection_name}")
                
    # 2. ทำการโหลดทีละ Collection - (เหมือนเดิม)
    logger.info(f"🔍 DEBUG: Attempting to load {len(target_collection_names)} total target collections: {target_collection_names}")
    
    for collection_name in target_collection_names:
        doc_type_for_check, enabler_for_check = manager._re_parse_collection_name(collection_name)
        
        if not vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=doc_type_for_check, enabler=enabler_for_check):
            logger.warning(f"🔍 DEBUG: Skipping collection '{collection_name}' (vectorstore_exists failed).")
            continue
            
        nr = NamedRetriever(
            doc_id=collection_name, 
            doc_type=collection_name, 
            top_k=top_k,
            final_k=final_k,
            base_path=base_path
        )
        all_retrievers.append(nr)
        logger.info(f"🔍 DEBUG: Successfully added retriever for collection '{collection_name}'.")

    # 3. จัดการ Hard Filter ID (ใช้ 64-char ID ดั้งเดิมโดยตรง)
    final_filter_ids = doc_ids # ⬅️ ใช้ doc_ids (64-char UUIDs) ที่ส่งมาโดยตรง
    
    if doc_ids:
        # ⚠️ ไม่จำเป็นต้องวนลูปหรือ Map อีกต่อไป
        # เพียงแค่ใช้ ID 64-char ที่ส่งมาจาก Evidence Mapping
        logger.info(f"✅ Hard Filter Enabled: Using {len(doc_ids)} original 64-char UUIDs for filtering.")
        
    # 4. ส่ง doc_ids ที่เป็น 64-char UUIDs ไปให้ MultiDocRetriever กรองต่อ
    logger.info(f"🔍 DEBUG: Final count of all_retrievers = {len(all_retrievers)}")
    if not all_retrievers:
        raise ValueError(f"No vectorstore collections found matching doc_types={doc_types} and evidence_enabler={evidence_enabler}")

    return MultiDocRetriever(
            retrievers_list=all_retrievers, 
            k_per_doc=top_k,
            doc_ids_filter=final_filter_ids 
        )

# -------------------- VECTORSTORE EXECUTOR SINGLETON (RETAINED) --------------------
# REQUIRED by ingest_batch.py for shared resource management.
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

# -------------------------
# FlashrankRerank class
# -------------------------
class FlashrankRerank:
    def __init__(self, top_n: int = 5, device: str = "mps"):
        """
        top_n: จำนวนเอกสารสูงสุดที่จะคืน
        device: 'cpu', 'cuda', 'mps' ตาม hardware
        """
        self.top_n = top_n
        self.device = device
        self.model: Optional[SentenceTransformer] = None
        self.util = None
        self._load_model()

    def _load_model(self):
        """โหลด SentenceTransformer model"""
        try:
            logger.info(f"📦 Loading SentenceTransformer on device: {self.device}")
            self.model = SentenceTransformer("intfloat/multilingual-e5-large", device=self.device)
            self.util = util
        except Exception as e:
            logger.error(f"⚠️ Failed to load SentenceTransformer: {e}")
            self.model = None
            self.util = None

    def rerank(self, query: str | Any, docs: list) -> list:
        """
        Rerank documents ตามความใกล้ query
        - query: str หรือ embedding tensor
        - docs: list ของ LcDocument
        คืน list ของ LcDocument top_n
        """
        if not self.model or not docs:
            logger.warning("⚠️ Model not loaded or docs empty. Returning original docs truncated.")
            return docs[:self.top_n]

        # embedding query
        query_emb = self.model.encode(query, convert_to_tensor=True) if isinstance(query, str) else query

        # embedding docs
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = self.model.encode(doc_texts, convert_to_tensor=True)

        # cosine similarity
        scores = self.util.cos_sim(query_emb, doc_embs)[0]  # [1, n] tensor
        sorted_docs = [doc for _, doc in sorted(zip(scores.tolist(), docs), key=lambda x: x[0], reverse=True)]
        return sorted_docs[:self.top_n]


# -------------------------
# Global reranker instance
# -------------------------
GLOBAL_RERANKER = FlashrankRerank(top_n=5, device="mps")  # ปรับ device ตาม hardware

def get_global_reranker(top_n: int | None = None) -> FlashrankRerank:
    """
    คืน reranker instance
    - top_n ถ้าไม่ระบุหรือเท่ากับ FINAL_K_RERANKED -> ใช้ global instance
    - top_n ถ้าระบุ -> สร้าง instance ใหม่
    """
    if top_n is None or top_n == 5:  # หรือใช้ FINAL_K_RERANKED
        return GLOBAL_RERANKER
    return FlashrankRerank(top_n=top_n, device="mps")
