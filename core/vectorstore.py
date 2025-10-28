import os
import platform
import logging
import threading
import multiprocessing
import json 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Union, Sequence, Any, Dict, Set, Tuple

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
MAPPING_FILE_PATH = "data/doc_id_mapping.json" 

# Safety: don't spawn too many processes by default
MAX_PARALLEL_WORKERS = int(os.getenv("MAX_PARALLEL_WORKERS", "2"))

# Env override to force mode: "thread" or "process"
ENV_FORCE_MODE = os.getenv("VECTOR_MODE", "").lower()  # "thread", "process", or ""

# Logging
logger = logging.getLogger(__name__)

# 🟢 FIX: บังคับ Logger Level ให้แสดงผล INFO และ DEBUG
logger.setLevel(logging.INFO)
logger.handlers = logging.root.handlers # ทำให้ Logger ใช้ Handler ของ Root Logger (ซึ่งคือ Console)
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

logger = logging.getLogger(__name__) # สมมติว่า logger ถูกตั้งค่าแล้ว


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
                        # query_instruction="query: ",
                        # 2. แนะนำให้ normalize embeddings สำหรับ E5 เพื่อความแม่นยำ
                        # encode_kwargs={'normalize_embeddings': True}
                    )

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
            # ใช้ Flashrank rerank
            ranked_results = self.ranker.rerank(run_input)
        except Exception as e:
            logger.warning(f"⚠️ Flashrank.rerank failed: {e}. Returning original docs.")
            # Fallback to original documents
            ranked_results = [{"id": i, "score": 0.0} for i in range(len(doc_list_for_rerank))]

        reranked_docs = []
        for res in ranked_results:
            idx = res.get("id", 0)
            score = res.get("score", 0.0)
            original_doc = documents[idx]
            # เพิ่ม relevance_score ใน metadata
            reranked_docs.append(LcDocument(page_content=original_doc.page_content, metadata={**original_doc.metadata, "relevance_score": score}))
        return reranked_docs


# -------------------- Vectorstore helpers (REVISED/CLEANED) --------------------

def get_vectorstore_path(doc_type: Optional[str] = None):
    # ปรับปรุง: ตอนนี้ใช้ doc_type เป็น collection_name ใน Chroma
    return VECTORSTORE_DIR

def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> List[str]:
    """
    Lists available Chroma collections (which are folders inside VECTORSTORE_DIR).
    """
    if not os.path.exists(base_path):
        return []
    
    # List folders inside VECTORSTORE_DIR
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    if doc_type:
        # ถ้ากำหนด doc_type (ซึ่งตอนนี้คือ collection name) ให้ตรวจสอบว่ามี folder นั้นหรือไม่
        return [doc_type] if doc_type in folders else []
        
    return folders


# ใน core/vectorstore.py
def vectorstore_exists(doc_id: str, base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None) -> bool:
    """
    Checks if a Chroma collection (doc_type) exists on disk.
    doc_type is the collection name (e.g., 'document').
    """
    if not doc_type:
        return False
    
    # 1. สร้างพาธไปยังโฟลเดอร์ Collection
    path = os.path.join(base_path, doc_type)
    file_path = os.path.join(path, "chroma.sqlite3")
    
    # 🟢 DEBUG: แสดงพาธที่กำลังจะตรวจสอบ (เพื่อยืนยันว่าถูกต้อง)
    logger.info(f"🔍 V-Exists Check: Checking path: {file_path} (from CWD: {os.getcwd()})")
    
    if not os.path.isdir(path):
        logger.warning(f"❌ V-Exists Check 1: Directory not found at {path}")
        return False
        
    # 2. ตรวจสอบไฟล์ฐานข้อมูลหลัก
    if os.path.isfile(file_path):
        logger.info(f"✅ V-Exists Check 2: Found required file at {file_path}")
        return True
        
    # 🚨 DEBUG: หากถึงตรงนี้ แสดงว่าหาไฟล์ไม่เจอ (คือบั๊กที่เรากำลังหา)
    logger.error(f"❌ V-Exists Check 3: FAILED to find file chroma.sqlite3 at {file_path}")
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
    
    # Cache for Doc ID Mapping (doc_id -> [UUIDs])
    # NOTE: type hint นี้สอดคล้องกับการโหลดใน _load_doc_id_mapping
    _doc_id_mapping: Dict[str, List[str]] = PrivateAttr({})
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
            # 💡 Initialization is handled internally by helper functions
            self._embeddings = get_hf_embeddings()
            
            # Load mapping on startup
            self._load_doc_id_mapping()
            
            logger.info(f"Initialized VectorStoreManager. Loaded {len(self._doc_id_mapping)} doc IDs.")
            VectorStoreManager._is_initialized = True

    def _load_doc_id_mapping(self):
        """Loads doc_id_mapping.json into memory."""
        self._doc_id_mapping = {}
        self._uuid_to_doc_id = {}
        try:
            with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
                mapping_data: Dict[str, List[str]] = json.load(f)
                self._doc_id_mapping = mapping_data
                
                # Create reverse mapping for quick lookup UUID -> Doc ID
                for doc_id, uuids in mapping_data.items():
                    for uid in uuids:
                        self._uuid_to_doc_id[uid] = doc_id
                        
            logger.info(f"✅ Loaded Doc ID Mapping: {len(self._doc_id_mapping)} original documents, {len(self._uuid_to_doc_id)} total chunks.")
        except FileNotFoundError:
            logger.warning(f"⚠️ Doc ID Mapping file not found at {MAPPING_FILE_PATH}. This is expected if no documents have been ingested yet.")
        except Exception as e:
            logger.error(f"❌ Failed to load Doc ID Mapping: {e}")

    def _load_chroma_instance(self, collection_name: str) -> Optional[Chroma]:
        """Loads a Chroma instance from disk or returns from cache."""
        if collection_name in self._chroma_cache:
            return self._chroma_cache[collection_name]

        with self._lock:
            # Re-check cache after acquiring lock
            if collection_name in self._chroma_cache:
                return self._chroma_cache[collection_name]
            
            persist_directory = os.path.join(self._base_path, collection_name)
            
            # if not vectorstore_exists(doc_id="N/A", base_path=self._base_path, doc_type=collection_name):
            if not vectorstore_exists(doc_id=collection_name, base_path=self._base_path, doc_type=collection_name): # ลองส่ง doc_type ชัดๆ
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

    def get_documents_by_id(self, doc_uuids: Union[str, List[str]], doc_type: str = "default_collection") -> List[LcDocument]:
        """
        Retrieves chunks (Documents) from a specific Chroma collection 
        using their UUIDs.
        """
        if isinstance(doc_uuids, str):
            doc_uuids = [doc_uuids]
            
        doc_uuids = [uid for uid in doc_uuids if uid] # filter out None/empty strings
        if not doc_uuids:
            return []
            
        chroma_instance = self._load_chroma_instance(doc_type)
        if not chroma_instance:
            logger.warning(f"Cannot retrieve documents: Collection '{doc_type}' is not loaded.")
            return []
        
        try:
            # 1. Get collection client
            collection = chroma_instance._collection
            
            # 2. Fetch data by IDs
            # NOTE: Chroma client.get() returns dict with 'ids', 'documents', 'metadatas'
            result = collection.get(
                ids=doc_uuids,
                include=['documents', 'metadatas'] 
            )
            
            # 3. Process results into LangChain Documents
            documents: List[LcDocument] = []
            for i, text in enumerate(result.get('documents', [])):
                if text:
                    metadata = result.get('metadatas', [{}])[i]
                    doc_id = self._uuid_to_doc_id.get(result.get('ids', [''])[i], "UNKNOWN")
                    
                    # Ensure metadata contains necessary keys
                    metadata["chunk_uuid"] = result.get('ids', [''])[i]
                    metadata["doc_id"] = doc_id
                    metadata["doc_type"] = doc_type
                    
                    documents.append(LcDocument(page_content=text, metadata=metadata))
            
            logger.info(f"✅ Retrieved {len(documents)} documents for {len(doc_uuids)} UUIDs from '{doc_type}'.")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents by UUIDs from collection '{doc_type}': {e}")
            return []

    # -------------------- Retriever Creation --------------------

    def get_retriever(self, collection_name: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED) -> Optional[BaseRetriever]:
        """
        Returns a ContextualCompressionRetriever for a given collection_name (doc_type).
        """
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            return None
        
        # preload reranker model for main/threads
        reranker_instance = preload_reranker_model()
        
        base_retriever = chroma_instance.as_retriever(search_kwargs={"k": top_k})
        
        if reranker_instance:
            try:
                compressor = CustomFlashrankCompressor(ranker=reranker_instance, top_n=final_k)
                retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)
                if multiprocessing.current_process().name == 'MainProcess': 
                    logger.info(f"✅ Loaded Reranking Retriever for collection={collection_name} with k={top_k}->{final_k}")
                return retriever
            except Exception as e:
                logger.warning(f"⚠️ CustomFlashrankCompressor failed for {collection_name}: {e}. Falling back to base retriever.")
                return base_retriever
        else:
            logger.warning("⚙️ Reranker model not available. Using base retriever only.")
            return base_retriever

    def get_all_collection_names(self) -> List[str]:
        """Returns a list of all available collection names (folders in VECTORSTORE_DIR)."""
        return list_vectorstore_folders(base_path=self._base_path)
    
    # ภายในคลาส VectorStoreManager
    def get_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], doc_type: str) -> List[LcDocument]:
        """
        Retrieves chunks (Documents) for a list of Stable Document IDs.
        """
        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
            
        stable_doc_ids = [uid for uid in stable_doc_ids if uid]
        if not stable_doc_ids:
            return []
            
        all_chunk_uuids = []
        skipped_docs = []
        found_stable_ids = []

        # 1. ค้นหา Chunk UUIDs ทั้งหมดจาก Mapping
        for stable_id in stable_doc_ids:
            if stable_id in self._doc_id_mapping:
                # 🚨 FIX 1: ตรวจสอบโครงสร้าง Dict ภายใน Mapping ที่ถูกต้อง
                doc_entry = self._doc_id_mapping[stable_id] 
                
                # โครงสร้างที่ถูกต้องควรเป็น {"stable_id": {"chunk_uuids": ["uuid1", "uuid2", ...]}}
                if isinstance(doc_entry, dict) and 'chunk_uuids' in doc_entry:
                    chunk_uuids = doc_entry['chunk_uuids']
                    if chunk_uuids:
                        all_chunk_uuids.extend(chunk_uuids)
                        found_stable_ids.append(stable_id)
                    else:
                        logger.warning(f"Mapping found for Stable ID '{stable_id}' but 'chunk_uuids' list is empty.")
                else:
                    logger.warning(f"Mapping entry for Stable ID '{stable_id}' is malformed.")
            else:
                skipped_docs.append(stable_id)
                
        if skipped_docs:
            logger.warning(f"Skipping Stable IDs not found in mapping: {skipped_docs}")

        if not all_chunk_uuids:
            # 🚨 FIX 2: เปลี่ยน ValueError เป็นการคืนค่าว่างเปล่า (ตามความยืดหยุ่นของ RAG) 
            # แต่ถ้าต้องการให้ fail ชัดเจน ต้องคง ValueError ไว้
            logger.warning(f"No valid chunk UUIDs found for provided Stable Document IDs: {stable_doc_ids}. Check doc_id_mapping.json.")
            return []
            
        # 2. โหลด Chroma Instance
        try:
            chroma_instance = self._load_chroma_instance(doc_type)
        except Exception as e:
            logger.error(f"❌ Error loading Chroma instance for collection '{doc_type}': {e}")
            return []

        if not chroma_instance:
            logger.error(f"Collection '{doc_type}' is not loaded.")
            return []

        # 3. Fetch data by Chunk IDs
        try:
            collection = chroma_instance._collection
            result = collection.get(
                ids=all_chunk_uuids,
                include=['documents', 'metadatas'] 
            )
            
            # 4. Process results into LangChain Documents (โค้ดเดิม)
            documents: List[LcDocument] = []
            
            # 🚨 FIX 3: ตรวจสอบว่ามีเอกสารจริง ๆ ถูกดึงออกมาหรือไม่
            if not result.get('documents'):
                logger.warning(f"ChromaDB returned 0 documents for {len(all_chunk_uuids)} chunk UUIDs.")
                return []
                
            for i, text in enumerate(result.get('documents', [])):
                if text:
                    metadata = result.get('metadatas', [{}])[i]
                    doc_id = self._uuid_to_doc_id.get(result.get('ids', [''])[i], "UNKNOWN")
                    
                    # Ensure metadata contains necessary keys
                    metadata["chunk_uuid"] = result.get('ids', [''])[i]
                    metadata["doc_id"] = doc_id
                    metadata["doc_type"] = doc_type
                    
                    documents.append(LcDocument(page_content=text, metadata=metadata))
            
            logger.info(f"✅ Retrieved {len(documents)} chunks for {len(found_stable_ids)} Stable IDs from '{doc_type}'.")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error retrieving documents by Chunk UUIDs from collection '{doc_type}': {e}")
            return []

# -------------------- Load single vectorstore retriever (REVISED) --------------------
def load_vectorstore(doc_id: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, doc_types: Union[list, str] = "default_collection", base_path: str = VECTORSTORE_DIR):
    """
    Loads a retriever instance for a specific collection name (doc_type). 
    """
    if isinstance(doc_types, str):
        collection_names = [doc_types]
    else:
        collection_names = doc_types
    
    manager = VectorStoreManager(base_path=base_path)
    retriever = None

    for collection_name in collection_names:
        # Check if the collection exists
        if vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=collection_name):
            # Use the manager to get the retriever
            retriever = manager.get_retriever(collection_name, top_k, final_k)
            if retriever:
                break
    
    if retriever is None:
        raise ValueError(f"❌ Vectorstore for doc_id/collection_name '{doc_id}' not found in any of {collection_names}")
    return retriever


# -------------------- MultiDoc / Parallel Retriever (REVISED) --------------------
class NamedRetriever(BaseModel):
    """Picklable wrapper storing minimal params to load retriever inside child process."""
    doc_id: str # doc_id ในตรรกะใหม่จะหมายถึง Collection Name
    doc_type: str # doc_type ก็หมายถึง Collection Name ด้วย
    top_k: int
    final_k: int
    # เพิ่ม Base path เข้าไปเพื่อความปลอดภัย
    base_path: str = VECTORSTORE_DIR

    def load_instance(self) -> BaseRetriever:
        """Load a retriever instance inside the current process using stored params."""
        # ใช้ Collection Name (doc_type) ในการโหลด
        manager = VectorStoreManager(base_path=self.base_path)
        retriever = manager.get_retriever(self.doc_type, top_k=self.top_k, final_k=self.final_k)
        if not retriever:
             raise ValueError(f"Retriever not found for collection '{self.doc_type}' at path '{self.base_path}'")
        return retriever


class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple NamedRetriever sources. Choose executor automatically (thread vs process).
    """

    _retrievers_list: list[NamedRetriever] = PrivateAttr()
    _k_per_doc: int = PrivateAttr()
    # เก็บ reference ไปยัง VectorStoreManager
    _manager: VectorStoreManager = PrivateAttr() 

    def __init__(self, retrievers_list: list[NamedRetriever], k_per_doc: int = INITIAL_TOP_K):
        super().__init__()
        self._retrievers_list = retrievers_list
        self._k_per_doc = k_per_doc
        self._manager = VectorStoreManager() # สร้าง/โหลด Manager

    @staticmethod
    def _static_retrieve_task(named_r: "NamedRetriever", query: str):
        """
        Static helper used in ProcessPoolExecutor. Executes inside child process.
        Must return list[Document] or None.
        """
        try:
            # load_instance จะสร้าง VectorStoreManager ใหม่ใน child process ซึ่งจะโหลด cache ใหม่
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
            # load_instance จะใช้ Manager เดียวกันกับ Main process
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


# -------------------- load_all_vectorstores --------------------
def load_all_vectorstores(doc_types: Optional[Union[str, List[str]]] = None,
                          top_k: int = INITIAL_TOP_K,
                          final_k: int = FINAL_K_RERANKED,
                          base_path: str = VECTORSTORE_DIR) -> MultiDocRetriever:
    """
    Load multiple vectorstore collections as MultiDocRetriever.
    doc_types = collection folder names (e.g., 'document')
    """
    if isinstance(doc_types, str):
        doc_type_filter = {doc_types}
    elif isinstance(doc_types, list):
        doc_type_filter = set(doc_types)
    else:
        doc_type_filter = set()

    manager = VectorStoreManager(base_path=base_path)
    all_retrievers: List[NamedRetriever] = []

    # 🟢 DEBUG 1: ตรวจสอบ Filter และ Collection ที่มีอยู่
    available_collections = manager.get_all_collection_names()
    logger.info(f"🔍 DEBUG: Target doc_types={doc_types}, Filter={doc_type_filter}")
    logger.info(f"🔍 DEBUG: Found available collections: {available_collections}")

    for collection_name in manager.get_all_collection_names():
        if doc_type_filter and collection_name not in doc_type_filter:
            logger.info(f"🔍 DEBUG: Skipping collection '{collection_name}' (not in filter)")
            continue
        if not vectorstore_exists(doc_id=collection_name, base_path=base_path, doc_type=collection_name): # FIX HERE
            logger.warning(f"🔍 DEBUG: Collection '{collection_name}' folder exists but FAILED vectorstore_exists check.")
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

    # 🟢 DEBUG 3: สรุปผลลัพธ์
    logger.info(f"🔍 DEBUG: Final count of all_retrievers = {len(all_retrievers)}")
    if not all_retrievers:
        raise ValueError(f"No vectorstore collections found matching doc_types={doc_types}")

    return MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k)

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