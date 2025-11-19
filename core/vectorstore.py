#core/vectorstore.py
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


# External libraries
# ✅ FIXED: นำเข้า PrivateAttr, ConfigDict, BaseModel โดยตรงจาก pydantic
from pydantic import PrivateAttr, ConfigDict, BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# 🟢 FIX (Option C): แทนที่ Flashrank ด้วย Sentence Transformers CrossEncoder
try:
    from sentence_transformers import CrossEncoder 
except ImportError:
    # Placeholder class to avoid crash if not installed
    logger.warning("⚠️ CrossEncoder not found. Install 'sentence-transformers' to use this reranker.")
    class CrossEncoder:
        def __init__(self, model_name, device, max_length): pass
        def predict(self, sentences, show_progress_bar): return [0] * len(sentences)

# Configure chromadb telemetry if available
try:
    chromadb.configure(anonymized_telemetry=False)
except Exception:
    try:
        chromadb.settings = Settings(anonymized_telemetry=False)
    except Exception:
        pass

# -------------------- Global Config --------------------
# NOTE: ต้องแน่ใจว่า import ถูกต้องตามโครงสร้างโปรเจกต์ของคุณ
# สมมติว่าไฟล์นี้ถูกเรียกใช้และ config.global_vars สามารถถูกเข้าถึงได้
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

# Global caches (per process)
_CACHED_EMBEDDINGS = None
_EMBED_LOCK = threading.Lock()
_MPS_WARNING_SHOWN = False

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
    """
    Return best device string for HuggingFaceEmbeddings: 'cuda'|'mps'|'cpu' when available.
    """
    # avoid importing torch at top-level if not installed; check safely
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # mac mps support: torch.backends.mps.is_available() may exist
        # 🟢 CLEANUP: ใช้ platform.system() แทน sys.platform ในการตรวจสอบ platform
        if platform.system().lower() == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    # If no accelerator is found (or torch not installed), default to 'cpu'.
    return "cpu"


def get_hf_embeddings(device_hint: Optional[str] = None):
    """
    Return a HuggingFaceEmbeddings instance (cached per process).
    """
    global _CACHED_EMBEDDINGS, _MPS_WARNING_SHOWN
    device = device_hint or _detect_torch_device()

    # Safety: MPS + multiprocessing is fragile on macOS.
    sys_info = _detect_system()
    force_mode = ENV_FORCE_MODE
    
    # ใช้วิธีการตรวจสอบแบบง่าย (ถ้าไม่ใช้ Thread pool ถือว่าเป็น Process-parallel)
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
                    model_name = "intfloat/multilingual-e5-large"
                    logger.info(f"📦 Creating HuggingFaceEmbeddings (model={model_name}, device={device})")
                    
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                        model_name=model_name, 
                        model_kwargs={"device": device},
                        # query_instruction="query: ", 
                        # encode_kwargs={'normalize_embeddings': True} 
                    )

                except Exception as e:
                    logger.warning(f"⚠️ Failed to create embeddings on device={device}: {e}. Falling back to CPU.")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    return _CACHED_EMBEDDINGS
    

# -------------------- Vectorstore helpers (REVISED/CLEANED) --------------------

def _get_collection_name(doc_type: str, enabler: Optional[str] = None) -> str:
    """
    Calculates the Chroma collection name and directory name based on doc_type and enabler.
    """
    doc_type_norm = doc_type.strip().lower()

    if doc_type_norm == EVIDENCE_DOC_TYPES:
        enabler_norm = (enabler or "km").strip().lower() 
        collection_name = f"{doc_type_norm}_{enabler_norm}"
    else:
        collection_name = doc_type_norm
        
    logger.critical(f"🧭 DEBUG: _get_collection_name(doc_type={doc_type}, enabler={enabler}) => {collection_name}")
    
    return collection_name

def get_vectorstore_path(doc_type: Optional[str] = None, enabler: Optional[str] = None) -> str:
    """
    Returns the full path to the base dir or the specific collection directory.
    Uses _get_collection_name logic.
    """
    if not doc_type:
        return VECTORSTORE_DIR
    
    collection_name = _get_collection_name(doc_type, enabler)
    return os.path.join(VECTORSTORE_DIR, collection_name)

def list_vectorstore_folders(base_path: str = VECTORSTORE_DIR, doc_type: Optional[str] = None, enabler: Optional[str] = None) -> List[str]:
    """
    Lists available Chroma collections (which are folders inside VECTORSTORE_DIR).
    Returns collection names (e.g., 'document', 'evidence_km').
    """
    if not os.path.exists(base_path):
        return []
    
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    
    if doc_type:
        doc_type_norm = doc_type.lower().strip()
        
        if doc_type_norm == EVIDENCE_DOC_TYPES and not enabler:
            # Special case: 'evidence' without enabler means list ALL evidence_*
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
# Custom HuggingFace Cross-Encoder Compressor
# =================================================================

class HuggingFaceCrossEncoderCompressor(BaseDocumentCompressor, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # 🟢 FIX Pydantic NameError: เปลี่ยนชื่อ fields เพื่อหลีกเลี่ยงข้อขัดแย้งกับ Pydantic/LangChain
    # rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_model: str = "intfloat/multilingual-e5-large" # <<-- แก้ไขตรงนี้
    rerank_device: str = _detect_torch_device() 
    rerank_max_length: int = 512
    _cross_encoder: Any = PrivateAttr(None)
    
    def __init__(self, **data):
        super().__init__(**data)
        # Initialization logic is now fully handled in get_global_reranker
        pass 
        
    def set_encoder_instance(self, encoder: Any):
        """Method to manually set the globally created CrossEncoder instance."""
        self._cross_encoder = encoder

    def compress_documents(
        self,
        documents: Sequence[LcDocument],
        query: str,
        top_n: int, 
        callbacks: Optional[Any] = None,
    ) -> List[LcDocument]:
        
        if not documents:
            return []
        # Check for predict method
        if self._cross_encoder is None or not hasattr(self._cross_encoder, 'predict'):
            logger.error("HuggingFace Cross-Encoder is not initialized. Returning truncated documents.")
            return list(documents)[:top_n]

        # 1. Prepare inputs: (query, document_text) pairs
        # Sentence Transformers CrossEncoder expects a list of [query, document] pairs
        sentence_pairs = [[query, doc.page_content] for doc in documents]

        # 2. Perform Reranking (Prediction)
        try:
            # Predict returns a list of scores (logits)
            scores = self._cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        except Exception as e:
            logger.error(f"❌ Cross-Encoder prediction failed: {e}. Returning truncated documents.")
            return list(documents)[:top_n]

        # 3. Combine documents and scores, then sort
        doc_scores = sorted(
            zip(documents, scores), key=lambda x: x[1], reverse=True
        )

        # 4. Map back to LcDocuments and apply top_n
        final_docs = []
        for doc, score in doc_scores[:top_n]:
            # Add the relevance score to the metadata
            doc.metadata["relevance_score"] = float(score)
            final_docs.append(doc)
            
        return final_docs


# B. แก้ไขฟังก์ชัน get_global_reranker: รับประกันการสร้าง Encoder ครั้งเดียว

_CACHED_RERANKER_INSTANCE: Optional[HuggingFaceCrossEncoderCompressor] = None
_CACHED_CROSS_ENCODER: Any = None # Instance ของ sentence_transformers.CrossEncoder จริงๆ

def get_global_reranker(final_k: int) -> Optional[HuggingFaceCrossEncoderCompressor]:
    """
    Return a global (cached) HuggingFaceCrossEncoderCompressor instance.
    The actual CrossEncoder is initialized only once.
    """
    global _CACHED_RERANKER_INSTANCE, _CACHED_CROSS_ENCODER
    
    if _CACHED_RERANKER_INSTANCE is None:
        try:
            # 1. สร้าง Compressor Wrapper Instance
            instance = HuggingFaceCrossEncoderCompressor()
            
            if _CACHED_CROSS_ENCODER is None:
                # 🟢 FIX: เรียกใช้ชื่อ fields ใหม่
                model_name = instance.rerank_model 
                device = instance.rerank_device     
                
                logger.info(f"📦 Initializing global CrossEncoder (model={model_name}, device={device})")
                
                # 2. Try to create the actual CrossEncoder object
                try: 
                    from sentence_transformers import CrossEncoder # Import again inside the function for robustness
                    
                    # Call the CrossEncoder constructor
                    _CACHED_CROSS_ENCODER = CrossEncoder(
                        model_name_or_path=model_name, # <--- **แก้ไขตรงนี้**
                        device=device,
                        # 🟢 FIX: เรียกใช้ชื่อ fields ใหม่
                        max_length=instance.rerank_max_length 
                    )
                    logger.info("✅ CrossEncoder initialized successfully.")

                except ImportError:
                    logger.error("❌ FATAL: sentence-transformers library not found. Cannot initialize CrossEncoder.")
                    _CACHED_CROSS_ENCODER = None
                    
                except Exception as encoder_e:
                    # Catch PyTorch/device/download errors
                    logger.error(f"❌ FATAL: CrossEncoder constructor failed: {encoder_e}. Try running in CPU mode.")
                    _CACHED_CROSS_ENCODER = None 
                    
            # 3. ตรวจสอบและกำหนด Encoder ให้กับ Singleton
            if _CACHED_CROSS_ENCODER and hasattr(_CACHED_CROSS_ENCODER, 'predict'):
                 instance.set_encoder_instance(_CACHED_CROSS_ENCODER)
                 _CACHED_RERANKER_INSTANCE = instance
                 logger.critical("✅ Reranker set to HuggingFace Cross-Encoder.")
            else:
                 logger.error("❌ HuggingFace Cross-Encoder failed to initialize or missing 'predict' method. Reranking disabled.")
                 return None
                 
        except Exception as e:
            logger.error(f"❌ Failed to create global HuggingFaceCrossEncoderCompressor: {e}")
            _CACHED_RERANKER_INSTANCE = None 
            return None
    
    return _CACHED_RERANKER_INSTANCE
    
    
# -------------------- VECTORSTORE MANAGER (SINGLETON) --------------------
class VectorStoreManager:
    """
    Singleton class to manage and cache Chroma vectorstore instances (collections).
    Handles initialization of Embeddings, Reranker, and Doc ID Mapping.
    """
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

    # เพิ่ม close/del method เพื่อจัดการ Executor ของ MultiDocRetriever (หากมี)
    def close(self):
        """Cleanly shuts down all managed resources, including MultiDocRetriever's executor."""
        with self._lock:
            # 1. Shutdown MultiDocRetriever's Executor
            if self._multi_doc_retriever and hasattr(self._multi_doc_retriever, 'shutdown'):
                logger.info("Closing MultiDocRetriever executor via VSM.")
                self._multi_doc_retriever.shutdown()
                self._multi_doc_retriever = None
                
            # 2. Clear caches
            self._chroma_cache = {}
            # 3. รีเซ็ตสถานะ Singleton
            VectorStoreManager._is_initialized = False

    def __del__(self):
        """Fallback cleanup for the VSM Singleton."""
        self.close()
    
    def _load_doc_id_mapping(self):
            """Loads doc_id_mapping.json into memory."""
            self._doc_id_mapping = {}
            self._uuid_to_doc_id = {}
            try:
                with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
                    mapping_data: Dict[str, Dict[str, Any]] = json.load(f)
                    
                    # FIX: ทำความสะอาด (strip) คีย์ทั้งหมดของ Dictionary ทันทีที่โหลด
                    cleaned_mapping = {k.strip(): v for k, v in mapping_data.items()}
                    
                    self._doc_id_mapping = cleaned_mapping
                    
                    for doc_id, doc_entry in cleaned_mapping.items(): 
                        if isinstance(doc_entry, dict) and 'chunk_uuids' in doc_entry and isinstance(doc_entry.get('chunk_uuids'), list):
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
            
        return collection_name_lower, None 

    def _load_chroma_instance(self, collection_name: str) -> Optional[Chroma]:
        """Loads a Chroma instance from disk or returns from cache."""
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

    def get_documents_by_id(self, stable_doc_ids: Union[str, List[str]], doc_type: str = "default_collection", enabler: Optional[str] = None) -> List[LcDocument]:
                """
                Retrieves chunks (Documents) from a specific Chroma collection 
                using their **Stable Document UUIDs** (64-char IDs).
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
                    
                    # FIX: ใช้ "doc_id" เป็นคีย์ใน where clause และรวม 'ids'
                    result = collection.get(
                        where={"stable_doc_uuid": {"$in": stable_doc_ids}}, 
                        include=['documents', 'metadatas']
                    )
                    
                    documents: List[LcDocument] = []
                    num_docs = len(result.get('documents', []))
                    
                    for i in range(num_docs):
                        text = result['documents'][i]
                        if text:
                            metadata = result.get('metadatas', [{}])[i]
                            chunk_uuid_from_result = result.get('ids', [''])[i]
                            
                            doc_id = metadata.get("doc_id", "UNKNOWN") 
                            
                            metadata["chunk_uuid"] = chunk_uuid_from_result
                            metadata["doc_id"] = doc_id
                            metadata["doc_type"] = doc_type 
                            
                            documents.append(LcDocument(page_content=text, metadata=metadata))
                    
                    logger.info(f"✅ Retrieved {len(documents)} documents for {len(stable_doc_ids)} Stable IDs from '{collection_name}'.")
                    return documents
                    
                except Exception as e:
                    logger.error(f"❌ Error retrieving documents by Stable IDs from collection '{collection_name}': {e}")
                    return []

    def retrieve_by_chunk_ids(self, chunk_ids: List[str], collection_name: str) -> List[LcDocument]:
            """
            [NEW] Retrieves a list of LangChain Document objects based on their internal chunk UUIDs 
            (ซึ่งถูกใช้เป็น internal ID ใน ChromaDB สำหรับ Persistent Mapping).

            Args:
                chunk_ids: List ของ chunk_uuid strings (Chroma IDs).
                collection_name: ชื่อของ Collection ที่จะค้นหา.
                
            Returns:
                List of LcDocument objects (หรือรายการว่างถ้าดึงข้อมูลไม่สำเร็จ).
            """
            if not chunk_ids:
                return []
            
            try:
                # 1. โหลด Chroma instance
                chroma_instance = self._load_chroma_instance(collection_name)

                if not chroma_instance:
                    logger.error(f"VSM: Collection '{collection_name}' ไม่ถูกโหลดสำหรับดึง chunk IDs.")
                    return []
                    
                # 2. เข้าถึง internal Chroma collection
                collection = chroma_instance._collection 
                    
                # 3. ดึงข้อมูลจาก Vector Store ด้วย ID (ซึ่งคือ chunk_id)
                retrieval_result = collection.get(
                    ids=chunk_ids,
                    # ดึง content, metadata และ IDs ภายในกลับมา
                    include=['documents', 'metadatas'] 
                )
                
                # 4. ประมวลผลผลลัพธ์ให้อยู่ในรูปแบบ LcDocument
                retrieved_docs: List[LcDocument] = []
                
                documents = retrieval_result.get('documents', [])
                metadatas = retrieval_result.get('metadatas', [])
                ids = retrieval_result.get('ids', []) # Internal IDs (chunk_uuid)
                
                num_results = len(documents)
                if num_results != len(chunk_ids):
                    logger.warning(f"VSM: ดึงเอกสารได้ {num_results} ชิ้น, ร้องขอ {len(chunk_ids)} ชิ้น.")
                    
                # วนซ้ำเพื่อสร้าง LcDocument
                for content, metadata, chunk_id in zip(documents, metadatas, ids):
                    if content and isinstance(metadata, dict):
                        
                        # 📌 ตั้งค่า chunk_uuid (Chroma ID)
                        metadata['chunk_uuid'] = chunk_id
                        
                        # 📌 ค้นหา Stable Doc ID จาก Mapping ที่โหลดไว้ใน __init__
                        stable_doc_id = self._uuid_to_doc_id.get(chunk_id, metadata.get('stable_doc_uuid', "UNKNOWN"))
                        metadata["doc_id"] = stable_doc_id 
                        
                        # ตั้งค่า doc_type ถ้าไม่มี
                        metadata["doc_type"] = metadata.get("doc_type", self._re_parse_collection_name(collection_name)[0])
                            
                        # สร้าง LcDocument
                        retrieved_docs.append(LcDocument(
                            page_content=content,
                            metadata=metadata
                        ))
                    
                logger.info(f"VSM: ดึงเอกสาร Priority ได้ {len(retrieved_docs)} ชิ้นจาก Persistent Map สำหรับ '{collection_name}'.")
                return retrieved_docs

            except Exception as e:
                logger.error(f"VSM: Error ในการดึงข้อมูลตาม chunk ID สำหรับ collection '{collection_name}': {e}")
                return []
        
    def get_limited_chunks_from_doc_ids(
            self, 
            stable_doc_ids: Union[str, List[str]], 
            query: Union[str, List[str]], # 📌 รับได้ทั้ง str และ List[str]
            doc_type: str, 
            enabler: Optional[str] = None, 
            limit_per_doc: int = 5 
        ) -> List[LcDocument]:
            """
            Retrieves a limited number of chunks (Documents) for a list of Stable Document IDs 
            by performing a similarity search on the documents' chunks.
            """
            if isinstance(stable_doc_ids, str):
                stable_doc_ids = [stable_doc_ids]
                
            stable_doc_ids = [uid for uid in stable_doc_ids if uid]
            if not stable_doc_ids:
                return []
                
            # 📌 FIX 1: เลือกใช้ Query ตัวแทน (Query แรก) สำหรับ Vector Search
            if isinstance(query, list):
                # ใช้ query ตัวแรกเป็นตัวแทนสำหรับ Similarity Search
                vector_search_query = query[0] if query else ""
            else:
                vector_search_query = query
                
            if not vector_search_query:
                logger.warning("Limited chunk search skipped: Query is empty.")
                return []
                
            collection_name = _get_collection_name(doc_type, enabler)
            
            # 1. โหลด Chroma Instance
            chroma_instance = self._load_chroma_instance(collection_name) 

            if not chroma_instance:
                logger.error(f"Collection '{collection_name}' is not loaded.")
                return []

            all_limited_documents: List[LcDocument] = []
            total_chunks_retrieved = 0
            
            # 2. วนซ้ำเพื่อทำ Similarity Search แยกตาม Stable ID แต่ละตัว
            for stable_id in stable_doc_ids:
                stable_id_clean = stable_id.strip()

                # 2.1 สร้าง Filter เพื่อค้นหาเฉพาะ Chunks ภายใน Stable ID นี้
                doc_filter = {
                    "stable_doc_uuid": stable_id_clean
                }
                
                # 2.2 ใช้ ChromaRetriever เพื่อดึงข้อมูลที่เกี่ยวข้องที่สุด K ชิ้น
                try:
                    # ใช้ ChromaRetriever เพื่อเข้าถึง similarity_search พร้อม filter
                    custom_retriever = ChromaRetriever(
                        vectorstore=chroma_instance,
                        k=limit_per_doc, # K ถูกจำกัดตาม limit_per_doc
                        filter=doc_filter
                    )
                    
                    # ทำ Similarity Search โดยใช้ Query ตัวแทน
                    limited_docs = custom_retriever.get_relevant_documents(query=vector_search_query) # 📌 FIX 2: ใช้ vector_search_query

                    # 2.3 เพิ่ม metadata สำหรับ tracking
                    for doc in limited_docs:
                        doc.metadata['priority_search_type'] = 'limited_vector_search'
                        doc.metadata['priority_limit'] = limit_per_doc
                        all_limited_documents.append(doc)
                        
                    total_chunks_retrieved += len(limited_docs)

                except Exception as e:
                    logger.error(f"❌ Error performing limited vector search for Stable ID '{stable_id_clean}': {e}")
                    continue 
            
            logger.info(f"✅ Retrieved {total_chunks_retrieved} limited chunks (max {limit_per_doc}/doc) for {len(stable_doc_ids)} Stable IDs from '{collection_name}' using vector search.")
            return all_limited_documents

# -------------------- Retriever Creation --------------------
    def get_retriever(self, collection_name: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, use_rerank: bool = True) -> Any:
        """
        Loads the base Chroma retriever for the given collection, and returns a wrapper
        that applies reranking and final truncation (final_k).
        """
        # 1. Load Chroma instance
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.warning(f"Retriever creation failed: Collection '{collection_name}' not loaded.")
            return None 
            
        # 2. Create the base retriever (Chroma as_retriever)
        search_kwargs = {"k": top_k}
        try:
            base_retriever = chroma_instance.as_retriever(search_type="similarity", search_kwargs=search_kwargs)
        except Exception as e:
            logger.error(f"Failed to create base retriever for '{collection_name}': {e}")
            return None

        # 3. Apply Rerank/Truncation Wrapper
        
        if use_rerank:
            # Wrapper Function: ใช้ invoke_with_rerank ที่รองรับการรับ config (search_kwargs) จาก MultiDocRetriever
            def invoke_with_rerank(query: str, config: Optional[Dict] = None):
                
                # Initialize docs to prevent NameError
                docs = [] 
                
                # 1. ดึง filter จาก config ที่ส่งมาจาก MultiDocRetriever
                chroma_filter = config.get('configurable', {}).get('search_kwargs', {}).get('filter') if config else None
                
                try: # ห่อหุ้มการเรียกใช้ base_retriever.invoke() ด้วย try-except
                    # 2. ถ้ามี filter ให้ override search_kwargs ของ base_retriever
                    if chroma_filter:
                        # ใช้ top_k เดิมของ collection นี้สำหรับการดึงก่อน Rerank
                        new_config = {'configurable': {"search_kwargs": {"k": top_k, "filter": chroma_filter}}}
                        docs = base_retriever.invoke(query, config=new_config)
                    else:
                        docs = base_retriever.invoke(query, config=config)
                except Exception as e:
                    logger.error(f"❌ Retrieval failed before rerank: {e}")
                    return [] # หากดึงข้อมูลล้มเหลว ให้คืนค่าลิสต์ว่างทันที (docs ยังคงเป็น [])
                
                # 3. Perform Reranking
                try:
                    # 4. ใช้ HuggingFaceCrossEncoderCompressor (ผ่าน get_global_reranker)
                    # NOTE: final_k ที่ถูกส่งไปใน get_global_reranker (ในบรรทัด 380) ถูกใช้แค่ในการ Init ตัว Compressor
                    reranker = get_global_reranker(final_k) 

                    if reranker and hasattr(reranker, 'compress_documents'):
                        # // 🟢 FIX: Rerank และ TRUNCATE ผลลัพธ์กลับไปที่ top_k (30/50) แทน final_k (5)
                        # // เพื่อให้ llm_data_utils.py ได้เอกสารเต็มจำนวนที่ถูกจัดเรียงแล้วไปประมวลผลต่อ
                        return reranker.compress_documents(docs, query, top_n=top_k) 
                    
                    logger.warning("⚠️ Reranker not available. Returning base docs truncated.")
                    return docs[:top_k] 
                    
                except Exception as e:
                    logger.warning(f"⚠️ Rerank failed: {e}. Returning base docs truncated to {top_k}")
                    return docs[:top_k] 

            # Wrapper class: ห่อหุ้มฟังก์ชัน invoke
            class SimpleRetrieverWrapper(BaseRetriever):
                model_config = ConfigDict(arbitrary_types_allowed=True)
                def invoke(self, query: str, config: Optional[Dict] = None):
                    return invoke_with_rerank(query, config=config)
                # สำหรับ BaseRetriever ใหม่, ต้องมี _get_relevant_documents
                def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LcDocument]:
                    config = None # LangChain config is typically passed via the 'invoke' method.
                    return invoke_with_rerank(query, config=config)


            return SimpleRetrieverWrapper()
        else:
            # No Rerank, just Truncate
            class TruncatedRetrieverWrapper(BaseRetriever):
                model_config = ConfigDict(arbitrary_types_allowed=True)
                def invoke(self, query: str, config: Optional[Dict] = None):
                    # ถ้ามีการส่ง filter มา ให้ override search_kwargs ของ base_retriever
                    chroma_filter = config.get('configurable', {}).get('search_kwargs', {}).get('filter') if config else None
                    if chroma_filter:
                        new_config = {'configurable': {"search_kwargs": {"k": top_k, "filter": chroma_filter}}}
                        docs = base_retriever.invoke(query, config=new_config)
                    else:
                        docs = base_retriever.invoke(query, config=config)
                        
                    return docs[:top_k] 
                
                def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LcDocument]:
                    config = run_manager.get_session_info() if run_manager else None
                    return self.invoke(query, config=config)

            return TruncatedRetrieverWrapper()

    def get_all_collection_names(self) -> List[str]:
        """Returns a list of all available collection names (folders in VECTORSTORE_DIR)."""
        return list_vectorstore_folders(base_path=self._base_path)
    
    def get_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], doc_type: str, enabler: Optional[str] = None) -> List[LcDocument]:
            """
            Retrieves chunks (Documents) for a list of Stable Document IDs from a specific collection.
            """
            if isinstance(stable_doc_ids, str):
                stable_doc_ids = [stable_doc_ids]
                
            stable_doc_ids = [uid for uid in stable_doc_ids if uid]
            if not stable_doc_ids:
                return []
                
            collection_name = _get_collection_name(doc_type, enabler)

            all_chunk_uuids = []
            skipped_docs = []
            found_stable_ids = []

            # 1. ค้นหา Chunk UUIDs ทั้งหมดจาก Mapping
            for stable_id in stable_doc_ids:
                stable_id_clean = stable_id.strip()

                if stable_id_clean in self._doc_id_mapping:
                    doc_entry = self._doc_id_mapping[stable_id_clean] 
                    
                    if isinstance(doc_entry, dict) and 'chunk_uuids' in doc_entry and isinstance(doc_entry.get('chunk_uuids'), list):
                        chunk_uuids = doc_entry['chunk_uuids']
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
                    logger.warning(f"Chroma DB returned 0 documents for {len(all_chunk_uuids)} chunk UUIDs in collection '{collection_name}'.")
                    return []
                    
                for i, text in enumerate(result.get('documents', [])):
                    if text:
                        metadata = result.get('metadatas', [{}])[i]
                        chunk_uuid_from_result = result.get('ids', [''])[i]
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

# Helper function to get the manager instance
def get_vectorstore_manager() -> VectorStoreManager:
    """Returns the singleton instance of VectorStoreManager."""
    return VectorStoreManager()
    
# Backward compatibility function 
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
            self.max_workers = MAX_PARALLEL_WORKERS
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
    
def load_all_vectorstores_dummy():
    """Placeholder for loading all vectorstores if needed globally."""
    pass


# -------------------- Custom Retriever for Chroma --------------------
class ChromaRetriever(BaseRetriever):
    """
    A simple custom retriever wrapper that uses the underlying Chroma vectorstore
    instance to retrieve documents with specific k and filter parameters.
    """
    vectorstore: Any
    k: int
    filter: Optional[Dict] = None
    
    # กำหนดให้ pydantic อนุญาตประเภทที่ไม่ใช่ Base Model
    model_config = ConfigDict(arbitrary_types_allowed=True) 

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
        """
        Synchronous retrieval method for the Chroma instance.
        """
        try:
            # self.vectorstore คือ Chroma instance
            # Chroma.similarity_search รองรับ query, k, และ filter โดยตรง
            return self.vectorstore.similarity_search(
                query=query, 
                k=self.k, 
                filter=self.filter 
            )
        except Exception as e:
            logger.error(f"❌ Chroma similarity_search failed in custom retriever: {e}")
            return []
    
    # Method สาธารณะที่โค้ดเรียกใช้ (get_relevant_documents)
    def get_relevant_documents(self, query: str, **kwargs) -> List[LcDocument]:
        """Public synchronous method for retrieval."""
        return self._get_relevant_documents(query, **kwargs)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
        # สำหรับ Chroma, เราสามารถเรียกใช้ sync method ได้โดยตรง
        return self._get_relevant_documents(query, run_manager=run_manager)

# -------------------- END Custom Retriever --------------------

# -------------------- MultiDoc / Parallel Retriever --------------------
class NamedRetriever(BaseModel):
    # BaseRetriever requires model_config
    model_config = ConfigDict(arbitrary_types_allowed=True) 
    
    doc_id: str
    doc_type: str
    top_k: int
    final_k: int
    base_path: str = VECTORSTORE_DIR
    enabler: Optional[str] = None

    def load_instance(self) -> Any:
        manager = VectorStoreManager(base_path=self.base_path)
        # set use_rerank=True 
        retriever = manager.get_retriever(
            collection_name=_get_collection_name(self.doc_type, self.enabler),
            top_k=self.top_k,
            final_k=self.final_k,
            use_rerank=True
        )
        if not retriever:
            raise ValueError(f"Retriever not found for collection '{_get_collection_name(self.doc_type, self.enabler)}' at path '{self.base_path}'")
        return retriever


class MultiDocRetriever(BaseRetriever):
    """
    Combine multiple NamedRetriever sources. Choose executor automatically (thread vs process).
    """
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
        
        # Initialize _all_retrievers และโหลด instance
        self._all_retrievers = {} 
        for named_r in retrievers_list:
            collection_name = _get_collection_name(named_r.doc_type, named_r.enabler)
            try:
                # โหลด Retriever Instance ด้วย Key ที่ถูกต้อง (collection_name)
                retriever_instance = named_r.load_instance()
                if retriever_instance:
                    self._all_retrievers[collection_name] = retriever_instance # 👈 ใช้ _all_retrievers
                    logger.info(f"✅ MultiDocRetriever cached collection: {collection_name}")
                else:
                    logger.warning(f"⚠️ Failed to load instance for {collection_name} during MDR init.")
            except Exception as e:
                logger.error(f"❌ Error loading instance {collection_name} into MDR cache: {e}")
        # -------------------- END NEW LOGIC --------------------

        # NEW LOGIC: สร้าง Chroma Filter จาก Doc IDs
        self._doc_ids_filter = doc_ids_filter
        self._chroma_filter = None
        if doc_ids_filter:
            # สร้าง Chroma DB Metadata Filter: ค้นหาเฉพาะเอกสารที่มี 'doc_id' อยู่ในลิสต์
            self._chroma_filter = {
                "doc_id": {"$in": doc_ids_filter} 
            }

            logger.info(f"✅ MultiDocRetriever initialized with doc_ids filter for {len(doc_ids_filter)} Stable IDs.")
            
        self._executor_type = self._choose_executor()
        logger.info(f"MultiDocRetriever will use executor type: {self._executor_type} (workers={MAX_PARALLEL_WORKERS})")
        
    
    # เมธอด _choose_executor ที่หายไป
    def _choose_executor(self) -> str:
        """
        Decide whether to use ProcessPoolExecutor or ThreadPoolExecutor.
        """
        sys_info = _detect_system()
        device = _detect_torch_device()
        force = ENV_FORCE_MODE

        # 1. Force mode if user set
        if force in ("thread", "process"):
            mode = force
            logger.info(f"VECTOR_MODE override: forcing '{mode}' executor")
            return mode
        
        # 2. MPS + Multiprocessing Safety on macOS
        if sys_info["platform"] == "darwin" and device == "mps":
            logger.warning("⚠️ Detected MPS on macOS: forcing executor -> thread to avoid multi-process failures.")
            return "thread"
            
        # 3. Low RAM check 
        if sys_info["total_ram_gb"] and sys_info["total_ram_gb"] < 12:
            logger.warning(f"⚠️ Detected low RAM ({sys_info['total_ram_gb']:.1f}GB): forcing executor -> thread.")
            return "thread"
            
        # 4. High Resource check: Prefer ProcessPoolExecutor
        if sys_info["cpu_count"] >= 8 and (sys_info["total_ram_gb"] or 0) >= 16:
            logger.info("High-resources machine detected -> choosing 'process' executor")
            return "process"

        # 5. Default
        logger.info("Defaulting to 'thread' executor")
        return "thread"

    def shutdown(self):
        """Cleanly shuts down the internal executor if it was created."""
        if self._executor:
            executor_type_name = "ProcessPoolExecutor" if self._executor_type == "process" else "ThreadPoolExecutor"
            workers = self._executor._max_workers if hasattr(self._executor, '_max_workers') else "N/A"
            logger.info(f"Shutting down MultiDocRetriever's {executor_type_name} executor ({workers} workers).")
            
            self._executor.shutdown(wait=True)
            self._executor = None
            
    def __del__(self):
        """Fallback cleanup. Attempts to shutdown the executor when the object is garbage collected."""
        self.shutdown() 
        
    def _get_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Returns the cached or newly created executor based on the chosen type."""
        if self._executor is None:
            workers = MAX_PARALLEL_WORKERS
            if self._executor_type == "process":
                self._executor = ProcessPoolExecutor(max_workers=workers)
                logger.info(f"🛠️ Using ProcessPoolExecutor with {workers} workers.")
            else:
                self._executor = ThreadPoolExecutor(max_workers=workers)
                logger.info(f"🛠️ Using ThreadPoolExecutor with {workers} workers.")
        return self._executor

    # แก้ไข _static_retrieve_task ให้รับ filter
    @staticmethod
    def _static_retrieve_task(named_r: NamedRetriever, query: str, chroma_filter: Optional[Dict]):
        """
        Static helper used in ProcessPoolExecutor. Executes inside child process.
        """
        try:
            retriever_instance = named_r.load_instance()
            
            # NEW: ส่ง filter ผ่าน search_kwargs
            search_kwargs = {"k": named_r.top_k, "filter": chroma_filter} if chroma_filter else {"k": named_r.top_k}
            config = {'configurable': {'search_kwargs': search_kwargs}}

            docs = retriever_instance.invoke(query, config=config)
            
            # 3. Add source info
            for doc in docs:
                doc.metadata["retrieval_source"] = named_r.doc_type
                doc.metadata["collection_name"] = _get_collection_name(named_r.doc_type, named_r.enabler)
            
            return docs
        except Exception as e:
            # ใช้ print สำหรับ exceptions ใน child processes
            print(f"❌ Child retrieval error for {named_r.doc_id} ({named_r.doc_type}): {e}")
            return []

    # แก้ไข _thread_retrieve_task ให้รับ filter
    def _thread_retrieve_task(self, named_r: NamedRetriever, query: str, chroma_filter: Optional[Dict]):
        """
        Retrieval performed in a thread inside the same process.
        """
        try:
            retriever_instance = named_r.load_instance()
            
            # NEW: ส่ง filter ผ่าน search_kwargs
            search_kwargs = {"k": named_r.top_k, "filter": chroma_filter} if chroma_filter else {"k": named_r.top_k}
            config = {'configurable': {'search_kwargs': search_kwargs}}
            
            docs = retriever_instance.invoke(query, config=config)

            # 3. Add source info
            for doc in docs:
                doc.metadata["retrieval_source"] = named_r.doc_type
                doc.metadata["collection_name"] = _get_collection_name(named_r.doc_type, named_r.enabler)
            
            return docs
        except Exception as e:
            logger.warning(f"⚠️ Thread retrieval error for {named_r.doc_id}: {e}")
            return []

    # แก้ไข _get_relevant_documents เพื่อส่ง filter เข้าไป
    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[LcDocument]:
        """
        Runs multiple retrievers in parallel using the chosen executor, and aggregates results.
        This method is required by BaseRetriever.
        """
        max_workers = min(len(self._retrievers_list), MAX_PARALLEL_WORKERS)
        if max_workers <= 0:
            max_workers = 1

        chosen = self._executor_type # ใช้ _executor_type ที่ถูก set ใน __init__

        logger.info(f"⚙️ Running MultiDocRetriever with {chosen} executor ({max_workers} workers) [Filter: {bool(self._chroma_filter)}]")

        all_docs: List[LcDocument] = []
        
        # ใช้ _get_executor() เพื่อสร้างหรือเรียก Executor ที่แคชไว้
        executor = self._get_executor() 
        
        futures = []
        for named_r in self._retrievers_list:
            if chosen == "process":
                # For process, use the static method
                # NEW: ส่ง self._chroma_filter เข้าไปใน task
                future = executor.submit(MultiDocRetriever._static_retrieve_task, named_r, query, self._chroma_filter)
            else:
                # For thread, use the instance method
                # NEW: ส่ง self._chroma_filter เข้าไปใน task
                future = executor.submit(self._thread_retrieve_task, named_r, query, self._chroma_filter)
            futures.append(future)

        # Wait for all tasks to complete and collect results
        for f in futures:
            try:
                docs = f.result()
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Future failed: {e}")

        # Combine results and deduplicate
        seen = set()
        unique_docs = []
        for d in all_docs:
            # dedupe key: source + chunk + doc_id + a snippet
            # ใช้ metadata ที่ถูกเพิ่มโดย worker (retrieval_source และ collection_name)
            src = d.metadata.get("retrieval_source") or ""
            chunk_uuid = d.metadata.get("chunk_uuid") or d.metadata.get("ids") or ""
            key = f"{src}_{chunk_uuid}_{d.page_content[:120]}" 
            
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)

        logger.info(f"📝 Query='{query[:80]}...' found {len(unique_docs)} unique docs across sources (Executor={chosen})")
        
        # Final log of top documents
        for d in unique_docs:
            score = d.metadata.get("relevance_score")
            if score is not None:
                logger.debug(f" - [Reranked] Source={d.metadata.get('doc_type')}, Score={score:.4f}, Content='{d.page_content[:80]}...'")
        
        return unique_docs
     
    # Required method for BaseRetriever
    def get_relevant_documents(self, query: str, **kwargs) -> List[LcDocument]:
        """Synchronous public method for retrieval."""
        return self._get_relevant_documents(query, **kwargs)


# -------------------- Load single vectorstore retriever (REVISED) --------------------
def load_vectorstore_retriever(doc_id: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, doc_types: Union[list, str] = "default_collection", base_path: str = VECTORSTORE_DIR, enabler: Optional[str] = None):
    """
    Loads a retriever instance for a specific collection name (doc_type) and optional enabler. 
    """
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
        raise ValueError(f"❌ Vectorstore for collection '{collection_name}' (derived from doc_type='{target_doc_type}' and enabler='{enabler}') not found.")
    return retriever


# -------------------- load_all_vectorstores (FINAL REVISED WITH CONDITIONAL ENABLER FIX) --------------------
def load_all_vectorstores(doc_types: Optional[Union[str, List[str]]] = None,
                          top_k: int = INITIAL_TOP_K,
                          final_k: int = FINAL_K_RERANKED,
                          base_path: str = VECTORSTORE_DIR,
                          evidence_enabler: Optional[str] = None,
                          doc_ids: Optional[List[str]] = None) -> VectorStoreManager: 
    
    """
    Load multiple vectorstore collections as MultiDocRetriever.
    """
    doc_types = [doc_types] if isinstance(doc_types, str) else doc_types or []
    doc_type_filter = {dt.strip().lower() for dt in doc_types}
    
    manager = VectorStoreManager(base_path=base_path)
    all_retrievers: List[NamedRetriever] = []
    
    # 1. กำหนดรายการ Collection ที่ต้องการโหลดจริง ๆ (Collection Name)
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
                
    # 2. ทำการโหลดทีละ Collection
    logger.info(f"🔍 DEBUG: Attempting to load {len(target_collection_names)} total target collections: {target_collection_names}")
    
    for collection_name in target_collection_names:
        doc_type_for_check, enabler_for_check = manager._re_parse_collection_name(collection_name)
        
        if not vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=doc_type_for_check, enabler=enabler_for_check):
            logger.warning(f"🔍 DEBUG: Skipping collection '{collection_name}' (vectorstore_exists failed).")
            continue
            
        nr = NamedRetriever(
            doc_id=collection_name, 
            doc_type=doc_type_for_check, # ใช้ doc_type ที่แยกออกมา (e.g., 'evidence')
            enabler=enabler_for_check,   # ใช้ enabler ที่แยกออกมา (e.g., 'KM')
            top_k=top_k,
            final_k=final_k,
            base_path=base_path
        )
        all_retrievers.append(nr)
        logger.info(f"🔍 DEBUG: Successfully added retriever for collection '{collection_name}'.")

    # 3. จัดการ Hard Filter ID (ใช้ 64-char ID ดั้งเดิมโดยตรง)
    final_filter_ids = doc_ids 
    
    if doc_ids:
        logger.info(f"✅ Hard Filter Enabled: Using {len(doc_ids)} original 64-char UUIDs for filtering.")
        
    # 4. สร้าง MultiDocRetriever
    logger.info(f"🔍 DEBUG: Final count of all_retrievers = {len(all_retrievers)}")
    if not all_retrievers:
        raise ValueError(f"No vectorstore collections found matching doc_types={doc_types} and evidence_enabler={evidence_enabler}")

    mdr = MultiDocRetriever(
            retrievers_list=all_retrievers, 
            k_per_doc=top_k,
            doc_ids_filter=final_filter_ids 
        )
        
    # ผูก MultiDocRetriever เข้ากับ VectorStoreManager
    manager._multi_doc_retriever = mdr
    
    # เปลี่ยนการเข้าถึงจาก .all_retrievers เป็น ._all_retrievers
    logger.info(f"✅ MultiDocRetriever loaded with {len(mdr._all_retrievers)} collections and cached in VSM.")

    # คืนค่า VectorStoreManager แทน
    return manager