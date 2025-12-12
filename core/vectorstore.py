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
from pathlib import Path
import hashlib
from threading import Lock

# system utils
try:
    import psutil
except ImportError:
    psutil = None

# LangChain-ish imports (adjust to your project's versions)
from langchain_core.documents import Document as LcDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables import Runnable 

# Pydantic helpers
from pydantic import PrivateAttr, ConfigDict, BaseModel

# Chroma / HF embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# 💡 NEW: Import Path Utilities (อัปเดตชื่อฟังก์ชันให้ตรงกับ utils/path_utils.py ใหม่)
from utils.path_utils import (
    get_doc_type_collection_key, # ใช้แทน _get_collection_name
    get_vectorstore_collection_path, 
    get_vectorstore_tenant_root_path,
    get_mapping_file_path, # ใช้แทนทั้ง year_specific และ tenant_root
    # ไม่ต้องใช้ get_vectorstore_collection_parent_dir แล้ว
)


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

# -------------------- Global Config (Path Vars Removed) --------------------
from config.global_vars import (
    # 💥 ลบ VECTORSTORE_DIR, MAPPING_BASE_DIR
    FINAL_K_RERANKED,
    INITIAL_TOP_K,
    EVIDENCE_DOC_TYPES,
    MAX_PARALLEL_WORKERS,
    DEFAULT_TENANT,
    DEFAULT_YEAR,
    DEFAULT_ENABLER,
    RERANKER_MODEL_NAME,
    EMBEDDING_MODEL_NAME
)
# -------------------- Vectorstore Constants --------------------
ENV_FORCE_MODE = os.getenv("VECTOR_MODE", "").lower()  # "thread", "process", or ""
ENV_DISABLE_ACCEL = os.getenv("VECTOR_DISABLE_ACCEL", "").lower() in ("1", "true", "yes")

# Global caches (per process)
_CACHED_EMBEDDINGS = None
_EMBED_LOCK = threading.Lock()
_MPS_WARNING_SHOWN = False

# -------------------- Helper: detect environment & device --------------------
@staticmethod
def _detect_system():
    cpu_count = os.cpu_count() or 4
    total_ram_gb = None
    if psutil:
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            total_ram_gb = None
    return {"cpu_count": cpu_count, "total_ram_gb": total_ram_gb, "platform": platform.system().lower()}

@staticmethod
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
    global _CACHED_EMBEDDINGS, _MPS_WARNING_SHOWN
    device = device_hint or _detect_torch_device()

    if _CACHED_EMBEDDINGS is None:
        # สมมติว่ามีการใช้ _EMBED_LOCK เพื่อจัดการ thread safe
        # with _EMBED_LOCK: 
        if _CACHED_EMBEDDINGS is None:
            
            # 🟢 เปลี่ยนมาใช้ Global Variable ที่กำหนดไว้ใน global_vars.py
            model_name = EMBEDDING_MODEL_NAME 

            logger.info(f"Loading BEST Thai RAG embedding 2025: {model_name} on {device}")
            logger.info("This model will be used to build ALL PEA 2568 vectorstores (evidence_km, document, etc.)")
            
            try:
                # 
                _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={
                        "device": device,
                        # BGE-M3 ไม่จำเป็นต้องมี prefix!
                    },
                    encode_kwargs={
                        "normalize_embeddings": True,
                        # การลบ 'prompt': 'query:' ออก สำหรับ BGE-M3 นั้น ถูกต้องแล้วครับ!
                    }
                )
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                logger.warning("Falling back to paraphrase-multilingual-MiniLM-L12-v2")
                # ใช้ Fallback model ตัวเดิม
                _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs={"device": "cpu"}
                )
    return _CACHED_EMBEDDINGS

# =================================================================
# HuggingFace Cross-Encoder Reranker wrapper (singleton)
# =================================================================
class HuggingFaceCrossEncoderCompressor(BaseDocumentCompressor, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rerank_model: str = RERANKER_MODEL_NAME
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

def get_global_reranker() -> Optional[HuggingFaceCrossEncoderCompressor]:
    """
    Returns a cached HuggingFaceCrossEncoderCompressor instance (singleton).
    """
    global _CACHED_RERANKER_INSTANCE, _CACHED_CROSS_ENCODER

    if _CACHED_RERANKER_INSTANCE is None:
        try:
            if not _HAS_SENT_TRANS:
                logging.warning("sentence-transformers not installed. Cross-Encoder reranker disabled.")
                return None

            instance = HuggingFaceCrossEncoderCompressor(
                # rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1"
                rerank_model=RERANKER_MODEL_NAME
            )

            from sentence_transformers import CrossEncoder
            cross_encoder_model = CrossEncoder(
                instance.rerank_model,
                device=instance.rerank_device
            )

            instance.set_encoder_instance(cross_encoder_model)

            _CACHED_RERANKER_INSTANCE = instance
            _CACHED_CROSS_ENCODER = cross_encoder_model

            logging.info(f"Initialized global Cross-Encoder reranker: {instance.rerank_model} on {instance.rerank_device}")

        except Exception as e:
            logging.warning(f"Failed to initialize global reranker: {e}")
            return None

    return _CACHED_RERANKER_INSTANCE

# -------------------- Path Helper Function (REVISED to use Path Utility) --------------------
# ⚠️ ลบฟังก์ชัน _get_collection_name (ใช้ get_doc_type_collection_key แทน)

def get_vectorstore_path(
    tenant: str, 
    year: Optional[int], 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None
) -> str:
    """
    Calculates the full persist directory path for the vector store instance
    based on the Centralized KM Logic by calling path_utils.
    """
    # 🎯 FIX: เปลี่ยนไปใช้ get_vectorstore_tenant_root_path
    if not doc_type:
        return get_vectorstore_tenant_root_path(tenant=tenant)
        
    # 🎯 FIX: เปลี่ยนไปใช้ get_vectorstore_collection_path
    return get_vectorstore_collection_path(
        tenant=tenant, 
        year=year, 
        doc_type=doc_type, 
        enabler=enabler, 
        # EVIDENCE_DOC_TYPES ไม่จำเป็นต้องส่งแล้ว ถูก hardcode ใน path_utils
    )

def vectorstore_exists(
    doc_id: str = "N/A", # รักษาไว้ตาม Signature เดิม
    tenant: str = DEFAULT_TENANT,
    year: Optional[int] = DEFAULT_YEAR, # <--- รองรับ None สำหรับ General Docs
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = "" # base_path ถูกละเลย เนื่องจากใช้ global VECTORSTORE_DIR ภายใน path_utils
) -> bool:
    """
    Checks if the Vector Store directory exists for the given context.
    """
    if not doc_type:
        return False
        
    # 1. Get the full path using the updated logic (เรียก get_vectorstore_path ใหม่)
    # NOTE: get_vectorstore_path จะคำนวณ Path ที่ถูกต้องโดยอิงจาก doc_type (มีปี/ไม่มีปี)
    path = get_vectorstore_path(tenant, year, doc_type, enabler) 
    
    # 2. Check for the actual data file created by Chroma
    file_path = os.path.join(path, "chroma.sqlite3")
    
    if not os.path.isdir(path):
        logger.warning(f"❌ V-Exists Check: Directory not found for doc_type '{doc_type}' at {path}")
        return False
    if os.path.isfile(file_path):
        return True
    logger.error(f"❌ V-Exists Check: FAILED to find file chroma.sqlite3 at {file_path} in {path}")
    return False

# ⚠️ ลบฟังก์ชัน _get_collection_parent_dir ออก (Logic ถูกยุบรวมและถูกใช้งานโดยตรงใน list_vectorstore_folders)

def list_vectorstore_folders(
    tenant: str, 
    year: int, # NOTE: ใช้ปีปัจจุบันที่รัน (int)
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = "" # base_path ถูกละเลย
) -> List[str]:
    """
    Lists the actual collection folder names (e.g., 'evidence_km', 'document')
    that exist under the specified tenant and year context.
    """
    # 🎯 FIX: ใช้ get_vectorstore_tenant_root_path แทน
    tenant_root = get_vectorstore_tenant_root_path(tenant) # VECTORSTORE_DIR / tenant
    
    # Scenario 1: Specific doc_type/enabler is requested
    if doc_type:
        doc_type_norm = doc_type.lower().strip()
        # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
        collection_name = get_doc_type_collection_key(doc_type_norm, enabler)
        
        # ต้องคำนวณ target_year ที่ถูกต้องก่อน (None สำหรับ General Docs)
        target_year = year
        if doc_type_norm != EVIDENCE_DOC_TYPES.lower():
            target_year = None
            
        # 🎯 FIX: ใช้ get_vectorstore_collection_path
        full_collection_path = get_vectorstore_collection_path(tenant, target_year, doc_type_norm, enabler)
        
        if os.path.isdir(full_collection_path) and os.path.isfile(os.path.join(full_collection_path, "chroma.sqlite3")):
            return [collection_name] 
        return []

    # Scenario 2: List ALL collections for the given tenant/year context (List All)
    
    collections: Set[str] = set()
    
    # 1. Scan the Year Root (สำหรับ Doc Type: evidence) - Path: V_ROOT/tenant/year
    root_year_evidence = os.path.join(tenant_root, str(year)) 
    if os.path.isdir(root_year_evidence):
        # ค้นหา evidence_... collections ภายในโฟลเดอร์ปี
        for sub_dir in os.listdir(root_year_evidence):
             sub_dir_lower = sub_dir.lower()
             if sub_dir_lower.startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"): 
                 full_collection_path = os.path.join(root_year_evidence, sub_dir)
                 if os.path.isfile(os.path.join(full_collection_path, "chroma.sqlite3")):
                    collections.add(sub_dir_lower) 

    # 2. Scan the Common Root (สำหรับ Doc Type: document, faq, ฯลฯ) - Path: V_ROOT/tenant
    root_common = tenant_root 
    if os.path.isdir(root_common):
        # ค้นหาโฟลเดอร์ Doc Type
        for sub_dir in os.listdir(root_common):
            sub_dir_lower = sub_dir.lower()
            
            # ข้ามโฟลเดอร์ที่เป็นตัวเลข (ปี) เพราะถูกสแกนในขั้นตอนที่ 1 แล้ว
            if sub_dir.isdigit():
                 continue 
            
            # ตรวจสอบว่าโฟลเดอร์นั้นเป็น Collection จริง (มีไฟล์ Chroma)
            full_collection_path = os.path.join(root_common, sub_dir)
            
            if os.path.isfile(os.path.join(full_collection_path, "chroma.sqlite3")):
                 collections.add(sub_dir_lower) 
    
    return sorted(list(collections))


# -------------------- VECTORSTORE MANAGER (SINGLETON) --------------------
class VectorStoreManager:
    _instance = None
    _is_initialized = False
    _lock = threading.Lock()

    # ใช้ default_factory แทนการใส่ {} หรือ None ตรงๆ
    _chroma_cache: Dict[str, Chroma] = PrivateAttr(default_factory=dict)
    _multi_doc_retriever: Optional['MultiDocRetriever'] = PrivateAttr(default=None)
    
    tenant: str = PrivateAttr(default=DEFAULT_TENANT)
    year: int = PrivateAttr(default=DEFAULT_YEAR)

    _doc_id_mapping: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)
    _uuid_to_doc_id: Dict[str, str] = PrivateAttr(default_factory=dict)

    _embeddings: Any = PrivateAttr(default=None)

    # สำคัญที่สุด: ต้องใช้ default_factory=dict หรือ default=None เท่านั้น!
    _client: Optional[chromadb.PersistentClient] = PrivateAttr(default=None)

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VectorStoreManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, base_path: str = "", tenant: str = DEFAULT_TENANT,  year: Optional[int] = None, enabler: Optional[str] = None, doc_type: str = EVIDENCE_DOC_TYPES,): # ⬅️ FIX: รับ enabler เข้ามาด้วย
        # 📌 FIX: ทำให้ init รับแค่ base_path และ tenant เพื่อให้คงความเป็น Singleton
        if not self._is_initialized:
            self._base_path = base_path
            self.tenant = tenant.lower()
            
            # 💡 FIX: ต้องกำหนดค่าเริ่มต้นให้ Attributes ที่จำเป็นต้องใช้ในเมธอดอื่น ๆ ของ Class
            #        โดยใช้ค่า Default จาก config
            self.year = year if year is not None else DEFAULT_YEAR    
            self.doc_type = doc_type
            self.enabler = enabler.upper() if enabler else DEFAULT_ENABLER # ⬅️ FIX: กำหนด enabler ให้ชัดเจน
            
            self._chroma_cache = {}
            self._embeddings = get_hf_embeddings()
            
            chroma_client_root = get_vectorstore_tenant_root_path(tenant=self.tenant)
            self._client = chromadb.PersistentClient(path=chroma_client_root)

            logger.info(f"ChromaDB Client initialized at TENANT ROOT PATH: {chroma_client_root}")
            
            self._load_doc_id_mapping()
            
            logger.info(f"Initialized VectorStoreManager (Tenant: {self.tenant})") 
            
            VectorStoreManager._is_initialized = True
    
    @property
    def doc_id_map(self) -> Dict[str, Dict[str, Any]]:
        """Provides access to the Stable Doc ID -> Chunk UUIDs mapping."""
        return self._doc_id_mapping

    @property
    def uuid_to_doc_id_map(self) -> Dict[str, str]:
        """Provides access to the Chunk UUID -> Stable Doc ID mapping."""
        return self._uuid_to_doc_id
    
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
        """
        โหลดและรวม Document ID Mapping จาก 2 Path (Global + Year/Enabler Specific) แบบ thread-safe
        เน้นความทนทานต่อ Worker Context (Handle ValueError และ Attribute Missing)
        """

        # Lock สำหรับ thread-safe update
        if not hasattr(self, "_mapping_lock") or self._mapping_lock is None:
            self._mapping_lock = Lock()

        self._doc_id_mapping = {}
        self._uuid_to_doc_id = {}

        # 🎯 CRITICAL: ดึงค่า attributes ปลอดภัยใน Worker Context
        current_tenant = getattr(self, 'tenant', 'default_tenant')
        current_year = getattr(self, 'year', None)
        current_enabler = getattr(self, 'enabler', None)
        # 🎯 FIX: ใช้ self.doc_type ที่ถูกกำหนดใน __init__ 
        # (ซึ่งถูกส่งมาจาก load_all_vectorstores อย่างถูกต้องแล้ว)
        current_doc_type = getattr(self, 'doc_type', EVIDENCE_DOC_TYPES) 

        logger.info(f"🔍 VSM MAP LOAD PARAMS: Tenant={current_tenant}, Year={current_year}, "
                    f"Enabler={current_enabler}, DocType={current_doc_type}")
        
        path_A = None # Specific Map
        path_B = None # Global Map

        # 1. PATH A: Year-Specific/Enabler Mapping
        try:
            path_A = get_mapping_file_path(
                doc_type=current_doc_type,
                tenant=current_tenant, 
                year=current_year, 
                enabler=current_enabler
            )
        except ValueError as e:
            logger.warning(f"⚠️ VSM MAP PATH A (Specific) failed generation: {e}. Skipping specific map.")
            path_A = None
        
        # 2. PATH B: Global/Tenant Root Mapping
        try:
             path_B = get_mapping_file_path(
                doc_type=current_doc_type,
                tenant=current_tenant,
                year=None, # บังคับเป็น None เพื่อให้ Path Logic ไปใช้แบบ Global
                enabler=None 
            )
        except ValueError as e:
            logger.warning(f"⚠️ VSM MAP PATH B (Global) failed generation: {e}. Skipping global map.")
            path_B = None
            
        # 🎯 FIX: ตรวจสอบความมีอยู่ของไฟล์และจัดลำดับการโหลด (Specific ก่อน Global, ถ้า Path ต่างกัน)
        paths_to_load = []
        # A ก่อน B เพื่อให้ Specific ทับ Global
        if path_A and os.path.exists(path_A):
            paths_to_load.append(path_A)
        # B ต้องไม่ซ้ำกับ A
        if path_B and path_B != path_A and os.path.exists(path_B):
            paths_to_load.append(path_B)


        # Log Path details
        logger.info(f"🔍 VSM MAP PATH A (Specific): {path_A} (Exists: {os.path.exists(path_A) if path_A else 'N/A'})")
        logger.info(f"🔍 VSM MAP PATH B (Global): {path_B} (Exists: {os.path.exists(path_B) if path_B else 'N/A'})")
        logger.info(f"🔍 VSM MAP Loading from {len(paths_to_load)} path(s): {paths_to_load}")
        
        total_loaded_docs = 0
        total_loaded_uuids = 0

        # เริ่มต้นการโหลด
        for path in paths_to_load:
            
            try:
                with open(path, "r", encoding="utf-8") as f:
                    mapping_data: Dict[str, Dict[str, Any]] = json.load(f)
                    
                # Thread-safe update
                with self._mapping_lock:
                    for doc_id, doc_entry in mapping_data.items():
                        doc_id_clean = doc_id.strip()
                        
                        self._doc_id_mapping[doc_id_clean] = doc_entry
                        
                        # สร้าง uuid to doc_id mapping
                        if isinstance(doc_entry, dict) and isinstance(doc_entry.get("chunk_uuids"), list):
                            for uid in doc_entry["chunk_uuids"]:
                                uid_clean = uid.replace("-", "")
                                
                                # 🎯 FIX: จัดการ UUID ซ้ำ (ตามคำแนะนำ)
                                if uid in self._uuid_to_doc_id and self._uuid_to_doc_id[uid] != doc_id_clean:
                                    logger.warning(f"⚠️ Duplicate UUID {uid} detected. Existing: {self._uuid_to_doc_id[uid]}, New: {doc_id_clean}")
                                    
                                self._uuid_to_doc_id[uid] = doc_id_clean
                                self._uuid_to_doc_id[uid_clean] = doc_id_clean
                                
                    current_total_docs = len(self._doc_id_mapping)
                    current_total_uuids = len(self._uuid_to_doc_id)
                
                logger.info(f"✅ Loaded {len(mapping_data)} documents from MAPPING: {path} (Current Total Docs: {current_total_docs}, Chunks: {current_total_uuids})")
                total_loaded_docs = current_total_docs
                total_loaded_uuids = current_total_uuids
                    
            except Exception as e:
                logger.error(f"❌ Failed to load Doc ID Mapping from {path}: {e}", exc_info=True)


        logger.info(f"Initialized Doc ID Mapping. Total documents loaded: {total_loaded_docs}, Total chunks mapped: {total_loaded_uuids}.")

    def _re_parse_collection_name(self, collection_name: str) -> Tuple[str, Optional[str]]:
        collection_name_lower = collection_name.strip().lower()
        if collection_name_lower.startswith(f"{EVIDENCE_DOC_TYPES}_"):
            parts = collection_name_lower.split("_", 1)
            return EVIDENCE_DOC_TYPES, parts[1].upper() if len(parts) == 2 else None
        return collection_name_lower, None

    def _load_chroma_instance(self, collection_name: str) -> Optional[Chroma]:
        # 1. Fast cache hit
        if collection_name in self._chroma_cache:
            return self._chroma_cache[collection_name]

        # 2. Thread-safe double-check
        with self._lock:
            if collection_name in self._chroma_cache:
                return self._chroma_cache[collection_name]

            # ------------------------------------------------------------------
            # 3. แยก doc_type กับ enabler ออกมา
            # ------------------------------------------------------------------
            doc_type, enabler = self._re_parse_collection_name(collection_name)

            # ------------------------------------------------------------------
            # 4. กำหนด target_year
            # ------------------------------------------------------------------
            if doc_type.lower() == EVIDENCE_DOC_TYPES.lower():
                target_year: Optional[int] = self.year
            else:
                target_year = None

            # ------------------------------------------------------------------
            # 5. สร้าง persist_directory ที่ถูกต้อง 100% (Full Path ของ Collection)
            # 🎯 FIX: เรียกใช้ get_vectorstore_collection_path
            # ------------------------------------------------------------------
            persist_directory = get_vectorstore_collection_path(
                tenant=self.tenant,
                year=target_year,
                doc_type=doc_type,
                enabler=enabler,
            )

            # ------------------------------------------------------------------
            # 6. ตรวจสอบว่ามี folder จริงไหม
            # ------------------------------------------------------------------
            if not os.path.exists(persist_directory):
                logger.warning(
                    f"Vectorstore directory NOT FOUND!\n"
                    f"   Collection   : {collection_name}\n"
                    f"   Expected path: {persist_directory}\n"
                    f"   tenant={self.tenant} | year={self.year} | doc_type={str(doc_type)} | enabler={enabler or 'None'}" # 💡 FIX: ครอบ doc_type ด้วย str()
                )
                # ลบ alt_path ออกเนื่องจาก Logic การสร้าง Path ได้รวมศูนย์แล้ว
                return None

            # ------------------------------------------------------------------
            # 7. ตรวจสอบว่า client ถูก init แล้ว (ไม่ต้องใช้ แต่ยังคงไว้)
            # ------------------------------------------------------------------
            if self._client is None:
                logger.error("Chroma PersistentClient is None! ต้อง init ก่อนใช้งาน")
            
            # 🎯 ดึง Global Embedding Model (768-dim) มาใช้โดยตรง
            try:
                correct_embeddings = get_hf_embeddings() 
            except Exception as e:
                logger.error(f"FATAL: Failed to get correct embeddings for Chroma init: {e}")
                return None

            try:
                # ------------------------------------------------------------------
                # 8. สร้าง Chroma instance (ใช้ Path แทน Client ตัวแม่)
                # ------------------------------------------------------------------
                
                # 🎯 FIX 6: บังคับให้ LangChain Chroma สร้าง PersistentClient ภายใน
                # ที่ชี้ไปที่ Collection Path โดยตรง เพื่อแก้ปัญหา Raw Retrieval: 0 Docs
                
                vectordb = Chroma(
                    # client=self._client,               # ⬅️ ลบ Client ตัวแม่ที่ Root ออก
                    embedding_function=correct_embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_directory  # ⬅️ ใช้ Full Path ของ Collection
                )

                # 🎯 FIX 7: บังคับโหลด Collection Object ทันที (แก้ปัญหา Lazy Loading ใน Worker)
                # การเรียก vectordb._collection จะเป็นการเรียก Collection.get_collection() ภายใน
                collection_test = vectordb._collection 
                # ทดสอบเรียก method หนึ่งครั้งเพื่อให้แน่ใจว่า collection ไม่ตาย
                collection_test.count()

                # Cache ไว้ใช้ครั้งต่อไป
                self._chroma_cache[collection_name] = vectordb

                logger.info(
                    f"Loaded Chroma collection '{collection_name}' → Path: {persist_directory} (Retrieval Test Pending)"
                )
                return vectordb

            except Exception as e:
                logger.error(
                    f"FAILED to load collection '{collection_name}' from {persist_directory}\n"
                    f"Error: {e}",
                    exc_info=True,
                )
                return None


    def get_documents_by_id(self, stable_doc_ids: Union[str, List[str]], doc_type: str = "default_collection", enabler: Optional[str] = None) -> List[LcDocument]:
        """
        Retrieve documents from Chroma collection by stable_doc_ids (64-char hash).

        🎯 FIX 21.0: ลบ 'ids' ออกจาก 'include' parameter ใน collection.get() เพื่อแก้ไข ValueError
        """
        import chromadb 
        from langchain_core.documents import Document as LcDocument
        from typing import Set, Dict, Any 

        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
            
        if not stable_doc_ids:
            return []

        # 1. กำหนดชื่อ Collection และโหลด Instance
        collection_name = get_doc_type_collection_key(doc_type=doc_type, enabler=enabler)
        chroma_instance = self._load_chroma_instance(collection_name)
        
        if not chroma_instance:
            logger.warning(f"VSM: Cannot load collection '{collection_name}' for document retrieval.")
            return []

        # 2. แปลง Stable Doc IDs เป็น Chunk UUIDs (เพื่อใช้ในการค้นหา Primary Key)
        chunk_uuids_for_search: List[str] = []
        
        for stable_id in stable_doc_ids:
            stable_id_clean = stable_id.strip() 
            map_entry = self.doc_id_map.get(stable_id_clean)
            if map_entry and map_entry.get("chunk_uuids"):
                chunk_uuids_for_search.extend(map_entry["chunk_uuids"])
            else:
                chunk_uuids_for_search.append(stable_id_clean) 
                
        # 3. จัดการ ID ซ้ำซ้อนและเตรียม Query สำหรับ Primary Key Search (Chunk UUIDs)
        search_ids_raw = list(set([
            str(i).strip()
            for i in chunk_uuids_for_search if str(i).strip()
        ]))
        
        # เพิ่ม Flexible UUID Search: ค้นหาทั้ง ID ที่มีขีดกลางและไม่มีขีดกลาง
        final_chunk_uuids_to_try: Set[str] = set()
        for chunk_id in search_ids_raw:
            final_chunk_uuids_to_try.add(chunk_id) 
            if "-" in chunk_id:
                final_chunk_uuids_to_try.add(chunk_id.replace("-", "")) 
                
        final_chunk_uuids_list = list(final_chunk_uuids_to_try)

        if not final_chunk_uuids_list:
             logger.warning(f"Hydration failed: No valid Chunk UUIDs derived from {len(stable_doc_ids)} Stable IDs.")
             return []


        try:
            collection = chroma_instance._collection
            documents: List[LcDocument] = []
            result: Dict[str, Any] = {}
            
            # --- Attempt 1: Primary Key Search (Chunk UUIDs) ---
            logger.info(f"Attempt 1/2: Primary Key Search ({len(final_chunk_uuids_list)} Chunk UUIDs)")
            result = collection.get(
                ids=final_chunk_uuids_list,
                include=["documents", "metadatas"] # <-- 🎯 FIX 21.0: ลบ "ids"
            )

            docs_result = result.get("documents", [])
            
            # 🎯 FINAL FIX 19.0: ถ้าได้ 0 chunks ให้ลอง Fallback Search ด้วย $or
            if not docs_result:
                
                # --- Attempt 2: Fallback Search (Metadata: stable_doc_uuid OR doc_id) ---
                logger.warning("Attempt 1 returned 0 chunks. Falling back to Robust Metadata Search (stable_doc_uuid / doc_id).")
                
                # ใช้ Stable Doc IDs ที่ Cleaned แล้ว เป็น Query สำหรับ Metadata Search
                stable_doc_ids_cleaned = list(set([uid.strip() for uid in stable_doc_ids if uid.strip()]))

                if stable_doc_ids_cleaned:
                    # ค้นหาด้วย $or: stable_doc_uuid หรือ doc_id
                    result = collection.get(
                        where={"$or": [
                            {"stable_doc_uuid": {"$in": stable_doc_ids_cleaned}},
                            {"doc_id": {"$in": stable_doc_ids_cleaned}}
                        ]},
                        include=["documents", "metadatas"] # <-- 🎯 FIX 21.0: ลบ "ids"
                    )
                    docs_result = result.get("documents", [])
                else:
                    logger.warning("Fallback Search failed: No valid Stable Doc IDs for metadata query.")
            
            # --- ประมวลผลผลลัพธ์ ---
            docs = docs_result
            metadatas = result.get("metadatas", [{}] * len(docs))
            # ids ยังคงถูกคืนมาเสมอ
            ids = result.get("ids", [""] * len(docs)) 

            for i, text in enumerate(docs):
                meta = metadatas[i].copy() if metadatas and metadatas[i] else {}
                chunk_uuid = ids[i] if ids else (meta.get("chunk_uuid") or "")
                
                if chunk_uuid:
                    meta["chunk_uuid"] = chunk_uuid

                # ใช้ map (uuid_to_doc_id_map) ในการหา Stable ID ที่แน่นอน
                stable_doc_id = self.uuid_to_doc_id_map.get(chunk_uuid) or meta.get("stable_doc_uuid") or meta.get("doc_id")
                
                # Fallback: หากหาไม่เจอ ให้ลองหาแบบไม่มีขีดด้วย
                if not stable_doc_id and "-" in chunk_uuid:
                    stable_doc_id = self.uuid_to_doc_id_map.get(chunk_uuid.replace("-", ""))
                
                if stable_doc_id:
                     meta["stable_doc_uuid"] = stable_doc_id

                doc = LcDocument(page_content=text, metadata=meta)
                documents.append(doc)
                
            logger.info(f"✅ Retrieved {len(documents)} documents for {len(stable_doc_ids)} Stable IDs from '{collection_name}' (Search Mode: {'Primary/Fallback'}).")
            return documents

        except Exception as e:
            logger.error(f"❌ Error retrieving documents by Stable/Chunk IDs from collection '{collection_name}': {e}", exc_info=True)
            return []

    def _ensure_chroma_client_is_valid(self):
        """
        🎯 FIX 4A: Re-initializes the Chroma client if it is None or lost during serialization (Worker Process).
        """
        # ตรวจสอบว่ามี _client attribute หรือไม่ และเป็น None หรือไม่
        if not hasattr(self, '_client') or self._client is None:
            self.logger.warning(f"Chroma client lost in worker process for tenant '{self.tenant}', re-initializing...")
            
            # ใช้ VSM attributes (tenant) ที่เราแก้ไขให้คงอยู่แล้วในการสร้าง Path
            tenant_root_path = get_vectorstore_tenant_root_path(self.tenant)
            
            # Re-initialize the Persistent Client
            try:
                self._client = chromadb.PersistentClient(path=tenant_root_path, settings=Settings(anonymized_telemetry=True))
                
                # เมื่อ Client ถูกสร้างใหม่, Collection Handles เก่าทั้งหมดต้องถือว่าใช้ไม่ได้
                self._collections = {} 
                
                self.logger.info(f"✅ ChromaDB Client re-initialized at TENANT ROOT PATH: {tenant_root_path}. Collections cleared.")
            except Exception as e:
                self.logger.error(f"FATAL: Failed to re-initialize Chroma Client in worker: {e}", exc_info=False)
                # ไม่ raise เพื่อให้โค้ดส่วนต่อไปสามารถจัดการกับ error ได้
            
    def retrieve_by_chunk_uuids(self, chunk_uuids: List[str], collection_name: Optional[str] = None) -> List[LcDocument]:
        """
        Hydrate documents by chunk UUIDs.
        - รองรับ UUID ทั้งแบบมี dash และไม่มี dash
        - Retry mechanism + cache clear เฉพาะเมื่อ retrieval fail
        """
        from core.vectorstore import get_doc_type_collection_key, EVIDENCE_DOC_TYPES
        import logging
        logger = logging.getLogger(__name__)

        self._ensure_chroma_client_is_valid()

        if not chunk_uuids:
            logger.info("VSM: No chunk_uuids provided for hydration.")
            return []

        if collection_name is None:
            collection_name = get_doc_type_collection_key(
                doc_type=EVIDENCE_DOC_TYPES, 
                enabler=getattr(self, 'enabler', 'km')
            )

        # Prepare UUIDs: no-dash + attempt 64-char dash formatting
        no_dash = [u.replace("-", "") for u in chunk_uuids if u]
        with_dash = []
        for u in no_dash:
            if len(u) == 64:
                part1, part2, part3, part4, part5 = u[:8], u[8:12], u[12:16], u[16:20], u[20:]
                with_dash.append(f"{part1}-{part2}-{part3}-{part4}-{part5}")

        all_formats = list(set(chunk_uuids + no_dash + with_dash))  # Remove duplicates

        result = {"documents": [], "metadatas": [], "ids": []}
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            chroma = self._load_chroma_instance(collection_name)
            if not chroma:
                logger.warning(f"Cannot load '{collection_name}' (attempt {attempt})")
                if attempt < max_retries and collection_name in self._chroma_cache:
                    logger.warning(f"VSM: Clearing Chroma cache for '{collection_name}' due to load failure.")
                    del self._chroma_cache[collection_name]
                    self._ensure_chroma_client_is_valid()
                continue

            try:
                logger.info(f"Hydration attempt {attempt}/{max_retries} → {len(all_formats)} UUIDs from '{collection_name}'")
                result = chroma.get(ids=all_formats, include=["documents", "metadatas", "ids"])

                if result.get("documents"):
                    logger.info(f"Success: Retrieved {len(result['documents'])} chunks on attempt {attempt}")
                    break

                logger.warning(f"Got 0 chunks on attempt {attempt}, retrying...")

            except Exception as e:
                logger.error(f"Hydration failed (attempt {attempt}): {e}")

            # Clear cache only on failure
            if attempt < max_retries and collection_name in self._chroma_cache:
                logger.warning(f"VSM: Clearing Chroma cache for '{collection_name}' for retry.")
                del self._chroma_cache[collection_name]
                self._ensure_chroma_client_is_valid()

        # Build LcDocument objects
        docs = []
        documents_raw = result.get("documents", [])
        metas = result.get("metadatas", [{}] * len(documents_raw))
        ids = result.get("ids", [])

        for i, text in enumerate(documents_raw):
            if not text or not text.strip():
                continue
            meta = metas[i].copy()
            id_clean = ids[i].replace("-", "") if ids[i] else ""
            meta["chunk_uuid"] = id_clean

            # Map to stable_doc_uuid
            stable = self._uuid_to_doc_id.get(id_clean) or meta.get("stable_doc_uuid")
            if stable:
                meta["stable_doc_uuid"] = stable

            docs.append(LcDocument(page_content=text.strip(), metadata=meta))

        logger.info(f"Hydration complete → Retrieved {len(docs)} full-text chunks (requested {len(chunk_uuids)})")
        return docs

            
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
        # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
        collection_name = get_doc_type_collection_key(doc_type, enabler)
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.error(f"Collection '{collection_name}' is not loaded.")
            return []
        all_limited_documents: List[LcDocument] = []
        total_chunks_retrieved = 0
        for stable_id in stable_doc_ids:
            stable_id_clean = stable_id.strip()
            # 🎯 FIX: ใช้ 'stable_doc_uuid' เป็น filter key แทน 'doc_id' เพื่อความแน่นอน
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

        # 1. สร้าง retriever ที่ควบคุม k ได้จริงทุกครั้ง
        def raw_retrieve(query: str, filter_dict: Optional[dict] = None, k: int = top_k) -> List[LcDocument]:
            try:
                original_query = query
                
                # BGE-M3 แนะนำให้ใส่ Query Instruction/Prefix
                bge_prefix = "เป็นคำถามสำหรับการค้นหาหลักฐานเพื่อประเมินเกณฑ์: "
                query_with_prefix = f"{bge_prefix}{query.strip()}"
                
                logger.info(f"[BGE-M3 PREFIX ADDED] Using prefixed query: '{query_with_prefix[:100]}...'")
                # -------------------------------------------------------------

                search_kwargs = {"k": k}
                if filter_dict:
                    search_kwargs["filter"] = filter_dict
                
                docs = chroma_instance.similarity_search(
                    query=query_with_prefix, # 🟢 ใช้ Query ที่มี Prefix แล้ว
                    k=k,
                    filter=filter_dict
                )
                logger.info(f"Raw retrieval: {len(docs)} docs (k={k}, filter={bool(filter_dict)})")
                return docs
            except Exception as e:
                logger.error(f"Raw retrieval failed: {e}", exc_info=True)
                return []

        # 2. Reranker wrapper (ฉบับสมบูรณ์แบบ)
        def retrieve_with_rerank(query: str, config: Optional[dict] = None):
            filter_dict = None
            if config and isinstance(config, dict):
                # 🎯 FIX: ต้องดึง filter จาก 'where' key หากมี
                search_kwargs = config.get("configurable", {}).get("search_kwargs", {})
                filter_dict = search_kwargs.get("filter") or search_kwargs.get("where")
            
            # ดึงเอกสารด้วย k เต็ม
            docs = raw_retrieve(query, filter_dict, k=top_k)

            # ถ้าไม่มี reranker → คืนตาม top_k
            reranker = get_global_reranker()
            if not (use_rerank and reranker and hasattr(reranker, "compress_documents")):
                return docs[:final_k]

            try:
                reranked = reranker.compress_documents(
                    documents=docs,
                    query=query,
                    top_n=final_k
                )
                # ดึง score จาก reranker (วิธีที่แน่นอนที่สุด)
                scores = getattr(reranker, "scores", None)
                # 🎯 FIX: scores ที่ได้จาก predict() ต้องถูกจัดเรียงก่อน
                if scores and len(scores) >= len(reranked):
                    # scores ที่ CrossEncoder.predict คืนมา ไม่ได้ถูกจัดเรียง
                    # เราต้องใช้ doc_scores จาก compress_documents เพื่อดึง score ที่ตรงกับ doc
                    doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                    
                    for i, (doc, score) in enumerate(doc_scores[:len(reranked)]):
                        # หา doc ที่ตรงกันใน reranked list เพื่ออัปเดต metadata
                        for reranked_doc in reranked:
                            # ตรวจสอบว่า page_content ตรงกัน (อาจจะไม่สมบูรณ์ 100% แต่ดีกว่าไม่มี)
                            if reranked_doc.page_content == doc.page_content:
                                score = float(score) if score is not None else 0.0
                                # ฉีดแค่ key เดียวที่แน่นอน → _rerank_score_force
                                reranked_doc.metadata["_rerank_score_force"] = score
                                # และ source_filename (สำคัญมากสำหรับ extraction)
                                orig = reranked_doc.metadata.get("source_filename", "UNKNOWN")
                                reranked_doc.metadata["source_filename"] = f"{orig}|SCORE:{score:.4f}"
                                break
                else:
                    # Fallback: ใช้ score จาก reranked doc.metadata ถ้ามี (จากการทำงานของ compress_documents)
                    for doc in reranked:
                        score = doc.metadata.get("relevance_score")
                        if score is not None:
                             orig = doc.metadata.get("source_filename", "UNKNOWN")
                             doc.metadata["source_filename"] = f"{orig}|SCORE:{score:.4f}"
                             doc.metadata["_rerank_score_force"] = score # ใส่ key ให้เหมือนกัน
                             
                logger.info(f"Reranking success → kept {len(reranked)} docs")
                return reranked

            except Exception as e:
                logger.warning(f"Rerank failed ({e}), fallback to raw")
                return docs[:final_k]

        # 3. สร้าง LangChain-compatible Retriever
        class UltimateRetriever(BaseRetriever):
            def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
                return retrieve_with_rerank(query, config=None)

            def invoke(self, query: str, config: Optional[dict] = None) -> List[LcDocument]:
                return retrieve_with_rerank(query, config=config)

        logger.info(f"Ultimate Retriever ready → {collection_name} | top_k={top_k} → final_k={final_k} | rerank={use_rerank}")
        return UltimateRetriever()

    def get_all_collection_names(self) -> List[str]:
        # 🎯 FIX: ลบ base_path ออกจากการเรียก list_vectorstore_folders
        return list_vectorstore_folders(tenant=self.tenant, year=self.year)


    def get_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], doc_type: str, enabler: Optional[str] = None) -> List[LcDocument]:
        import chromadb # Import locally if not already imported
        from langchain_core.documents import Document as LcDocument

        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
        stable_doc_ids = [uid for uid in stable_doc_ids if uid]
        if not stable_doc_ids:
            return []
        
        # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
        collection_name = get_doc_type_collection_key(doc_type, enabler)
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
                    # [Minor Fix] แก้ไขตัวแปรจาก stable_doc_clean เป็น stable_id_clean
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
            
            # 🎯 FINAL FIX 16.0: ลบ "ids" ออกจาก include เพื่อแก้ ChromaDB API Error
            result = collection.get(ids=all_chunk_uuids, include=["documents", "metadatas"]) 
            
            documents: List[LcDocument] = []
            if not result.get("documents"):
                logger.warning(f"Chroma DB returned 0 documents for {len(all_chunk_uuids)} chunk UUIDs in collection '{collection_name}'.")
                return []
                
            for i, text in enumerate(result.get("documents", [])):
                if text:
                    metadata = result.get("metadatas", [{}])[i]
                    # IDs ถูกคืนมาโดยอัตโนมัติ ไม่ต้อง include
                    chunk_uuid_from_result = result.get("ids", [""])[i] 
                    
                    # NOTE: ใช้ self._uuid_to_doc_id
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
def get_vectorstore_manager(
    doc_type: str = "all",           # เพิ่มค่า default
    tenant: str = DEFAULT_TENANT,
    year: Optional[int] = None,
    enabler: Optional[str] = None,
) -> VectorStoreManager:
    """
    สร้างหรือคืน VectorStoreManager (รองรับการค้นทุก doc_type)
    """
    return VectorStoreManager(
        # doc_type=doc_type,
        tenant=tenant,
        # year=year or DEFAULT_YEAR,
        # enabler=enabler
    )

def load_vectorstore(doc_type: str, enabler: Optional[str] = None) -> Optional[Chroma]:
    collection_name = get_doc_type_collection_key(doc_type, enabler)
    vsm = get_vectorstore_manager(
        doc_type=doc_type,           # เพิ่มบรรทัดนี้!
        enabler=enabler
    )
    return vsm._load_chroma_instance(collection_name)

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
        
        # 🎯 DEBUG: ตรวจสอบ Collection Count และ Embedding (โค้ดนี้คือส่วนที่เราเพิ่ม)
        try:
            raw_collection = self.vectorstore._collection
            count = raw_collection.count() 
            logger.critical(f"🎯 [DEBUG CHROMA] Collection Count: {count}")
            
            # ตรวจสอบการสร้าง Query Embedding
            if hasattr(self.vectorstore, '_embedding_function') and self.vectorstore._embedding_function:
                embedding_function = self.vectorstore._embedding_function
                # ทดสอบ embed query
                query_embedding = embedding_function.embed_query(query)
                logger.critical(f"🎯 [DEBUG CHROMA] Query Embedding Success. Vector size: {len(query_embedding)}")
            else:
                logger.critical("🎯 [DEBUG CHROMA] Cannot access _embedding_function.")

        except Exception as debug_e:
            logger.critical(f"❌ [DEBUG CHROMA] Debug check failed (Skip search): {debug_e}")
            return []
        # END DEBUG

        try:
            # รัน similarity search
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
    """
    Defines a single retriever configuration, mapping a document type/enabler
    to a specific VectorStore collection context (tenant/year).
    """
    doc_id: str
    doc_type: str
    top_k: int = INITIAL_TOP_K
    final_k: int = FINAL_K_RERANKED
    # 🎯 FIX: ลบ base_path ออก
    enabler: Optional[str] = None
    tenant: str = DEFAULT_TENANT
    year: Optional[int] = DEFAULT_YEAR # <--- 🎯 FIX: เปลี่ยนเป็น Optional[int] เพื่อรองรับ None
    
    # load_instance ต้องเรียก VSM โดยใช้ self.tenant และ self.year
    def load_instance(self) -> Any:
        """
        Loads the actual VectorStore Retriever instance using VectorStoreManager 
        with the correct tenant and year context.
        """
        # ⚠️ VSM Singleton ต้องถูก init ด้วยปีเสมอ เพื่อโหลด Doc ID Mapping
        # ถ้า self.year เป็น None (เช่น เป็น General Docs ที่ใช้ร่วมกันทุกปี) ให้ใช้ DEFAULT_YEAR
        manager = VectorStoreManager(
            # 🎯 FIX: ลบ base_path ออก
            tenant=self.tenant, 
            year=self.year if self.year is not None else DEFAULT_YEAR # <--- 🎯 FIX: ใช้ DEFAULT_YEAR ถ้า self.year เป็น None
        ) 
        # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
        collection_name = get_doc_type_collection_key(self.doc_type, self.enabler)
        
        retriever = manager.get_retriever(collection_name=collection_name, top_k=self.top_k, final_k=self.final_k) 
        
        if not retriever:
            raise ValueError(f"Retriever not found for collection '{collection_name}' at path based on tenant={self.tenant}, year={self.year}")
        
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

        # 🎯 FIX 2: ดึง Context จาก NamedRetriever ตัวแรก
        tenant_context = retrievers_list[0].tenant if retrievers_list else DEFAULT_TENANT
        year_context = retrievers_list[0].year if retrievers_list else DEFAULT_YEAR

        # 🎯 FIX 2A: สร้าง VSM ด้วย Context ที่ถูกต้อง (สำหรับ Doc ID Mapping)
        # ถ้า year_context เป็น None (จาก NamedRetriever ที่เป็น General doc) ให้ใช้ DEFAULT_YEAR
        if year_context is None:
             year_context = DEFAULT_YEAR 
             
        self._manager = VectorStoreManager(tenant=tenant_context, year=year_context)
        
        self._all_retrievers = {}
        for named_r in retrievers_list:
            # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
            collection_name = get_doc_type_collection_key(named_r.doc_type, named_r.enabler)
            try:
                # Load the RerankRetriever instance
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
            # Chroma filter applied during retrieval
            self._chroma_filter = {"doc_id": {"$in": doc_ids_filter}}
            logger.info(f"✅ MultiDocRetriever initialized with doc_ids filter for {len(doc_ids_filter)} Stable IDs.")

        # Using a simpler executor choice that doesn't rely on undefined imports
        self._executor_type = self._choose_executor() 
        logger.info(f"MultiDocRetriever will use executor type: {self._executor_type} (workers={MAX_PARALLEL_WORKERS})")
    
    def _choose_executor(self) -> str:
        """Selects the executor type based on basic platform info."""
        # Simplify executor choice to avoid dependency on undefined imports
        if platform.system() == "Windows":
             return "process"
        # Defaulting to thread pool for efficiency on other platforms unless specified otherwise
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
        """Static task method for ProcessPoolExecutor."""
        try:
            # load_instance ensures the correct VSM context is used
            retriever_instance = named_r.load_instance()
            if not retriever_instance:
                return []
                
            # Prepare config for the invoke method of the RerankRetriever
            search_kwargs = {"k": named_r.top_k}
            if chroma_filter:
                # The Chroma filter is applied as 'where' in Chroma's implementation
                # 🎯 FIX: ต้องใช้ key 'where' ใน config ของ invoke()
                search_kwargs["where"] = chroma_filter 
                
            # 🎯 FIX: ต้องส่ง filter เป็น 'where' ใน config ด้วย
            config = {"configurable": {"search_kwargs": {"where": chroma_filter}}}
            if not chroma_filter:
                config = {"configurable": {"search_kwargs": {}}} # ส่ง config เปล่าถ้าไม่มี filter
            
            # retriever_instance is a RerankRetriever (which implements Runnable.invoke)
            docs = retriever_instance.invoke(query, config=config)
            
            for doc in docs:
                doc.metadata["retrieval_source"] = named_r.doc_type
                # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
                doc.metadata["collection_name"] = get_doc_type_collection_key(named_r.doc_type, named_r.enabler)
                
                # 🎯 FIX 1A: Ensure chunk_uuid is present for final filtering.
                chunk_uuid = doc.metadata.get("chunk_uuid") 
                
                if not chunk_uuid:
                    # Try to find the ID from common Langchain/Chroma internal keys
                    potential_uuid = doc.metadata.get("id") or doc.metadata.get("_id") 
                    
                    if potential_uuid:
                        doc.metadata["chunk_uuid"] = str(potential_uuid)
                    else:
                        # Final Fallback: Use a stable hash of the content/metadata for deduplication/ID
                        key_content = f"{doc.page_content}{doc.metadata.get('doc_id')}"
                        hashed_uuid = hashlib.sha256(key_content.encode('utf-8')).hexdigest()
                        doc.metadata["chunk_uuid"] = hashed_uuid[:32] # Use first 32 chars for uniqueness
                        
            return docs
        except Exception as e:
            # Use print here as logger might not be configured correctly in child process
            print(f"❌ Child retrieval error for {named_r.doc_id} ({named_r.doc_type}): {e}")
            return []

    def _thread_retrieve_task(self, named_r: NamedRetriever, query: str, chroma_filter: Optional[Dict]):
        """Instance method for ThreadPoolExecutor."""
        try:
            # load_instance ensures the correct VSM context is used
            retriever_instance = named_r.load_instance()
            if not retriever_instance:
                return []
                
            # Prepare config for the invoke method of the RerankRetriever
            search_kwargs = {"k": named_r.top_k}
            if chroma_filter:
                # The Chroma filter is applied as 'where' in Chroma's implementation
                # 🎯 FIX: ต้องใช้ key 'where' ใน config ของ invoke()
                search_kwargs["where"] = chroma_filter 
                
            # 🎯 FIX: ต้องส่ง filter เป็น 'where' ใน config ด้วย
            config = {"configurable": {"search_kwargs": {"where": chroma_filter}}}
            if not chroma_filter:
                config = {"configurable": {"search_kwargs": {}}} # ส่ง config เปล่าถ้าไม่มี filter

            # retriever_instance is a RerankRetriever (which implements Runnable.invoke)
            docs = retriever_instance.invoke(query, config=config)
            
            for doc in docs:
                doc.metadata["retrieval_source"] = named_r.doc_type
                # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
                doc.metadata["collection_name"] = get_doc_type_collection_key(named_r.doc_type, named_r.enabler)

                # 🎯 FIX 1A: Ensure chunk_uuid is present for final filtering.
                chunk_uuid = doc.metadata.get("chunk_uuid") 
                
                if not chunk_uuid:
                    # Try to find the ID from common Langchain/Chroma internal keys
                    potential_uuid = doc.metadata.get("id") or doc.metadata.get("_id") 
                    
                    if potential_uuid:
                        doc.metadata["chunk_uuid"] = str(potential_uuid)
                    else:
                        # Final Fallback: Use a stable hash of the content/metadata for deduplication/ID
                        key_content = f"{doc.page_content}{doc.metadata.get('doc_id')}"
                        hashed_uuid = hashlib.sha256(key_content.encode('utf-8')).hexdigest()
                        doc.metadata["chunk_uuid"] = hashed_uuid[:32] # Use first 32 chars for uniqueness
                    
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
                # Use static method for ProcessPoolExecutor
                future = executor.submit(MultiDocRetriever._static_retrieve_task, named_r, query, self._chroma_filter)
            else:
                # Use instance method for ThreadPoolExecutor
                future = executor.submit(self._thread_retrieve_task, named_r, query, self._chroma_filter)
            futures.append(future)
            
        for f in futures:
            try:
                docs = f.result()
                if docs:
                    all_docs.extend(docs)
            except Exception as e:
                logger.warning(f"Future failed: {e}")
                
        # Deduplication using chunk metadata
        seen = set()
        unique_docs = []
        for d in all_docs:
            src = d.metadata.get("retrieval_source") or ""
            # Use 'chunk_uuid' or 'ids' (which is the UUID from Chroma) for unique identification
            # NOTE: chunk_uuid should now be present due to the fix in the task methods
            chunk_uuid = d.metadata.get("chunk_uuid") or d.metadata.get("ids") or "" 
            
            # Fallback to content if UUIDs are missing (less reliable)
            if not chunk_uuid:
                 # Use a hash or truncated content as a fallback unique key
                 key = f"{src}_{d.page_content[:120]}_{d.metadata.get('doc_id', 'no_doc_id')}"
            else:
                 key = f"{src}_{chunk_uuid}"
                 
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
def load_all_vectorstores( 
    tenant: str, 
    year: int,    
    doc_types: Optional[Union[str, List[str]]] = None, 
    top_k: int = INITIAL_TOP_K, 
    final_k: int = FINAL_K_RERANKED, 
    base_path: Path = "", # 🎯 FIX: base_path ถูกละเลย
    evidence_enabler: Optional[str] = None, 
    doc_ids: Optional[List[str]] = None
) -> 'VectorStoreManager':
    """
    Loads all relevant vectorstore collections based on tenant, year, and document types.
    Handles segregation logic for year-specific (evidence) and general (standard) documents.
    """
    
    doc_types = [doc_types] if isinstance(doc_types, str) else doc_types or []
    doc_type_filter = {dt.strip().lower() for dt in doc_types}
    
    # 📌 1. กำหนด doc_type หลักสำหรับ VSM
    # เราคาดหวังว่าสำหรับ Engine จะส่ง EVIDENCE_DOC_TYPES เข้ามาเป็นตัวแรกใน List
    # ใช้ "all" เป็น fallback (ซึ่งเป็นค่า default ภายในของ VSM ถ้าไม่กำหนด)
    primary_doc_type = doc_types[0] if doc_types else "all" 

    # 🎯 FIX 1 & 1.1: สร้าง VSM โดยใส่ doc_type ที่ถูกต้อง
    manager = VectorStoreManager(
        tenant=tenant, 
        year=year,
        enabler=evidence_enabler, # ใส่ enabler เข้าไปเพื่อความสมบูรณ์
        # 🟢 CRITICAL FIX: ส่ง doc_type ที่ถูกต้องเข้าไปในการสร้าง VSM
        doc_type=primary_doc_type 
    ) 
    
    all_retrievers: List['NamedRetriever'] = []
    target_collection_names: Set[str] = set()

    # --- 1. Collection Discovery ---
    if not doc_type_filter:
        logger.error("Must specify doc_types for multi-year compatibility.")
        raise ValueError("Must specify doc_types when using multi-tenant setup.")
    
    # NEW: เราไม่สามารถ list collections ทั้งหมดได้โดยไม่รู้ว่าเอกสารไหนใช้ปี/ไม่ใช้ปี 
    # เราจึงใช้ doc_type_filter ในการสร้าง collection_name แทนการ list จาก folder
    for dt_norm in doc_type_filter:
        if dt_norm == EVIDENCE_DOC_TYPES.lower(): 
            if evidence_enabler:
                # ✅ FIX: Specific evidence collection: ใช้ year และ enabler
                collection_name = get_doc_type_collection_key(EVIDENCE_DOC_TYPES, evidence_enabler)
                target_collection_names.add(collection_name)
                logger.info(f"🔍 Added specific evidence collection: {collection_name} (Year-Specific)")
            else:
                # All evidence collections: ต้อง list จาก folder ภายใต้ tenant/year
                evidence_collections = list_vectorstore_folders(tenant=tenant, year=year, doc_type=EVIDENCE_DOC_TYPES)
                target_collection_names.update(evidence_collections)
                logger.info(f"🔍 Added all evidence collections found: {evidence_collections} (Year-Specific)")
        else:
            # Standard Collections: ไม่ใช้ year
            collection_name = get_doc_type_collection_key(dt_norm, None)
            target_collection_names.add(collection_name)
            logger.info(f"🔍 Added standard collection: {collection_name} (Shared/General)")
    
    logger.info(f"🔍 DEBUG: Attempting to load {len(target_collection_names)} total target collections: {target_collection_names}")
    
    # --- 2. Retriever List Creation & Existence Check ---
    for collection_name in target_collection_names:
        doc_type_for_check, enabler_for_check = manager._re_parse_collection_name(collection_name)
        
        # 🎯 FIX 2A: Logic กำหนด target_year (None สำหรับ General Docs)
        target_year: Optional[int] = year
        if doc_type_for_check.lower() != EVIDENCE_DOC_TYPES.lower() and enabler_for_check is None:
            # นี่คือเอกสารทั่วไปที่ไม่ควรถูกแยกตามปี
            target_year = None # <--- กำหนดเป็น None
            
        # 🎯 FIX 2B: ส่ง target_year เข้าไปในการตรวจสอบการมีอยู่ของ Vectorstore
        if not vectorstore_exists(doc_id="N/A", tenant=tenant, year=target_year, doc_type=doc_type_for_check, enabler=enabler_for_check):
            logger.warning(f"🔍 DEBUG: Skipping collection '{collection_name}' (vectorstore_exists failed at tenant={tenant}, year={target_year}).")
            continue
            
        # 🎯 FIX 2C: ส่ง target_year เข้าไปใน NamedRetriever
        nr = NamedRetriever(
            doc_id=collection_name, 
            doc_type=doc_type_for_check, 
            enabler=enabler_for_check, 
            top_k=top_k, 
            final_k=final_k, 
            tenant=tenant, 
            year=target_year # <--- ส่ง None สำหรับ General Docs
        )
        all_retrievers.append(nr)
        logger.info(f"🔍 DEBUG: Successfully added retriever for collection '{collection_name}' (Year={target_year}).")

    final_filter_ids = doc_ids
    if doc_ids:
        logger.info(f"✅ Hard Filter Enabled: Using {len(doc_ids)} original 64-char UUIDs for filtering.")
    logger.info(f"🔍 DEBUG: Final count of all_retrievers = {len(all_retrievers)}")

    if not all_retrievers:
        raise ValueError(f"No vectorstore collections found matching tenant={tenant}, year={year}, doc_types={doc_types} and evidence_enabler={evidence_enabler}")
        
    mdr = MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k, doc_ids_filter=final_filter_ids)
    manager._multi_doc_retriever = mdr
    logger.info(f"✅ MultiDocRetriever loaded with {len(mdr._all_retrievers)} collections and cached in VSM.")
    return manager


def get_multi_doc_retriever(
    tenant: str = DEFAULT_TENANT,
    year: int = DEFAULT_YEAR,
    doc_types: List[str] = [],
    doc_ids: Optional[List[str]] = None,
    evidence_enabler: Optional[str] = None,
    base_path: str = "", # 🎯 FIX: base_path ถูกละเลย
    top_k: int = INITIAL_TOP_K,
    final_k: int = FINAL_K_RERANKED
) -> MultiDocRetriever:
    """
    Factory function to create a MultiDocRetriever based on configuration.
    It determines which NamedRetrievers to initialize based on the tenant, year, and doc_types.
    """
    all_retrievers: List[NamedRetriever] = []

    # 1. Dynamic Check for Year-Specific Collections
    # Loop through requested doc_types and check against the target year
    target_year = year
    for doc_type_for_check in doc_types:
        # 🎯 FIX: ใช้ get_doc_type_collection_key แทน _get_collection_name
        collection_name = get_doc_type_collection_key(doc_type_for_check, evidence_enabler)
        
        enabler_for_check = evidence_enabler
        
        # Check if collection exists for the specific year
        if not vectorstore_exists(tenant=tenant, year=target_year, doc_type=doc_type_for_check, enabler=enabler_for_check):
            logger.warning(f"🔍 DEBUG: Skipping collection '{collection_name}' (vectorstore_exists failed at tenant={tenant}, year={target_year}).")
            continue
            
        # 🎯 FIX 2C: ส่ง target_year เข้าไปใน NamedRetriever
        nr = NamedRetriever(
            doc_id=collection_name, 
            doc_type=doc_type_for_check, 
            enabler=enabler_for_check, 
            top_k=top_k, 
            final_k=final_k, 
            # 🎯 FIX: ลบ base_path ออก
            tenant=tenant, 
            year=target_year # <--- ส่งค่าปีที่ถูกต้อง
        )
        all_retrievers.append(nr)
        logger.info(f"🔍 DEBUG: Successfully added retriever for collection '{collection_name}' (Year={target_year}).")

    final_filter_ids = doc_ids
    if doc_ids:
        logger.info(f"✅ Hard Filter Enabled: Using {len(doc_ids)} original 64-char UUIDs for filtering.")
    logger.info(f"🔍 DEBUG: Final count of all_retrievers = {len(all_retrievers)}")

    if not all_retrievers:
        raise ValueError(f"No vectorstore collections found matching tenant={tenant}, year={year}, doc_types={doc_types} and evidence_enabler={evidence_enabler}")
        
    mdr = MultiDocRetriever(retrievers_list=all_retrievers, k_per_doc=top_k, doc_ids_filter=final_filter_ids)
    return mdr