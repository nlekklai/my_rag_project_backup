# core/vectorstore.py
import os
import platform
import logging
import threading
from threading import Lock
import multiprocessing
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Union, Sequence, Any, Dict, Set, Tuple
from pathlib import Path
import hashlib
import uuid

# system utils
try:
    import psutil
except ImportError:
    psutil = None

# LangChain-ish imports
from langchain_core.documents import Document as LcDocument
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import BaseDocumentCompressor
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables import Runnable
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
# from langchain.retrievers import EnsembleRetriever

try:
    # สำหรับ Mac (เวอร์ชันเก่า) หรือ Server (ถ้าลงตัวหลักไว้)
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    # สำหรับ Server (เวอร์ชันใหม่ v0.2+)
    from langchain_community.retrievers import EnsembleRetriever

# Thai Tokenizer
from pythainlp.tokenize import word_tokenize

# Pydantic helpers
from pydantic import PrivateAttr, ConfigDict, BaseModel, Field

# Chroma / HF embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings

# CrossEncoder
try:
    from sentence_transformers import CrossEncoder
    _HAS_SENT_TRANS = True
except Exception:
    CrossEncoder = None
    _HAS_SENT_TRANS = False
    logging.warning("⚠️ sentence-transformers CrossEncoder not available. Reranker will be disabled.")

# Path utils
from utils.path_utils import (
    get_doc_type_collection_key,
    get_vectorstore_collection_path,
    get_vectorstore_tenant_root_path,
    get_mapping_file_path,
    _n
)

# Global config
from config.global_vars import (
    FINAL_K_RERANKED,
    INITIAL_TOP_K,
    EVIDENCE_DOC_TYPES,
    MAX_PARALLEL_WORKERS,
    DEFAULT_TENANT,
    DEFAULT_YEAR,
    DEFAULT_ENABLER,
    RERANKER_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    USE_HYBRID_SEARCH,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT
)

# Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Chroma telemetry
try:
    chromadb.configure(anonymized_telemetry=False)
except Exception:
    try:
        chromadb.settings = Settings(anonymized_telemetry=False)
    except Exception:
        pass

# -------------------- Vectorstore Constants --------------------
ENV_FORCE_MODE = os.getenv("VECTOR_MODE", "").lower()
ENV_DISABLE_ACCEL = os.getenv("VECTOR_DISABLE_ACCEL", "").lower() in ("1", "true", "yes")

# Global caches
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
            return "mps"
    except Exception:
        pass
    return "cpu"

# -------------------- HuggingFace Embeddings --------------------
def get_hf_embeddings(device_hint: Optional[str] = None):
    global _CACHED_EMBEDDINGS
    device = device_hint or _detect_torch_device()

    if _CACHED_EMBEDDINGS is None:
        with _EMBED_LOCK:
            if _CACHED_EMBEDDINGS is None:
                model_name = EMBEDDING_MODEL_NAME
                logger.info(f"Loading HF Embedding: {model_name} on {device}")
                try:
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": device},
                        encode_kwargs={"normalize_embeddings": True}
                    )
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    logger.warning("Falling back to paraphrase-multilingual-MiniLM-L12-v2")
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                        model_kwargs={"device": "cpu"}
                    )
    return _CACHED_EMBEDDINGS

# -------------------- Thai Tokenizer --------------------
def thai_tokenizer_for_bm25(text: str) -> List[str]:
    return word_tokenize(text.lower().strip())

# -------------------- HuggingFace CrossEncoder Reranker --------------------
class HuggingFaceCrossEncoderCompressor(BaseDocumentCompressor):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    rerank_model: str = RERANKER_MODEL_NAME
    rerank_device: str = Field(default_factory=lambda: _detect_torch_device())
    rerank_max_length: int = 512
    top_n: int = FINAL_K_RERANKED
    
    _cross_encoder: Optional[Any] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        detected_device = _detect_torch_device()
        object.__setattr__(self, 'rerank_device', detected_device)

        try:
            if not _HAS_SENT_TRANS:
                raise ImportError("sentence-transformers not installed")
            encoder = CrossEncoder(
                model_name_or_path=self.rerank_model,
                device=self.rerank_device,
                max_length=self.rerank_max_length
            )
            object.__setattr__(self, '_cross_encoder', encoder)
        except Exception as e:
            logger.error(f"❌ Error loading Reranker: {e}", exc_info=True)
            object.__setattr__(self, '_cross_encoder', None)

    def compress_documents(
        self,
        documents: Sequence[LcDocument],
        query: str,
        callbacks: Optional[Any] = None,
        top_n: Optional[int] = None  # <- เพิ่มบรรทัดนี้
    ) -> Sequence[LcDocument]:
        if not self._cross_encoder or not documents:
            return documents

        # ใช้ top_n ที่ส่งเข้ามา ถ้าไม่มีให้ fallback เป็น self.top_n
        current_top_n = min(len(documents), top_n or self.top_n)

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._cross_encoder.predict(pairs)

        ranked_docs = []
        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            ranked_docs.append(doc)

        ranked_docs.sort(key=lambda x: x.metadata["rerank_score"], reverse=True)
        final_docs = ranked_docs[:current_top_n]

        if final_docs:
            logger.info(f"📊 Reranking Stats | Top Score: {final_docs[0].metadata['rerank_score']:.4f} | Selected: {len(final_docs)} docs")

        return final_docs


# -------------------- Global Reranker Singleton --------------------
_global_reranker_instance = None
_global_reranker_lock = threading.Lock()

def get_global_reranker() -> Optional[HuggingFaceCrossEncoderCompressor]:
    global _global_reranker_instance
    with _global_reranker_lock:
        if _global_reranker_instance is None:
            try:
                _global_reranker_instance = HuggingFaceCrossEncoderCompressor(
                    rerank_model=RERANKER_MODEL_NAME,
                    top_n=FINAL_K_RERANKED
                )
            except Exception as e:
                logger.error(f"Failed to create global reranker: {e}")
                _global_reranker_instance = None
        return _global_reranker_instance

# -------------------- Path Helper Function (REVISED to use Path Utility) --------------------

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


def list_vectorstore_folders(
    tenant: str, 
    year: int, 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = "" 
) -> List[str]:
    """
    Lists the actual collection names that exist under the specified tenant and year context.
    Fixed: Checks for chroma.sqlite3 at the DB Root level instead of inside collection folders.
    """
    tenant_root = get_vectorstore_tenant_root_path(tenant) 
    
    # Scenario 1: Specific doc_type/enabler requested
    if doc_type:
        doc_type_norm = doc_type.lower().strip()
        collection_name = get_doc_type_collection_key(doc_type_norm, enabler)
        
        target_year = year if doc_type_norm == EVIDENCE_DOC_TYPES.lower() else None
        
        # Path ไปยังโฟลเดอร์ที่เก็บ Collection (e.g., .../2568/evidence_km)
        full_collection_path = get_vectorstore_collection_path(tenant, target_year, doc_type_norm, enabler)
        # Path ไปยัง DB Root (e.g., .../2568/)
        db_root_path = os.path.dirname(full_collection_path.rstrip('/'))
        
        # ✅ FIX: เช็คว่ามีโฟลเดอร์ collection และมี chroma.sqlite3 อยู่ใน DB Root
        if os.path.isdir(full_collection_path) and os.path.isfile(os.path.join(db_root_path, "chroma.sqlite3")):
            return [collection_name] 
        return []

    # Scenario 2: List ALL collections
    collections: Set[str] = set()
    
    # 1. Scan the Year Root (สำหรับ evidence) - Path: V_ROOT/tenant/year
    root_year = os.path.join(tenant_root, str(year)) 
    if os.path.isdir(root_year):
        # ✅ FIX: เช็คว่าในโฟลเดอร์ปีมีไฟล์ DB หลักไหม
        has_db_file = os.path.isfile(os.path.join(root_year, "chroma.sqlite3"))
        
        if has_db_file:
            for sub_dir in os.listdir(root_year):
                 sub_dir_lower = sub_dir.lower()
                 # ถ้าเป็นโฟลเดอร์ และขึ้นต้นด้วย evidence_ (เช่น evidence_km)
                 if sub_dir_lower.startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"): 
                     if os.path.isdir(os.path.join(root_year, sub_dir)):
                        collections.add(sub_dir_lower) 

    # 2. Scan the Common Root (สำหรับ document, faq) - Path: V_ROOT/tenant
    if os.path.isdir(tenant_root):
        has_common_db = os.path.isfile(os.path.join(tenant_root, "chroma.sqlite3"))
        
        for sub_dir in os.listdir(tenant_root):
            if sub_dir.isdigit(): continue # ข้ามโฟลเดอร์ปี
            
            sub_dir_lower = sub_dir.lower()
            full_path = os.path.join(tenant_root, sub_dir)
            
            # ✅ FIX: ถ้าเป็นโฟลเดอร์ และมีไฟล์ DB อยู่ที่ระดับ tenant root
            if os.path.isdir(full_path) and has_common_db:
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

    def __init__(self, base_path: str = "", tenant: str = DEFAULT_TENANT, 
                 year: Optional[int] = None, enabler: Optional[str] = None, 
                 doc_type: str = EVIDENCE_DOC_TYPES):
        if not self._is_initialized:
            with self._lock:
                if not self._is_initialized:
                    # --- Basic Setup ---
                    self._base_path = base_path
                    self.tenant = tenant.lower()
                    self.year = year if year is not None else DEFAULT_YEAR    
                    self.doc_type = doc_type
                    self.enabler = enabler.upper() if enabler else DEFAULT_ENABLER 

                    # --- Caches ---
                    self._chroma_cache: Dict[str, Any] = {}
                    self._multi_doc_retriever: Optional[Any] = None
                    self._doc_id_mapping: Dict[str, Dict[str, Any]] = {}
                    self._uuid_to_doc_id: Dict[str, str] = {}
                    self._hybrid_retriever_cache: Dict[str, Any] = {}
                    self._bm25_docs_cache: Dict[str, List[Document]] = {}

                    # --- Core Components ---
                    self._embeddings = get_hf_embeddings()
                    self._client: Optional[chromadb.PersistentClient] = None

                    # --- Logger (สำคัญ!) ---
                    self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
                    self.logger.info(f"VectorStoreManager initialized for tenant={self.tenant}, year={self.year}")

                    # --- Initialize Client ---
                    try:
                        client_base_path = self._get_chroma_client_base_path(tenant, year)
                        self._client = chromadb.PersistentClient(path=client_base_path)
                        self.logger.info(f"ChromaDB Client initialized at: {client_base_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize ChromaDB client: {e}")
                        self._client = None

                    # --- Load Mapping ---
                    try:
                        self._load_doc_id_mapping()
                        self.logger.info(f"Loaded doc_id_mapping: {len(self._doc_id_mapping)} documents")
                    except Exception as e:
                        self.logger.error(f"Failed to load doc_id_mapping: {e}")

                    self._is_initialized = True  # ← ใช้ instance variable
                    self.logger.info(f"VectorStoreManager fully initialized (Tenant: {self.tenant})")
    
    def _get_chroma_client_base_path(self, tenant: str, year: Optional[int]) -> str:
        """
        Determines the base path for the Chroma PersistentClient.
        - Global Docs (document, seam): ชี้ไปที่ root ของ vectorstore
        - Evidence Docs (KM): ชี้ไปที่โฟลเดอร์ปี (เช่น vectorstore/2568)
        """
        # ดึง root path ของ tenant (เช่น .../data_store/pea/vectorstore)
        root_path = get_vectorstore_tenant_root_path(tenant) 
        
        # ดึงค่า doc_type มา normalize เพื่อเปรียบเทียบ
        current_dt = _n(getattr(self, 'doc_type', EVIDENCE_DOC_TYPES))
        evidence_type = _n(EVIDENCE_DOC_TYPES)

        # 🎯 FIX LOGIC:
        # เฉพาะกรณีที่เป็น Evidence และมีการระบุปีเท่านั้น ถึงจะชี้เข้าโฟลเดอร์ปี
        if current_dt == evidence_type and year is not None:
            target_path = os.path.join(root_path, str(year))
            self.logger.info(f"📂 VSM Path Mode: YEARLY -> {target_path}")
            return target_path
        
        # นอกเหนือจากนั้น (เช่น document, seam) ให้ใช้ Root Path เสมอ
        self.logger.info(f"📂 VSM Path Mode: GLOBAL -> {root_path}")
        return root_path
    
    # -------------------- START FIXES (3 Functions) --------------------
    
    def set_multi_doc_retriever(self, mdr: 'MultiDocRetriever'):
        """
        Sets the active MultiDocRetriever instance.
        NOTE: This is the setter for the PrivateAttr _multi_doc_retriever.
        (FIX: แก้ไข AttributeError ใน load_all_vectorstores)
        """
        # 🎯 FIX: ใช้ object.__setattr__ เพื่อตั้งค่า PrivateAttr
        object.__setattr__(self, '_multi_doc_retriever', mdr)
        logger.info("✅ MultiDocRetriever has been set in VectorStoreManager.")

    def get_multi_doc_retriever(self) -> Optional['MultiDocRetriever']:
        """Gets the active MultiDocRetriever instance."""
        return self._multi_doc_retriever
    
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
        โหลด Document ID Mapping โดยเลือก Path ที่ถูกต้องที่สุดเพียงหนึ่งเดียว (Simplified Version)
        - Evidence: บังคับใช้ Path รายปี/ราย Enabler
        - อื่นๆ: ใช้ Path กลางระดับ Tenant Root
        """
        from threading import Lock

        # 1. เตรียม Lock และโครงสร้างข้อมูลภายใน (Internal State)
        if not hasattr(self, "_mapping_lock") or self._mapping_lock is None:
            self._mapping_lock = Lock()

        self._doc_id_mapping = {}
        self._uuid_to_doc_id = {}

        # 2. ดึงค่า Attributes จาก Instance
        current_tenant = getattr(self, 'tenant', 'default_tenant')
        current_year = getattr(self, 'year', None)
        current_enabler = getattr(self, 'enabler', None)
        current_doc_type = getattr(self, 'doc_type', EVIDENCE_DOC_TYPES) 

        # 3. ตัดสินใจเลือก Path เดียว (Single Path Decision)
        target_path = None
        
        # ใช้ _n() เพื่อป้องกันปัญหา NFD/NFC บน macOS และความแตกต่างของตัวพิมพ์
        if _n(current_doc_type) == EVIDENCE_DOC_TYPES.lower():
            # สาย Evidence: กฎใน path_utils บังคับว่าต้องมี year และ enabler
            try:
                target_path = get_mapping_file_path(
                    doc_type=current_doc_type,
                    tenant=current_tenant, 
                    year=current_year, 
                    enabler=current_enabler
                )
            except ValueError:
                # กรณี year/enabler เป็น None จะไม่พ่น Warning แต่จะปล่อยให้ target_path เป็น None
                target_path = None
        else:
            # สาย Global (seam, faq, policy, etc.): ใช้ Path กลาง ไม่ต้องระบุปี
            try:
                target_path = get_mapping_file_path(
                    doc_type=current_doc_type,
                    tenant=current_tenant,
                    year=None,
                    enabler=None 
                )
            except ValueError:
                target_path = None

        # 4. Validation: ตรวจสอบความมีอยู่ของไฟล์ก่อนอ่าน
        if not target_path or not os.path.exists(target_path):
            logger.warning(f"⚠️ No mapping file found for type '{current_doc_type}' at: {target_path}")
            return

        # 5. กระบวนการโหลดและสร้างดัชนี (Indexing)
        logger.info(f"📂 Loading mapping from: {target_path}")

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
                
            with self._mapping_lock:
                for doc_id, doc_entry in mapping_data.items():
                    doc_id_clean = doc_id.strip()
                    self._doc_id_mapping[doc_id_clean] = doc_entry
                    
                    # สร้าง UUID Lookup Table เพื่อให้ RAG ทราบชื่อไฟล์ต้นทางจากก้อนเนื้อหา (Chunk)
                    if isinstance(doc_entry, dict) and "chunk_uuids" in doc_entry:
                        for uid in doc_entry["chunk_uuids"]:
                            uid_clean = uid.replace("-", "")
                            # เก็บทั้งแบบมีขีดและไม่มีขีดเพื่อความแม่นยำในการค้นหา
                            self._uuid_to_doc_id[uid] = doc_id_clean
                            self._uuid_to_doc_id[uid_clean] = doc_id_clean
            
            logger.info(f"✅ Success: Loaded {len(self._doc_id_mapping)} documents into Memory.")
                
        except Exception as e:
            logger.error(f"❌ Failed to load mapping: {e}")

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
                    f"   Collection      : {collection_name}\n"
                    f"   Expected path: {persist_directory}\n"
                    f"   tenant={self.tenant} | year={self.year} | doc_type={str(doc_type)} | enabler={enabler or 'None'}" 
                )
                return None

            # ------------------------------------------------------------------
            # 7. ตรวจสอบว่า client ถูก init แล้ว (ใช้ self.client ที่ถูกแก้ไขแล้ว)
            # ------------------------------------------------------------------
            if self.client is None: # ใช้ property client ที่เรียก _ensure_chroma_client_is_valid
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
                
                vectordb = Chroma(
                    # client=self._client,       # ⬅️ ลบ Client ตัวแม่ที่ Root ออก
                    embedding_function=correct_embeddings,
                    collection_name=collection_name,
                    persist_directory=persist_directory  # ⬅️ ใช้ Full Path ของ Collection
                )

                # 🎯 FIX 7: บังคับโหลด Collection Object ทันที (แก้ปัญหา Lazy Loading ใน Worker)
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
        เวอร์ชันปรับปรุง: เน้นความเร็วในการค้นหาและความถูกต้องของ Metadata
        รองรับทั้งการหาผ่าน Primary Keys (Chunk IDs) และ Metadata (Stable IDs)
        """
        from langchain_core.documents import Document as LcDocument
        
        # 1. ปรับเตรียม Input
        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
        
        stable_doc_ids_cleaned = list(set([uid.strip() for uid in stable_doc_ids if uid.strip()]))
        if not stable_doc_ids_cleaned:
            return []

        # 2. เตรียม Collection และตรวจสอบ Client (Worker Safety)
        collection_name = get_doc_type_collection_key(doc_type=doc_type, enabler=enabler)
        self._ensure_chroma_client_is_valid() # 🛡️ มั่นใจว่า Client ไม่หลุดใน Worker
        chroma_instance = self._load_chroma_instance(collection_name)
        
        if not chroma_instance:
            logger.error(f"❌ VSM: Collection '{collection_name}' load failed.")
            return []

        collection = chroma_instance._collection
        documents: List[LcDocument] = []

        try:
            # 3. สร้างรายการ Chunk IDs ที่เป็นไปได้จาก Doc ID Map
            search_ids: Set[str] = set()
            for s_id in stable_doc_ids_cleaned:
                map_entry = self.doc_id_map.get(s_id)
                if map_entry and map_entry.get("chunk_uuids"):
                    search_ids.update(map_entry["chunk_uuids"])
                # เผื่อกรณี s_id ที่ส่งมาเป็น Chunk ID โดยตรง
                search_ids.add(s_id)
            
            # ทำความสะอาด ID (รองรับทั้งแบบมี dash และไม่มี dash)
            final_ids = list(search_ids)
            for cid in list(search_ids):
                if "-" in cid: final_ids.append(cid.replace("-", ""))
            
            # --- Attempt 1: ค้นหาด้วย Primary Key (IDs) ---
            logger.info(f"🔄 Attempt 1: Fetching {len(final_ids)} IDs from {collection_name}")
            result = collection.get(ids=final_ids, include=["documents", "metadatas"])

            # --- Attempt 2: Fallback ด้วย Metadata Search (กรณี ID Map ไม่อัปเดต) ---
            if not result.get("documents"):
                logger.warning("⚠️ Primary key search empty. Falling back to Metadata filter...")
                result = collection.get(
                    where={"$or": [
                        {"stable_doc_uuid": {"$in": stable_doc_ids_cleaned}},
                        {"doc_id": {"$in": stable_doc_ids_cleaned}}
                    ]},
                    include=["documents", "metadatas"]
                )

            # 4. ประมวลผลและสร้าง LcDocument
            docs_raw = result.get("documents", [])
            metas_raw = result.get("metadatas", [])
            ids_raw = result.get("ids", [])

            for i, text in enumerate(docs_raw):
                meta = metas_raw[i].copy() if metas_raw and metas_raw[i] else {}
                current_id = ids_raw[i]
                
                p_val = meta.get("page_label") or meta.get("page_number") or meta.get("page") or "N/A"
                meta["page"] = str(p_val)
                meta["page_label"] = str(p_val) # UI มักเรียกใช้ตัวนี้
                
                            # ฝัง ID ที่แท้จริงกลับเข้าไป
                meta["chunk_uuid"] = current_id
                
                # พยายาม Map กลับหา Stable ID ที่ถูกต้องที่สุด
                stable_ref = (
                    self.uuid_to_doc_id_map.get(current_id) or 
                    self.uuid_to_doc_id_map.get(current_id.replace("-", "")) or
                    meta.get("stable_doc_uuid") or 
                    meta.get("doc_id")
                )
                if stable_ref:
                    meta["stable_doc_uuid"] = stable_ref

                documents.append(LcDocument(page_content=text, metadata=meta))

            logger.info(f"✅ Success: Retrieved {len(documents)} chunks from '{collection_name}'")
            return documents

        except Exception as e:
            logger.error(f"❌ Error in get_documents_by_id: {str(e)}", exc_info=True)
            return []

    def get_chunks_by_page(self, collection_name: str, stable_doc_uuid: str, page_label: str) -> List[LcDocument]:
        """
        [NEW] ดึง Chunks ทั้งหมดของเลขหน้าที่ระบุ (Exact Metadata Match)
        ใช้สำหรับดึงบริบทข้างเคียง (Neighbor Context) เพื่อแก้ปัญหาข้อมูล Act (A) ขาดหาย
        """
        try:
            # 1. โหลด Chroma Instance ผ่าน Cache/Logic เดิมของ VSM
            self._ensure_chroma_client_is_valid()
            chroma_instance = self._load_chroma_instance(collection_name)
            
            if not chroma_instance:
                self.logger.error(f"❌ Neighbor Fetch: ไม่พบ Collection {collection_name}")
                return []

            # 2. เข้าถึง Collection ระดับต่ำ (Chroma native collection) เพื่อใช้ filter
            collection = chroma_instance._collection

            # 🎯 สร้าง Filter เจาะจงไฟล์และหน้า
            # หมายเหตุ: page_label ต้องเป็น String ตามมาตรฐานการ Ingest ของเรา
            where_filter = {
                "$and": [
                    {"stable_doc_uuid": {"$eq": str(stable_doc_uuid)}},
                    {"page_label": {"$eq": str(page_label)}}
                ]
            }

            # 3. ดึงข้อมูล (ตั้ง limit=10 เพื่อให้ครอบคลุมกรณี 1 หน้ามีหลาย chunks)
            results = collection.get(
                where=where_filter,
                limit=10, 
                include=["documents", "metadatas", "ids"]
            )

            extra_docs = []
            if results and results['documents']:
                for idx, text in enumerate(results['documents']):
                    meta = results['metadatas'][idx].copy() if results['metadatas'] else {}
                    
                    # ทำความสะอาด Metadata ให้พร้อมใช้งานเหมือนฟังก์ชันหลักอื่นๆ
                    p_val = meta.get("page_label") or meta.get("page_number") or "N/A"
                    meta["page_label"] = str(p_val)
                    meta["chunk_uuid"] = results['ids'][idx].replace("-", "")
                    
                    extra_docs.append(LcDocument(page_content=text, metadata=meta))
            
            if extra_docs:
                self.logger.info(f"➕ Neighbor Fetch: พบข้อมูลหน้า {page_label} ในไฟล์ {stable_doc_uuid} ({len(extra_docs)} chunks)")
            
            return extra_docs

        except Exception as e:
            self.logger.error(f"❌ Error ใน get_chunks_by_page: {str(e)}", exc_info=True)
            return []
        
    def _ensure_chroma_client_is_valid(self):
        """
        Re-initializes the Chroma client if it is None or lost during serialization (Worker Process).
        """
        # ตรวจสอบว่ามี _client attribute หรือไม่ และเป็น None หรือไม่
        if not hasattr(self, '_client') or self._client is None:
            logger.warning(f"Chroma client lost in worker process for tenant '{self.tenant}', re-initializing...")
            
            # ใช้ VSM attributes (tenant) ที่เราแก้ไขให้คงอยู่แล้วในการสร้าง Path
            tenant_root_path = get_vectorstore_tenant_root_path(self.tenant)
            
            # Re-initialize the Persistent Client
            try:
                # 🎯 FIX: ใช้ logger ที่ถูก import เข้ามา
                self._client = chromadb.PersistentClient(path=tenant_root_path, settings=Settings(anonymized_telemetry=True))
                
                # เมื่อ Client ถูกสร้างใหม่, Collection Handles เก่าทั้งหมดต้องถือว่าใช้ไม่ได้
                # เนื่องจากเราใช้ _chroma_cache แทน _collections เราจึงต้องล้าง cache
                self._chroma_cache = {} 
                
                logger.info(f"✅ ChromaDB Client re-initialized at TENANT ROOT PATH: {tenant_root_path}. Collections cache cleared.")
            except Exception as e:
                logger.error(f"FATAL: Failed to re-initialize Chroma Client in worker: {e}", exc_info=False)
                # ไม่ raise เพื่อให้โค้ดส่วนต่อไปสามารถจัดการกับ error ได้
            
    def retrieve_by_chunk_uuids(self, chunk_uuids: List[str], collection_name: Optional[str] = None) -> List[LcDocument]:
        """
        Hydrate documents by chunk UUIDs.
        - รองรับ UUID ทั้งแบบมี dash และไม่มี dash
        - Retry mechanism + cache clear เฉพาะเมื่อ retrieval fail
        """
        # (ต้องมั่นใจว่ามีการ import ที่จำเป็นอยู่แล้ว)

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
            if len(u) == 32: # UUID4 is 32 chars, not 64
                # สมมติว่า UUID ที่คุณใช้มีรูปแบบมาตรฐาน 8-4-4-4-12 = 32
                try:
                    uuid_obj = uuid.UUID(u, version=4) # ทดสอบแปลงเป็น UUID
                    with_dash.append(str(uuid_obj))
                except ValueError:
                    # ถ้าแปลงไม่ได้ ให้ข้าม
                    pass 
            # ถ้าเป็น 64-char hash (ซึ่งไม่น่าเป็น UUID) ให้เก็บไว้ในรูปแบบไม่มีขีดต่อไป
            # โค้ดที่ให้มาพยายามทำ 64-char hash เป็นแบบขีดกลาง ซึ่งอาจไม่ถูกต้องตามมาตรฐาน UUID
            # แต่เพื่อรักษาโครงสร้างเดิม:
            elif len(u) == 64:
                 part1, part2, part3, part4, part5 = u[:8], u[8:12], u[12:16], u[16:20], u[20:]
                 with_dash.append(f"{part1}-{part2}-{part3}-{part4}-{part5}")
        
        # All formats
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

            p_val = meta.get("page_label") or meta.get("page_number") or meta.get("page") or "N/A"
            meta["page"] = str(p_val)
            meta["page_label"] = str(p_val)

            docs.append(LcDocument(page_content=text.strip(), metadata=meta))

        logger.info(f"Hydration complete → Retrieved {len(docs)} full-text chunks (requested {len(chunk_uuids)})")
        return docs

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 10,
        filter_doc_ids: Optional[Set[str]] = None,
        metadata_filter: Optional[Dict[str, Any]] = None  # 👈 เพิ่มมาเพื่อรับ Rubric Filter
    ) -> List[LcDocument]:
        """
        เวอร์ชันอัปเกรด: รองรับ Hybrid + Flexible Post-filtering
        """
        self.logger.info(f"🔍 VSM: Retrieving from {collection_name} | Query: {query[:50]}...")

        # 1. เรียกใช้ get_retriever
        retriever = self.get_retriever(
            collection_name=collection_name, 
            top_k=top_k, 
            use_hybrid=True
        )

        if not retriever:
            return []

        # 2. ค้นหาข้อมูล
        docs = retriever.invoke(query)

        # 3. จัดการ Filter (แบบ Flexible)
        if filter_doc_ids:
            # ทำความสะอาด ID ทั้งหมดให้เป็น lowercase string และลบช่องว่าง
            clean_targets = {str(tid).lower().strip() for tid in filter_doc_ids}
            
            filtered_docs = []
            for d in docs:
                m = d.metadata or {}
                # ดึง ID จากทุก key ที่เป็นไปได้
                m_stable = str(m.get("stable_doc_uuid", "")).lower().strip()
                m_doc = str(m.get("doc_id", "")).lower().strip()
                
                if m_stable in clean_targets or m_doc in clean_targets:
                    filtered_docs.append(d)
                # Fallback: เช็คชื่อไฟล์เผื่อกรณี ID หลุด
                elif any(tid in str(m.get("source", "")).lower() for tid in clean_targets):
                    filtered_docs.append(d)
            
            docs = filtered_docs

        # 4. จัดการ Metadata Filter (สำหรับ Rubric/Enabler)
        if metadata_filter:
            for key, value in metadata_filter.items():
                docs = [d for d in docs if d.metadata.get(key) == value]

        return docs[:top_k]


    def create_hybrid_retriever(self, collection_name: str, top_k: int = 20) -> EnsembleRetriever:
        """
        สร้างและ Cache Hybrid Retriever (Vector + BM25)
        เวอร์ชันปรับปรุง: รองรับการตัดคำภาษาไทยและป้องกัน Metadata เป็น None
        """
        # 1. ตรวจสอบ Cache เพื่อประหยัดทรัพยากร
        if collection_name in self._hybrid_retriever_cache:
            logger.info(f"♻️ Using cached Hybrid Retriever for: {collection_name}")
            return self._hybrid_retriever_cache[collection_name]
            
        logger.info(f"🏗️ Creating NEW Hybrid Retriever for: {collection_name}...")

        try:
            # 2. โหลด Chroma Instance
            chroma_instance = self._load_chroma_instance(collection_name) 
            if not chroma_instance:
                raise ValueError(f"Chroma instance for '{collection_name}' failed to load.")
            
            # 3. สร้าง Vector Retriever (Dense)
            # เราจะตั้งค่า k ให้สูงกว่า top_k เล็กน้อยเพื่อให้ Ensemble มีตัวเลือกในการคำนวณคะแนน
            vector_retriever = chroma_instance.as_retriever(
                search_kwargs={"k": top_k}
            )

            # 4. ดึง Documents ทั้งหมดมาเตรียมทำ BM25 Index (Sparse)
            if collection_name in self._bm25_docs_cache:
                langchain_docs = self._bm25_docs_cache[collection_name]
                logger.info(f"📦 Loaded {len(langchain_docs)} docs for BM25 from cache.")
            else:
                logger.info(f"🔍 Fetching docs from Chroma collection '{collection_name}' for BM25 indexing...")
                
                # ดึงข้อมูลดิบจาก Chroma (ดึงเฉพาะที่จำเป็น)
                raw_data = chroma_instance._collection.get(
                    include=["documents", "metadatas"]
                )
                
                texts = raw_data.get("documents", [])
                metas = raw_data.get("metadatas", [])
                
                # ป้องกันกรณี metas เป็น None หรือยาวไม่เท่ากับ texts
                if not metas:
                    metas = [{} for _ in texts]
                
                # แปลงเป็น LangChain Document Objects
                langchain_docs = [
                    Document(page_content=text, metadata=meta if meta else {})
                    for text, meta in zip(texts, metas)
                ]
                
                # เก็บลง Cache เพื่อไม่ให้ต้องดึงใหม่บ่อยๆ
                self._bm25_docs_cache[collection_name] = langchain_docs
                logger.info(f"✅ Indexed {len(langchain_docs)} documents for BM25.")

            # 5. สร้าง BM25 Retriever พร้อมตัวตัดคำภาษาไทย
            if not langchain_docs:
                logger.warning(f"⚠️ Collection '{collection_name}' is empty. Returning vector retriever only.")
                return vector_retriever

            bm25_retriever = BM25Retriever.from_documents(
                langchain_docs, 
                preprocess_func=word_tokenize # 🎯 FIX: ใช้ pythainlp ตัดคำเพื่อให้ Search ภาษาไทยแม่นยำ
            )
            bm25_retriever.k = top_k

            # 6. รวมร่างเป็น Ensemble Retriever (Hybrid)
            # โดยปกติ Vector 0.7 และ BM25 0.3 เป็นค่าเริ่มต้นที่ดีสำหรับงาน RAG
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3] # หรือใช้ค่าจาก global_vars
            )
            
            # 7. เก็บเข้า Cache และส่งออก
            self._hybrid_retriever_cache[collection_name] = ensemble_retriever
            logger.info(f"🚀 Hybrid Retriever for '{collection_name}' is ready (Vector + BM25).")
            return ensemble_retriever
        
        except Exception as e:
            logger.error(f"❌ Failed to create Hybrid Retriever for '{collection_name}': {str(e)}", exc_info=True)
            # กรณีพลาด ให้คืนค่าเป็น Vector Retriever ปกติเพื่อไม่ให้ระบบล่ม
            try:
                return chroma_instance.as_retriever(search_kwargs={"k": top_k})
            except:
                return None
        
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

    def get_retriever(self, collection_name: str, top_k: int = INITIAL_TOP_K, final_k: int = FINAL_K_RERANKED, use_rerank: bool = USE_HYBRID_SEARCH, use_hybrid: bool = True) -> Any:
        """
        สร้าง Retriever ที่รองรับ Hybrid Search (Vector + BM25) และ Reranking 
        โดยมีการจัดการ Scope ของฟังก์ชันภายในให้ถูกต้อง
        """
        
        # โหลด Chroma Instance
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.warning(f"Retriever creation failed: Collection '{collection_name}' not loaded.")
            return None

        # --- [INTERNAL HELPER 1]: Reranker Wrapper ---
        # ประกาศไว้บนสุดเพื่อให้ทั้ง Hybrid และ Fallback เรียกใช้ได้
        def retrieve_with_rerank(docs: List[LcDocument], query: str) -> List[LcDocument]:
            reranker = get_global_reranker()
            if not (use_rerank and reranker and hasattr(reranker, "compress_documents")):
                return docs[:final_k]

            try:
                reranked = reranker.compress_documents(documents=docs, query=query, top_n=final_k)
                # Inject score กลับเข้าไปใน metadata เพื่อแสดงผลหรือ debug
                scores = getattr(reranker, "scores", None)
                if scores and len(scores) >= len(reranked):
                    # เรียงลำดับตาม score
                    doc_scores = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
                    for i, (doc, score) in enumerate(doc_scores[:len(reranked)]):
                        for r_doc in reranked:
                            if r_doc.page_content == doc.page_content:
                                score_val = float(score) if score is not None else 0.0
                                r_doc.metadata["_rerank_score_force"] = score_val
                                orig = r_doc.metadata.get("source_filename", "UNKNOWN")
                                r_doc.metadata["source_filename"] = f"{orig}|SCORE:{score_val:.4f}"
                                break
                logger.info(f"Reranking success → kept {len(reranked)} docs")
                return reranked
            except Exception as e:
                logger.warning(f"Rerank failed: {e}, fallback to raw")
                return docs[:final_k]

        # --- [INTERNAL HELPER 2]: Raw Vector Retrieve ---
        def raw_vector_retrieve(query: str, filter_dict: Optional[dict] = None, k: int = top_k) -> List[LcDocument]:
            try:
                # เพิ่ม Prefix สำหรับ BGE-M3 (ถ้ามี)
                bge_prefix = "เป็นคำถามสำหรับการค้นหาหลักฐานเพื่อประเมินเกณฑ์: "
                query_with_prefix = f"{bge_prefix}{query.strip()}"
                
                docs = chroma_instance.similarity_search(
                    query=query_with_prefix,
                    k=k,
                    filter=filter_dict
                )
                return docs
            except Exception as e:
                logger.error(f"Vector retrieval failed: {e}")
                return []

        # 1. สร้าง Vector Retriever พื้นฐาน
        vector_retriever = chroma_instance.as_retriever(search_kwargs={"k": top_k})

        # 2. กรณีใช้ Hybrid (BM25 + Vector)
        if use_hybrid:
            try:
                # 🟢 FIX CRITICAL: ใช้ _collection โดยตรง
                if not hasattr(chroma_instance, "_collection"):
                    raise ValueError("chroma_instance has no _collection attribute.")
                
                collection = chroma_instance._collection
                
                # 🟢 FIX: ดึงข้อมูลเพื่อทำ BM25 Index (ลบ "ids" ออกจาก include)
                result = collection.get(include=["documents", "metadatas"])
                texts = result.get("documents", [])
                metadatas = result.get("metadatas", [])

                if texts:
                    langchain_docs = [
                        LcDocument(page_content=text, metadata=meta or {})
                        for text, meta in zip(texts, metadatas)
                    ]

                    # 🟢 KEY FIX: ใส่ Tokenizer ภาษาไทย (pythainlp)
                    from pythainlp.tokenize import word_tokenize as thai_tokenizer
                    bm25_retriever = BM25Retriever.from_documents(
                        langchain_docs,
                        preprocess_func=thai_tokenizer # หรือใช้ชื่อ tokenizer ตามที่คุณตั้งไว้
                    )
                    bm25_retriever.k = top_k

                    # รวมร่าง Ensemble
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[vector_retriever, bm25_retriever],
                        weights=[HYBRID_VECTOR_WEIGHT, HYBRID_BM25_WEIGHT]
                    )

                    # คลาสสำหรับ Hybrid + Rerank
                    class UltimateHybridRetriever(BaseRetriever):
                        def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
                            # ดึงผลลัพธ์ผ่าน Ensemble
                            docs = ensemble_retriever.invoke(query)
                            # ส่งไป Rerank ผ่านฟังก์ชัน Helper ที่ประกาศไว้ด้านบน
                            return retrieve_with_rerank(docs, query)

                        def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> List[LcDocument]:
                            return self._get_relevant_documents(query)
                    
                    return UltimateHybridRetriever()

            except Exception as e:
                logger.error(f"Hybrid setup failed for '{collection_name}': {e}", exc_info=False)
                # หาก Hybrid มีปัญหา ให้ไหลลงไปใช้ Fallback ด้านล่าง
                pass

        # 3. Fallback: กรณี Rerank อย่างเดียว หรือ Hybrid พัง
        if use_rerank and get_global_reranker():
            class SimpleVectorRerankRetriever(BaseRetriever):
                def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[LcDocument]:
                    # ดึงผลลัพธ์ผ่าน Vector Search
                    docs = raw_vector_retrieve(query, filter_dict=None, k=top_k)
                    # ส่งไป Rerank
                    return retrieve_with_rerank(docs, query)
            
                def invoke(self, query: str, config: Optional[dict] = None, **kwargs) -> List[LcDocument]:
                    return self._get_relevant_documents(query)
            
            return SimpleVectorRerankRetriever()
        
        # 4. สุดท้าย: คืนค่า Vector Retriever ธรรมดา
        return vector_retriever

    def get_all_collection_names(self) -> List[str]:
        # 🎯 FIX: ลบ base_path ออกจากการเรียก list_vectorstore_folders
        return list_vectorstore_folders(tenant=self.tenant, year=self.year)

    def get_chunks_from_doc_ids(self, stable_doc_ids: Union[str, List[str]], doc_type: str, enabler: Optional[str] = None) -> List[LcDocument]:
        """
        ดึง Chunk ของเอกสารทั้งหมดจาก ChromaDB โดยใช้ Stable Document IDs
        ผ่านการ Mapping จาก doc_id_mapping.json
        """
        import chromadb
        from langchain_core.documents import Document as LcDocument

        # 1. เตรียม input ให้เป็น List และจัดการค่าว่าง
        if isinstance(stable_doc_ids, str):
            stable_doc_ids = [stable_doc_ids]
        stable_doc_ids = [uid.strip() for uid in stable_doc_ids if uid and isinstance(uid, str)]
        
        if not stable_doc_ids:
            logger.warning("No valid Stable Document IDs provided.")
            return []
        
        # 2. ระบุชื่อ Collection
        # 🎯 FIX: ใช้ get_doc_type_collection_key เพื่อความถูกต้องของชื่อตามโครงสร้างระบบ
        collection_name = get_doc_type_collection_key(doc_type, enabler)
        
        all_chunk_uuids = []
        skipped_docs = []
        found_stable_ids = []
        
        # 3. ค้นหา Chunk UUIDs จาก Mapping
        for stable_id in stable_doc_ids:
            if stable_id in self._doc_id_mapping:
                doc_entry = self._doc_id_mapping[stable_id]
                
                # ตรวจสอบโครงสร้างข้อมูลใน Mapping
                if isinstance(doc_entry, dict) and "chunk_uuids" in doc_entry:
                    chunk_uuids = doc_entry["chunk_uuids"]
                    if isinstance(chunk_uuids, list) and chunk_uuids:
                        all_chunk_uuids.extend(chunk_uuids)
                        found_stable_ids.append(stable_id)
                    else:
                        logger.warning(f"Stable ID '{stable_id}' has an empty or invalid chunk_uuids list.")
                else:
                    logger.warning(f"Mapping for Stable ID '{stable_id}' is malformed or missing 'chunk_uuids'.")
            else:
                skipped_docs.append(stable_id)
                
        if skipped_docs:
            logger.warning(f"Skipping Stable IDs not found in mapping: {skipped_docs}")
            
        if not all_chunk_uuids:
            logger.warning(f"No valid chunk UUIDs found in collection '{collection_name}' for provided IDs.")
            return []
            
        # 4. โหลด Chroma Instance
        chroma_instance = self._load_chroma_instance(collection_name)
        if not chroma_instance:
            logger.error(f"Collection '{collection_name}' could not be loaded.")
            return []
            
        try:
            # 🟢 FIX CRITICAL: เข้าถึง _collection โดยตรง และลบ "ids" ออกจาก include 
            # เพื่อป้องกัน TypeError ใน ChromaDB version ใหม่
            collection = chroma_instance._collection
            result = collection.get(
                ids=all_chunk_uuids, 
                include=["documents", "metadatas"]
            ) 
            
            documents: List[LcDocument] = []
            retrieved_texts = result.get("documents", [])
            retrieved_metas = result.get("metadatas", [])
            retrieved_ids = result.get("ids", []) # IDs จะถูกคืนมาให้โดยอัตโนมัติ
            
            if not retrieved_texts:
                logger.warning(f"Chroma DB returned 0 documents for {len(all_chunk_uuids)} chunk UUIDs in '{collection_name}'.")
                return []
                
            # 5. ประกอบร่าง LangChain Documents
            for i, text in enumerate(retrieved_texts):
                if text:
                    metadata = retrieved_metas[i] if i < len(retrieved_metas) else {}
                    chunk_uuid = retrieved_ids[i]
                    
                    # ค้นหา doc_id ดั้งเดิมจาก uuid_to_doc_id mapping
                    doc_id = self._uuid_to_doc_id.get(chunk_uuid, "UNKNOWN") 
                    
                    # ฉีด Metadata เพิ่มเติมสำหรับการแสดงผลและ Traceability
                    metadata["chunk_uuid"] = chunk_uuid
                    metadata["doc_id"] = doc_id
                    metadata["doc_type"] = doc_type
                    
                    documents.append(LcDocument(page_content=text, metadata=metadata))
            
            # เรียงลำดับเอกสารตาม Chunk Order (ถ้ามีข้อมูล index ใน metadata) เพื่อความต่อเนื่อง
            try:
                documents.sort(key=lambda x: (x.metadata.get("doc_id", ""), x.metadata.get("chunk_index", 0)))
            except:
                pass

            logger.info(f"✅ Successfully retrieved {len(documents)} chunks from '{collection_name}'.")
            return documents

        except Exception as e:
            logger.error(f"❌ Error retrieving chunks from collection '{collection_name}': {e}", exc_info=True)
            return []
    
    @property
    def client(self) -> Optional[chromadb.PersistentClient]:
        """
        Provides access to the underlying Chroma Persistent Client (Re-validate in worker).
        (FIX: แก้ไข AttributeError ใน get_retriever)
        """
        # 🎯 FIX: เรียก ensure เพื่อให้มั่นใจว่า client ไม่ได้หายไปใน worker context
        self._ensure_chroma_client_is_valid()
        return self._client
    
    @property
    def doc_id_map(self) -> Dict[str, Dict[str, Any]]:
        """Provides access to the Stable Doc ID -> Chunk UUIDs mapping."""
        return self._doc_id_mapping

    @property
    def uuid_to_doc_id_map(self) -> Dict[str, str]:
        """Provides access to the Chunk UUID -> Stable Doc ID mapping."""
        return self._uuid_to_doc_id
    

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

def get_vectorstore(
    collection_name: str, 
    tenant: str, 
    year: Optional[int],
    # 💡 เพิ่ม Argument ที่จำเป็นทั้งหมดเพื่อสร้าง Chroma Client ที่ถูกต้อง
    # ถ้าคุณใช้ Embedding Model ภายใน VectorStoreExecutorSingleton
    # อาจจะต้องเพิ่ม embedding_function หรืออื่นๆ ด้วย
) -> VectorStoreExecutorSingleton:
    """
    Wrapper function สำหรับเรียกใช้ VectorStoreExecutorSingleton 
    และส่งผ่าน Argument ที่จำเป็นในการระบุ Path และ Collection Name.
    """
    
    # 🎯 FIX: ส่งผ่าน Argument ไปยัง Constructor ของคลาสหลัก
    return VectorStoreExecutorSingleton(
        collection_name=collection_name, 
        tenant=tenant, 
        year=year
    )

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

class MultiDocRetriever(BaseRetriever): # FIX: ไม่มี BaseModel เพื่อเลี่ยง Metaclass Conflict
    # 🎯 FIX: Pydantic Fields (ใช้ชื่อตัวแปรที่ไม่มี _ นำหน้าสำหรับการรับ Input)
    retrievers_list: List[NamedRetriever] = Field(default_factory=list)
    k_per_doc: int = Field(default=INITIAL_TOP_K)
    doc_ids_filter: Optional[Set[str]] = Field(default=None) 
    
    # Reranking fields 
    compressor: Optional[BaseDocumentCompressor] = Field(default=None)
    final_k: int = Field(default=FINAL_K_RERANKED)
    
    # 🎯 Internal Fields (ประกาศทุกตัวที่ต้องการกำหนดค่าใน __init__ และ exclude=True)
    _executor: Optional[Union[ThreadPoolExecutor, ProcessPoolExecutor]] = Field(default=None, exclude=True)
    _executor_type: Optional[str] = Field(default=None, exclude=True) 
    _executor_mode: Optional[str] = Field(default=None, exclude=True)
    _all_retrievers: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    _doc_ids_filter_list: Optional[List[str]] = Field(default=None, exclude=True) 
    _chroma_filter: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    _manager: Optional['VectorStoreManager'] = Field(default=None, exclude=True) 
    _is_running: bool = Field(default=False, exclude=True) # เพิ่ม is_running สำหรับ cleanup
    _lock: threading.Lock = Field(default_factory=threading.Lock, exclude=True) # เพิ่ม lock
    
    # NOTE: _retrievers_list เป็นชื่อที่ใช้ใน _get_relevant_documents
    # ต้องมั่นใจว่ามันถูกกำหนดค่าจาก self.retrievers_list
    _retrievers_list: List[NamedRetriever] = Field(default_factory=list, exclude=True) 

    class Config:
        arbitrary_types_allowed = True
    
    # -------------------- Property: num_workers (FIXED) --------------------
    @property
    def num_workers(self) -> int:
        """Calculates the optimal number of workers for the current executor type."""
        # ดึง MAX_PARALLEL_WORKERS จาก globals()
        max_workers_from_config = globals().get('MAX_PARALLEL_WORKERS', 4) 
        
        # ดึง _executor_type ที่ถูกกำหนดค่าแล้ว
        # ใช้ getattr เพื่อความปลอดภัยถ้า __init__ ยังทำงานไม่เสร็จสมบูรณ์
        executor_type = getattr(self, '_executor_type', 'thread') 
        
        if executor_type == "process":
            # สำหรับ Process Pool ควรใช้จำนวนที่จำกัด
            return max(1, min(max_workers_from_config, os.cpu_count() - 1 if os.cpu_count() else 4))
        # สำหรับ Thread Pool สามารถใช้จำนวนที่สูงกว่าได้
        return max_workers_from_config

    # -------------------- Initializer --------------------
    def __init__(self, **data: Any) -> None:
        """Initializes the MultiDocRetriever and its internal state."""
        
        # 1. เรียก Pydantic init ก่อน
        super().__init__(**data)

        # 2. กำหนดค่าให้กับ Internal Fields โดยใช้ object.__setattr__
        #    เพื่อหลีกเลี่ยงการถูกดักโดย Pydantic V1 __setattr__ (FIXED)
        
        # กำหนดค่าที่จำเป็น
        object.__setattr__(self, '_retrievers_list', self.retrievers_list)
        
        # กำหนด Executor Type
        executor_type_val = self._choose_executor()
        object.__setattr__(self, '_executor_type', executor_type_val)
        
        # กำหนด Executor instance
        object.__setattr__(self, '_executor', self._initialize_executor())
        object.__setattr__(self, '_is_running', True)
        
        # 3. เตรียมโครงสร้าง Retriever
        object.__setattr__(self, '_all_retrievers', {
            r.doc_id: r for r in self.retrievers_list
        })
        
        # 4. เตรียม Doc ID Filter
        if self.doc_ids_filter:
            doc_ids_list = list(self.doc_ids_filter)
            object.__setattr__(self, '_doc_ids_filter_list', doc_ids_list)
            # สร้าง Chroma Filter
            chroma_filter = {"$or": [{"chunk_uuid": {"$in": doc_ids_list}}]}
            object.__setattr__(self, '_chroma_filter', chroma_filter)
        
        # 5. Logging (ตอนนี้ num_workers สามารถใช้ได้แล้ว)
        if self._executor_type == "process":
            logger.info(f"Initialized MultiDocRetriever using ProcessPoolExecutor ({self.num_workers} workers).")
        else:
            logger.info(f"Initialized MultiDocRetriever using ThreadPoolExecutor ({self.num_workers} threads).")
            
    # -------------------- Executor Management --------------------
    
    def _initialize_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        """Initializes the appropriate executor."""
        # NOTE: เนื่องจาก _get_executor มี logic การสร้างและเก็บ instance อยู่
        # เราจึงเรียกมันมาตรงๆ ได้เลย
        return self._get_executor() 
        
    def _choose_executor(self) -> str:
        # NOTE: ต้องมั่นใจว่า ENV_FORCE_MODE, _detect_system(), _detect_torch_device() ถูก import
        # ดึง ENV_FORCE_MODE จาก globals()
        ENV_FORCE_MODE = globals().get('ENV_FORCE_MODE', None) 
        
        if ENV_FORCE_MODE == "process":
            return "process"
        if ENV_FORCE_MODE == "thread":
            return "thread"
            
        # NOTE: ถ้า _detect_system, _detect_torch_device เป็น Global functions
        # คุณต้องเข้าถึงมันผ่าน globals() หรือ import มันมาตรงๆ
        system = globals().get('_detect_system', lambda: {'platform': 'unknown', 'cpu_count': 1, 'total_ram_gb': 1})()
        _detect_torch_device_func = globals().get('_detect_torch_device', lambda: 'cpu')
        
        if _detect_torch_device_func() == "mps" or system['platform'] == 'darwin':
            return "thread"
        
        if system['cpu_count'] >= 4 and (system['total_ram_gb'] is None or system['total_ram_gb'] > 8):
            return "process"
            
        return "thread" 

    def _get_executor(self) -> Union[ThreadPoolExecutor, ProcessPoolExecutor]:
        if self._executor is None:
            # ใช้ self.num_workers ที่เป็น Property คำนวณแล้ว
            workers = self.num_workers 
            
            # 📌 FIX: ใช้ object.__setattr__ ในการกำหนดค่า self._executor ใน lazy init
            if self._executor_type == "process":
                new_executor = ProcessPoolExecutor(max_workers=workers)
                logger.info(f"🛠️ Using ProcessPoolExecutor with {workers} workers.")
            else:
                new_executor = ThreadPoolExecutor(max_workers=workers)
                logger.info(f"🛠️ Using ThreadPoolExecutor with {workers} workers.")
            
            object.__setattr__(self, '_executor', new_executor)
            
        return self._executor
    
    # (เมธอด get_relevant_documents, _choose_executor, shutdown, __del__, 
    # _static_retrieve_task, _thread_retrieve_task, _get_relevant_documents 
    # ที่เหลือยังคงใช้โค้ดที่คุณให้มาล่าสุด)

    # ... (ส่วนที่เหลือของคลาส) ...
    
    def shutdown(self):
        with self._lock: # ใช้ lock เพื่อป้องกัน race condition
            if self._executor and self._is_running:
                executor_type_name = "ProcessPoolExecutor" if self._executor_type == "process" else "ThreadPoolExecutor"
                # ตอนนี้เราใช้ self.num_workers ได้แล้ว
                workers = self.num_workers
                
                logger.info(f"Shutting down MultiDocRetriever's {executor_type_name} executor ({workers} workers).")
                self._executor.shutdown(wait=True)
                object.__setattr__(self, '_executor', None)
                object.__setattr__(self, '_is_running', False)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


    @staticmethod
    def _static_retrieve_task(named_r: NamedRetriever, query: str, chroma_filter: Optional[Dict]):
        """Static task method for ProcessPoolExecutor."""
        # NOTE: โค้ดถูกคัดลอกมาเหมือนเดิม
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
        # NOTE: โค้ดถูกคัดลอกมาเหมือนเดิม
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
        # ใช้ self.num_workers แทนการคำนวณ max_workers ในเมธอดนี้
        max_workers = self.num_workers
        # NOTE: self._retrievers_list ถูกกำหนดค่าใน __init__ แล้ว
        num_retrievers = len(self._retrievers_list) 
        
        # ปรับ max_workers ให้ไม่เกินจำนวน retriever
        max_workers = min(num_retrievers, self.num_workers)
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

# -------------------- END OF MultiDocRetriever --------------------
# -------------------- load_all_vectorstores --------------------
def load_all_vectorstores(
    tenant: str = DEFAULT_TENANT, 
    year: int = DEFAULT_YEAR, 
    doc_ids: Optional[Set[str]] = None,
    doc_types: Optional[Union[str, List[str]]] = None,
    enabler_filter: Optional[str] = None,
    top_k: int = INITIAL_TOP_K,
    final_k: int = FINAL_K_RERANKED
) -> 'VectorStoreManager':
    """
    Initializes the VSM and the main MultiDocRetriever for the current assessment context.
    Improved with Case-Insensitive matching for Mac/Linux environments.
    """
    # 1. Initialize VSM (Singleton)
    manager = VectorStoreManager(
        tenant=tenant, 
        year=year, 
    )
    
    # 2. Prepare the list of target collection keys
    target_collection_keys: Set[str] = set()
    
    # สแกนหา collections ที่มีอยู่จริงใน Tenant/Year
    existing_collections = list_vectorstore_folders(tenant, year, doc_type=None, enabler=None) 
    
    # 🎯 [FIX] สร้าง Mapping แบบ Case-Insensitive (key=ตัวเล็ก, value=ชื่อจริงในเครื่อง)
    # เพื่อรองรับกรณีในเครื่องเป็น 'evidence_km' แต่ระบบส่ง 'evidence_KM'
    existing_map = {c.lower(): c for c in existing_collections}
    
    # Filtering Logic
    if doc_types:
        if isinstance(doc_types, str):
            doc_types = [dt.strip() for dt in doc_types.split(',')]
        
        for dt in doc_types:
            dt_norm = dt.lower().strip()
            
            if dt_norm == EVIDENCE_DOC_TYPES.lower():
                # ถ้าเป็น doc_type 'evidence' และมีการกรอง enabler
                if enabler_filter:
                    # แปลง enabler เป็น list และล้างค่าช่องว่าง
                    enabler_list = [e.strip() for e in enabler_filter.split(',')]
                    for enabler in enabler_list:
                        # สร้าง key ที่คาดหวัง (มักจะได้ evidence_KM หรือ evidence_km)
                        key_expected = get_doc_type_collection_key(dt_norm, enabler).lower()
                        
                        if key_expected in existing_map:
                            # ✅ ใช้ชื่อจริงที่เจอใน Folder (เช่น 'evidence_km')
                            target_collection_keys.add(existing_map[key_expected])
                        else:
                            logger.warning(
                                f"🔍 DEBUG: Skipping collection '{key_expected}' "
                                f"(Not found in existing: {list(existing_map.keys())})."
                            )
                else:
                    # ถ้าไม่ระบุ enabler ให้เอา evidence ทั้งหมดที่ขึ้นต้นด้วย evidence_
                    for c_low, c_orig in existing_map.items():
                        if c_low.startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"):
                            target_collection_keys.add(c_orig)
            else: 
                # สำหรับ doc_type อื่นๆ ที่ไม่ได้แยกตามปี (เช่น document, faq)
                key_gen = get_doc_type_collection_key(dt_norm, None).lower()
                if key_gen in existing_map:
                     target_collection_keys.add(existing_map[key_gen])
                
                # รองรับกรณี collection ชื่อ doc_type_all
                key_all = get_doc_type_collection_key(dt_norm, "ALL").lower()
                if key_all in existing_map:
                     target_collection_keys.add(existing_map[key_all])
    
    else:
        # หากไม่ระบุ doc_types เลย ให้โหลดทั้งหมดที่สแกนเจอ
        target_collection_keys.update(existing_collections)

    logger.info(f"🔍 DEBUG: Attempting to load {len(target_collection_keys)} total target collections: {target_collection_keys}")

    # 3. Build NamedRetriever objects
    all_retrievers: List[NamedRetriever] = []
    
    for collection_name in target_collection_keys:
        # แยกชื่อ collection เพื่อหา doc_type และ enabler (e.g., 'evidence_km' -> ['evidence', 'km'])
        parts = collection_name.split('_')
        doc_type_for_check = parts[0]
        # รักษา Case ของ enabler ตามที่ระบบต้องการ (มักจะเป็นตัวใหญ่)
        enabler_for_check = parts[1].upper() if len(parts) > 1 else None
        
        # 🎯 กำหนดปีที่ถูกต้อง: evidence ใช้ปีจาก config, อื่นๆ (Global) ไม่ใช้ปี
        target_year = year
        if doc_type_for_check.lower() != EVIDENCE_DOC_TYPES.lower():
            target_year = None
            
        nr = NamedRetriever(
            doc_id=collection_name, 
            doc_type=doc_type_for_check, 
            enabler=enabler_for_check, 
            top_k=top_k, 
            final_k=final_k, 
            tenant=tenant, 
            year=target_year
        )
        all_retrievers.append(nr)
        logger.info(f"🔍 DEBUG: Added retriever for '{collection_name}' (Year={target_year}, Enabler={enabler_for_check}).")

    if not all_retrievers:
        # พ่น Error ที่ระบุ Path ชัดเจนเพื่อช่วยในการ Debug หน้างาน
        debug_vstore_path = f"data_store/{tenant}/vectorstore/{year}"
        raise ValueError(
            f"No vectorstore collections found matching:\n"
            f" - Path: {debug_vstore_path}\n"
            f" - DocTypes: {doc_types}\n"
            f" - Enabler: {enabler_filter}\n"
            f"Please check if ChromaDB folders exist in the path above."
        )
        
    # 4. Initialize MultiDocRetriever (MDR)
    
    # 4.1 Prepare Reranker (Compressor)
    reranker = None
    if final_k > 0:
        reranker = get_global_reranker()
        if reranker is None:
             logger.warning("❌ Reranker failed to initialize. Reranking disabled.")
             final_k = top_k 
        else:
             logger.info(f"✅ Reranker initialized ({reranker.rerank_model}).")
             

    # 4.2 Create MDR instance
    mdr = MultiDocRetriever( 
        retrievers_list=all_retrievers, 
        k_per_doc=top_k, 
        doc_ids_filter=doc_ids,
        compressor=reranker, 
        final_k=final_k
    )
    
    # 5. Set MDR in VSM (Singleton)
    manager._multi_doc_retriever = mdr
    
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