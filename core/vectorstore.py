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
    MAPPING_BASE_DIR,
    FINAL_K_RERANKED,
    INITIAL_TOP_K,
    EVIDENCE_DOC_TYPES,
    MAX_PARALLEL_WORKERS,
    DEFAULT_TENANT,
    DEFAULT_YEAR,
    DEFAULT_ENABLER
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

    # ... (โค้ดเดิมทั้งหมดจนถึงตรงนี้)

    if _CACHED_EMBEDDINGS is None:
        with _EMBED_LOCK:
            if _CACHED_EMBEDDINGS is None:
                # เปลี่ยนตรงนี้เด็ดขาด!!!
                model_name = "intfloat/multilingual-e5-base"  # หรือ large ถ้าเครื่องแรง
                
                logger.info(f"Loading BEST Thai RAG embedding 2025: {model_name} on {device}")
                logger.info("This model was used to build ALL PEA 2568 vectorstores (evidence_km, document, etc.)")
                
                try:
                    _CACHED_EMBEDDINGS = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={
                            "device": device,
                            # สำคัญมาก: e5 ต้องใช้ prefix!
                            # ถ้าไม่ใส่ → คะแนนตกฮวบ!
                        },
                        encode_kwargs={
                            "normalize_embeddings": True,
                            # สำหรับ e5 series ต้องใส่ prefix เท่านั้น!!!
                            "prompt": "query: "  # สำหรับ query
                            # หรือถ้าจะ embed เอกสาร → ใช้ "passage: "
                        }
                    )
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
                    logger.warning("Falling back to paraphrase-multilingual-MiniLM-L12-v2")
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
                rerank_model="mixedbread-ai/mxbai-rerank-xsmall-v1"
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


# -------------------- Path Helper Function (REVISED for Lowercase Path Suffix and Optional Year) --------------------
def _build_vectorstore_path_by_doc_type(tenant: str, year: Optional[int], doc_type: str, enabler: Optional[str] = None) -> str:
    """
    สร้าง Full Path สำหรับ Collection โดยใช้ตรรกะ EVIDENCE_DOC_TYPES
    - Evidence (มี year): VECTORSTORE_DIR / tenant / year / collection_name
    - Docs/FAQ (ไม่มี year): VECTORSTORE_DIR / tenant / collection_name
    """
    doc_type_lower = doc_type.lower()
    collection_name = _get_collection_name(doc_type, enabler)
    
    path_segments = [VECTORSTORE_DIR, tenant.lower()]
    
    # 🎯 FIX: ตรวจสอบปี (year is not None) สำหรับ Evidence เท่านั้น
    if doc_type_lower == EVIDENCE_DOC_TYPES.lower() and year is not None:
        # Path สำหรับ evidence คือ /tenant/year/collection_name
        path_segments.append(str(year))
        path_segments.append(collection_name)
    else:
        # Path สำหรับ document/faq (ไม่ใช้ปี)
        path_segments.append(collection_name)
        
    return os.path.join(*path_segments)

# -----------------------------------------------------------

def _get_collection_name(doc_type: str, enabler: Optional[str] = None) -> str:
    """
    Calculates the Chroma collection name (Internal identifier).
    """
    doc_type_norm = doc_type.strip().lower()
    
    if doc_type_norm == EVIDENCE_DOC_TYPES.lower():
        # สำหรับ Evidence: ชื่อ Collection ต้องรวม enabler เสมอ
        enabler_norm = (enabler or "km").strip().lower()
        
        # 🎯 FIX: รวมตรรกะให้เป็นบรรทัดเดียว: evidence + enabler 
        collection_name = f"{doc_type_norm}_{enabler_norm}"
        
    else:
        # สำหรับเอกสารทั่วไป: ชื่อ Collection คือ doc_type เท่านั้น
        collection_name = doc_type_norm
        
    logger.debug(f"🧭 DEBUG: _get_collection_name(doc_type={doc_type}, enabler={enabler}) => {collection_name}")
    return collection_name

# 📌 REVISED: เพิ่ม tenant และ year, ใช้ Logic Centralized KM
def get_vectorstore_path(
    tenant: str, 
    year: Optional[int], 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None
) -> str:
    """
    Calculates the full persist directory path for the vector store instance
    based on the Centralized KM Logic.
    """
    if not doc_type:
        # ถ้าไม่มี doc_type ให้ return root ของ tenant
        return os.path.join(VECTORSTORE_DIR, tenant.lower()) 
        
    # 🎯 FIX: เรียกใช้ _build_vectorstore_path_by_doc_type ที่แก้ไขแล้ว
    return _build_vectorstore_path_by_doc_type(tenant, year, doc_type, enabler)

# 📌 REVISED: เพิ่ม tenant และ year, ใช้ Logic Centralized KM
def vectorstore_exists(
    doc_id: str, # รักษาไว้ตาม Signature เดิมของผู้ใช้
    tenant: str,
    year: Optional[int], # <--- รองรับ None สำหรับ General Docs
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR # base_path จะถูกละเลยเนื่องจากใช้ global VECTORSTORE_DIR
) -> bool:
    """
    Checks if the Vector Store directory exists for the given context.
    """
    if not doc_type:
        return False
        
    # 1. Get the full path using the updated logic
    path = get_vectorstore_path(tenant, year, doc_type, enabler) 
    
    # 2. Check for the actual data file created by Chroma
    file_path = os.path.join(path, "chroma.sqlite3")
    
    if not os.path.isdir(path):
        logger.warning(f"❌ V-Exists Check: Directory not found for doc_type '{doc_type}' at {path}")
        return False
    if os.path.isfile(file_path):
        return True
    logger.error(f"❌ V-Exists Check: FAILED to find file chroma.sqlite3 at {file_path}")
    return False

def _get_collection_parent_dir(tenant: str, year: Optional[int], doc_type: str) -> str: 
    """
    Calculates the parent directory where the collection folder resides.
    - evidence: VECTORSTORE_DIR / tenant / year 
    - others:   VECTORSTORE_DIR / tenant 
    """
    doc_type_lower = doc_type.lower()
    path_segments = [VECTORSTORE_DIR, tenant.lower()]
    
    # 🎯 FIX: ตรวจสอบปี (year is not None) สำหรับ Evidence เท่านั้น
    if doc_type_lower == EVIDENCE_DOC_TYPES.lower() and year is not None:
        # Parent ของ evidence คือมีปี
        path_segments.append(str(year))
        
    return os.path.join(*path_segments)

def list_vectorstore_folders(
    tenant: str, 
    year: int, # NOTE: ใช้ปีปัจจุบันที่รัน (int)
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR
) -> List[str]:
    """
    Lists the actual collection folder names (e.g., 'evidence_km', 'document')
    that exist under the specified tenant and year context.
    """
    
    # Scenario 1: Specific doc_type/enabler is requested
    if doc_type:
        doc_type_norm = doc_type.lower().strip()
        collection_name = _get_collection_name(doc_type_norm, enabler)
        
        # 🎯 FIX 4: ใช้ get_vectorstore_path ที่แก้ไขแล้ว
        full_collection_path = get_vectorstore_path(tenant, year, doc_type_norm, enabler)
        
        if os.path.isdir(full_collection_path):
            return [collection_name] 
        return []

    # Scenario 2: List ALL collections for the given tenant/year context (List All)
    
    collections: Set[str] = set()
    
    # 1. Scan the Year Root (สำหรับ Doc Type: evidence)
    root_year = _get_collection_parent_dir(tenant, year, EVIDENCE_DOC_TYPES) 
    if os.path.isdir(root_year):
        # ค้นหา evidence_... collections ภายในโฟลเดอร์ปี
        for sub_dir in os.listdir(root_year):
             # 🎯 FIX 5: ใช้ startsWith เพื่อจับ 'evidence_km' (หรือ evidence_xxx)
             if sub_dir.lower().startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"): 
                 collections.add(sub_dir.lower()) 

    # 2. Scan the Common Root (สำหรับ Doc Type: document, faq, ฯลฯ)
    # Common Root: VECTORSTORE_DIR / tenant
    root_common = _get_collection_parent_dir(tenant, year=None, doc_type="document") 
    if os.path.isdir(root_common):
        # ค้นหาโฟลเดอร์ Doc Type
        for sub_dir in os.listdir(root_common):
            sub_dir_lower = sub_dir.lower()
            if sub_dir_lower == EVIDENCE_DOC_TYPES.lower() or sub_dir.isdigit():
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

    def __init__(self, base_path: str = VECTORSTORE_DIR, tenant: str = DEFAULT_TENANT, year: int = DEFAULT_YEAR):
        if not self._is_initialized:
            self._base_path = base_path
            self.tenant = tenant.lower()        # ต้อง .lower() ด้วยนะ!
            self.year = year
            self.enabler = DEFAULT_ENABLER   
            self._chroma_cache = {}
            self._embeddings = get_hf_embeddings()
            
            # แก้ตรงนี้: Chroma Client ต้องชี้ที่ root ของ tenant เท่านั้น!
            # chroma_root_path = os.path.join(base_path, self.tenant)   # ไม่มี str(year)!!!
            # chroma_root_path = os.path.join(base_path, self.tenant, str(self.year))  # เพิ่ม year เข้าไป!
            # 1. สร้างชื่อ Collection ที่ถูกต้อง (evidence_km)
            collection_name = _get_collection_name(EVIDENCE_DOC_TYPES, self.enabler)
            
            # 2. ใช้ Path Helper ที่คุณมีอยู่ เพื่อสร้าง Full Path
            #    ผลลัพธ์: /vectorstore/pea/2568/evidence_km
            chroma_root_path = _build_vectorstore_path_by_doc_type(
                tenant=self.tenant, 
                year=self.year, 
                doc_type=EVIDENCE_DOC_TYPES, 
                enabler=self.enabler
            )
            
            self._client = chromadb.PersistentClient(path=chroma_root_path)
            logger.info(f"ChromaDB Client initialized at FULL COLLECTION PATH: {chroma_root_path}")
            # Note: เราใช้ Chroma Client ในโหมดนี้เพื่อโหลด DB ที่สร้างเป็นราย collection
            
            # โหลด mapping หลังจากตั้งค่า tenant/year แล้ว
            self._load_doc_id_mapping() 
            
            logger.info(f"Initialized VectorStoreManager (Tenant: {self.tenant}, Year: {self.year}). "
                        f"Loaded {len(self._doc_id_mapping)} stable doc IDs.")
            
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
        
        # NOTE: Doc ID Mapping ต้องใช้ Path ที่ระบุปีเสมอ
        # mapping_filename = f"{self.tenant.lower()}_{self.year}_doc_id_mapping.json"
        mapping_filename = f"{self.tenant.lower()}_{self.year}_{self.enabler.lower()}_doc_id_mapping.json"
        
        doc_id_mapping_path = os.path.join(
            MAPPING_BASE_DIR, 
            self.tenant.lower(), 
            str(self.year), 
            mapping_filename
        )
  
        try:
            with open(doc_id_mapping_path, "r", encoding="utf-8") as f:
                mapping_data: Dict[str, Dict[str, Any]] = json.load(f)
                cleaned_mapping = {k.strip(): v for k, v in mapping_data.items()}
                self._doc_id_mapping = cleaned_mapping
                for doc_id, doc_entry in cleaned_mapping.items():
                    if isinstance(doc_entry, dict) and "chunk_uuids" in doc_entry and isinstance(doc_entry.get("chunk_uuids"), list):
                        for uid in doc_entry["chunk_uuids"]:
                            self._uuid_to_doc_id[uid] = doc_id
            logger.info(f"✅ Loaded Doc ID Mapping from {doc_id_mapping_path}: {len(self._doc_id_mapping)} original documents, {len(self._uuid_to_doc_id)} total chunks.")
        except FileNotFoundError:
            logger.warning(f"⚠️ Doc ID Mapping file not found at {doc_id_mapping_path}.")
        except Exception as e:
            logger.error(f"❌ Failed to load Doc ID Mapping from {doc_id_mapping_path}: {e}")

    def _re_parse_collection_name(self, collection_name: str) -> Tuple[str, Optional[str]]:
        collection_name_lower = collection_name.strip().lower()
        if collection_name_lower.startswith(f"{EVIDENCE_DOC_TYPES}_"):
            parts = collection_name_lower.split("_", 1)
            return EVIDENCE_DOC_TYPES, parts[1].upper() if len(parts) == 2 else None
        return collection_name_lower, None

    def _load_chroma_instance(self, collection_name: str) -> Optional[Chroma]:
        """
        โหลด (หรือคืนจาก cache) Chroma instance โดยใช้ PersistentClient เดียวกันทุกครั้ง
        รองรับทั้ง year-specific (evidence_*) และ general/policy collections
        """
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
            # 4. กำหนด target_year → นี่คือจุดที่เคยฆ่าคุณมาตลอด!!!
            # ------------------------------------------------------------------
            # evidence_* ทุกตัว (ไม่ว่าจะมี enabler หรือไม่) ต้องใช้ year ของ tenant เท่านั้น!!!
            if doc_type.startswith("evidence"):
                target_year: Optional[int] = self.year
            else:
                # general / policy / หรือ collection ที่มี enabler (เช่น risk_control_xxx)
                # ให้เป็น general (ไม่มี year)
                target_year = None

            # ------------------------------------------------------------------
            # 5. สร้าง persist_directory ที่ถูกต้อง 100%
            # ------------------------------------------------------------------
            persist_directory = get_vectorstore_path(
                tenant=self.tenant,
                year=target_year,
                doc_type=doc_type,
                enabler=enabler,
            )

            # ------------------------------------------------------------------
            # 6. ตรวจสอบว่ามี folder จริงไหม (debug สะดวกสุด ๆ)
            # ------------------------------------------------------------------
            if not os.path.exists(persist_directory):
                logger.warning(
                    f"Vectorstore directory NOT FOUND!\n"
                    f"   Collection   : {collection_name}\n"
                    f"   Expected path: {persist_directory}\n"
                    f"   tenant={self.tenant} | year={self.year} | doc_type={doc_type} | enabler={enabler or 'None'}"
                )
                # ลองเช็ค path อีกแบบเผื่อใครตั้งชื่อ folder ผิด
                alt_path = get_vectorstore_path(self.tenant, self.year, doc_type, enabler)
                if os.path.exists(alt_path):
                    logger.warning(f"   BUT found at alternative path: {alt_path} ← อาจตั้งค่า year ผิด?")
                return None

            # ------------------------------------------------------------------
            # 7. ตรวจสอบว่า client ถูก init แล้ว
            # ------------------------------------------------------------------
            if self._client is None:
                logger.error("Chroma PersistentClient is None! ต้อง init ก่อนใช้งาน")
                return None

            try:
                # ------------------------------------------------------------------
                # 8. สร้าง Chroma instance (ใช้ client เดียวกัน → สำคัญสุด!)
                # ------------------------------------------------------------------
                vectordb = Chroma(
                    client=self._client,                     # ต้องใช้ client เดียวกันทุกครั้ง!!!
                    embedding_function=self._embeddings,
                    collection_name=collection_name,
                )

                # Cache ไว้ใช้ครั้งต่อไป
                self._chroma_cache[collection_name] = vectordb

                logger.info(
                    f"Loaded Chroma collection '{collection_name}' → {persist_directory}"
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

    def retrieve_by_chunk_uuids(self, chunk_uuids: List[str], collection_name: Optional[str] = None) -> List[LcDocument]:
            """
            Retrieves documents from Chroma collection based on a list of unique chunk_uuids (IDs).
            Includes de-duplication logic to prevent ChromaDB DuplicateIDError.
            """
            if not chunk_uuids:
                logger.info("VSM: No chunk_uuids provided for hydration.")
                return []
            
            # กำหนดชื่อ collection เริ่มต้นหากไม่ได้ระบุ
            if collection_name is None:
                collection_name = f"evidence_{getattr(self, 'enabler', 'km').lower()}"

            # โหลด Chroma instance
            chroma_instance = self._load_chroma_instance(collection_name)
            if not chroma_instance:
                logger.warning(f"VSM: Cannot load collection '{collection_name}' for hydration.")
                return []

            # DEBUG (ตามที่คุณมีอยู่)
            logger.info(f"VSM: Attempting hydration with {len(chunk_uuids)} UUIDs from '{collection_name}'")
            logger.info(f"VSM: First 5 UUIDs → {chunk_uuids[:5]}")

            try:
                collection = chroma_instance._collection

                # 1. Clean IDs (แปลงเป็น str และ strip)
                clean_ids = [str(uuid).strip() for uuid in chunk_uuids if uuid and str(uuid).strip()]

                # 2. 🎯 FIX: ทำการ De-duplicate IDs ก่อนส่งให้ ChromaDB
                unique_chunk_uuids = list(set(clean_ids))
                
                if len(unique_chunk_uuids) < len(clean_ids):
                    duplicated_count = len(clean_ids) - len(unique_chunk_uuids)
                    logger.warning(f"VSM: De-duplicated {duplicated_count} repeated UUIDs before calling ChromaDB get.")
                
                if not unique_chunk_uuids:
                    logger.warning("VSM: All UUIDs became empty after cleaning or duplication removal!")
                    return []

                # 3. เรียกใช้ ChromaDB ด้วย IDs ที่ไม่ซ้ำซ้อน
                results = collection.get(
                    ids=unique_chunk_uuids,
                    include=["documents", "metadatas"]
                )

                found_count = len(results["ids"]) if results["ids"] else 0
                logger.info(f"VSM: Successfully retrieved {found_count}/{len(unique_chunk_uuids)} chunks by UUID from '{collection_name}'")

                # 4. แปลงผลลัพธ์เป็น LcDocument
                docs = []
                for i, doc_id in enumerate(results["ids"]):
                    content = results["documents"][i]
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    doc = LcDocument(page_content=content, metadata=meta.copy())
                    # สำคัญ: doc_id ที่ได้จาก Chroma คือ chunk_uuid
                    doc.metadata["chunk_uuid"] = doc_id 
                    docs.append(doc)
                
                return docs

            except Exception as e:
                logger.error(f"VSM: FATAL Error in retrieve_by_chunk_uuids: {e}", exc_info=True)
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

        # 1. สร้าง retriever ที่ควบคุม k ได้จริงทุกครั้ง
        def raw_retrieve(query: str, filter_dict: Optional[dict] = None, k: int = top_k) -> List[LcDocument]:
            try:
                original_query = query
                
                # สำคัญมาก: PEA 2568 ingest ด้วย paraphrase-multilingual-MiniLM-L12-v2 → ไม่มี prefix
                # ดังนั้นห้ามใส่ query: เด็ดขาด!
                query = query.strip()
                logger.critical(f"[NO PREFIX FOR PEA] Using raw query: '{query[:100]}...'")

                search_kwargs = {"k": k}
                if filter_dict:
                    search_kwargs["filter"] = filter_dict
                
                docs = chroma_instance.similarity_search(
                    query=query,
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
                filter_dict = config.get("configurable", {}).get("search_kwargs", {}).get("filter")

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
                if scores and len(scores) >= len(reranked):
                    for doc, score in zip(reranked, scores[:len(reranked)]):
                        score = float(score) if score is not None else 0.0
                        # ฉีดแค่ key เดียวที่แน่นอน → _rerank_score_force
                        doc.metadata["_rerank_score_force"] = score
                        # และ source_filename (สำคัญมากสำหรับ extraction)
                        orig = doc.metadata.get("source_filename", "UNKNOWN")
                        doc.metadata["source_filename"] = f"{orig}|SCORE:{score:.4f}"

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
        return list_vectorstore_folders(tenant=self.tenant, year=self.year, base_path=self._base_path)

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
    doc_id: str
    doc_type: str
    top_k: int = INITIAL_TOP_K
    final_k: int = FINAL_K_RERANKED
    base_path: str = VECTORSTORE_DIR
    enabler: Optional[str] = None
    tenant: str = DEFAULT_TENANT
    year: Optional[int] = DEFAULT_YEAR # <--- 🎯 FIX: เปลี่ยนเป็น Optional[int] เพื่อรองรับ None
    
    # load_instance ต้องเรียก VSM โดยใช้ self.tenant และ self.year
    def load_instance(self) -> Any:
        # VSM ถูกสร้างด้วย context ของ collection นี้ (year อาจเป็น None)
        # ⚠️ VSM Singleton ต้องถูก init ด้วยปีเสมอ เพื่อโหลด Doc ID Mapping
        manager = VectorStoreManager(base_path=self.base_path, tenant=self.tenant, year=self.year if self.year is not None else DEFAULT_YEAR) 
        collection_name = _get_collection_name(self.doc_type, self.enabler)
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
def load_vectorstore_retriever(
    doc_id: str, 
    top_k: int = INITIAL_TOP_K, 
    final_k: int = FINAL_K_RERANKED, 
    doc_types: Union[list, str] = "default_collection", 
    base_path: str = VECTORSTORE_DIR, 
    enabler: Optional[str] = None, 
    tenant: str = DEFAULT_TENANT,
    year: int = DEFAULT_YEAR      
):
    if isinstance(doc_types, str):
        target_doc_type = doc_types
    elif isinstance(doc_types, list) and doc_types:
        target_doc_type = doc_types[0]
    else:
        raise ValueError("doc_types must be a single string or a non-empty list containing the target doc_type.")
        
    collection_name = _get_collection_name(target_doc_type, enabler)
    
    # 🎯 FIX 3: ส่ง tenant และ year เข้าไปใน VectorStoreManager
    manager = VectorStoreManager(base_path=base_path, tenant=tenant, year=year) 
    retriever = None
    
    # NOTE: load_vectorstore_retriever ใช้ปีเสมอ เนื่องจากเป็น context ที่เรียกมา
    if vectorstore_exists(doc_id="N/A", base_path=base_path, doc_type=target_doc_type, enabler=enabler, tenant=tenant, year=year):
        retriever = manager.get_retriever(collection_name, top_k, final_k)
        
    if retriever is None:
        raise ValueError(f"❌ Vectorstore for collection '{collection_name}' not found.")
    return retriever

def load_all_vectorstores(
    tenant: str, 
    year: int,    
    doc_types: Optional[Union[str, List[str]]] = None, 
    top_k: int = INITIAL_TOP_K, 
    final_k: int = FINAL_K_RERANKED, 
    base_path: str = VECTORSTORE_DIR, 
    evidence_enabler: Optional[str] = None, 
    doc_ids: Optional[List[str]] = None
) -> VectorStoreManager:
    
    doc_types = [doc_types] if isinstance(doc_types, str) else doc_types or []
    doc_type_filter = {dt.strip().lower() for dt in doc_types}
    
    # 🎯 FIX 1: VSM ต้องสร้างด้วย tenant/year ของ RUN (ปีที่ใช้สำหรับ Doc ID Mapping)
    manager = VectorStoreManager(base_path=base_path, tenant=tenant, year=year) 
    
    all_retrievers: List[NamedRetriever] = []
    target_collection_names: Set[str] = set()

    # --- 1. Collection Discovery ---
    if not doc_type_filter:
        logger.error("Must specify doc_types for multi-year compatibility.")
        raise ValueError("Must specify doc_types when using multi-tenant setup.")
    
    # 🎯 NEW: เราไม่สามารถ list collections ทั้งหมดได้โดยไม่รู้ว่าเอกสารไหนใช้ปี/ไม่ใช้ปี 
    # เราจึงใช้ doc_type_filter ในการสร้าง collection_name แทนการ list จาก folder
    for dt_norm in doc_type_filter:
        if dt_norm == EVIDENCE_DOC_TYPES.lower(): 
            if evidence_enabler:
                # Specific evidence collection: ใช้ year
                collection_name = _get_collection_name(EVIDENCE_DOC_TYPES, evidence_enabler)
                target_collection_names.add(collection_name)
                logger.info(f"🔍 Added specific evidence collection: {collection_name} (Year-Specific)")
            else:
                # All evidence collections: ต้อง list จาก folder ภายใต้ tenant/year
                evidence_collections = list_vectorstore_folders(tenant=tenant, year=year, doc_type=EVIDENCE_DOC_TYPES, base_path=base_path)
                target_collection_names.update(evidence_collections)
                logger.info(f"🔍 Added all evidence collections found: {evidence_collections} (Year-Specific)")
        else:
            # Standard Collections: ไม่ใช้ year
            collection_name = _get_collection_name(dt_norm, None)
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
            base_path=base_path,
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