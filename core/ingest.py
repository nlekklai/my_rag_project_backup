import os
import re
import logging
import unicodedata 
import json 
import uuid 
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Iterable, Dict, Any, Union, Tuple, TypedDict
import pandas as pd
import shutil 
import numpy as np 
import glob 
from pydantic import ValidationError


from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredPowerPointLoader
)

# For PDF inspection
import fitz  # PyMuPDF

# [NEW] เพิ่มส่วนนี้ไว้ในส่วน Helper: Normalization utility
import hashlib

# --- Document Info Model ---
class DocInfo(TypedDict):
    """
    ข้อมูลสถานะของเอกสารที่ใช้สำหรับ API
    """
    doc_id: str             # Stable UUID
    doc_id_key: str         # Filename ID Key (normalized name)
    filename: str
    filepath: str
    doc_type: str           # Collection name (e.g., 'document', 'evidence')
    enabler: Optional[str]  # Enabler code (e.g., 'KM')
    upload_date: str        # ISO format
    chunk_count: int
    status: str             # "Ingested" | "Pending" | "Error"
    size: int               # File size in bytes

# แก้ไข: ลอง Import UnstructuredLoader ใหม่ก่อน (ตาม Deprecation Warning)
try:
    from langchain_unstructured.document_loaders import UnstructuredLoader as UnstructuredFileLoader
except ImportError:
    try:
        from langchain_community.document_loaders import UnstructuredFileLoader
    except ImportError:
        # Fallback to base UnstructuredFileLoader if necessary
        from langchain.document_loaders import UnstructuredFileLoader 
    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter 

# 🟢 Import Chroma และ Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Optional OCR (Omitted for brevity)
_HAS_PDF2IMAGE = False 

# Try to import helper for filtering metadata
try:
    from langchain_community.vectorstores.utils import filter_complex_metadata as _imported_filter_complex_metadata
except Exception:
    _imported_filter_complex_metadata = None


# -------------------- Config --------------------
DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
MAPPING_FILE_PATH = "data/doc_id_mapping.json" 
SUPPORTED_TYPES = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"]

# 🔴 แก้ไข: ใช้ 'statement' ตามที่ผู้ใช้แจ้งมา
SUPPORTED_DOC_TYPES = ["document", "policy", "report", "statement", "evidence", "feedback", "faq"] 

# 📌 [NEW] Enabler Configuration
DEFAULT_ENABLER = "KM"
SUPPORTED_ENABLERS = ["CG", "L", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"]

# Logging
logging.basicConfig(
    filename="ingest.log",
    level=logging.DEBUG, # ✅ แก้ไขจาก INFO เป็น DEBUG
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------- [NEW] Path & Collection Utilities --------------------

def get_target_dir(doc_type: str, enabler: Optional[str] = None) -> str:
    """
    Calculates the target directory name / Chroma Collection Name.
    e.g., ("evidence", "KM") -> "evidence_km"
    e.g., ("document", None) -> "document"
    """
    doc_type_norm = doc_type.strip().lower()

    if doc_type_norm == "evidence":
        # 1. กำหนดค่า enabler สุดท้าย (ใช้ enabler ที่ให้มา หรือ DEFAULT_ENABLER)
        # Note: ต้องกำหนด DEFAULT_ENABLER ใน core/ingest.py
        final_enabler = (enabler or DEFAULT_ENABLER) 
        
        # 🟢 FIX: ตรวจสอบความสมบูรณ์ของ enabler ขั้นสุดท้าย
        if not final_enabler or not final_enabler.strip():
             # หากมาถึงจุดนี้โดยที่ enabler เป็น None, Empty String, หรือ Whitespace เท่านั้น
             # ให้ยกเลิกการทำงานทันที เพราะจะทำให้ชื่อ Collection ผิดพลาด
             raise ValueError(
                 "CRITICAL: Evidence document chunk cannot be indexed: Final enabler is empty or missing. "
                 "Check DEFAULT_ENABLER definition or ensure enabler is provided."
             )

        # 2. จัดรูปแบบชื่อ Collection
        enabler_norm = final_enabler.strip().lower()
        # คืนค่าเป็น evidence_km
        return f"{doc_type_norm}_{enabler_norm}"
        
    # Default logic (for document, faq, policy, etc.)
    return doc_type_norm

def _parse_collection_name(collection_name: str) -> Tuple[str, Optional[str]]:
    """
    Parses a collection/folder name back into doc_type and enabler.
    e.g., "evidence_km" -> ("evidence", "KM")
    e.g., "document"    -> ("document", None)
    """
    collection_name_lower = collection_name.lower()
    
    if collection_name_lower.startswith("evidence_"):
        parts = collection_name_lower.split("_", 1)
        if len(parts) == 2:
            return "evidence", parts[1].upper()
    
    # Check if the collection name is a simple doc_type
    if collection_name_lower in SUPPORTED_DOC_TYPES:
        return collection_name_lower, None
        
    return collection_name_lower, None # Fallback

def _get_source_dir(
    doc_type: str, 
    enabler: Optional[str] = None, 
    base_data_dir: str = DATA_DIR
) -> str:
    """
    Calculates the specific source directory path based on doc_type and enabler.
    e.g., ("evidence", "KM") -> "data/evidence_km"
    e.g., ("document", None) -> "data/document"
    """
    collection_name = get_target_dir(doc_type, enabler)
    return os.path.join(base_data_dir, collection_name)

# -------------------- Helper: safe metadata filter --------------------
def _safe_filter_complex_metadata(meta: Any) -> Dict[str, Any]:
    """
    Ensure metadata is serializable and safe for Chroma / storage.
    Adds a fix to convert single-item lists (like ['eng']) to a single primitive value.
    """
    
    if not isinstance(meta, dict):
        # Attempt to convert to dict if possible
        if hasattr(meta, "items"):
            meta_dict = dict(meta.items())
        else:
            return {} # If it's not a dict, we can't process it safely
    else:
        meta_dict = meta
        
    clean = {}

    for k, v in meta_dict.items():
        if v is None:
            continue
            
        # 1. Handle primitive types (str, int, float, bool)
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        # 2. Handle nested dictionary (Chroma does not support, so stringify)
        elif isinstance(v, dict):
            try:
                # แปลง Dict เป็น JSON string
                clean[k] = json.dumps(v)
            except TypeError:
                clean[k] = str(v)
                
        # 3. Handle list or tuple
        elif isinstance(v, (list, tuple)):
            # 📌 FIX: ตรวจสอบและแปลง List ที่มีสมาชิกเดียวให้เป็น Primitive Type
            if len(v) == 1:
                item = v[0]
                # ถ้าเป็น String/Int/Float/Bool ให้ใช้ค่าเดียวโดดๆ
                if isinstance(item, (str, int, float, bool)):
                    clean[k] = item # เช่น ['eng'] -> 'eng'
                    continue 
                # ถ้าเป็น Numpy type ให้แปลงเป็น Python native type
                elif isinstance(item, (np.floating, np.integer)):
                    clean[k] = item.item() 
                    continue 
            
            # Fallback สำหรับ List ที่มีหลายรายการ หรือมีรายการที่ซับซ้อน
            try:
                # Stringify ทุก element ใน List แล้ว dump เป็น JSON string เพื่อความปลอดภัย
                clean[k] = json.dumps([str(x) for x in v]) 
            except Exception:
                clean[k] = str(v)
                
        # 4. Handle other complex objects (e.g., numpy types)
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
        else:
            try:
                clean[k] = str(v)
            except Exception:
                continue
                
    # Use LangChain's filter utility as a final safety layer if available
    if _imported_filter_complex_metadata:
        try:
            return _imported_filter_complex_metadata(clean)
        except Exception as e:
            logger.debug(f"LangChain filter failed after local cleanup: {e}")
            pass

    return clean

# -------------------- Normalization utility --------------------
def _normalize_doc_id(raw_id: str, file_content: bytes = None) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9]', '', raw_id).lower()
    if len(normalized) > 28:
        normalized = normalized[:28]
    hash_suffix = '000000'
    if file_content:
        hash_suffix = hashlib.sha1(file_content).hexdigest()[:6]
    final_id = (normalized + hash_suffix).ljust(34, '0')
    return final_id


# -------------------- Text Cleaning --------------------
def clean_text(text: str) -> str:
    """
    Basic text cleaning utility.
    """
    if not text: return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u00ad', '')
    text = re.sub(r'[\uFFFD\u2000-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]', '', text)
    # Remove excessive spaces between Thai characters
    text = re.sub(r'([ก-๙])\s{1,3}(?=[ก-๙])', r'\1', text)
    ocr_replacements = {"สำนักงน": "สำนักงาน", "คณะกรรมกร": "คณะกรรมการ"}
    for bad, good in ocr_replacements.items(): text = text.replace(bad, good)
    # Filter out non-printable ASCII except standard ones and Thai characters
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0E00-\u0E7F]', '', text)
    text = re.sub(r'\(\s+', '(', text); text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\r\n', '\n', text); text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text); text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

# -------------------- Loaders Helper --------------------

def _is_pdf_image_only(file_path: str) -> bool:
    """
    ตรวจสอบว่า PDF เป็น image-only หรือมี text layer
    คืนค่า True ถ้าเป็น image-only
    """
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text().strip()
            if text:
                return False  # มี text layer
        return True  # ไม่มี text layer → image-only
    except Exception as e:
        logger.warning(f"Cannot check PDF text layer for {file_path}: {e}")
        return True  # ป้องกัน fail → treat as image-only

def _load_document_with_loader(file_path: str, loader_class: Any) -> List[Document]:
    """
    Helper function to load a document using a specific LangChain loader class.
    - CSV → CSVLoader
    - PDF → auto detect text layer
    - Image → OCR with UnstructuredFileLoader
    """
    try:
        ext = file_path.lower().split('.')[-1]
        
        if loader_class is CSVLoader:
            loader = loader_class(file_path, encoding='utf-8')
        
        elif ext == "pdf":
            if _is_pdf_image_only(file_path):
                logger.info(f"PDF is image-only, using OCR loader: {file_path}")
                loader = UnstructuredPDFLoader(file_path, mode="elements", languages=['tha','eng'])
            else:
                logger.info(f"PDF has text layer, using PyPDFLoader: {file_path}")
                loader = PyPDFLoader(file_path)
        
        elif loader_class is UnstructuredFileLoader:
            # สำหรับไฟล์รูปภาพ
            loader = loader_class(file_path, mode="elements", languages=['tha','eng'])
        
        else:
            loader = loader_class(file_path)
        
        return loader.load()
    
    except Exception as e:
        # 📌 [FIXED: Detailed Loader Failure Logging] บันทึก Exception ที่เกิดขึ้นในการโหลด
        logger.error(f"❌ LOADER FAILED: {os.path.basename(file_path)} - {loader_class.__name__} raised: {type(e).__name__} ({e})")
        return []    

# -------------------- Loaders --------------------

# 📌 [FIXED] Full implementation using imported loaders and helper
FILE_LOADER_MAP = {
    # 🟢 FIX: เปลี่ยนมาใช้ UnstructuredFileLoader เพื่อรองรับ OCR และเพิ่มความทนทาน
    ".pdf": lambda p: _load_document_with_loader(p, UnstructuredFileLoader), 
    
    ".docx": lambda p: _load_document_with_loader(p, UnstructuredWordDocumentLoader),
    ".txt": lambda p: _load_document_with_loader(p, TextLoader),
    ".xlsx": lambda p: _load_document_with_loader(p, UnstructuredExcelLoader),
    ".pptx": lambda p: _load_document_with_loader(p, UnstructuredPowerPointLoader),
    ".md": lambda p: _load_document_with_loader(p, TextLoader), 
    ".csv": lambda p: _load_document_with_loader(p, CSVLoader),
    
    # สำหรับรูปภาพ (ใช้ UnstructuredFileLoader ที่ถูกแก้ไขให้เปิด OCR ภาษาไทย/อังกฤษแล้ว)
    ".jpg": lambda p: _load_document_with_loader(p, UnstructuredFileLoader),
    ".jpeg": lambda p: _load_document_with_loader(p, UnstructuredFileLoader),
    ".png": lambda p: _load_document_with_loader(p, UnstructuredFileLoader),
}

# -------------------- Normalization utility --------------------
def normalize_loaded_documents(raw_docs: List[Any], source_path: Optional[str] = None) -> List[Document]:
    """
    Converts raw loaded documents into clean LangChain Document objects.
    """
    normalized: List[Document] = []
    for idx, item in enumerate(raw_docs):
        try:
            if isinstance(item, Document): doc = item
            else: doc = Document(page_content=str(item), metadata={})
            
            # 📌 [FIX] ตรวจสอบและจัดการ page_content ที่เป็น None หรือ Empty (แก้ ValidationError จาก OCR/UnstructuredLoader)
            if not doc.page_content or doc.page_content.strip() == "":
                logger.warning(f"⚠️ Doc #{idx} from loader has no content (Empty/None). Skipping normalization for this document.")
                continue # <-- จุดสำคัญ: ข้ามเอกสารที่ไม่มีเนื้อหา
            
            if not isinstance(doc.metadata, dict): doc.metadata = {"_raw_meta": str(doc.metadata)}
            if source_path: doc.metadata.setdefault("source_file", os.path.basename(source_path))
            try: doc.metadata = _safe_filter_complex_metadata(doc.metadata)
            except Exception: doc.metadata = {"source_file": os.path.basename(source_path)} if source_path else {}
            normalized.append(doc)
        except Exception as e:
            logger.warning(f"normalize_loaded_documents: skipping item #{idx} due to error: {e}")
            continue
    return normalized

# 📌 Global Text Splitter Configuration (สำหรับใช้ใน Load & Chunk)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    separators=["\n\n", "\n", " ", ""]
)

# -------------------- Load & Chunk Document --------------------
def load_and_chunk_document(
    file_path: str,
    doc_id_key: str, 
    stable_doc_uuid: str, # ⬅️ เพิ่ม Stable UUID เข้ามา
    year: Optional[int] = None,
    version: str = "v1",
    metadata: Optional[Dict[str, Any]] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> List[Document]:
    """
    Load document, inject metadata, clean, and split into chunks.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_func = FILE_LOADER_MAP.get(file_extension)
    
    if not loader_func:
        logger.error(f"No loader found for extension: {file_extension} at {file_path}")
        return []
        
    raw_docs = [] # <-- ตั้งค่าเริ่มต้น
    try:
        # 🟢 [Step 1] เรียกใช้ Loader เพื่อโหลดเอกสาร
        raw_docs = loader_func(file_path)
        
    except ValidationError as e:
        # 📌 [FIX] ดักจับ Pydantic Crash จาก Unstructured Loader/OCR ทันที (สาเหตุหลัก 65%)
        # Error นี้เกิดก่อนที่โค้ดจะส่งค่าไป normalize_loaded_documents
        loader_name = str(loader_func)
        if 'Unstructured' in loader_name or 'Unstructured' in str(loader_func):
             logger.warning(f"⚠️ OCR Crash Handled: {os.path.basename(file_path)} - Loader raised Pydantic ValidationError. Treating as 0 documents loaded.")
             raw_docs = [] # ให้ถือว่าไม่มีเอกสารถูกโหลด แต่ไม่ Crash ทั้งไฟล์
        else:
             # ถ้าไม่ใช่ Unstructured Loader ให้โยน Error ไปต่อ (เพื่อตรวจสอบ)
             raise e 
             
    except Exception as e:
        # ดักจับ Error อื่นๆ ที่อาจเกิดขึ้นระหว่างการโหลดไฟล์
        logger.error(f"❌ Critical error during file loading: {file_path}. Error: {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning(f"Loader returned 0 documents for {os.path.basename(file_path)}. Skipping chunking.")
        return []
        
    # 2. Normalize และทำความสะอาด (ใช้ normalize_loaded_documents ที่เราแก้ไขแล้ว)
    docs = normalize_loaded_documents(raw_docs, source_path=file_path)

    # Inject metadata before splitting
    for d in docs:
        if metadata:
            try:
                new_meta = d.metadata
                new_meta.update(metadata) 
                d.metadata = _safe_filter_complex_metadata(new_meta)
            except Exception:
                d.metadata["injected_metadata"] = str(metadata)
        
        if year: d.metadata["year"] = year
        d.metadata["version"] = version
        d.metadata["doc_id"] = stable_doc_uuid 
        d.metadata["source"] = d.metadata.get("source_file", os.path.basename(file_path))
        d.metadata = _safe_filter_complex_metadata(d.metadata) # Final filter

    # 3. Split into Chunks
    try:
        chunks = TEXT_SPLITTER.split_documents(docs) 
    except Exception as e:
        logger.error(f"Error during document splitting for {os.path.basename(file_path)}: {e}")
        chunks = docs # Fallback to using raw docs as chunks

    for idx, c in enumerate(chunks, start=1):
        c.page_content = clean_text(c.page_content)
        c.metadata["chunk_index"] = idx
        c.metadata = _safe_filter_complex_metadata(c.metadata) 
        
    logger.info(f"Loaded and chunked {os.path.basename(file_path)} -> {len(chunks)} chunks.")
    return chunks


# -------------------- [REVISED] Process single document --------------------
def process_document(
    file_path: str,
    file_name: str,
    stable_doc_uuid: str, # ⬅️ (Warning: This is the 64-char SHA256 Hash)
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None, 
    base_path: str = "vectorstore_data", # Assuming default value
    year: Optional[int] = None,
    version: str = "v1",
    metadata: dict = None,
    source_name_for_display: Optional[str] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> Tuple[List[Document], str, str]: 
    
    raw_doc_id_input = os.path.splitext(file_name)[0]
    filename_doc_id_key = _normalize_doc_id(raw_doc_id_input) # ⬅️ นี่คือ ID 34-char ที่เราต้องการ!
            
    doc_type = doc_type or "document"
    
    # 📌 Determine resolved enabler
    resolved_enabler = None
    if doc_type.lower() == "evidence":
        # Ensure DEFAULT_ENABLER is defined or handle its absence
        DEFAULT_ENABLER = "KM" # Mocking if not defined
        resolved_enabler = (enabler or DEFAULT_ENABLER).upper()

    # 📌 Inject enabler into metadata if it exists
    injected_metadata = metadata or {}
    if resolved_enabler:
        injected_metadata["enabler"] = resolved_enabler
        
    # --- DEBUG LOGS เพื่อยืนยัน ID ---
    filter_id_value = filename_doc_id_key # ค่า 34-char ที่จะถูกเก็บ
    logger.critical(f"================== START DEBUG INGESTION: {file_name} ==================")
    logger.critical(f"🔍 DEBUG ID (stable_doc_uuid, 64-char Hash): {len(stable_doc_uuid)}-char: {stable_doc_uuid[:34]}...")
    logger.critical(f"✅ FINAL ID TO STORE (34-char Ref ID): {len(filter_id_value)}-char: {filter_id_value[:34]}...")
    # ---------------------------------

    chunks = load_and_chunk_document(
        file_path=file_path,
        doc_id_key=filename_doc_id_key, 
        stable_doc_uuid=stable_doc_uuid,
        year=year,
        version=version,
        metadata=injected_metadata, 
        ocr_pages=ocr_pages
    )
    
    for c in chunks:
        # [2] เก็บ ID 64-char (ที่ใช้ใน Ingestion เดิม)
        c.metadata["doc_id"] = stable_doc_uuid          
        c.metadata["stable_doc_uuid"] = stable_doc_uuid 
        
        # ✅ การแก้ไขที่สำคัญ: เพิ่มคีย์สำหรับ ID 32-char (ใช้สำหรับกรองเดิม)
        c.metadata["original_stable_id"] = filename_doc_id_key[:32].lower()  
        
        c.metadata["doc_type"] = doc_type 
        if resolved_enabler:
             c.metadata["enabler"] = resolved_enabler

        # 🎯 FINAL FIX: ใช้ filename_doc_id_key (34-char) สำหรับ Hard Filter
        c.metadata["assessment_filter_id"] = filter_id_value # <--- ถูกต้องแล้ว
        
        c.metadata = _safe_filter_complex_metadata(c.metadata)
        logger.debug(f"Chunk metadata preview: {c.metadata}")
        
    return chunks, stable_doc_uuid, doc_type

# -------------------- Vectorstore / Mapping Utilities --------------------

_VECTORSTORE_SERVICE_CACHE = {} 

def get_vectorstore(collection_name: str = "default", base_path: str = VECTORSTORE_DIR) -> Chroma:
    """
    Loads or creates the Chroma vector store instance for a given collection.
    """
    cache_key = f"{collection_name}_{base_path}"
    if cache_key in _VECTORSTORE_SERVICE_CACHE:
        return _VECTORSTORE_SERVICE_CACHE[cache_key]
    if len(collection_name) < 3:
        raise ValueError(f"Collection name '{collection_name}' is too short. Chroma DB requires at least 3 characters.")
    
    # Load embeddings model once
    embeddings = _VECTORSTORE_SERVICE_CACHE.get("embeddings_model")
    if not embeddings:
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        _VECTORSTORE_SERVICE_CACHE["embeddings_model"] = embeddings
        
    persist_directory = os.path.join(base_path, collection_name)
    os.makedirs(persist_directory, exist_ok=True)
    try:
        vectorstore = Chroma(
            collection_name=collection_name, 
            persist_directory=persist_directory, 
            embedding_function=embeddings
        )
        _VECTORSTORE_SERVICE_CACHE[cache_key] = vectorstore
        logger.info(f"✅ Loaded Vector Store service for collection: {collection_name}")
        return vectorstore
    except Exception as e:
        logger.critical(f"❌ Failed to load Chroma for collection '{collection_name}': {e}")
        raise RuntimeError(f"Failed to initialize Chroma DB: {e}")
    

def create_vectorstore_from_documents(
    chunks: List[Document], 
    collection_name: str, 
    doc_mapping_db: Dict[str, Dict[str, Any]],
    base_path: str = VECTORSTORE_DIR
) -> Chroma:
    """
    Adds documents (chunks) to a vector store collection and manages the document ID mapping.
    """
    vectorstore = get_vectorstore(collection_name, base_path)
    if not chunks:
        logger.warning(f"No chunks provided for collection {collection_name}. Skipping indexing.")
        return vectorstore
        
    texts = [c.page_content for c in chunks]; metadatas = [c.metadata for c in chunks]
    
    try:
        ids = [str(uuid.uuid4()) for _ in texts]
        stable_doc_uuids_to_delete = set(); old_chunk_ids_to_delete = []
        
        # 1. Prepare to delete existing chunks for documents being updated
        for m in metadatas:
            s_uuid = m.get("stable_doc_uuid")
            if s_uuid and s_uuid in doc_mapping_db:
                existing_chunk_uuids = doc_mapping_db[s_uuid].get("chunk_uuids")
                if existing_chunk_uuids and isinstance(existing_chunk_uuids, list):
                    stable_doc_uuids_to_delete.add(s_uuid)
                    old_chunk_ids_to_delete.extend(existing_chunk_uuids)
                    doc_mapping_db[s_uuid]["chunk_uuids"] = [] # Clear mapping entry temporarily

        if old_chunk_ids_to_delete:
            vectorstore.delete(ids=old_chunk_ids_to_delete)
            logger.info(f"🧹 Deleted {len(old_chunk_ids_to_delete)} old chunks from Chroma.")
            
        # 2. Add new chunks
        vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        
        # 3. Update mapping with new chunk IDs
        for i, chunk_uuid in enumerate(ids):
            stable_doc_uuid = metadatas[i].get("stable_doc_uuid") 
            if stable_doc_uuid:
                if stable_doc_uuid not in doc_mapping_db:
                    # Should not happen if logic in ingest_all_files is correct, but safe check
                    doc_mapping_db[stable_doc_uuid] = {"chunk_uuids": []} 
                if chunk_uuid not in doc_mapping_db[stable_doc_uuid]["chunk_uuids"]:
                    doc_mapping_db[stable_doc_uuid]["chunk_uuids"].append(chunk_uuid)
                metadatas[i]["chunk_uuid"] = chunk_uuid 
                
        logger.info(f"✅ Indexed {len(ids)} chunks and updated mapping for {collection_name}. Persist finished.")
    except Exception as e:
        logger.error(f"❌ Error during Chroma indexing or persisting for {collection_name}: {e}")
    return vectorstore


def save_doc_id_mapping(mapping_data: Dict[str, Dict[str, Any]], path: str):
    """Saves the document ID mapping file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Successfully saved Doc ID Mapping to: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to save Doc ID Mapping: {e}")

def load_doc_id_mapping(path: str) -> Dict[str, Dict[str, Any]]:
    """Loads the document ID mapping file."""
    if not os.path.exists(path): return {} 
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load Doc ID Mapping from {path}: {e}")
        return {}
        
# -------------------- API Helper: Get UUIDs for RAG Filtering --------------------
def get_stable_uuids_by_doc_type(doc_types: List[str]) -> List[str]:
    """
    Retrieves Stable UUIDs for RAG filtering based on document types.
    """
    if not doc_types: return []
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    target_uuids = []
    doc_type_set = {dt.lower() for dt in doc_types}
    for s_uuid, entry in doc_mapping_db.items():
        doc_type = entry.get("doc_type", "default")
        if doc_type.lower() in doc_type_set:
            target_uuids.append(s_uuid)
    return target_uuids
        

def create_stable_uuid_from_path(filepath: str, ref_id_key: Optional[str] = None) -> str:
    """
    สร้าง UUID ที่มั่นคงจากพาธไฟล์เต็ม, ขนาดไฟล์, และ Ref ID Key เพื่อให้แน่ใจว่า
    แต่ละเอกสารมี ID ที่ไม่ซ้ำกันอย่างแท้จริง
    """
    try:
        full_path = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath)
    except FileNotFoundError:
        # Fallback ที่ปลอดภัยกว่า: ใช้ชื่อไฟล์และ Ref ID Key
        full_path = os.path.abspath(filepath)
        file_size = 0
    
    # 💡 FIX: รวม Ref ID Key ใน unique_string เพื่อแยกไฟล์ที่มีชื่อต่างกัน
    unique_string = f"{full_path}-{file_size}-{ref_id_key or 'NO_REF'}" 
    
    # ใช้ SHA-256 เพื่อสร้าง Hash ที่ยาวและมั่นคงกว่า SHA-1
    hash_object = hashlib.sha256(unique_string.encode('utf-8'))
    
    # แปลง Hash เป็น UUID-like string
    return hash_object.hexdigest()

# -------------------- [REVISED] ingest_all_files --------------------
def ingest_all_files(
    data_dir: str = DATA_DIR,
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR,
    exclude_dirs: Set[str] = set(),
    year: Optional[int] = None,
    version: str = "v1",
    sequential: bool = True,
    skip_ext: Optional[List[str]] = None,
    log_every: int = 50,
    batch_size: int = 500
) -> List[Dict[str, Any]]:
    """
    Ingest documents into vectorstore based on doc_type and enabler filters.
    Handles filenames with spaces, (), [], and unicode characters.
    """
    skip_ext = skip_ext or []
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(base_path, exist_ok=True)
    files_to_process = []
    
    doc_type_req = (doc_type or "all").lower()
    enabler_req = (enabler or (DEFAULT_ENABLER if doc_type_req == "evidence" else None))
    if enabler_req:
        enabler_req = enabler_req.upper()

    logger.info(f"Starting ingest_all_files: doc_type_req='{doc_type_req}', enabler_req='{enabler_req}'")

    # 📌 [FIXED: Flexible Source Dir] กำหนด Root Path สำหรับการสแกน
    scan_roots: List[str] = []
    
    if doc_type_req == "all":
        # ถ้าเป็น 'all' ให้สแกนทุกโฟลเดอร์ที่รองรับ
        for dt in SUPPORTED_DOC_TYPES:
             if dt == "evidence":
                for ena in SUPPORTED_ENABLERS:
                    scan_roots.append(_get_source_dir(dt, ena, data_dir))
             else:
                scan_roots.append(_get_source_dir(dt, None, data_dir))
    elif doc_type_req in SUPPORTED_DOC_TYPES:
        # ถ้าเป็น Specific Doc Type ให้คำนวณพาธเฉพาะ
        if doc_type_req == "evidence" and not enabler_req:
             # ถ้าเป็น evidence แต่ไม่ระบุ enabler ให้สแกนทุก enabler
             for ena in SUPPORTED_ENABLERS:
                 scan_roots.append(_get_source_dir(doc_type_req, ena, data_dir))
        else:
             scan_roots = [_get_source_dir(doc_type_req, enabler_req, data_dir)]
    else:
        logger.error(f"Invalid doc_type_req for ingestion: {doc_type_req}")
        return []

    # 1. รวบรวมไฟล์ทั้งหมด based on scanning calculated source dir(s)
    for root_to_scan in set(scan_roots):
        if not os.path.isdir(root_to_scan):
             logger.warning(f"⚠️ Source directory not found: {root_to_scan}. Skipping scan.")
             continue
             
        # Collection name คือชื่อโฟลเดอร์ที่สแกน (e.g., 'document' หรือ 'evidence_km')
        current_collection_name = os.path.basename(root_to_scan) 
        original_doc_type, resolved_enabler = _parse_collection_name(current_collection_name)
        
        logger.info(f"Scanning source directory: {root_to_scan} (Maps to: {current_collection_name})")

        for root, dirs, filenames in os.walk(root_to_scan):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            # สแกนแค่ชั้นแรกของแต่ละ source_dir
            if root != root_to_scan: continue 

            for f in filenames:
                # Skip hidden files
                if f.startswith('.'):
                    continue

                file_path = os.path.join(root, f)
                file_extension = os.path.splitext(f)[1].lower()

                # Skip unsupported types
                if file_extension not in SUPPORTED_TYPES:
                    logger.info(f"⚠️ Skipping unsupported file type {file_extension}: {f}")
                    continue

                # Skip extensions in skip_ext
                if skip_ext and file_extension in skip_ext:
                    logger.info(f"⚠️ Skipping excluded extension {file_extension}: {f}")
                    continue

                # ✅ Append file 
                files_to_process.append({
                    "file_path": file_path,
                    "file_name": f,
                    "doc_type": original_doc_type,
                    "enabler": resolved_enabler,
                    "collection_name": current_collection_name
                })

    if not files_to_process:
        logger.warning("⚠️ No files found to ingest!")
        return []

    # 2. Load Mapping DB และกำหนด Stable IDs (***REVISED LOGIC***)
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    
    # 📌 REVISED: ใช้ Stable UUID ที่สร้างจาก File Path เป็นคีย์ในการค้นหา
    uuid_from_path_lookup: Dict[str, str] = {
        entry["filepath"]: s_uuid 
        for s_uuid, entry in doc_mapping_db.items() 
        if "filepath" in entry # ใช้ filepath ในการค้นหาไฟล์เดิม
    } 

    # 📌 REVISED: กำหนด Stable UUID ทีละไฟล์โดยอิงจาก Path
    for file_info in files_to_process:
        # 0. คำนวณ filename_doc_id_key (ใช้เป็น REF ID)
        filename_doc_id_key = _normalize_doc_id(os.path.splitext(file_info["file_name"])[0])
        file_info["doc_id_key"] = filename_doc_id_key # เก็บ Ref ID ไว้ใน file_info

        # 🟢 A. สร้าง UUID ที่มั่นคงจาก File Path (ใหม่)
        # 💡 FIX: ส่ง filename_doc_id_key เข้าไปเป็น Ref ID Key
        stable_uuid_from_path = create_stable_uuid_from_path(
            file_info["file_path"], 
            ref_id_key=filename_doc_id_key # <--- ส่ง Ref ID Key เข้าไป
        ) 
        
        # 🟢 B. ค้นหา ID เดิมจาก Mapping โดยใช้ File Path เป็น Key
        stable_doc_uuid = uuid_from_path_lookup.get(file_info["file_path"])
        
        if stable_doc_uuid and stable_doc_uuid in doc_mapping_db:
            # ใช้ UUID เดิมที่พบ
            file_info["stable_doc_uuid"] = stable_doc_uuid
            # อัปเดตข้อมูลที่อาจเปลี่ยนไป
            doc_mapping_db[stable_doc_uuid].update({
                "filename": file_info["file_name"],
                "doc_type": file_info["doc_type"],
                "enabler": file_info["enabler"],
                "doc_id_key": filename_doc_id_key, # อัปเดต Ref ID Key
            })
        else:
            # 🟢 C. ถ้าไม่เจอ ID เดิม 
            new_uuid = stable_uuid_from_path # ใช้ Stable ID ใหม่
            file_info["stable_doc_uuid"] = new_uuid
            
            # 🟢 D. สร้าง Mapping entry ใหม่
            doc_mapping_db[new_uuid] = {
                "file_name": file_info["file_name"],
                "doc_type": file_info["doc_type"],
                "enabler": file_info["enabler"],
                "doc_id_key": filename_doc_id_key, # เก็บ Ref ID Key
                "filepath": file_info["file_path"], # บันทึก File Path
                "notes": "",
                "statement_id": "",
                "chunk_uuids": []
            }
            uuid_from_path_lookup[file_info["file_path"]] = new_uuid # อัปเดต Lookup

    # 3. Load & Chunk (sequential or threaded)
    all_chunks: List[Document] = []
    results: List[Dict[str, Any]] = []

    def _process_file_task(file_info: Dict[str, str]):
        return process_document(
            file_path=file_info["file_path"],
            file_name=file_info["file_name"],
            stable_doc_uuid=file_info["stable_doc_uuid"],
            doc_type=file_info["doc_type"],
            enabler=file_info["enabler"],
            base_path=base_path,
            year=year,
            version=version
        )

    if sequential:
        for idx, file_info in enumerate(files_to_process, 1):
            f = file_info["file_name"]
            stable_doc_uuid = file_info["stable_doc_uuid"]
            try:
                chunks, doc_id, dt = _process_file_task(file_info)
                all_chunks.extend(chunks)
                results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks)})
                if idx % log_every == 0:
                    logger.info(f"Processed {idx}/{len(files_to_process)} files...")
            except Exception as e:
                results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": file_info["doc_type"], "status": "failed_chunk", "error": str(e)})
                # 📌 [FIXED: Detailed Process/Chunk Failure Logging]
                logger.error(f"❌ CHUNK/PROCESS FAILED: {f} (ID: {stable_doc_uuid}) - {type(e).__name__} ({e})")
    else:
        max_workers = os.cpu_count() or 4
        logger.info(f"Using ThreadPoolExecutor with max_workers={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(_process_file_task, fi): fi for fi in files_to_process}
            for idx, future in enumerate(as_completed(future_to_file), 1):
                fi = future_to_file[future]
                f = fi["file_name"]
                stable_doc_uuid = fi["stable_doc_uuid"]
                try:
                    chunks, doc_id, dt = future.result()
                    all_chunks.extend(chunks)
                    results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks)})
                except Exception as e:
                    results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": fi["doc_type"], "status": "failed_chunk", "error": str(e)})
                    # 📌 [FIXED: Detailed Process/Chunk Failure Logging]
                    logger.error(f"❌ CHUNK/PROCESS FAILED: {f} (ID: {stable_doc_uuid}) - {type(e).__name__} ({e})")
                if idx % log_every == 0:
                    logger.info(f"Processed {idx}/{len(files_to_process)} files...")

    # 4. Group & Index Chunks
    chunks_by_collection: Dict[str, List[Document]] = {}
    for chunk in all_chunks:
        dt = chunk.metadata.get("doc_type", "default")
        ena = chunk.metadata.get("enabler")
        coll_name = get_target_dir(dt, ena)
        chunks_by_collection.setdefault(coll_name, []).append(chunk)

    for coll_name, coll_chunks in chunks_by_collection.items():
        if not coll_chunks:
            logger.warning(f"Skipping collection '{coll_name}' - 0 chunks found.")
            continue
        logger.info(f"--- Indexing collection '{coll_name}' ({len(coll_chunks)} chunks) ---")
        for i in range(0, len(coll_chunks), batch_size):
            batch = coll_chunks[i:i+batch_size]
            logger.info(f"Indexing chunks {i+1} to {i+len(batch)} of {len(coll_chunks)}")
            try:
                create_vectorstore_from_documents(
                    chunks=batch,
                    collection_name=coll_name,
                    doc_mapping_db=doc_mapping_db,
                    base_path=base_path
                )
            except Exception as e:
                logger.error(f"Error indexing chunks {i+1}-{i+len(batch)}: {e}")

    save_doc_id_mapping(doc_mapping_db, MAPPING_FILE_PATH)
    logger.info("✅ Batch ingestion process finished.")
    return results


# -------------------- [REVISED] wipe_vectorstore --------------------
def wipe_vectorstore(
    doc_type_to_wipe: str = 'all', 
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR
):
    """Wipes the vector store directory/collection(s) and updates the doc_id_mapping file."""
    
    doc_type_to_wipe = doc_type_to_wipe.lower()
    
    collections_to_delete: List[str] = []
    
    if doc_type_to_wipe == 'all':
        logger.warning(f"Wiping ALL collections in {base_path}")
        if os.path.exists(base_path):
            # Only list directories that are likely collections (avoid .gitkeep, etc.)
            collections_to_delete = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    elif doc_type_to_wipe == 'evidence':
        if enabler:
            # Wipe specific evidence enabler
            collection_name = get_target_dir("evidence", enabler)
            collections_to_delete.append(collection_name)
        else:
            # Wipe ALL evidence enablers
            logger.warning("Wiping ALL evidence_* collections.")
            evidence_paths = glob.glob(os.path.join(base_path, "evidence_*"))
            collections_to_delete = [os.path.basename(p) for p in evidence_paths]
    elif doc_type_to_wipe in SUPPORTED_DOC_TYPES:
        collection_name = get_target_dir(doc_type_to_wipe, None)
        collections_to_delete.append(collection_name)
    else:
        logger.error(f"Invalid doc_type_to_wipe: {doc_type_to_wipe}")
        return

    # 1. Delete collection folders
    for col_name in collections_to_delete:
        col_path = os.path.join(base_path, col_name)
        if os.path.exists(col_path):
            try:
                shutil.rmtree(col_path)
                logger.info(f"✅ Deleted vector store collection: {col_path}")
            except OSError as e:
                logger.error(f"❌ Error deleting collection {col_name}: {e}")

    # 2. Update Mapping file
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    uuids_to_keep = {}
    deletion_count = 0
    
    for s_uuid, entry in doc_mapping_db.items():
        entry_doc_type = entry.get("doc_type")
        entry_enabler = entry.get("enabler")
        entry_collection = get_target_dir(entry_doc_type, entry_enabler)
        
        if entry_collection not in collections_to_delete:
            uuids_to_keep[s_uuid] = entry
        else:
            deletion_count += 1
            
    if deletion_count > 0:
        save_doc_id_mapping(uuids_to_keep, MAPPING_FILE_PATH)
        logger.info(f"🧹 Removed {deletion_count} entries from mapping file for deleted collections.")
        
    if doc_type_to_wipe == 'all' and os.path.exists(MAPPING_FILE_PATH):
         # Also delete the mapping file itself if wiping 'all'
         try:
            os.remove(MAPPING_FILE_PATH)
            logger.info(f"✅ Deleted Doc ID mapping file: {MAPPING_FILE_PATH}")
         except OSError as e:
            logger.error(f"❌ Error deleting mapping file: {e}")


# -------------------- [REVISED] Document Management Utilities --------------------

def delete_document_by_uuid(
    stable_doc_uuid: str, 
    collection_name: Optional[str] = None, 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR
) -> bool:
    """
    Deletes all chunks associated with the given Stable Document UUID.
    """
    
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    
    if stable_doc_uuid not in doc_mapping_db:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} not found in mapping.")
        return False

    doc_entry = doc_mapping_db[stable_doc_uuid]
    all_chunk_uuids = doc_entry.get("chunk_uuids", [])
    
    # 📌 Determine collection name from mapping
    doc_type_from_map = doc_entry.get("doc_type")
    enabler_from_map = doc_entry.get("enabler")
    
    # Use provided doc_type/enabler as override, otherwise use mapping
    final_doc_type = doc_type or doc_type_from_map
    final_enabler = enabler or enabler_from_map
    
    if not final_doc_type:
         logger.error(f"Cannot delete {stable_doc_uuid}: doc_type not found in mapping or provided.")
         return False
         
    final_collection_name = get_target_dir(final_doc_type, final_enabler)
    
    if not all_chunk_uuids:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} has no chunk UUIDs recorded.")
        del doc_mapping_db[stable_doc_uuid]
        save_doc_id_mapping(doc_mapping_db, MAPPING_FILE_PATH)
        return True

    persist_directory = os.path.join(base_path, final_collection_name)
    if not os.path.isdir(persist_directory):
        logger.warning(f"Vectorstore directory not found for collection '{final_collection_name}'. Skipping Chroma deletion.")
    else:
        try:
            vectorstore = get_vectorstore(final_collection_name, base_path)
            vectorstore.delete(ids=all_chunk_uuids)
            logger.info(f"✅ Successfully deleted {len(all_chunk_uuids)} chunks from collection '{final_collection_name}' for UUID: {stable_doc_uuid}")
        except Exception as e:
            logger.error(f"❌ Error during Chroma deletion for collection '{final_collection_name}' (UUID: {stable_doc_uuid}): {e}")
            
    # Delete from mapping regardless of vectorstore success
    del doc_mapping_db[stable_doc_uuid]
    save_doc_id_mapping(doc_mapping_db, MAPPING_FILE_PATH)
        
    return True

def list_documents(
    doc_types: Optional[List[str]] = None,
    enabler: Optional[str] = None, 
    show_results: str = "ingested" 
# ) -> List[Any]: # <--- [OLD] คืนค่าเป็น List
) -> Dict[str, Any]: # <--- [NEW] คืนค่าเป็น Dict
    
    doc_mapping_db = load_doc_id_mapping(MAPPING_FILE_PATH)
    # all_docs: List[Any] = [] # <--- [OLD]
    all_docs: Dict[str, Any] = {} # <--- [NEW] ใช้ Dictionary
    
    # 🟢 [FIX: ใช้ Filepath ที่เป็น Unique ในการค้นหา Stable UUID แทน doc_id_key]
    filepath_to_stable_uuid: Dict[str, str] = {
        entry["filepath"]: s_uuid 
        for s_uuid, entry in doc_mapping_db.items() 
        if "filepath" in entry # ใช้ filepath ในการค้นหาไฟล์เดิม
    } 
            
    # Filtering setup
    doc_type_reqs = {dt.lower() for dt in doc_types} if doc_types and doc_types[0] and doc_types[0].lower() != "all" else set()
    enabler_req = (enabler or "").upper()
    
    # 📌 [FIXED: Flexible Source Dir] กำหนด Root Path สำหรับการสแกน
    source_dirs_to_scan: List[str] = []

    if not doc_type_reqs: # 'all'
        # ... (ส่วนนี้เหมือนเดิม) ...
        for dt in SUPPORTED_DOC_TYPES:
            if dt == "evidence":
                if enabler_req:
                     source_dirs_to_scan.append(_get_source_dir(dt, enabler_req))
                else:
                     for ena in SUPPORTED_ENABLERS:
                         source_dirs_to_scan.append(_get_source_dir(dt, ena))
            else:
                source_dirs_to_scan.append(_get_source_dir(dt, None))
    else:
        # ... (ส่วนนี้เหมือนเดิม) ...
        for dt in doc_type_reqs:
            if dt == "evidence":
                if enabler_req:
                    source_dirs_to_scan.append(_get_source_dir(dt, enabler_req))
                else:
                    for ena in SUPPORTED_ENABLERS:
                        source_dirs_to_scan.append(_get_source_dir(dt, ena))
            elif dt in SUPPORTED_DOC_TYPES:
                source_dirs_to_scan.append(_get_source_dir(dt, None))
                
    # 1. Scan files and build DocInfo list (all_docs)
    for root_to_scan in set(source_dirs_to_scan): # ใช้ set เพื่อป้องกันการสแกนซ้ำ
        if not os.path.isdir(root_to_scan):
            logger.debug(f"Source directory not found: {root_to_scan}. Skipping scan.")
            continue
            
        current_collection_name = os.path.basename(root_to_scan)
        original_doc_type, resolved_enabler = _parse_collection_name(current_collection_name)
        
        for root, _, filenames in os.walk(root_to_scan):
            if root != root_to_scan: continue # สแกนแค่ชั้นแรก
            
            for f in filenames:
                file_extension = os.path.splitext(f)[1].lower()
                if f.startswith('.') or file_extension not in SUPPORTED_TYPES:
                    continue
                    
                file_path = os.path.join(root, f)
                file_name_no_ext = os.path.splitext(f)[0]
                filename_doc_id_key = _normalize_doc_id(file_name_no_ext)
                
                # 🟢 [FIX: ใช้ filepath ในการ lookup Stable UUID]
                stable_doc_uuid = filepath_to_stable_uuid.get(file_path)
                
                doc_entry = doc_mapping_db.get(stable_doc_uuid) if stable_doc_uuid else None
                
                chunk_uuids = doc_entry.get("chunk_uuids", []) if doc_entry else []
                chunk_count = len(chunk_uuids)
                is_ingested = chunk_count > 0
                
                final_doc_id = stable_doc_uuid or f"TEMP_ID__{filename_doc_id_key}"
                
                try:
                    upload_date = datetime.fromtimestamp(os.path.getmtime(file_path), timezone.utc).isoformat()
                    file_size = os.path.getsize(file_path)
                except FileNotFoundError:
                    upload_date = datetime.now(timezone.utc).isoformat()
                    file_size = 0

                doc_info: Any = {
                    "doc_id": final_doc_id,
                    "doc_id_key": filename_doc_id_key,
                    "filename": f,
                    "filepath": file_path,
                    "doc_type": original_doc_type,
                    "enabler": resolved_enabler, 
                    "upload_date": upload_date,
                    "chunk_count": chunk_count,
                    "status": "Ingested" if is_ingested else "Failed", 
                    "size": file_size,
                }
                # all_docs.append(doc_info) # <--- [OLD]
                all_docs[final_doc_id] = doc_info # <--- [NEW] เพิ่มลงใน Dictionary ด้วย doc_id เป็น Key

    # NEW DEBUG LOG: ตรวจสอบจำนวนไฟล์ทั้งหมดที่พบหลังจาก Initial Filter
    logger.info(f"DEBUG: Total physical files found (len(all_docs)): {len(all_docs)}")

    # START OF COUNTING LOGIC 
    # Y: Total number of supported files found after doc_type/enabler filter
    total_supported_files = len(all_docs) 
    
    # 2. Apply show_results filtering and calculate X
    show_results_lower = show_results.lower()
    
    # 📌 [NEW] ใช้ Dictionary เพื่อเก็บผลลัพธ์ที่ถูกกรอง
    filtered_docs_dict: Dict[str, Any] = {}
    
    if total_supported_files == 0:
        doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
        logger.warning(f"⚠️ No documents found in DATA_DIR matching the requested type '{doc_types_str}' (Enabler: {enabler_req or 'ALL'}).")
        return filtered_docs_dict # คืนค่า Dictionary ว่าง
    
    # 📌 [NEW] ปรับ Logic การกรองให้ทำงานกับ Dictionary
    if show_results_lower == "ingested":
        filtered_docs_dict = {
            k: d for k, d in all_docs.items() 
            if d.get('status', '').lower() == 'ingested' and not k.startswith('TEMP_ID__')
        }
        filter_name = "INGESTED (Successful / Unique Doc IDs)"
        display_count_x = len(filtered_docs_dict)
        
    elif show_results_lower == "failed":
        filtered_docs_dict = {
            k: d for k, d in all_docs.items() 
            if d.get('status', '').lower() == 'failed'
        }
        display_count_x = len(filtered_docs_dict) 
        filter_name = "FAILED/SKIPPED (Physical Files)"
        
    elif show_results_lower == "full":
        filtered_docs_dict = all_docs
        display_count_x = len(all_docs) 
        filter_name = "FULL (All Supported Files)"
        
    else:
        # Default: ingested
        filtered_docs_dict = {
            k: d for k, d in all_docs.items() 
            if d.get('status', '').lower() == 'ingested' and not k.startswith('TEMP_ID__')
        }
        display_count_x = len(filtered_docs_dict)
        filter_name = "INGESTED (Default / Unique Doc IDs)"


    # 3. Format for console output and printing
    
    display_list = []
    # 📌 [NEW] ดึง values() จาก Dictionary เพื่อใช้ในการแสดงผล
    for doc_info in filtered_docs_dict.values(): 
        file_size_mb = doc_info['size'] / (1024 * 1024)
        enabler_display = doc_info['enabler'] if doc_info['enabler'] is not None else '-'
        
        display_list.append({
            "doc_id": doc_info["doc_id"],
            "file_name": doc_info["filename"],
            "doc_type": doc_info["doc_type"],
            "enabler": enabler_display,
            "size_mb": file_size_mb,
            "status": doc_info["status"],
            "chunk_count": doc_info["chunk_count"],
            "ref_doc_id": doc_info["doc_id_key"]
        })
        
    display_list.sort(key=lambda x: (x["doc_type"], x["file_name"]))
    
    # ... (ส่วนการ Print Output เหมือนเดิม) ...
    doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
    
    print(f"\nFound {display_count_x}/{total_supported_files} supported documents for type '{doc_types_str}' (Filter: {filter_name}):\n")

    if not display_list:
        print("--- No documents found matching the filter criteria to display ---")
        return filtered_docs_dict # คืนค่า Dictionary ว่าง

    print("-" * 155)
    print(f"{'DOC ID (Stable/Temp)':<38} | {'FILENAME':<35} | {'EXT':<5} | {'TYPE':<10} | {'ENB':<5} | {'SIZE(MB)':<9} | {'STATUS':<10} | {'CHUNKS':<8} | {'REF ID (Old Key)'}")
    print("-" * 155)
    
    for info in display_list:
        short_doc_id = info['doc_id'][:38] 
        file_name, file_ext = os.path.splitext(info['file_name'])
        short_filename = file_name[:33] if len(file_name) > 33 else file_name 
        file_ext = file_ext[1:].upper() if file_ext else '-' 
        size_str = f"{info['size_mb']:.2f}"
        short_ref_doc_id = info['ref_doc_id'][:20] if len(info['ref_doc_id']) > 20 else info['ref_doc_id']
        enabler_display = info['enabler'] 
        
        print(
            f"{short_doc_id:<38} | "
            f"{short_filename:<35} | " 
            f"{file_ext:<5} | "
            f"{info['doc_type']:<10} | "
            f"{enabler_display:<5} | " 
            f"{size_str:<9} | "
            f"{info['status']:<10} | "
            f"{info['chunk_count']:<8} | "
            f"{short_ref_doc_id}"
        )
    print("-" * 155)
    print(f"\nFound {display_count_x}/{total_supported_files} supported documents for type '{doc_types_str}' (Filter: {filter_name}):\n")

    
    # 4. Return the filtered list
    # return filtered_docs # <--- [OLD]
    return filtered_docs_dict # <--- [NEW] คืนค่า Dictionary ที่ถูกกรอง