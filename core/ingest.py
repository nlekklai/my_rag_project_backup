# core/ingest.py
# เวอร์ชันเต็ม: Multi-Tenant + Multi-Year (รัฐวิสาหกิจไทย Ready)
# รวมการแก้ไข: Path Isolation, get_vectorstore, ingest_all_files, list_documents, wipe_vectorstore

import os
import re
import sys
import logging
import unicodedata
import json
import uuid
import glob
import hashlib
import shutil
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Set, Iterable, Dict, Any, Union, Tuple, TypedDict
import pandas as pd
import numpy as np
from pydantic import ValidationError


# LangChain loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader
)

import fitz  # PyMuPDF

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter 

# 💡 แก้ไข: ใช้ langchain_chroma และ langchain_huggingface แทน
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain_community.vectorstores.utils import filter_complex_metadata as _imported_filter_complex_metadata
except ImportError:
    _imported_filter_complex_metadata = None


# -------------------- Global Config --------------------
# 📌 ASSUME: config.global_vars มีการกำหนดค่าที่ถูกต้องตามที่ใช้
from config.global_vars import (
    DATA_DIR,
    VECTORSTORE_DIR,
    MAPPING_BASE_DIR, 
    SUPPORTED_TYPES,
    SUPPORTED_DOC_TYPES,
    DEFAULT_ENABLER,
    SUPPORTED_ENABLERS,
    EVIDENCE_DOC_TYPES,
    DEFAULT_DOC_TYPES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TENANT, 
    DEFAULT_YEAR,
    EVIDENCE_MAPPING_FILENAME_SUFFIX,
    EMBEDDING_MODEL_NAME
)

# -------------------- [NEW] Import Path Utilities --------------------
# 🎯 FIX: นำเข้าฟังก์ชัน Path จาก utils/path_utils.py เพื่อแทนที่ฟังก์ชันภายใน
from utils.path_utils import (
    get_document_source_dir,
    get_doc_type_collection_key,
    get_vectorstore_collection_path,
    get_mapping_file_path,
    get_vectorstore_tenant_root_path, # ใช้สำหรับ wipe
    get_evidence_mapping_file_path, # ใช้สำหรับ Evidence Map
    load_doc_id_mapping,
    save_doc_id_mapping
)
# ---------------------------------------------------------------------

# Logging
logging.basicConfig(
    filename="ingest.log",
    level=logging.DEBUG, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

try:
    import pytesseract
    # 📌 Comment out or adjust path based on target OS
    # pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract' 
    logger.info("✅ Pytesseract module loaded.")
except ImportError:
    logger.warning("Pytesseract not installed. Tesseract OCR may fail.")
except Exception as e:
    logger.error(f"Failed to set pytesseract path: {e}")

# --- Document Info Model ---
class DocInfo(TypedDict):
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

# -------------------- Log Noise Suppression (NEW) --------------------
import warnings

warnings.filterwarnings(
    "ignore", 
    "Cannot set gray non-stroke color because", 
    category=UserWarning,
    module='pdfminer' 
)
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfminer.pdfinterp').setLevel(logging.ERROR)
logging.getLogger('unstructured').setLevel(logging.ERROR)
logging.getLogger('pypdf').setLevel(logging.ERROR)

# -------------------- [REMOVED/REPLACED] Path Builders --------------------
# 📌 ถูกลบออกเนื่องจากถูกแทนที่ด้วยฟังก์ชันจาก utils/path_utils.py:
# build_tenant_base_path
# get_collection_parent_dir
# get_target_dir (ถูกแทนที่ด้วย get_doc_type_collection_key)
# _get_source_dir (ถูกแทนที่ด้วย get_document_source_dir)
# --------------------------------------------------------------------------

def _parse_collection_name(
    collection_name: str, 
) -> Tuple[str, Optional[str]]:
    """
    Parses a collection name back into doc_type and enabler, handling both 
    Multi-Tenant/Year structure (Fallback) and the simple structure.
    
    Collection IDs ที่คาดหวัง: 'evidence_km', 'document'
    """
    collection_name_lower = collection_name.lower()
    
    # 📌 NEW: จัดการ Prefix 'rag_' ก่อน (เพื่อให้เข้ากันได้กับ VSM)
    if collection_name_lower.startswith("rag_"):
         collection_name_lower = collection_name_lower[4:]

    # 1. ลองหาในรูปแบบ DocType_Enabler (เช่น evidence_km)
    if collection_name_lower.startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"):
        # Split แค่ครั้งเดียว: evidence_km -> ['evidence', 'km']
        parts_old = collection_name_lower.split("_", 1) 
        if len(parts_old) == 2:
            doc_type = parts_old[0]
            enabler_candidate = parts_old[1].upper()
            
            if enabler_candidate in SUPPORTED_ENABLERS: # 🎯 FIX: เช็ค Enabler กับ Global List
                 return doc_type, enabler_candidate
        
    # 2. ลองหาในรูปแบบ DocType (เช่น document)
    if collection_name_lower in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
        return collection_name_lower, None
        
    # 3. Fallback to the original name if no match is found (ถือว่าเป็น Doc Type ฐาน)
    return collection_name_lower, None


# -------------------- Helper: safe metadata filter --------------------
def _safe_filter_complex_metadata(meta: Any) -> Dict[str, Any]:
    """Ensure metadata is serializable and safe for Chroma / storage. (No Change)"""
    if not isinstance(meta, dict):
        if hasattr(meta, "items"):
            meta_dict = dict(meta.items())
        else:
            return {} 
    else:
        meta_dict = meta
        
    clean = {}

    for k, v in meta_dict.items():
        if v is None:
            continue
            
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        elif isinstance(v, dict):
            try:
                clean[k] = json.dumps(v)
            except TypeError:
                clean[k] = str(v)
                
        elif isinstance(v, (list, tuple)):
            if len(v) == 1:
                item = v[0]
                if isinstance(item, (str, int, float, bool)):
                    clean[k] = item 
                    continue 
                elif isinstance(item, (np.floating, np.integer)):
                    clean[k] = item.item() 
                    continue 
            
            try:
                clean[k] = json.dumps([str(x) for x in v]) 
            except Exception:
                clean[k] = str(v)
                
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = v.item()
        else:
            try:
                clean[k] = str(v)
            except Exception:
                continue
                
    if _imported_filter_complex_metadata:
        try:
            return _imported_filter_complex_metadata(clean)
        except Exception as e:
            logger.debug(f"LangChain filter failed after local cleanup: {e}")
            pass

    return clean

# -------------------- Normalization utility --------------------
def _normalize_doc_id(raw_id: str, file_content: bytes = None) -> str:
    """Generates the 34-character reference ID Key. (No Change)"""
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
    """Basic text cleaning utility. (No Change in Logic)"""
    if not text: return ""
    text = text.replace('\xa0', ' ').replace('\u200b', '').replace('\u00ad', '')
    text = re.sub(r'[\uFFFD\u2000-\u200F\u2028-\u202F\u2060-\u206F\uFEFF]', '', text)
    text = re.sub(r'([ก-๙])\s{1,3}(?=[ก-๙])', r'\1', text) 
    ocr_replacements = {"สำนักงน": "สำนักงาน", "คณะกรรมกร": "คณะกรรมการ"}
    for bad, good in ocr_replacements.items(): text = text.replace(bad, good)
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u0E00-\u0E7F]', '', text) 
    text = re.sub(r'\(\s+', '(', text); text = re.sub(r'\s+\)', ')', text)
    text = re.sub(r'\r\n', '\n', text); text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text); text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def _is_pdf_image_only(file_path: str) -> bool:
    """ตรวจสอบว่า PDF เป็น image-only หรือมี text layer (No Change)"""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text().strip()
            if text:
                return False  
        return True  
    except Exception as e:
        logger.warning(f"Cannot check PDF text layer for {file_path}: {e}")
        return True  


def _load_document_with_loader(file_path: str, loader_class: Any) -> List[Document]:
    """Helper function to load a document using a specific LangChain loader class. (Modified for Image Fallback)"""
    raw_docs: List[Any] = [] 
    ext = "." + file_path.lower().split('.')[-1]
    
    # --- 1. Handle Known Loaders (CSV) ---
    if loader_class is CSVLoader:
        try:
            loader = loader_class(file_path, encoding='utf-8')
            raw_docs = loader.load()
        except Exception as e:
            logger.error(f"❌ LOADER FAILED: CSVLoader for {os.path.basename(file_path)} raised: {type(e).__name__} ({e})")
            return []
    
    # --- 2. Handle PDF (Text/Image-Only) ---
    elif ext == ".pdf":
        try:
            if _is_pdf_image_only(file_path):
                logger.info(f"PDF is image-only, using OCR loader: {file_path}")
                # 📌 FIX: เปลี่ยน mode="elements" เป็น mode="single" สำหรับ Image-Only PDF
                loader = UnstructuredFileLoader(file_path, mode="single", languages=['tha','eng'])
            else:
                logger.info(f"PDF has text layer, using PyPDFLoader: {file_path}")
                # PyPDFLoader มักจะให้ผลลัพธ์ที่ดีกว่าสำหรับ PDF ที่มี text layer 
                loader = PyPDFLoader(file_path) 
            raw_docs = loader.load()
        except Exception as e:
             logger.error(f"❌ LOADER FAILED: PDF Loader for {os.path.basename(file_path)} raised: {type(e).__name__} ({e})")
             return []
    
    # --- 3. Handle Images (JPG, PNG) with Fallback ---
    elif ext in [".jpg", ".jpeg", ".png"]:
        
        # 3.1 Primary Attempt: UnstructuredFileLoader (Robust OCR, but can fail with TypeError)
        try:
            logger.info(f"Reading image file using UnstructuredFileLoader (Primary OCR): {file_path} ...")
            
            # 📌 FIX 1: ใช้ mode="elements" (ดีที่สุด) และ languages
            loader = UnstructuredFileLoader(file_path, mode="elements", languages=['tha','eng']) 
            raw_docs = loader.load()
            
            # ตรวจสอบว่าได้เอกสารที่มีเนื้อหาจริงหรือไม่
            if any(doc.page_content and doc.page_content.strip() for doc in raw_docs):
                return raw_docs
            
            # หากไม่มีเนื้อหา (OCR failed silently), ลอง Fallback
            raise RuntimeError("Unstructured OCR failed to extract text content.") 
            
        except Exception as primary_e:
            
            # 📌 FIX 3: ใช้ UnstructuredFileLoader (mode="single") เป็น Fallback OCR
            try:
                logger.warning(
                    f"⚠️ Primary image loader failed with {type(primary_e).__name__}. "
                    f"Falling back to simpler UnstructuredFileLoader (mode='single') for {os.path.basename(file_path)}."
                )
                # ใช้ mode="single" ซึ่งง่ายกว่า mode="elements" 
                loader = UnstructuredFileLoader(file_path, mode="single", languages=['tha','eng']) 
                raw_docs = loader.load()
                
                if raw_docs and raw_docs[0].page_content.strip():
                     logger.info("✅ Fallback Unstructured OCR (single mode) successful.")
                     return raw_docs

            except Exception as fallback_e:
                logger.error(
                    f"❌ FALLBACK FAILED: Simpler Unstructured OCR also failed for {os.path.basename(file_path)} "
                    f"with {type(fallback_e).__name__}."
                )
                return [] # Image file fully failed to load
            
            # ถ้า Fallback แล้วยังไม่มีเนื้อหา ก็คืนค่าว่าง
            return []

    # --- 4. Handle Other File Types ---
    else:
        try:
            loader = loader_class(file_path)
            raw_docs = loader.load()
        except Exception as e:
            loader_name = getattr(loader_class, '__name__', 'UnknownLoader')
            logger.error(f"❌ LOADER FAILED: {os.path.basename(file_path)} - {loader_name} raised: {type(e).__name__} ({e})")
            return []
        
    
    # --- 5. Post-Processing & Filtering ---
    if raw_docs:
        original_count = len(raw_docs)
        
        filtered_docs = [
            doc for doc in raw_docs 
            if isinstance(doc, Document) and doc.page_content is not None and doc.page_content.strip()
        ]
        
        if len(filtered_docs) < original_count:
            logger.warning(
                f"⚠️ Loader returned {original_count - len(filtered_docs)} empty/None documents "
                f"for {os.path.basename(file_path)}. Filtered to {len(filtered_docs)} valid documents."
            )
        
        if not filtered_docs and original_count > 0:
             logger.warning(f"⚠️ Loader returned documents but all were empty/invalid for {os.path.basename(file_path)}. Returning 0 valid documents.")
             return []
             
        return filtered_docs
    
    return [] # Return empty list if raw_docs is empty (e.g., file was empty or loading failed silently)

# -------------------- Loaders --------------------
FILE_LOADER_MAP = {
    ".pdf": lambda p: _load_document_with_loader(p, UnstructuredFileLoader), 
    ".docx": lambda p: _load_document_with_loader(p, UnstructuredWordDocumentLoader),
    ".txt": lambda p: _load_document_with_loader(p, TextLoader),
    ".xlsx": lambda p: _load_document_with_loader(p, UnstructuredExcelLoader),
    ".pptx": lambda p: _load_document_with_loader(p, UnstructuredPowerPointLoader),
    ".md": lambda p: _load_document_with_loader(p, TextLoader), 
    ".csv": lambda p: _load_document_with_loader(p, CSVLoader),
    ".jpg": lambda p: _load_document_with_loader(p, UnstructuredFileLoader), 
    ".jpeg": lambda p: _load_document_with_loader(p, UnstructuredFileLoader),
    ".png": lambda p: _load_document_with_loader(p, UnstructuredFileLoader),
}

# -------------------- Normalization utility --------------------
def normalize_loaded_documents(raw_docs: List[Any], source_path: Optional[str] = None) -> List[Document]:
    """Converts raw loaded documents into clean LangChain Document objects. (No Change)"""
    normalized: List[Document] = []
    for idx, item in enumerate(raw_docs):
        try:
            if isinstance(item, Document): doc = item
            else: doc = Document(page_content=str(item), metadata={})
            
            doc.page_content = unicodedata.normalize("NFKC", doc.page_content or "").strip() 
            
            if not doc.page_content:
                logger.warning(f"⚠️ Doc #{idx} from loader has no content (Empty/None). Skipping normalization for this document.")
                continue 
            
            if not isinstance(doc.metadata, dict): doc.metadata = {"_raw_meta": str(doc.metadata)}
            if source_path: doc.metadata.setdefault("source_file", os.path.basename(source_path))
            try: doc.metadata = _safe_filter_complex_metadata(doc.metadata)
            except Exception: doc.metadata = {"source_file": os.path.basename(source_path)} if source_path else {}
            normalized.append(doc)
            
        except Exception as e:
            logger.warning(f"normalize_loaded_documents: skipping item #{idx} due to error: {e}")
            continue
    return normalized

# 📌 Global Text Splitter Configuration (No Change)
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,              
    chunk_overlap=CHUNK_OVERLAP,            
    separators=[
        "\n\n",                   
        "\n- ",                   
        "\n• ",                   
        " ",                      
        ""
    ]   ,
    length_function=len,
    is_separator_regex=False
)

# -------------------- Load & Chunk Document --------------------
# ------------------------------------------------------------------
# SE-AM Sub-topic Mapping (จากหน้า 3-15 ของ SE-AM Manual Book 2566)
# ------------------------------------------------------------------
SEAM_SUBTOPIC_MAP = {
    # CG
    "1.1": "CG-1.1", "1-1": "CG-1.1",
    # SP
    "2.1": "SP-2.1", "2-1": "SP-2.1",
    # RM&IC
    "3.1": "RMIC-3.1", "3-1": "RMIC-3.1",
    # SCM
    "4.1": "SCM-4.1", "4-1": "SCM-4.1",
    # DT
    "5.1": "DT-5.1", "5-1": "DT-5.1",
    # HCM
    "6.1": "HCM-6.1", "6-1": "HCM-6.1", "6.2": "HCM-6.2", "6.3": "HCM-6.3", "6.4": "HCM-6.4",
    "6.5": "HCM-6.5", "6.6": "HCM-6.6", "6.7": "HCM-6.7",
    # KM & IM
    "7.1": "KM-7.1", "7-1": "KM-7.1",
    "7.20": "IM-7.20", "7-20": "IM-7.20",
    # IA
    "8.1": "IA-8.1", "8-1": "IA-8.1",
}

# Keywords ที่บ่งบอกว่าเป็นเกณฑ์ระดับต่าง ๆ
LEVEL_KEYWORDS = ["ระดับ 1", "ระดับ 2", "ระดับ 3", "ระดับ 4", "ระดับ 5"]

def _detect_sub_topic_and_page(text: str) -> Dict[str, Any]:
    """
    ตรวจจับ sub_topic และ page number จากข้อความของ chunk
    """
    result = {"sub_topic": None, "page_number": None}

    # 1. จับ page number (เช่น "หน้า 1-1", "หน้า 243")
    page_match = re.search(r'หน้า\s*(\d+(?:-\d+)?)', text)
    if page_match:
        result["page_number"] = page_match.group(1)

    # 2. จับ sub_topic เช่น "4.1", "7-20", "KM topic 4.1"
    for pattern, code in [
        (r'(?:KM|topic)?\s*(\d+\.\d+)', None),
        (r'(\d+-\d+)', None),
        (r'(\d+\.\d+)', None),
    ]:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            key = match.group(1).replace("-", ".")
            if key in SEAM_SUBTOPIC_MAP:
                result["sub_topic"] = SEAM_SUBTOPIC_MAP[key]
                break

    # 3. ถ้ายังไม่เจอ ให้ลองจับจากหัวข้อเต็ม (เช่น "4.1 กระบวนการจัดการความรู้ที่เป็นระบบ")
    if not result["sub_topic"]:
        for key, code in SEAM_SUBTOPIC_MAP.items():
            if key.replace(".", "-") in text or key in text:
                result["sub_topic"] = code
                break

    return result


# ------------------------------------------------------------------
# load_and_chunk_document – เวอร์ชันสมบูรณ์สุดสำหรับ SE-AM
# ------------------------------------------------------------------
def load_and_chunk_document(
    file_path: str,
    stable_doc_uuid: str,
    doc_type: str,
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    year: Optional[int] = None,
    version: str = "v1",
    metadata: Optional[Dict[str, Any]] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> List[Document]:
    """
    Load + Clean + Chunk + ใส่ sub_topic + page_number อัตโนมัติ
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_func = FILE_LOADER_MAP.get(file_extension)
    
    if not loader_func:
        logger.error(f"No loader found for {file_extension}")
        return []

    # --- Load Document ---
    try:
        raw_docs = loader_func(file_path)
    except ValidationError as e:
        if 'Unstructured' in str(loader_func):
            logger.warning(f"OCR crash handled: {os.path.basename(file_path)}")
            raw_docs = []
        else:
            raise e
    except Exception as e:
        logger.error(f"Load failed: {file_path} | {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning(f"No content loaded from {os.path.basename(file_path)}")
        return []

    # --- Normalize to Document objects ---
    docs = []
    for doc in raw_docs:
        if isinstance(doc, Document):
            docs.append(doc)
        else:
            logger.warning(f"Non-Document object skipped: {type(doc)}")

    # --- Inject Base Metadata ---
    base_metadata = {
        "doc_type": doc_type,
        "stable_doc_uuid": stable_doc_uuid,
        "source_filename": os.path.basename(file_path),
        "source": os.path.basename(file_path),
        "version": version,
    }
    if enabler: base_metadata["enabler"] = enabler
    if subject: base_metadata["subject"] = subject.strip()
    if year: base_metadata["year"] = year
    if metadata: base_metadata.update(metadata)

    for d in docs:
        d.metadata.update(base_metadata)
        d.metadata = _safe_filter_complex_metadata(d.metadata)

    # --- Split into chunks ---
    try:
        chunks = TEXT_SPLITTER.split_documents(docs)
    except Exception as e:
        logger.error(f"Split failed: {e}")
        chunks = docs

    # --- Clean text & Inject per-chunk metadata ---
    final_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, Document):
            continue

        # Clean text
        chunk.page_content = clean_text(chunk.page_content)

        # Detect sub_topic & page_number
        detected = _detect_sub_topic_and_page(chunk.page_content)
        if detected["sub_topic"]:
            chunk.metadata["sub_topic"] = detected["sub_topic"]
        if detected["page_number"]:
            chunk.metadata["page_number"] = detected["page_number"]

        # Unique chunk ID
        chunk.metadata["chunk_uuid"] = f"{stable_doc_uuid}_{idx}"
        chunk.metadata["chunk_index"] = idx
        chunk.metadata = _safe_filter_complex_metadata(chunk.metadata)

        final_chunks.append(chunk)

    logger.info(f"Loaded {os.path.basename(file_path)} → {len(final_chunks)} chunks | "
                f"sub_topic detected: {len([c for c in final_chunks if c.metadata.get('sub_topic')])}")
    return final_chunks

# -------------------- [REVISED] Process single document (Cleaned & Final) --------------------
def process_document(
    file_path: str,
    file_name: str,
    stable_doc_uuid: str, 
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None, 
    subject: Optional[str] = None,  
    base_path: str = VECTORSTORE_DIR, 
    year: Optional[int] = None,
    tenant: Optional[str] = None, 
    version: str = "v1",
    metadata: dict = None,
    source_name_for_display: Optional[str] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> Tuple[List[Document], str, str]: 
    
    raw_doc_id_input = os.path.splitext(file_name)[0]
    filename_doc_id_key = _normalize_doc_id(raw_doc_id_input) 
            
    doc_type = doc_type or DEFAULT_DOC_TYPES
    
    resolved_enabler = None
    if doc_type.lower() == EVIDENCE_DOC_TYPES.lower():
        resolved_enabler = (enabler or DEFAULT_ENABLER).upper()

    # 🟢 FIX: รวบรวม Metadata ที่จำเป็นทั้งหมดไว้ใน injected_metadata ณ จุดนี้
    injected_metadata = metadata or {}
    
    # 1. ข้อมูลที่ถูก Resolve
    injected_metadata["doc_type"] = doc_type
    injected_metadata["original_stable_id"] = filename_doc_id_key[:32].lower() # ใช้เป็น Reference ID
    
    if resolved_enabler:
        injected_metadata["enabler"] = resolved_enabler
    if tenant: 
        injected_metadata["tenant"] = tenant
    if subject: 
        injected_metadata["subject"] = subject
        
    filter_id_value = filename_doc_id_key 
    logger.critical(f"================== START DEBUG INGESTION: {file_name} ==================")
    logger.critical(f"🔍 DEBUG ID (stable_doc_uuid, 64-char Hash): {len(stable_doc_uuid)}-char: {stable_doc_uuid[:34]}...")
    logger.critical(f"✅ FINAL ID TO STORE (34-char Ref ID): {len(filter_id_value)}-char: {filter_id_value[:34]}...")

    # 🎯 FIX: ส่ง Metadata ทั้งหมดผ่าน dict ไปให้ load_and_chunk_document
    chunks = load_and_chunk_document(
        file_path=file_path,
        stable_doc_uuid=stable_doc_uuid,
        doc_type=doc_type, 
        enabler=resolved_enabler, 
        subject=subject, 
        year=year,
        version=version,
        metadata=injected_metadata, 
        ocr_pages=ocr_pages
    )
    
    if chunks:
         logger.debug(f"Chunk metadata preview: {chunks[0].metadata}")
        
    return chunks, stable_doc_uuid, doc_type


# -------------------- Vectorstore / Mapping Utilities --------------------

_VECTORSTORE_SERVICE_CACHE: dict = {}

def get_vectorstore(
    collection_name: str = "default",
    tenant: str = "pea",
    year: int = 2568,
    base_path: str = VECTORSTORE_DIR
) -> Chroma:
    """
    เวอร์ชัน Multi-Tenant/Multi-Year ที่ใช้ Path Utility ในการสร้าง Path
    """

    # === 1. ใช้ชื่อตรง ๆ ไม่ต้องเติม prefix อัตโนมัติ ===
    if len(collection_name) < 3:
        logger.warning(
            f"Collection name '{collection_name}' ค่อนข้างสั้น แนะนำให้ใช้ชื่ออย่างน้อย 6 ตัวอักษร "
            f"(เช่น evidence_km, km42l103) เพื่อป้องกันการชนกันของชื่อ"
        )

    # === 2. สร้าง path ที่ถูกต้องตามโครงสร้าง PEA โดยใช้ Path Utility ===
    try:
        doc_type_for_path, enabler_for_path = _parse_collection_name(collection_name)
        
        # 🎯 FIX: ใช้ get_vectorstore_collection_path จาก path_utils.py
        persist_directory = get_vectorstore_collection_path(
            tenant=tenant,
            year=year, # ส่งปีไป, path_utils จะตัดสินใจใช้หรือไม่ใช้เอง
            doc_type=doc_type_for_path,
            enabler=enabler_for_path
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate vectorstore path using path_utils: {e}. Using default fallback path.")
        # Fallback path (หาก path_utils มีปัญหา)
        persist_directory = os.path.join(base_path, tenant, str(year), "km", collection_name) 

    cache_key = persist_directory

    # === 3. Cache HIT ===
    if cache_key in _VECTORSTORE_SERVICE_CACHE:
        logger.debug(f"Cache HIT → Reusing vectorstore: {persist_directory}")
        return _VECTORSTORE_SERVICE_CACHE[cache_key]

    # === 4. Embedding model (แชร์ตัวเดียวตลอด process) ===
    embeddings = _VECTORSTORE_SERVICE_CACHE.get("embeddings_model")

    if not embeddings:
        logger.info(f"กำลังโหลด {EMBEDDING_MODEL_NAME} (SOTA Multilingual 2024) เพื่อปรับปรุง Retrieval")

        # 🟢 FIX: ลบ E5PrefixWrapper ออก และใช้ BGE-M3 โดยตรง
        try:
            embeddings = HuggingFaceEmbeddings(
                model_name= EMBEDDING_MODEL_NAME,
                model_kwargs={
                    "device": "cpu", # เปลี่ยนเป็น "cuda" ถ้ามี GPU 
                },  
                encode_kwargs={
                    "normalize_embeddings": True, 
                    "batch_size": 32,
                    # BGE-M3 ไม่ต้องการ 'prompt': 'query:' 
                }
            )
            _VECTORSTORE_SERVICE_CACHE["embeddings_model"] = embeddings
            logger.info(f"{EMBEDDING_MODEL_NAME} โหลดสำเร็จและแชร์ตลอด process")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {EMBEDDING_MODEL_NAME}: {e}")
            logger.warning("⚠️ Falling back to paraphrase-multilingual-MiniLM-L12-v2")
            # ใช้ Fallback model ตัวเดิม
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={"device": "cpu"}
            )
            _VECTORSTORE_SERVICE_CACHE["embeddings_model"] = embeddings


    # === 5. สร้างหรือโหลด Chroma ===
    os.makedirs(persist_directory, exist_ok=True)

    vectorstore = Chroma(
        collection_name=collection_name,           # ใช้ชื่อเดิมตรง ๆ
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    _VECTORSTORE_SERVICE_CACHE[cache_key] = vectorstore

    logger.info(
        f"Vectorstore พร้อมใช้งาน!\n"
        f"   Collection  : {collection_name}\n"
        f"   Path        : {persist_directory}"
    )

    return vectorstore


# -------------------- Load / Save / Management Mapping DB --------------------

def _get_doc_map_key(doc_type: str, enabler: Optional[str]) -> str:
    """Helper for internal dict key for mapping DB."""
    doc_type_lower = doc_type.lower()
    if doc_type_lower == EVIDENCE_DOC_TYPES.lower() and enabler:
        return f"{doc_type_lower}_{enabler.upper()}"
    return doc_type_lower


_MAPPING_DB_CACHE: Dict[str, Dict[str, Any]] = {}

def load_doc_id_mapping(
    doc_type: str, 
    tenant: str = DEFAULT_TENANT, 
    year: Optional[int] = DEFAULT_YEAR, 
    enabler: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load document ID mapping database from JSON file (Multi-Tenant/Year supported).
    """
    map_key = _get_doc_map_key(doc_type, enabler)
    cache_key = f"{tenant.lower()}_{year}_{map_key}"

    # 1. Cache HIT
    if cache_key in _MAPPING_DB_CACHE:
        logger.debug(f"Cache HIT → Reusing mapping DB: {cache_key}")
        return _MAPPING_DB_CACHE[cache_key]

    # 2. Determine path
    # 🎯 FIX: ใช้ get_mapping_file_path จาก path_utils.py
    mapping_file_path = get_mapping_file_path(tenant, year, enabler)
    
    mapping_db = {}
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, "r", encoding="utf-8") as f:
                mapping_db = json.load(f)
            logger.debug(f"Loaded {len(mapping_db)} entries from {mapping_file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding mapping file {mapping_file_path}: {e}")
        except Exception as e:
             logger.error(f"Error loading mapping file {mapping_file_path}: {e}")

    # 3. Cache and Return
    _MAPPING_DB_CACHE[cache_key] = mapping_db
    return mapping_db


def save_doc_id_mapping(
    mapping_db: Dict[str, Dict[str, Any]],
    doc_type: str, 
    tenant: str = DEFAULT_TENANT, 
    year: Optional[int] = DEFAULT_YEAR, 
    enabler: Optional[str] = None
) -> None:
    """
    Save document ID mapping database to JSON file.
    """
    map_key = _get_doc_map_key(doc_type, enabler)
    cache_key = f"{tenant.lower()}_{year}_{map_key}"

    # 1. Determine path
    # 🎯 FIX: ใช้ get_mapping_file_path จาก path_utils.py
    mapping_file_path = get_mapping_file_path(tenant, year, enabler)
    
    # Ensure directory exists (handled by path_utils)
    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    try:
        with open(mapping_file_path, "w", encoding="utf-8") as f:
            json.dump(mapping_db, f, indent=4, ensure_ascii=False)
        
        logger.debug(f"Saved {len(mapping_db)} entries to {mapping_file_path}")
        
        # 🎯 FIX: บังคับ Flush Output ทันทีหลังจากการเขียนไฟล์ I/O เสร็จสิ้น
        # เพื่อให้แน่ใจว่า Log ถูกแสดงผลออกมา
        sys.stdout.flush() 
        
        # 2. Update Cache
        _MAPPING_DB_CACHE[cache_key] = mapping_db
        
    except Exception as e:
        logger.error(f"Error saving mapping file {mapping_file_path}: {e}")

def load_evidence_mapping(
    tenant: str = DEFAULT_TENANT, 
    year: int = DEFAULT_YEAR, 
    enabler: str = DEFAULT_ENABLER
) -> Dict[str, Any]:
    """
    Load the persistent map for Evidence Statements (for RAG hydration).
    """
    # 🎯 FIX: ใช้ get_evidence_mapping_file_path จาก path_utils.py
    mapping_file_path = get_evidence_mapping_file_path(tenant, year, enabler)
    
    evidence_map = {}
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, "r", encoding="utf-8") as f:
                evidence_map = json.load(f)
            logger.debug(f"Loaded {len(evidence_map)} entries from evidence mapping: {mapping_file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding evidence mapping file {mapping_file_path}: {e}")
        except Exception as e:
            logger.error(f"Error loading evidence mapping file {mapping_file_path}: {e}")

    return evidence_map


def save_evidence_mapping(
    evidence_map: Dict[str, Any],
    tenant: str = DEFAULT_TENANT, 
    year: int = DEFAULT_YEAR, 
    enabler: str = DEFAULT_ENABLER
) -> None:
    """
    Save the persistent map for Evidence Statements.
    """
    # 🎯 FIX: ใช้ get_evidence_mapping_file_path จาก path_utils.py
    mapping_file_path = get_evidence_mapping_file_path(tenant, year, enabler)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    try:
        with open(mapping_file_path, "w", encoding="utf-8") as f:
            json.dump(evidence_map, f, indent=4, ensure_ascii=False)
        logger.debug(f"Saved {len(evidence_map)} entries to evidence mapping: {mapping_file_path}")
    except Exception as e:
        logger.error(f"Error saving evidence mapping file {mapping_file_path}: {e}")


# -------------------- API Helper: Get UUIDs for RAG Filtering --------------------
# 📌 REVISED: เพิ่ม tenant และ year
def get_stable_uuids_by_doc_type(doc_types: List[str], tenant: str = "pwa", year: int = 2568) -> List[str]:
    """Retrieves Stable UUIDs for RAG filtering based on document types (Multi-Tenant/Year)."""
    if not doc_types: return []
    doc_type_set = {dt.lower() for dt in doc_types}
    target_uuids = []

    # โหลด mapping db ทั้งหมดสำหรับ doc_types ที่ร้องขอ
    for dt_req in doc_type_set:
        
        doc_mapping_db = load_doc_id_mapping(dt_req, tenant, year)
        
        for s_uuid, entry in doc_mapping_db.items():
            # NOTE: การกรอง Tenant/Year/Doc Type ถูกทำใน load_doc_id_mapping แล้ว
            target_uuids.append(s_uuid)

    return list(set(target_uuids))

# 📌 REVISED: เพิ่ม tenant และ year
def create_stable_uuid_from_path(
    filepath: str, 
    ref_id_key: Optional[str] = None,
    tenant: Optional[str] = None,
    year: Optional[int] = None
) -> str:
    """
    สร้าง UUID ที่เสถียรสำหรับไฟล์ (64-char Hash)
    - ใช้ SHA-256 ของ (basename + size + modification time + tenant + year)
    """
    if not os.path.exists(filepath):
        logger.warning(f"File not found for UUID creation: {filepath}")
        return str(uuid.uuid4()) # Fallback to random UUID

    file_name = os.path.basename(filepath)
    try:
        file_size = os.path.getsize(filepath)
        mod_time = os.path.getmtime(filepath)
    except Exception as e:
        logger.error(f"Error getting file metadata for {filepath}: {e}")
        file_size = 0
        mod_time = 0

    # Key fields that define the 'stable' identity of the document
    key_fields = (
        file_name, 
        file_size, 
        mod_time,
        tenant or "",
        year or ""
    )
    
    # Combine fields into a single string for hashing
    hash_input = ":".join(map(str, key_fields)).encode('utf-8')

    # Generate SHA-256 hash
    stable_hash = hashlib.sha256(hash_input).hexdigest()
    
    # Return the 64-character hash
    return stable_hash


# -------------------- CORE INGESTION LOGIC --------------------
DOC_TYPES_WITH_YEAR_AND_ENABLER = [EVIDENCE_DOC_TYPES.lower()]
# -------------------- CORE INGESTION LOGIC --------------------
# 📌 ASSUME:
# - Helper functions: get_document_source_dir, load_doc_id_mapping, save_doc_id_mapping, 
#   get_doc_type_collection_key, create_stable_uuid_from_path, process_document, get_vectorstore ถูกนิยามและ import แล้ว
# - Global variables: DOC_TYPES_WITH_YEAR_AND_ENABLER, EVIDENCE_DOC_TYPES, SUPPORTED_DOC_TYPES, 
#   SUPPORTED_ENABLERS, DEFAULT_TENANT, DEFAULT_YEAR, SUPPORTED_TYPES, logger ถูกนิยามและ import แล้ว
# - import os, sys, shutil, json, uuid, List, Dict, Any, Optional, Union, Tuple, Set, 
#   ThreadPoolExecutor, as_completed, Document ถูกทำไว้แล้ว
# --------------------------------------------------------------

# DOC_TYPES_WITH_YEAR_AND_ENABLER (Global Variable)
# e.g., DOC_TYPES_WITH_YEAR_AND_ENABLER = [EVIDENCE_DOC_TYPES.lower()]

def ingest_all_files(
    tenant: str = DEFAULT_TENANT, 
    year: int = DEFAULT_YEAR,
    doc_type: Union[str, List[str]] = "all",
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    dry_run: bool = False,
    sequential: bool = False,
    skip_ext: Optional[Set[str]] = None,
    log_every: int = 5,
) -> List[Dict[str, Any]]:
    """
    Scan files in the source directories and ingest them into the relevant Vector Store collections.
    """
    
    if isinstance(doc_type, list):
        if len(doc_type) > 1:
            logger.warning("Multiple doc_types provided. Using only the first one for logic control.")
        doc_type_req = doc_type[0].lower()
    else:
        doc_type_req = doc_type.lower()
        
    enabler_req = enabler.upper() if enabler else None
    
    # --- 🎯 FIX: Path Mismatch Logic ---
    final_year: Optional[int] = year
    final_enabler: Optional[str] = enabler_req
    
    if doc_type_req.lower() not in DOC_TYPES_WITH_YEAR_AND_ENABLER:
        # ถ้าเป็น Global Doc Type (document, seam, faq) ให้บังคับใช้ final_year=None, final_enabler=None
        if year is not None:
            logger.warning(f"⚠️ Warning: Year '{year}' provided for doc_type='{doc_type_req}'. Year is usually ignored for non-evidence types. Setting year to None.")
        if enabler_req is not None:
            logger.warning(f"⚠️ Enabler ({enabler_req}) is ignored for Global Doc Type: {doc_type_req}. Setting enabler to None.")
        final_year = None
        final_enabler = None
    # -----------------------------------

    logger.info(f"Starting ingest_all_files: Tenant={tenant}, Year={final_year}, doc_type_req='{doc_type_req}', enabler_req='{final_enabler}', subject_req='{subject}'")

    # 1. รวบรวมไฟล์ (ใช้ final_year/final_enabler ในการกำหนด Path)
    scan_roots: List[str] = []
    
    if doc_type_req == "all":
        # Scan ทุก Doc Type และทุก Enabler
        for dt in SUPPORTED_DOC_TYPES:
            dt_lower = dt.lower()
            if dt_lower == EVIDENCE_DOC_TYPES.lower():
                for ena in SUPPORTED_ENABLERS:
                    # Evidence: ใช้ final_year และ Enabler
                    scan_roots.append(get_document_source_dir(
                        tenant=tenant, year=final_year, doc_type=dt_lower, enabler=ena
                    ))
            else:
                # Global: ใช้ year=None และ Enabler=None
                scan_roots.append(get_document_source_dir(
                    tenant=tenant, year=None, doc_type=dt_lower, enabler=None
                ))

    elif doc_type_req in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
        if doc_type_req == EVIDENCE_DOC_TYPES.lower():
            # ... (Logic สำหรับ Evidence - ใช้ final_year/enabler)
            if final_enabler and final_enabler in SUPPORTED_ENABLERS:
                scan_roots = [get_document_source_dir(
                    tenant=tenant, year=final_year, doc_type=doc_type_req, enabler=final_enabler
                )]
            else:
                for ena in SUPPORTED_ENABLERS:
                    scan_roots.append(get_document_source_dir(
                        tenant=tenant, year=final_year, doc_type=doc_type_req, enabler=ena
                    ))
        else:
             # Doc Type อื่น ๆ (document, faq, seam, other) เป็น Global
             scan_roots = [get_document_source_dir(
                tenant=tenant, year=None, doc_type=doc_type_req, enabler=None
            )]
    else:
        logger.error(f"❌ Doc type '{doc_type_req}' is not supported.")
        return []

    files_to_process: List[Dict[str, Any]] = []
    exclude_dirs = ['.DS_Store', '__pycache__', 'backup']

    for root_to_scan in scan_roots:
        # ... Logic เพื่อหา doc_type_from_path และ resolved_enabler จาก root_to_scan
        path_segments = root_to_scan.lower().split(os.path.sep)
        doc_type_from_path = None 
        resolved_enabler = None

        if EVIDENCE_DOC_TYPES.lower() in path_segments:
            doc_type_from_path = EVIDENCE_DOC_TYPES.lower()
            # Find resolved_enabler from path_segments
            for ena in SUPPORTED_ENABLERS:
                 if ena.lower() in path_segments:
                     resolved_enabler = ena
                     break
            
        elif path_segments and path_segments[-1] in [dt.lower() for dt in SUPPORTED_DOC_TYPES if dt.lower() != EVIDENCE_DOC_TYPES.lower()]:
            doc_type_from_path = path_segments[-1]
        
        if not os.path.exists(root_to_scan) or doc_type_from_path is None:
            logger.warning(f"Source directory not found or Doc Type unresolved: {root_to_scan}. Skipping.")
            continue
            
        current_collection_name = get_doc_type_collection_key(doc_type_from_path, resolved_enabler)
        logger.info(f"Scanning source directory: {root_to_scan} (Maps to Collection: {current_collection_name})")

        for root, dirs, filenames in os.walk(root_to_scan):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            if root != root_to_scan: continue 

            for f in filenames:
                file_extension = os.path.splitext(f)[1].lower()
                if f.startswith('.') or file_extension not in SUPPORTED_TYPES or (skip_ext and file_extension in skip_ext): continue

                files_to_process.append({
                    "file_path": os.path.join(root, f),
                    "file_name": f,
                    "doc_type": doc_type_from_path,
                    "enabler": resolved_enabler, 
                    "collection_name": current_collection_name
                })
    
    if not files_to_process:
        logger.warning("⚠️ No files found to ingest!")
        return []

    # 2. Load Mapping DB และกำหนด Stable IDs
    doc_mapping_dbs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    loading_contexts = set()
    for f in files_to_process:
        dt, ena = f['doc_type'], f['enabler'] 
        # 📌 FIX: ใช้ final_year ที่ถูกกำหนดให้เป็น None สำหรับ Global Doc Type
        # NOTE: final_year จะเป็น year หรือ None ขึ้นอยู่กับ Doc Type
        loading_contexts.add((dt, ena, final_year)) 

    for dt, ena, yr in loading_contexts:
        dt_ena_key = get_doc_type_collection_key(dt, ena) 
        try:
            # โหลด Mapping โดยใช้ final_year/ena ที่ถูกต้อง
            mapping_db = load_doc_id_mapping(dt, tenant, yr, ena)
            doc_mapping_dbs[dt_ena_key] = mapping_db
        except FileNotFoundError:
            doc_mapping_dbs[dt_ena_key] = {}
        except Exception as e:
            logger.error(f"❌ Error loading mapping for {dt} / {ena or 'None'} / Year {yr or 'None'}: {e}")

    uuid_from_path_lookup: Dict[str, str] = {}
    for dt_ena_key, db in doc_mapping_dbs.items():
        for s_uuid, entry in db.items():
            # NOTE: entry["filepath"] ควรจะเป็น Full Path
            if "filepath" in entry and str(entry.get("tenant")).lower() == tenant.lower():
                # ตรวจสอบว่าบริบทใน entry ตรงกับบริบทที่โหลดหรือไม่ (ปี/enabler จะเป็น None สำหรับ Global)
                entry_dt = entry.get("doc_type")
                entry_ena = entry.get("enabler")
                entry_year = entry.get("year")
                
                # Logic ตรวจสอบบริบท (ซับซ้อน แต่พยายามให้ครอบคลุม)
                if entry_dt == dt_ena_key.split('_')[0] and entry_year == final_year:
                    # ตรวจสอบ Enabler สำหรับ Evidence
                    if entry_dt == EVIDENCE_DOC_TYPES.lower():
                        if entry_ena == dt_ena_key.split('_')[1].upper():
                            uuid_from_path_lookup[entry["filepath"]] = s_uuid
                    # สำหรับ Global Doc Type
                    else:
                        uuid_from_path_lookup[entry["filepath"]] = s_uuid


    # Pre-calculate Stable UUIDs
    for file_info in files_to_process:
        file_path = file_info["file_path"] # Full Path
        stable_doc_uuid = uuid_from_path_lookup.get(file_path)

        if not stable_doc_uuid:
            # 📌 FIX: ใช้ final_year ในการสร้าง UUID เพื่อให้ Global Doc Type ได้ UUID ที่ไม่มีปี
            # NOTE: create_stable_uuid_from_path ต้องรับ file_path ที่เป็น Full Path
            stable_doc_uuid = create_stable_uuid_from_path(file_path, tenant=tenant, year=final_year)
        
        file_info["stable_doc_uuid"] = stable_doc_uuid


    # 3. Process files (Load + Chunk)
    all_chunks: List[Document] = []
    results: List[Dict[str, Any]] = []
    
    # ------------------------------------------------------------------
    # 📌 Worker Function
    # ------------------------------------------------------------------
    
    def _process_file_task(file_info: Dict[str, Any]) -> Tuple[List[Document], str, str]:
        """Worker function for parallel processing."""
        file_path = file_info["file_path"] # Full Path
        file_name = file_info["file_name"]
        stable_doc_uuid = file_info["stable_doc_uuid"]
        doc_type = file_info["doc_type"]
        enabler = file_info["enabler"]
        
        # NOTE: process_document ต้องถูกเรียกใช้ด้วย final_year เพื่อใส่ใน metadata
        # process_document ต้องใส่ file_path (Full Path) ลงใน chunk.metadata['filepath'] ด้วย
        chunks, doc_id, dt = process_document(
            file_path=file_path,
            file_name=file_name,
            stable_doc_uuid=stable_doc_uuid,
            doc_type=doc_type,
            enabler=enabler,
            subject=subject,
            year=final_year, # 💡 ใช้ final_year ใน Metadata
            tenant=tenant
        )
        return chunks, doc_id, dt

    # ... (โค้ด Sequential/Parallel processing) ...
    if sequential or dry_run:
        # Sequential processing (Debugging/Dry Run)
        for idx, file_info in enumerate(files_to_process, 1):
            f, stable_doc_uuid = file_info["file_name"], file_info["stable_doc_uuid"]
            try:
                chunks, doc_id, dt = _process_file_task(file_info)
                all_chunks.extend(chunks)
                results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks), "tenant": tenant, "year": final_year, "enabler": file_info["enabler"]})
            except Exception as e:
                results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": file_info["doc_type"], "status": "failed_chunk", "error": str(e), "tenant": tenant, "year": final_year, "enabler": file_info["enabler"]})
                logger.error(f"❌ Failed to process {f}: {e}")
            if idx % log_every == 0: logger.info(f"Processed {idx}/{len(files_to_process)} files...")
                
    else:
        # Parallel processing (Production)
        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
            future_to_file = {executor.submit(_process_file_task, file_info): file_info for file_info in files_to_process}
            for idx, future in enumerate(as_completed(future_to_file), 1):
                file_info = future_to_file[future]
                f, stable_doc_uuid = file_info["file_name"], file_info["stable_doc_uuid"]
                try:
                    chunks, doc_id, dt = future.result()
                    all_chunks.extend(chunks)
                    results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks), "tenant": tenant, "year": final_year, "enabler": file_info["enabler"]})
                except Exception as e:
                    results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": file_info["doc_type"], "status": "failed_chunk", "error": str(e), "tenant": tenant, "year": final_year, "enabler": file_info["enabler"]})
                    logger.error(f"❌ Failed to process {f}: {e}")
                if idx % log_every == 0: logger.info(f"Processed {idx}/{len(files_to_process)} files...")


    # 4. Ingest Chunks to Vectorstore
    if dry_run or not all_chunks:
        logger.warning(f"Dry run or no chunks created. Skipping ingestion to VectorStore.")
        return results

    chunks_by_collection: Dict[str, List[Document]] = {}
    for chunk in all_chunks:
        chunk_doc_type = chunk.metadata.get("doc_type")
        chunk_enabler = chunk.metadata.get("enabler")
        collection_name = get_doc_type_collection_key(chunk_doc_type, chunk_enabler)
        if collection_name:
            chunks_by_collection.setdefault(collection_name, []).append(chunk)

    ingested_uuids = set()

    for collection_name, chunks in chunks_by_collection.items():
        logger.info(f"Ingesting {len(chunks)} chunks into collection '{collection_name}'...")
        
        # NOTE: ใช้ final_year ในการสร้าง VS object
        vs_year = final_year if collection_name.startswith(EVIDENCE_DOC_TYPES.lower()) else DEFAULT_YEAR
        vectorstore = get_vectorstore(collection_name, tenant, vs_year) 
        
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        try:
            chunk_uuids = [meta['chunk_uuid'] for meta in metadatas]
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=chunk_uuids)
            logger.info(f"✅ Ingested {len(chunks)} chunks successfully into '{collection_name}'.")
            ingested_uuids.update(chunk_uuids)

        except Exception as e:
            logger.error(f"❌ Failed to ingest chunks into '{collection_name}': {e}", exc_info=True)


    # 5. Update Mapping Database
    
    updated_contexts: Set[Tuple[str, Optional[str], Optional[int]]] = set()
    for r in results:
        if r['status'] == 'chunked':
            # 📌 ใช้ final_year ในการกำหนด Context ที่จะบันทึก
            updated_contexts.add((r['doc_type'], r.get('enabler'), final_year))
                
    for current_dt, current_enabler, current_year_for_map in updated_contexts:
        
        dt_ena_key = get_doc_type_collection_key(current_dt, current_enabler)
        doc_mapping_db = doc_mapping_dbs.get(dt_ena_key)
        
        if not doc_mapping_db:
             # Logic เพื่อโหลดซ้ำหากไม่พบ
             try:
                 doc_mapping_db = load_doc_id_mapping(current_dt, tenant, final_year, current_enabler)
                 doc_mapping_dbs[dt_ena_key] = doc_mapping_db
             except Exception as e:
                 logger.error(f"❌ Error reloading mapping DB for save: {e}")
                 continue

        updated_count = 0
        doc_chunks: Dict[str, Dict[str, Any]] = {} 
        
        # Group chunks ที่ถูก Ingest สำเร็จ
        for chunk in all_chunks:
            s_uuid = chunk.metadata.get("stable_doc_uuid")
            chunk_dt = chunk.metadata.get("doc_type")
            chunk_ena = chunk.metadata.get("enabler")
            
            if chunk_dt == current_dt and chunk_ena == current_enabler:
                 doc_chunks.setdefault(s_uuid, {})["chunk_uuids"] = doc_chunks.setdefault(s_uuid, {}).get("chunk_uuids", []) + [chunk.metadata['chunk_uuid']]
                 doc_chunks[s_uuid]['metadata'] = chunk.metadata 

        for stable_doc_uuid, data in doc_chunks.items():
            
            new_ids = [uid for uid in data["chunk_uuids"] if uid in ingested_uuids]
            if not new_ids: continue
                
            metadata = data["metadata"]
            entry = doc_mapping_db.get(stable_doc_uuid)
            
            # 🎯 FIX: Year/Enabler ที่จะบันทึกใน Mapping DB
            year_to_save = final_year
            enabler_to_save = current_enabler if current_dt == EVIDENCE_DOC_TYPES.lower() else None

            # =========================================================
            # 💡 START FIX: Reconstruction Logic (ใช้ path_utils.py)
            # =========================================================
            filepath_from_metadata = metadata.get("filepath", metadata.get("source_filename"))
            final_filepath_to_save = filepath_from_metadata
            
            # ตรวจสอบ: ถ้า filepath ไม่ใช่ Absolute Path (เช่น "2566-PEA.pdf")
            if filepath_from_metadata and not os.path.isabs(filepath_from_metadata):
                 # 3. ใช้ get_document_source_dir จาก utils.path_utils.py
                 doc_root_dir = get_document_source_dir(
                     tenant=tenant, 
                     year=year_to_save,       # final_year (None สำหรับ Global Doc Type)
                     enabler=enabler_to_save, # None สำหรับ Global Doc Type
                     doc_type=current_dt
                 )
                 # 4. Reconstruct Full Path (Absolute Path)
                 final_filepath_to_save = os.path.join(doc_root_dir, metadata.get("source_filename"))
                 logger.warning(f"⚠️ Reconstructed relative filepath for {metadata.get('source_filename')} (Doc Type: {current_dt}) to: {final_filepath_to_save}")
            # =========================================================
            # 💡 END FIX: Reconstruction Logic
            # =========================================================

            if entry:
                # Update existing entry
                entry['chunk_uuids'].extend(new_ids)
                entry['chunk_uuids'] = list(set(entry['chunk_uuids']))
                entry['chunk_count'] = len(entry['chunk_uuids'])
                entry['status'] = "Ingested"
                entry['year'] = year_to_save # อัปเดตปี/enabler เผื่อมีการแก้ไข
                entry['enabler'] = enabler_to_save
                entry['filepath'] = final_filepath_to_save # 🎯 UPDATE FILEPATH สำหรับ Entry เก่าด้วย
                
            else:
                # Create new entry
                entry = {
                    "doc_id": stable_doc_uuid,
                    "file_name": metadata.get("source_filename"),
                    "file_type": os.path.splitext(metadata.get("source_filename", ""))[1].lower(),
                    "filepath": final_filepath_to_save, # 🎯 ใช้ Absolute Path ที่แก้ไขแล้ว
                    "doc_type": current_dt, 
                    "enabler": enabler_to_save,
                    "tenant": tenant,
                    "year": year_to_save,
                    "notes": "CREATED_DURING_INGEST",
                    "statement_id": "", 
                    "chunk_uuids": list(set(new_ids)),
                    "status": "Ingested",
                    "chunk_count": len(new_ids)
                }
                doc_mapping_db[stable_doc_uuid] = entry
                
            updated_count += 1
        
        # 📌 บันทึก Mapping DB คืน (ใช้ final_year ที่เป็น None สำหรับ Global Doc Type)
        if updated_count > 0:
             # ใช้ current_enabler ในการกำหนด Path สำหรับ Evidence, ใช้ None สำหรับ Global
             save_doc_id_mapping(doc_mapping_db, current_dt, tenant, final_year, enabler=current_enabler) 
             
        logger.info(f"Updated mapping DB for {updated_count} documents in doc type '{current_dt}' (Enabler: {current_enabler or 'None'}) for tenant '{tenant}/{final_year}'.")
        
    
    logger.info(f"✅ Batch ingestion process finished (dry_run={dry_run}) for {tenant}/{final_year}.")
    
    sys.stdout.flush() 

    return results

# -------------------- Wipe Vectorstore / Mapping --------------------

def wipe_vectorstore(
    doc_type_to_wipe: str = "all",
    enabler: Optional[str] = None,
    tenant: str = DEFAULT_TENANT,
    year: int = DEFAULT_YEAR,
    base_path: str = VECTORSTORE_DIR
) -> None:
    """
    Deletes the Vector Store collection(s) and associated mapping files.
    """
    logger.critical(f"⚠️ !!! เริ่มกระบวนการ WIPE Vectorstore และ Mapping Files !!!")
    
    doc_type_to_wipe_lower = doc_type_to_wipe.lower()
    final_year: Optional[int] = year

    # 🎯 FIX: สำหรับ DocType Global (เช่น document) ให้ใช้ final_year=None
    if doc_type_to_wipe_lower not in DOC_TYPES_WITH_YEAR_AND_ENABLER and doc_type_to_wipe_lower != 'all':
        final_year = None
    
    final_enabler = enabler.upper() if enabler else None

    # 1. กำหนด Collections ที่ต้องการลบ (ใช้ Path Utility)
    collections_to_delete: List[Tuple[str, Optional[int], Optional[str], str]] = [] # (doc_type, year, enabler, collection_name)
    
    doc_types_to_check = []
    if doc_type_to_wipe_lower == 'all':
        doc_types_to_check.extend([dt.lower() for dt in SUPPORTED_DOC_TYPES])
    else:
        doc_types_to_check.append(doc_type_to_wipe_lower)
        
    for dt in doc_types_to_check:
        
        map_year_to_use = year if dt == EVIDENCE_DOC_TYPES.lower() else final_year
        
        if dt == EVIDENCE_DOC_TYPES.lower():
            enablers_to_check = []
            if final_enabler and final_enabler in SUPPORTED_ENABLERS:
                enablers_to_check.append(final_enabler)
            elif doc_type_to_wipe_lower == 'all' or (doc_type_to_wipe_lower == EVIDENCE_DOC_TYPES.lower() and not final_enabler):
                enablers_to_check.extend(SUPPORTED_ENABLERS)
                
            for ena in enablers_to_check:
                collections_to_delete.append((
                    dt, map_year_to_use, ena, get_doc_type_collection_key(dt, ena) # 🎯 FIX: ใช้ get_doc_type_collection_key
                ))
        else:
            if doc_type_to_wipe_lower == 'all' or doc_type_to_wipe_lower == dt:
                collections_to_delete.append((
                    dt, map_year_to_use, None, get_doc_type_collection_key(dt, None) # 🎯 FIX: ใช้ get_doc_type_collection_key
                ))


    if not collections_to_delete and doc_type_to_wipe_lower != 'all':
        logger.warning(f"⚠️ ไม่พบ Collection ที่ตรงกับเงื่อนไข Tenant='{tenant}', Year='{year}', Doc Type='{doc_type_to_wipe}', Enabler='{enabler}'.")
        return

    # 2. ลบ Collection Folder และปรับปรุง Doc ID Mapping DB
    deletion_count = 0
    
    for dt, map_year_to_use, ena, col_name in collections_to_delete:
        
        # 2a. ลบ Vector Store Folder
        try:
            # 🎯 FIX: ใช้ get_vectorstore_collection_path จาก path_utils.py
            collection_path = get_vectorstore_collection_path(tenant, map_year_to_use, dt, ena)
            
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
                logger.info(f"🗑️ Deleted Collection Folder: {col_name} at {collection_path}")
                deletion_count += 1
            else:
                logger.info(f"Collection Folder ไม่พบ: {col_name} ({collection_path}).")
        except Exception as e:
            logger.error(f"❌ Error deleting vectorstore folder {col_name}: {e}")

        # 2b. ลบ Entry จาก Doc ID Mapping
        doc_mapping_db = load_doc_id_mapping(dt, tenant, map_year_to_use, ena) 
        uuids_to_keep = {}
        
        # กรอง Entry ที่ไม่เกี่ยวข้องกับ Collection ที่เพิ่งถูกลบ
        for s_uuid, entry in doc_mapping_db.items():
            
            entry_col_name = get_doc_type_collection_key(entry.get('doc_type'), entry.get('enabler')) # 🎯 FIX: ใช้ get_doc_type_collection_key

            if entry.get("tenant").lower() != tenant.lower():
                 uuids_to_keep[s_uuid] = entry
                 continue
                 
            if entry_col_name != col_name:
                uuids_to_keep[s_uuid] = entry
            else:
                logger.debug(f"Removed mapping entry for {s_uuid} (Collection: {col_name})")


        if not uuids_to_keep and len(doc_mapping_db) > 0:
            # ถ้าไม่มี Entry เหลืออยู่เลย และไฟล์ Mapping มีอยู่จริง ให้ลบไฟล์ Mapping
            # 🎯 FIX: ใช้ get_mapping_file_path
            mapping_path = get_mapping_file_path(tenant, map_year_to_use, ena) 
            if os.path.exists(mapping_path):
                try:
                    os.remove(mapping_path)
                    logger.info(f"✅ Deleted (empty) Doc ID mapping file: {mapping_path} (Via Step 2)")
                except OSError as e:
                    logger.error(f"❌ Error deleting mapping file: {e}")
        else:
            save_doc_id_mapping(uuids_to_keep, dt, tenant, map_year_to_use, enabler=ena) 

        # 2c. ลบไฟล์ Evidence Mapping ถ้าเป็น Evidence และเงื่อนไขตรง
        if dt == EVIDENCE_DOC_TYPES.lower():
            # 🎯 FIX: ใช้ get_evidence_mapping_file_path
            evidence_map_path = get_evidence_mapping_file_path(tenant, map_year_to_use, ena) 
            if os.path.exists(evidence_map_path):
                 try:
                    os.remove(evidence_map_path)
                    logger.info(f"✅ Deleted Evidence Mapping file: {evidence_map_path}")
                 except OSError as e:
                    logger.error(f"❌ Error deleting evidence mapping file: {e}")
            
        logger.info(f"🧹 Removed {len(doc_mapping_db) - len(uuids_to_keep)} entries from mapping file for deleted collection (Doc Type: {dt}/{ena}) of {tenant}/{map_year_to_use}.")

    # 3. ลบ Root Directory ของ Vector Store และ Mapping หากว่างเปล่า
    try:
        # ลบ Vectorstore Root
        # 🎯 FIX: ใช้ get_vectorstore_tenant_root_path
        tenant_root_path = get_vectorstore_tenant_root_path(tenant) 
        if os.path.isdir(tenant_root_path):
            if not any(f for f in os.listdir(tenant_root_path) if not f.startswith('.')):
                try:
                    shutil.rmtree(tenant_root_path) 
                    logger.info(f"✅ Deleted empty Vector Store root directory: {tenant_root_path}")
                except OSError as e:
                    logger.debug(f"Vector Store directory {tenant_root_path} not empty or cannot be deleted: {e}")

        # ลบ Mapping Root
        mapping_dir_year = os.path.join(MAPPING_BASE_DIR, tenant.lower(), str(year))
        if os.path.isdir(mapping_dir_year):
            if not any(f for f in os.listdir(mapping_dir_year) if not f.startswith('.')):
                try:
                    os.rmdir(mapping_dir_year)
                    logger.info(f"✅ Deleted empty Doc ID mapping directory: {mapping_dir_year}")
                except OSError as e:
                    logger.debug(f"Mapping directory {mapping_dir_year} not empty or cannot be deleted: {e}")
            else: 
                logger.info(f"Mapping directory {mapping_dir_year} is not completely empty. Keeping.")
                
    except Exception as e:
        logger.warning(f"Error during final mapping directory cleanup: {e}")

# -------------------- [REVISED] Document Management Utilities --------------------
def delete_document_by_uuid(
    stable_doc_uuid: str, 
    tenant: str = "pwa", 
    year: int = 2568, 
    collection_name: Optional[str] = None, 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR
) -> bool:
    """Deletes all chunks associated with the given Stable Document UUID (Multi-Tenant/Year)."""
    if not doc_type:
        logger.error(f"Cannot delete {stable_doc_uuid}: doc_type must be provided for mapping file isolation.")
        return False
        
    # =================================================================
    # 🎯 FIX 1: OVERRIDE LOGIC สำหรับ Global Doc Types
    # =================================================================
    doc_type_lower = doc_type.lower()
    final_year_for_map: Optional[int] = year
    final_enabler: Optional[str] = enabler.upper() if enabler else None
    
    if doc_type_lower not in DOC_TYPES_WITH_YEAR_AND_ENABLER:
        logger.warning(f"⚠️ Year ({year}) is ignored for Global Doc Type deletion: {doc_type_lower}. Setting year to None for Mapping.")
        final_year_for_map = None
        final_enabler = None

    # 💡 ใช้ final_year_for_map ในการโหลด Mapping
    doc_mapping_db = load_doc_id_mapping(doc_type, tenant, final_year_for_map, final_enabler) 

    entry = doc_mapping_db.get(stable_doc_uuid)
    if not entry:
        logger.warning(f"UUID {stable_doc_uuid} not found in mapping DB for {doc_type}/{final_year_for_map}.")
        return False

    final_doc_type = entry.get("doc_type", doc_type_lower)
    final_enabler = entry.get("enabler", final_enabler)
    chunk_uuids = entry.get("chunk_uuids", [])
    
    if not chunk_uuids:
        logger.warning(f"No chunks found for UUID {stable_doc_uuid}. Deleting mapping entry only.")
        del doc_mapping_db[stable_doc_uuid]
        
    else:
        # 1. ลบ Chunks ออกจาก Vectorstore
        try:
            # 🎯 FIX: ใช้ get_doc_type_collection_key
            col_name = get_doc_type_collection_key(final_doc_type, final_enabler)
            
            # 💡 ใช้ final_year_for_map (จะเป็น year หรือ None)
            vectorstore = get_vectorstore(col_name, tenant, final_year_for_map or DEFAULT_YEAR) 
            
            vectorstore.delete(ids=chunk_uuids)
            logger.info(f"✅ Deleted {len(chunk_uuids)} chunks for {stable_doc_uuid} from collection '{col_name}'.")
        except Exception as e:
            logger.error(f"❌ Failed to delete chunks from Vectorstore for {stable_doc_uuid}: {e}")

        # 🎯 FIX: ลบ Entry ออกจาก Mapping DB ที่โหลดมา
        del doc_mapping_db[stable_doc_uuid]
        
    # 📌 FIX: บันทึก Mapping DB คืน (ต้องมีการแยก Enabler ถ้าเป็น Evidence)
    if final_doc_type.lower() == EVIDENCE_DOC_TYPES.lower():
        # ... (Logic การบันทึก Evidence - โค้ดเดิม, ใช้ year ที่รับเข้ามา)
        # ในกรณีนี้ final_year_for_map จะเท่ากับ year
        db_to_save: Dict[str, Dict[str, Any]] = {}
        for s_uuid, entry in doc_mapping_db.items():
            # 📌 FIX: ต้องเช็ค Tenant/Year/Enabler ด้วย
            if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year_for_map and entry.get("enabler") == final_enabler:
                db_to_save[s_uuid] = entry
        save_doc_id_mapping(db_to_save, final_doc_type, tenant, final_year_for_map, enabler=final_enabler)
    else:
        # สำหรับ Doc Type อื่นๆ (เช่น document) บันทึกเฉพาะ Entry ที่เกี่ยวข้องกับ Tenant/Year
        # ในกรณีนี้ final_year_for_map จะเป็น None
        db_to_save = {
            s_uuid: entry for s_uuid, entry in doc_mapping_db.items() 
            if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year_for_map
        }
        save_doc_id_mapping(db_to_save, final_doc_type, tenant, final_year_for_map)
        
    return True

def list_documents(
    doc_types: Optional[List[str]] = None,
    enabler: Optional[str] = None,
    tenant: str = "pwa",
    year: Union[int, str] = 2568, 
    show_results: str = "ingested"
) -> Dict[str, Any]:
    """
    Scans data directory and mapping DBs to list documents.
    """
    
    # 💡 NOTE: สมมติว่ามีการกำหนดค่าเหล่านี้ในระดับ Global/Module
    DEFAULT_DOC_TYPES = ["document", "faq"]
    SUPPORTED_DOC_TYPES = ["document", "faq", "evidence", "seam", "other"]
    SUPPORTED_ENABLERS = ["KM", "RM", "IC", "SM", "SP"]
    SUPPORTED_TYPES = ['.pdf', '.docx', '.xlsx', '.pptx', '.txt', '.csv', '.png', '.jpg', '.jpeg']
    EVIDENCE_DOC_TYPES = "evidence"
    
    # หากมีการใช้ `_normalize_doc_id` และ `get_document_source_dir`
    # ต้องมั่นใจว่ามีการ import/defined ไว้ หรือแทนที่ด้วย logic ของคุณ
    
    logger.info(f"Listing documents for Tenant={tenant}, Year={year}, Doc Types={doc_types}, Enabler={enabler}, Show={show_results}")

    # --- 1. Resolve Parameters ---
    year_int: Optional[int]
    try:
        year_int = int(year) if year is not None and str(year).isdigit() else None
    except ValueError:
        year_int = None 
        
    doc_types_to_load = {dt.lower() for dt in doc_types or DEFAULT_DOC_TYPES if dt.lower() in [s.lower() for s in SUPPORTED_DOC_TYPES]}
    if not doc_types_to_load: doc_types_to_load = {dt.lower() for dt in DEFAULT_DOC_TYPES}
    
    enabler_req = enabler.upper() if enabler else None
    
    # กำหนด Context (dt, ena, yr) ที่ต้องโหลด Mapping DB และ Scan File
    load_contexts: Set[Tuple[str, Optional[str], Optional[int]]] = set()
    
    for dt in doc_types_to_load:
        # 🎯 แก้ไข 1: Case 1 เหลือไว้แค่ EVIDENCE_DOC_TYPES เท่านั้นที่ต้องมี Year/Enabler
        if dt.lower() == EVIDENCE_DOC_TYPES.lower(): 
            # ต้องมีปี
            if year_int is not None and year_int > 0:
                 # ถ้ากำหนด Enabler
                 if enabler_req and enabler_req in SUPPORTED_ENABLERS:
                    load_contexts.add((dt, enabler_req, year_int))
                 else:
                    # Scan All Enablers for the year
                    for ena in SUPPORTED_ENABLERS:
                        load_contexts.add((dt, ena, year_int))
            else:
                 logger.warning(f"⚠️ Cannot list {dt} without a specific year. Skipping {dt} listing.")
        else:
            # Case 2: Global Doc Types (document, faq, seam, other): Year และ Enabler ควรเป็น None
            load_contexts.add((dt, None, None)) 

    # --- 2. Load Doc ID Mapping Files (using load_contexts) ---
    doc_mapping_db: Dict[str, Dict[str, Any]] = {}
    # สมมติว่า load_doc_id_mapping ดำเนินการและคืนค่า Dict ของ UUID: Entry
    for dt, ena, yr in load_contexts:
        ena_code = ena
        try:
            # NOTE: load_doc_id_mapping ต้องถูก Import/Defined
            mapping_db = load_doc_id_mapping(
                doc_type=dt,
                tenant=tenant,
                year=yr,
                enabler=ena_code
            )
            doc_mapping_db.update(mapping_db)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"❌ Error loading mapping for {dt} / {ena_code or 'None'} / Year {yr or 'None'}: {e}")

    # --- 3. Filtering Doc ID Mapping for Physical Scan ---
    filepath_to_stable_uuid: Dict[str, str] = {}
    for s_uuid, entry in doc_mapping_db.items():
        if "filepath" not in entry or str(entry.get("tenant")).lower() != str(tenant).lower():
            continue
            
        doc_type_in_map = entry.get("doc_type", "").lower()
        if doc_type_in_map not in doc_types_to_load:
            continue
            
        doc_year_in_map = entry.get("year")
        is_context_match = False
        
        # 🎯 แก้ไข 2: DocType ที่ขึ้นกับ Year และ Enabler เหลือไว้แค่ EVIDENCE_DOC_TYPES
        if doc_type_in_map == EVIDENCE_DOC_TYPES.lower():
            # Case 1: Year-Specific Type (Evidence): ต้องตรงกับปีที่ร้องขอ (year_int > 0)
            if year_int is not None and year_int > 0 and doc_year_in_map == year_int:
                is_context_match = True
        else:
            # Case 2: Global Types (รวมถึง 'seam'): ต้องไม่มีส่วนประกอบของปีใน mapping (year=None หรือ 0)
            if doc_year_in_map in [None, 0]:
                is_context_match = True
                
        if is_context_match:
            filepath_to_stable_uuid[entry["filepath"]] = s_uuid

    # --- 4. Physical File Scan and Status Check (using load_contexts) ---
    all_docs: Dict[str, Any] = {}
    
    # โหลดไฟล์ที่มีอยู่จริงในระบบ
    for dt_lower, resolved_enabler, resolved_year in load_contexts:
        
        # 🟢 FIX 1: เพิ่มการตรวจสอบเพื่อข้ามบริบทที่ Doc Type เป็น None
        if dt_lower is None:
            logger.error(f"❌ Critical Error: Found None in load_contexts for Doc Type. Skipping context: Enabler={resolved_enabler or 'None'}, Year={resolved_year or 'None'}")
            continue
            
        # 🎯 FIX 2 (Argument Swap): แก้ไขการสลับอาร์กิวเมนต์: (tenant, year, enabler, doc_type)
        # NOTE: get_document_source_dir ต้องถูก Import/Defined
        scan_dir = get_document_source_dir(tenant, resolved_year, resolved_enabler, dt_lower)
        
        if not os.path.exists(scan_dir):
            logger.warning(f"Source directory not found: {scan_dir}. Skipping scan.")
            continue
            
        for root, dirs, filenames in os.walk(scan_dir):
             dirs[:] = [d for d in dirs if d not in ['.DS_Store', '__pycache__', 'backup']]
             
             for f in filenames:
                file_path = os.path.join(root, f)
                file_extension = os.path.splitext(f)[1].lower()
                
                if f.startswith('.') or file_extension not in SUPPORTED_TYPES: continue

                # 🎯 FIX 3 (UnboundLocalError): คำนวณ filename_doc_id_key เสมอ
                raw_doc_id_input = os.path.splitext(f)[0]
                # NOTE: _normalize_doc_id ต้องถูก Import/Defined
                filename_doc_id_key = _normalize_doc_id(raw_doc_id_input) 

                # Determine status from mapping DB
                stable_doc_uuid = filepath_to_stable_uuid.get(file_path)
                
                if stable_doc_uuid and stable_doc_uuid in doc_mapping_db:
                    mapping_entry = doc_mapping_db[stable_doc_uuid]
                    chunk_count = mapping_entry.get("chunk_count", 0)
                    original_doc_type = mapping_entry.get("doc_type", dt_lower)
                    
                else:
                    # File exists but not in mapping DB (Pending Ingestion)
                    stable_doc_uuid = None
                    chunk_count = 0
                    original_doc_type = dt_lower # Use resolved doc type from scan context

                # Calculate temporary values if needed
                if stable_doc_uuid and chunk_count == 0:
                    chunk_count = 1 # Assume 1 chunk if mapped but count is 0
                    
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
                    "enabler": resolved_enabler, # int/None
                    "tenant": tenant,
                    "year": resolved_year, # int/None
                    "upload_date": upload_date,
                    "chunk_count": chunk_count,
                    "status": "Ingested" if is_ingested else "Pending",
                    "size": file_size,
                }
                all_docs[final_doc_id] = doc_info

    # --- 5. Final Filtering and Data Preparation ---
    total_supported_files = len(all_docs)
    show_results_lower = show_results.lower()
    filtered_docs_dict: Dict[str, Any] = {}
    
    # 🎯 FIX 4 (Pydantic): List สำหรับพิมพ์ผลเท่านั้น
    display_list_for_print: List[Dict[str, Any]] = []

    # กำหนด String สำหรับแสดง Year
    year_request_str = str(year_int) if year_int is not None and year_int > 0 else "Global"

    if total_supported_files == 0:
        doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
        logger.warning(f"⚠️ No documents found in DATA_DIR matching the requested type '{doc_types_str}' (Enabler: {enabler_req or 'ALL'}, Year: {year_request_str}) for {tenant}.")
        return filtered_docs_dict

    for doc_id, info in all_docs.items():
        info_status_lower = info['status'].lower()
        
        # Filtering logic
        if show_results_lower == "all":
            is_match = True
        elif show_results_lower == "ingested" and info_status_lower == "ingested":
            is_match = True
        elif show_results_lower == "pending" and info_status_lower == "pending":
            is_match = True
        elif show_results_lower == "failed" and info_status_lower.startswith("failed"):
            is_match = True
        else:
            is_match = False
            
        if is_match:
            # 🎯 สร้างสำเนาสำหรับ display และแปลง int/None เป็น string '-'
            display_info = info.copy()
            
            display_info["size_mb"] = display_info["size"] / (1024 * 1024)
            display_info["year_display"] = display_info["year"] if display_info["year"] is not None else "-"
            display_info["enabler_display"] = display_info["enabler"] if display_info["enabler"] is not None else "-"
            
            display_list_for_print.append(display_info)
            
            # เก็บข้อมูลต้นฉบับ (info) ที่ year เป็น int/None ไว้สำหรับ Pydantic
            filtered_docs_dict[doc_id] = info


    if not display_list_for_print:
        logger.info(f"--- Found {total_supported_files} supported files but none matched the filter criteria to display ---")
        return filtered_docs_dict
        
    # --- 6. Print Results ---
    
    # Sort for consistent display
    display_list_for_print.sort(key=lambda x: (x['doc_type'], x['enabler_display'], x['filename']))

    UUID_COL_WIDTH = 65
    NEW_TABLE_WIDTH = 197
    
    print("-" * NEW_TABLE_WIDTH)
    print(f"--- Document List (Tenant: {tenant.upper()}, Year: {year_request_str}, Filter: {show_results.upper()}) ---")
    print("-" * NEW_TABLE_WIDTH)
    print(f"{'DOC ID (Stable/Temp)':<{UUID_COL_WIDTH}} | {'TENANT':<7} | {'YEAR':<4} | {'FILENAME':<30} | {'EXT':<5} | {'TYPE':<10} | {'ENB':<5} | {'SIZE(MB)':<9} | {'STATUS':<10}")
    print("-" * NEW_TABLE_WIDTH)
    
    for info in display_list_for_print:
        full_doc_id = info['doc_id']
        file_name, file_ext = os.path.splitext(info['filename'])
        short_filename = file_name[:28] if len(file_name) > 28 else file_name
        file_ext = file_ext[1:].upper() if file_ext else '-'
        size_str = f"{info['size_mb']:.2f}"
        
        # ใช้ตัวแปรสำหรับ display
        enabler_display = info['enabler_display'] 
        year_display = info['year_display']
        
        # Truncate UUID if too long
        display_doc_id = full_doc_id
        if len(display_doc_id) > UUID_COL_WIDTH:
             display_doc_id = display_doc_id[:UUID_COL_WIDTH-3] + "..."

        print(
            f"{display_doc_id:<{UUID_COL_WIDTH}} | "
            f"{info['tenant']:<7} | "
            f"{year_display:<4} | "
            f"{short_filename:<30} | "
            f"{file_ext:<5} | "
            f"{info['doc_type'][:10]:<10} | "
            f"{enabler_display:<5} | "
            f"{size_str:<9} | "
            f"{info['status'][:10]:<10}"
        )
        
        if len(file_name) > 28:
            # 🎯 FIX 5 (Format Specifier): แก้ไข '>30-3' เป็น '>27'
            print(f"{'':<{UUID_COL_WIDTH}} | {'':<7} | {'':<4} | ...{file_name[28:]:>27}")

    print("-" * NEW_TABLE_WIDTH)
    logger.info(f"--- Displaying {len(display_list_for_print)} documents of {total_supported_files} supported files ---")
    
    return filtered_docs_dict # คืนค่า Dict ของ doc_id: info

# -------------------- Main Execution --------------------

if __name__ == "__main__":
    try:
        import argparse
        
        # ... (Argument Parser setup - โค้ดเดิม) ...
        parser = argparse.ArgumentParser(description="Multi-Tenant RAG Ingestion and Management Tool (SE-AM Ready)")
        
        # Global Settings
        parser.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant code (e.g., 'pea', 'pwa').")
        parser.add_argument("--year", type=str, default=str(DEFAULT_YEAR), help="Assessment year (e.g., '2568').")
        parser.add_argument("--doc-type", nargs='+', default=[EVIDENCE_DOC_TYPES], help="Document type(s) to process ('evidence', 'document', 'all').")
        parser.add_argument("--enabler", type=str, default=DEFAULT_ENABLER, help="Enabler code (e.g., 'KM', 'HCM').")
        parser.add_argument("--subject", type=str, default=None, help="Subject/Topic tag for documents (optional).")

        # Ingest Mode
        parser.add_argument("--ingest", action="store_true", help="Run ingestion mode.")
        parser.add_argument("--dry-run", action="store_true", help="Simulate ingestion without writing to Chroma/DB.")
        parser.add_argument("--sequential", action="store_true", help="Run ingestion in sequential mode (for debugging).")
        parser.add_argument("--skip-wipe", action="store_true", help="Skip the wiping of vector store before ingestion.")

        # List Mode
        parser.add_argument("--list", action="store_true", help="Run document listing mode.")
        parser.add_argument("--show-results", type=str, default="all", choices=["all", "ingested", "pending"], help="Filter results for list mode.")
        
        # Wipe Mode
        parser.add_argument("--wipe", action="store_true", help="Wipe (delete) vector store and mapping files for the specified context.")
        
        args = parser.parse_args()
        
        logger.info(f"--- STARTING EXECUTION: Tenant={args.tenant}, Year={args.year}, DocType={args.doc_type}, Enabler={args.enabler} ---")
        
        doc_type_for_ingest_wipe = args.doc_type[0] if isinstance(args.doc_type, list) and args.doc_type else DEFAULT_DOC_TYPES

        if args.ingest:
            logger.info("--- INGESTION MODE ACTIVATED ---")
            
            if not args.skip_wipe and not args.dry_run:
                # 1. WIPE LOGIC (Optional)
                logger.warning("⚠️ Wiping Vector Store before ingestion!!!")
                wipe_vectorstore(
                    doc_type_to_wipe=doc_type_for_ingest_wipe,
                    enabler=args.enabler, 
                    tenant=args.tenant, 
                    year=int(args.year) if args.year.isdigit() else DEFAULT_YEAR
                )
            
            # 2. INGEST LOGIC
            ingest_all_files(
                tenant=args.tenant,
                year=int(args.year) if args.year.isdigit() else DEFAULT_YEAR,
                doc_type=doc_type_for_ingest_wipe,
                enabler=args.enabler,
                subject=args.subject, 
                dry_run=args.dry_run,
                sequential=args.sequential
            )
            
        elif args.list:
            logger.info("--- LIST MODE ACTIVATED ---\n")
            
            # 3. LIST LOGIC
            list_documents(
                doc_types=args.doc_type, # list_documents รับเป็น list ของ doc_type ได้
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=args.year, # ส่งเป็น string เพื่อให้ list_documents จัดการแปลงเป็น int/None เอง
                show_results=args.show_results 
            )
            
        elif args.wipe:
            logger.info("--- WIPE MODE ACTIVATED ---")
            logger.critical("⚠️ Wiping Vector Store and Mapping Files as requested!!!")
            wipe_vectorstore(
                doc_type_to_wipe=doc_type_for_ingest_wipe,
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=int(args.year) if args.year.isdigit() else DEFAULT_YEAR
            )
            
        else:
            print("\nUsage: Specify --ingest, --list, or --wipe mode.")
            parser.print_help()

        logger.info("Execution finished.")
        
    except ImportError:
         print("--- RUNNING SCRIPT STANDALONE FAILED: Missing argparse module ---")
         
    except Exception as e:
         logger.critical(f"FATAL ERROR DURING MAIN EXECUTION: {e}", exc_info=True)
         print(f"--- FATAL ERROR: Check ingest.log for details. ---")