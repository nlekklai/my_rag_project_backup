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
from typing import List, Optional, Set, Iterable, Dict, Any, Union, Tuple, TypedDict, Literal # 🟢 FIX: เพิ่ม Literal
import pandas as pd
import numpy as np
from pydantic import ValidationError
from collections import defaultdict # 🟢 FIX 1: เพิ่ม defaultdict


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
    EMBEDDING_MODEL_NAME,
    DATA_STORE_ROOT,
    SUPPORTED_DOC_TYPES
)

# -------------------- [NEW] Import Path Utilities --------------------
from utils.path_utils import (
    get_document_source_dir,
    get_doc_type_collection_key,
    get_vectorstore_collection_path,
    get_mapping_file_path,
    get_vectorstore_tenant_root_path, # ใช้สำหรับ wipe
    get_evidence_mapping_file_path, # ใช้สำหรับ Evidence Map
    load_doc_id_mapping,
    save_doc_id_mapping,
    # 💡 FIX: เพิ่ม load/save_evidence_mapping
    load_evidence_mapping,
    save_evidence_mapping,
    get_normalized_metadata,
    create_stable_uuid_from_path,
    parse_collection_name,
    get_mapping_tenant_root_path,
    _update_evidence_mapping,
    get_mapping_key_from_physical_path
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
            # 💡 FIX: เพิ่ม csv_args={"delimiter": "|"} เพื่อรองรับการใช้ Pipe
            # 📌 หมายเหตุ: คุณจะต้องสร้างไฟล์ FAQ .csv โดยใช้ | เป็นตัวแบ่งแทน ,
            loader = loader_class(
                file_path, 
                encoding='utf-8', 
                csv_args={"delimiter": "|"} 
            )
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
# load_and_chunk_document – เวอร์ชันสมบูรณ์สุดสำหรับ SE-AM (ใช้ Key เดียว: chunk_uuid)
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
        # สมมติว่า loader_func มีการจัดการ OCR และ Error ได้ดี
        raw_docs = loader_func(file_path)
    except Exception as e:
        # Handle exceptions including ValidationError (ถ้ามี)
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
    # base_metadata ที่นี่คือ base_metadata ที่จะถูกส่งผ่านไปทุก chunk
    base_metadata = {
        "doc_type": doc_type,
        "doc_id": stable_doc_uuid, # ใช้ doc_id เป็นชื่อหลักในการอ้างถึง Stable UUID
        "stable_doc_uuid": stable_doc_uuid, # ยังคงเก็บไว้ในชื่อเดิม (เผื่อใช้)
        "source_filename": os.path.basename(file_path),
        "source": os.path.basename(file_path), # อาจจะเปลี่ยนเป็น Path ที่ clean กว่านี้
        "version": version,
    }
    if enabler: base_metadata["enabler"] = enabler
    if subject: base_metadata["subject"] = subject.strip()
    if year: base_metadata["year"] = year
    
    # รวม injected_metadata ที่ส่งมาจาก process_document
    if metadata: 
        base_metadata.update(metadata) 

    for d in docs:
        d.metadata.update(base_metadata)
        d.metadata = _safe_filter_complex_metadata(d.metadata)

    # --- Split into chunks ---
    try:
        # TEXT_SPLITTER ควรเป็น LangChain RecursiveCharacterTextSplitter หรือ seggmenter ที่คล้ายกัน
        chunks = TEXT_SPLITTER.split_documents(docs)
    except Exception as e:
        logger.error(f"Split failed: {e}")
        chunks = docs

    # --- Clean text & Inject per-chunk metadata ---
    final_chunks = []
    # 💡 FIX: ใช้ start=1 เหมือนเดิม และใช้ format string เพื่อให้ index มีความยาวคงที่ (เช่น 0001)
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
            page_val = detected["page_number"]
            chunk.metadata["page_number"] = page_val
            chunk.metadata["page"] = f"P{page_val}"  # แสดงเป็น P45 ทันที!

        # 🟢 CRITICAL FIX: ใช้ Key "chunk_uuid" เป็น Key หลักเพียง Key เดียว
        chunk_id_prefix = stable_doc_uuid[:16] # ใช้แค่ 16 ตัวแรกของ UUID ก็เพียงพอ
        
        # 1. กำหนด ID หลักเป็น chunk_uuid
        unique_chunk_id = f"{chunk_id_prefix}-{idx:04d}" 
        chunk.metadata["chunk_uuid"] = unique_chunk_id
        
        # 2. ✅ ลบ chunk_id ทิ้งเพื่อความสะอาดและใช้ Key เดียว
        if "chunk_id" in chunk.metadata:
            del chunk.metadata["chunk_id"]
        
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
    base_path: str = "", # 💡 FIX: เปลี่ยน Default จาก VECTORSTORE_DIR เป็น String ว่างเปล่า
    year: Optional[int] = None,
    tenant: Optional[str] = None, 
    version: str = "v1",
    metadata: dict = None,
    source_name_for_display: Optional[str] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> Tuple[List[Document], str, str]: 
    
    # 📌 ASSUME: _normalize_doc_id, DEFAULT_DOC_TYPES, EVIDENCE_DOC_TYPES, DEFAULT_ENABLER ถูก Import อย่างถูกต้อง
    raw_doc_id_input = os.path.splitext(file_name)[0]
    filename_doc_id_key = _normalize_doc_id(raw_doc_id_input) 
            
    doc_type = doc_type or DEFAULT_DOC_TYPES
    
    resolved_enabler = None
    if doc_type.lower() == EVIDENCE_DOC_TYPES.lower():
        resolved_enabler = (enabler or DEFAULT_ENABLER).upper()

    # 🟢 รวบรวม Metadata ที่จำเป็นทั้งหมดไว้ใน injected_metadata ณ จุดนี้
    injected_metadata = metadata or {}
    
    # 1. ข้อมูลที่ถูก Resolve
    injected_metadata["doc_type"] = doc_type
    injected_metadata["original_stable_id"] = filename_doc_id_key[:32].lower() # ใช้เป็น Reference ID
    
    if resolved_enabler:
        injected_metadata["enabler"] = resolved_enabler
    if tenant: 
        injected_metadata["tenant"] = tenant
        
    # 💡 FIX: ต้องเพิ่ม year เข้าไปใน injected_metadata ด้วย (ถ้ามีค่า)
    if year is not None: 
        injected_metadata["year"] = year
        
    if subject: 
        injected_metadata["subject"] = subject
        
    filter_id_value = filename_doc_id_key 
    logger.critical(f"================== START DEBUG INGESTION: {file_name} ==================")
    logger.critical(f"🔍 DEBUG ID (stable_doc_uuid, 64-char Hash): {len(stable_doc_uuid)}-char: {stable_doc_uuid[:34]}...")
    logger.critical(f"✅ FINAL ID TO STORE (34-char Ref ID): {len(filter_id_value)}-char: {filter_id_value[:34]}...")

    # 🎯 ส่ง Metadata ทั้งหมดผ่าน dict ไปให้ load_and_chunk_document
    chunks = load_and_chunk_document(
        file_path=file_path,
        stable_doc_uuid=stable_doc_uuid,
        doc_type=doc_type, 
        enabler=resolved_enabler, 
        subject=subject, 
        # base_path ไม่ถูกส่งแล้ว เพราะไม่จำเป็น
        version=version,
        metadata=injected_metadata, # <--- **สำคัญ:** มี year อยู่ในนี้แล้ว
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
    base_path: str = "" # 💡 FIX: แก้ไข VECTORSTORE_DIR ที่ไม่ได้นิยาม ให้เป็น String ว่างเปล่า
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
        # 🎯 REVISED: ใช้ parse_collection_name จาก path_utils.py
        doc_type_for_path, enabler_for_path = parse_collection_name(collection_name)
        
        # 🎯 FIX: ใช้ get_vectorstore_collection_path จาก path_utils.py
        persist_directory = get_vectorstore_collection_path(
            tenant=tenant,
            # Path Utility จะตัดสินใจใช้ year/enabler ก็ต่อเมื่อ doc_type เป็น Evidence
            year=year, 
            doc_type=doc_type_for_path,
            enabler=enabler_for_path
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to generate vectorstore path using path_utils: {e}. Using simple fallback path.")
        # Fallback Path ที่ไม่มี Dependency กับ Global Constant เดิม
        persist_directory = os.path.join(tenant, str(year), collection_name)
        logger.warning(f"⚠️ Warning: Fallback path used. Result: {persist_directory}")

    cache_key = persist_directory

    # === 3. Cache HIT ===
    if cache_key in _VECTORSTORE_SERVICE_CACHE:
        logger.debug(f"Cache HIT → Reusing vectorstore: {persist_directory}")
        return _VECTORSTORE_SERVICE_CACHE[cache_key]

    # === 4. Embedding model (แชร์ตัวเดียวตลอด process) ===
    embeddings = _VECTORSTORE_SERVICE_CACHE.get("embeddings_model")

    if not embeddings:
        # 📌 ASSUME: EMBEDDING_MODEL_NAME ถูก Import จาก config/global_vars
        logger.info(f"กำลังโหลด {EMBEDDING_MODEL_NAME} (SOTA Multilingual 2024) เพื่อปรับปรุง Retrieval")

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name= EMBEDDING_MODEL_NAME,
                model_kwargs={
                    "device": "cpu", # เปลี่ยนเป็น "cuda" ถ้ามี GPU 
                },  
                encode_kwargs={
                    "normalize_embeddings": True, 
                    "batch_size": 32,
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
    # สร้าง directory ตาม Path ที่คำนวณจาก Path Utility (รวมถึง DATA_STORE_ROOT)
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

def ingest_all_files(
    doc_types: List[str],
    tenant: str = DEFAULT_TENANT,
    year: Optional[Union[str, int]] = None,
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    dry_run: bool = False,
    sequential: bool = False,
    skip_ext: Optional[List[str]] = None,
) -> Dict[str, Any]:
    logger.info("--- STARTING INGESTION PROCESS ---")
    import unicodedata
    from collections import defaultdict # ต้องมั่นใจว่ามีการ import defaultdict ใน core/ingest.py

    tenant_clean = unicodedata.normalize('NFKC', tenant.lower().replace(" ", "_"))

    files_to_process: List[Dict[str, Any]] = []
    context_to_files: Dict[Tuple[str, Optional[str], Optional[int]], List[Dict]] = defaultdict(list)

    # คำนวณ context ใหม่ทุก doc_type ← จุดสำคัญที่สุด
    for dt in doc_types:
        dt_lower = dt.lower()

        resolved_year, resolved_enabler = get_normalized_metadata(
            doc_type=dt_lower,
            year_input=year,
            enabler_input=enabler,
            default_enabler=DEFAULT_ENABLER,
        )

        # Evidence ต้องมี year เสมอ
        if dt_lower == EVIDENCE_DOC_TYPES.lower() and resolved_year is None:
            logger.error("Evidence requires --year. Skipping evidence ingestion.")
            continue

        root_path = get_document_source_dir(tenant_clean, resolved_year, resolved_enabler, dt_lower)
        collection_name = get_doc_type_collection_key(dt_lower, resolved_enabler)

        logger.info(f" [SCAN] '{dt_lower}' → Collection: {collection_name} | Path: {root_path}")

        if not os.path.exists(root_path):
            logger.warning(f"Directory not found: {root_path}")
            continue

        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in ['.DS_Store', '__pycache__', 'backup']]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if f.startswith('.') or ext not in SUPPORTED_TYPES:
                    continue
                if skip_ext and ext in skip_ext:
                    continue

                file_path_abs = os.path.join(root, f)

                # สร้าง UUID เสถียรด้วย NFKC + relative path
                # try:
                #     rel_path = os.path.relpath(file_path_abs, DATA_STORE_ROOT)
                # except ValueError:
                #     rel_path = file_path_abs
                # normalized_path = unicodedata.normalize('NFKC', rel_path)
                # stable_doc_uuid = create_stable_uuid_from_path(normalized_path)
    
                stable_doc_uuid = create_stable_uuid_from_path(
                    file_path_abs, 
                    tenant=tenant_clean, 
                    year=resolved_year, 
                    enabler=resolved_enabler
                )

                info = {
                    "file_path": file_path_abs,
                    "file_name": f,
                    "doc_type": dt_lower,
                    "enabler": resolved_enabler,
                    "year": resolved_year,
                    "tenant": tenant_clean,
                    "stable_doc_uuid": stable_doc_uuid,
                    "collection_name": collection_name,
                }

                files_to_process.append(info)
                context_to_files[(dt_lower, resolved_enabler, resolved_year)].append(info)

    if not files_to_process:
        logger.warning("--- NO FILES FOUND ---")
        return {"updated_count": 0}

    # โหลด mapping ทุก context
    full_mapping: Dict[str, Dict] = {}
    for ctx in context_to_files:
        dt, ena, yr = ctx
        try:
            full_mapping.update(load_doc_id_mapping(dt, tenant_clean, yr, ena))
        except FileNotFoundError:
            pass

    # กรองไฟล์ที่ต้อง ingest
    files_to_ingest = [
        f for f in files_to_process
        if f["stable_doc_uuid"] not in full_mapping
        or full_mapping[f["stable_doc_uuid"]].get("chunk_count", 0) == 0
    ]

    logger.info(f"Will ingest {len(files_to_ingest)} files.")

    # Process + Index (เหมือนเดิม)
    chunks_by_collection: Dict[str, List[Document]] = defaultdict(list)
    results: List[Dict] = []

    def process_one(info: Dict):
        # NOTE: process_document ต้องเรียก load_and_chunk_document ที่แก้ไขแล้ว
        chunks, doc_uuid, _ = process_document(
            file_path=info["file_path"],
            file_name=info["file_name"],
            stable_doc_uuid=info["stable_doc_uuid"],
            doc_type=info["doc_type"],
            enabler=info["enabler"],
            tenant=tenant_clean,
            year=info["year"],
            subject=subject,
        )
        if chunks:
            for c in chunks:
                c.metadata.update({"tenant": tenant_clean, "year": info["year"]})
            chunks_by_collection[info["collection_name"]].extend(chunks)
            results.append({"file": info["file_name"], "doc_id": doc_uuid, "chunks": len(chunks)})

    if dry_run:
        return {"updated_count": 0}
    elif sequential:
        for info in files_to_ingest:
            process_one(info)
    else:
        with ThreadPoolExecutor() as ex:
            list(ex.map(process_one, files_to_ingest))

    # Index
    for coll, chunks in chunks_by_collection.items():
        if not chunks:
            continue
        # 1. Initialize Vector Store
        # ดึงปีจาก Chunk แรกเพื่อใช้กำหนด Vector Store (ถ้ามี)
        vs = get_vectorstore(coll, tenant_clean, chunks[0].metadata.get("year")) 
        
        total_chunks = len(chunks)
        
        # 2. Batch Indexing
        for i in range(0, total_chunks, 500):
            batch = chunks[i:i+500]
            
            # --- ✅ Log ---
            start_index = i + 1
            end_index = min(i + 500, total_chunks)
            logger.info(f"Indexing batch {start_index}-{end_index}/{total_chunks} chunks into Collection: {coll}")
            # ---------------------------
            
            vs.add_texts(
                texts=[c.page_content for c in batch],
                metadatas=[c.metadata for c in batch],
                # 🎯 FIX 1: เปลี่ยนมาใช้ chunk_uuid
                ids=[c.metadata["chunk_uuid"] for c in batch], 
            )
        
        # Log สรุปหลังจบ Collection นั้น
        logger.info(f"✅ Indexed {total_chunks} chunks successfully into Collection: {coll}")

    # Update mapping (แยก context)
    updated = 0
    for ctx, infos in context_to_files.items():
        dt, ena, yr = ctx
        mapping = load_doc_id_mapping(dt, tenant_clean, yr, ena)

        for info in infos:
            chunks = [c for c in chunks_by_collection[info["collection_name"]]
                     if c.metadata.get("doc_id") == info["stable_doc_uuid"]]
            if not chunks:
                continue

            rel_path = unicodedata.normalize('NFKC',
                os.path.relpath(info["file_path"], DATA_STORE_ROOT))

            mapping[info["stable_doc_uuid"]] = {
                "doc_id": info["stable_doc_uuid"],
                "file_name": info["file_name"],
                "filepath": rel_path,
                "doc_type": dt,
                "enabler": ena,
                "tenant": tenant_clean,
                "year": yr,
                "chunk_count": len(chunks),
                "status": "Ingested",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "size": os.path.getsize(info["file_path"]),
                # 🎯 FIX 2: บันทึก chunk_uuids สำหรับการลบ (Deletion)
                "chunk_uuids": [c.metadata["chunk_uuid"] for c in chunks], 
            }
            updated += 1

        save_doc_id_mapping(mapping, dt, tenant_clean, yr, enabler=ena)

    logger.info(f"--- INGESTION COMPLETE | Updated mapping: {updated} docs ---")
    return {"updated_count": updated, "results": results}

# -------------------- Wipe Vectorstore / Mapping --------------------
def wipe_vectorstore(
    doc_type_to_wipe: str = "all",
    enabler: Optional[str] = None,
    tenant: str = DEFAULT_TENANT,
    year: Union[int, str] = DEFAULT_YEAR, # รับ int หรือ str เข้ามา
    base_path: Optional[str] = None # ลบตัวแปรนี้ออกเพื่อให้โค้ด clean ขึ้น
) -> None:
    """
    Deletes the Vector Store collection(s) and associated mapping files.
    """
    logger.critical(f"⚠️ !!! เริ่มกระบวนการ WIPE Vectorstore และ Mapping Files !!!")
    
    doc_type_to_wipe_lower = doc_type_to_wipe.lower()
    enabler_req = enabler.upper() if enabler else None
    
    # 1. กำหนด Doc Types ที่ต้องตรวจสอบ
    doc_types_to_check: List[str] = []
    supported_doc_types_lower = [dt.lower() for dt in SUPPORTED_DOC_TYPES]
    
    if doc_type_to_wipe_lower == 'all':
        doc_types_to_check.extend(supported_doc_types_lower)
    elif doc_type_to_wipe_lower in supported_doc_types_lower:
        doc_types_to_check.append(doc_type_to_wipe_lower)
    else:
        logger.warning(f"⚠️ Invalid Doc Type '{doc_type_to_wipe}'. Skipping wipe.")
        return

    # 1a. วนลูปเพื่อหาบริบทที่ Normalize แล้วสำหรับแต่ละ Collection
    collections_to_delete: Set[Tuple[str, Optional[Union[str, int]], Optional[str], str]] = set() 
    
    year_int: Optional[int]
    try:
        year_int = int(year) if year is not None and str(year).isdigit() else None
    except ValueError:
        year_int = None
        
    for dt in doc_types_to_check:
        
        is_evidence = dt == EVIDENCE_DOC_TYPES.lower()
        
        enablers_to_iterate: List[Optional[str]] = []
        
        if is_evidence:
            if enabler_req:
                enablers_to_iterate.append(enabler_req)
            elif doc_type_to_wipe_lower == 'all' or (doc_type_to_wipe_lower == EVIDENCE_DOC_TYPES.lower() and not enabler_req):
                # ถ้าล้าง 'all' หรือ 'evidence' ทั้งหมด ให้วนลูป Enablers ที่รองรับ
                enablers_to_iterate.extend(SUPPORTED_ENABLERS)
        else:
            # Global Doc Types สนใจแค่ None
            enablers_to_iterate.append(None) 
            
        enablers_to_iterate = list(set(enablers_to_iterate)) # ลบซ้ำ

        if not enablers_to_iterate:
             continue
             
        for ena_req in enablers_to_iterate:
            
            final_year_wipe, final_enabler_wipe = get_normalized_metadata(
                doc_type=dt,
                year_input=year_int,
                enabler_input=ena_req, 
                default_enabler=DEFAULT_ENABLER
            )
            
            # กรองบริบทที่ไม่ได้ถูกใช้ (เช่น Evidence ที่ไม่มี Enabler/Year)
            if is_evidence and (final_year_wipe is None or final_enabler_wipe is None):
                logger.debug(f"Skipping wipe context: {dt}/{final_enabler_wipe}/{final_year_wipe} (Missing year/enabler for evidence).")
                continue
                
            # เพิ่มบริบทที่ Normalize แล้วเข้าไปใน Set
            col_name = get_doc_type_collection_key(dt, final_enabler_wipe)
            collections_to_delete.add((dt, final_year_wipe, final_enabler_wipe, col_name))


    if not collections_to_delete:
        logger.warning(f"⚠️ ไม่พบ Collection ที่ตรงกับเงื่อนไข Tenant='{tenant}', Year='{year}', Doc Type='{doc_type_to_wipe}', Enabler='{enabler}'. Skipping wipe.")
        return

    # 2. ลบ Collection Folder และปรับปรุง Doc ID Mapping DB
    tenant_clean = tenant.lower().replace(" ", "_")
    
    for dt, map_year_to_use, map_enabler_to_use, col_name in collections_to_delete:
        
        # 2a. ลบ Vector Store Folder
        try:
            # ใช้ค่าที่ Normalize แล้วในการคำนวณ Path
            collection_path = get_vectorstore_collection_path(tenant_clean, map_year_to_use, dt, map_enabler_to_use)
            
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
                logger.info(f"🗑️ Deleted Collection Folder: {col_name} at {collection_path}")
            else:
                logger.info(f"Collection Folder ไม่พบ: {col_name} ({collection_path}).")
        except Exception as e:
            logger.error(f"❌ Error deleting vectorstore folder {col_name}: {e}")

        # 2b. ลบ Entry จาก Doc ID Mapping
        
        # โหลด Mapping ที่ถูกต้องตามบริบท (ซึ่งอาจมี Entry อื่นๆ อยู่)
        try:
            doc_mapping_db = load_doc_id_mapping(dt, tenant_clean, map_year_to_use, map_enabler_to_use) 
        except FileNotFoundError:
             logger.info(f"Mapping file for {dt}/{map_enabler_to_use}/{map_year_to_use} not found. Skipping mapping update.")
             continue
        except Exception as e:
            logger.error(f"❌ Error loading mapping file for {dt}/{map_enabler_to_use}/{map_year_to_use}: {e}. Skipping mapping update.")
            continue
            
        uuids_to_keep = {}
        
        # กรอง Entry ที่ไม่เกี่ยวข้องกับ Collection ที่เพิ่งถูกลบ
        for s_uuid, entry in doc_mapping_db.items():
            
            entry_doc_type = entry.get('doc_type', dt).lower()
            entry_enabler = entry.get('enabler')
            entry_tenant = entry.get("tenant", tenant_clean).lower()
            
            # การเปรียบเทียบปีต้องทำกับค่าที่ Normalize แล้ว (str/None)
            entry_year_str = str(entry.get("year")) if entry.get("year") is not None else None
            map_year_to_use_str = str(map_year_to_use) if map_year_to_use is not None else None

            # ใช้ get_doc_type_collection_key เพื่อตรวจสอบว่า Entry เป็นของ Collection นี้หรือไม่
            entry_col_name = get_doc_type_collection_key(entry_doc_type, entry_enabler)
            
            # เงื่อนไขการเก็บ: 
            # 1. Entry เป็นของ Tenant อื่น
            # 2. Entry ไม่ได้เป็นของ Collection ที่กำลังลบ (col_name) 
            # 3. Entry เป็นของปีอื่น (สำหรับ Evidence)
            
            is_match = (
                entry_tenant == tenant_clean and
                entry_col_name == col_name and
                (entry_year_str == map_year_to_use_str or dt != EVIDENCE_DOC_TYPES.lower()) # สำหรับ Global Docs ปีไม่สำคัญ
            )

            if not is_match:
                 uuids_to_keep[s_uuid] = entry
            else:
                logger.debug(f"Removed mapping entry for {s_uuid} (Collection: {col_name})")

        # บันทึกหรือลบไฟล์ Mapping
        removed_count = len(doc_mapping_db) - len(uuids_to_keep)
        if removed_count > 0:
            if not uuids_to_keep:
                mapping_path = get_mapping_file_path(tenant_clean, map_year_to_use, map_enabler_to_use) 
                if os.path.exists(mapping_path):
                    try:
                        os.remove(mapping_path)
                        logger.info(f"✅ Deleted (empty) Doc ID mapping file: {mapping_path}")
                    except OSError as e:
                        logger.error(f"❌ Error deleting mapping file: {e}")
            else:
                # บันทึกส่วนที่เหลือ
                save_doc_id_mapping(uuids_to_keep, dt, tenant_clean, map_year_to_use, enabler=map_enabler_to_use) 
                logger.info(f"✅ Saved updated mapping file for {dt}/{map_enabler_to_use}/{map_year_to_use}. Entries left: {len(uuids_to_keep)}.")
        
        logger.info(f"🧹 Removed {removed_count} entries from mapping file for deleted collection (Doc Type: {dt}/{map_enabler_to_use}) of {tenant}/{map_year_to_use}.")

        # 2c. ลบไฟล์ Evidence Mapping ถ้าเป็น Evidence
        if dt == EVIDENCE_DOC_TYPES.lower():
            # ใช้ค่าที่ Normalize แล้วในการหา Path
            if map_year_to_use is not None and map_enabler_to_use is not None:
                evidence_map_path = get_evidence_mapping_file_path(tenant_clean, map_year_to_use, map_enabler_to_use) 
                if os.path.exists(evidence_map_path):
                     try:
                        os.remove(evidence_map_path)
                        logger.info(f"✅ Deleted Evidence Mapping file: {evidence_map_path}")
                     except OSError as e:
                        logger.error(f"❌ Error deleting evidence mapping file: {e}")
            
    # 3. ลบ Root Directory ของ Vector Store และ Mapping หากว่างเปล่า
    # (โค้ดส่วนนี้ที่ส่งมาดูดีแล้ว)
    try:
        # Vectorstore cleanup
        tenant_root_path = get_vectorstore_tenant_root_path(tenant_clean) 
        if os.path.isdir(tenant_root_path):
            if not any(f for f in os.listdir(tenant_root_path) if not f.startswith('.')):
                try:
                    shutil.rmtree(tenant_root_path) 
                    logger.info(f"✅ Deleted empty Vector Store tenant directory: {tenant_root_path}")
                except OSError as e:
                    logger.debug(f"Vector Store directory {tenant_root_path} not empty or cannot be deleted: {e}")

        # Mapping cleanup
        mapping_dir_tenant = get_mapping_tenant_root_path(tenant_clean)
        if os.path.isdir(mapping_dir_tenant):
            if not any(f for f in os.listdir(mapping_dir_tenant) if not f.startswith('.')):
                try:
                    shutil.rmtree(mapping_dir_tenant)
                    logger.info(f"✅ Deleted empty Doc ID mapping tenant directory: {mapping_dir_tenant}")
                except OSError as e:
                    logger.debug(f"Mapping directory {mapping_dir_tenant} not empty or cannot be deleted: {e}")
            else: 
                logger.info(f"Mapping directory {mapping_dir_tenant} is not completely empty. Keeping.")
                
    except Exception as e:
        logger.warning(f"Error during final mapping directory cleanup: {e}")
        
    logger.critical("✅ !!! กระบวนการ WIPE Vectorstore และ Mapping Files เสร็จสิ้น !!!")

# -------------------- [REVISED] Document Management Utilities --------------------
def delete_document_by_uuid(
    stable_doc_uuid: str, 
    tenant: str = "pwa", 
    year: Union[int, str] = DEFAULT_YEAR, 
    collection_name: Optional[str] = None, 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
    base_path: Optional[str] = None
) -> bool:
    """Deletes all chunks associated with the given Stable Document UUID (Multi-Tenant/Year)."""
    if not doc_type:
        logger.error(f"Cannot delete {stable_doc_uuid}: doc_type must be provided for mapping file isolation.")
        return False
        
    tenant_clean = tenant.lower().replace(" ", "_")
    
    # 🎯 FIX 1: ใช้ get_normalized_metadata เพื่อหาบริบทที่ถูกต้อง
    doc_type_lower = doc_type.lower()
    
    year_int: Optional[int]
    try:
        year_int = int(year) if year is not None and str(year).isdigit() else None
    except ValueError:
        year_int = None
        
    final_year_for_map, final_enabler = get_normalized_metadata(
        doc_type=doc_type_lower,
        year_input=year_int,
        enabler_input=enabler,
        default_enabler=DEFAULT_ENABLER
    )

    # โหลด Mapping (ต้องมี try-except)
    try:
        doc_mapping_db = load_doc_id_mapping(doc_type_lower, tenant_clean, final_year_for_map, final_enabler) 
    except FileNotFoundError:
        logger.warning(f"Mapping file not found for context {doc_type_lower}/{final_year_for_map}/{final_enabler}. Cannot delete entry {stable_doc_uuid}.")
        return False
    except Exception as e:
        logger.error(f"❌ Error loading mapping file: {e}")
        return False

    entry = doc_mapping_db.get(stable_doc_uuid)
    if not entry:
        logger.warning(f"UUID {stable_doc_uuid} not found in mapping DB for {doc_type_lower}/{final_year_for_map}/{final_enabler}. No action taken.")
        return False

    # 📌 ใช้ metadata จาก Entry เพื่อความแม่นยำในการเรียก Vectorstore/Mapping
    final_doc_type = entry.get("doc_type", doc_type_lower)
    final_enabler_from_entry = entry.get("enabler", final_enabler)
    final_year_from_entry = entry.get("year", final_year_for_map)
    chunk_uuids = entry.get("chunk_uuids", [])
    
    if not chunk_uuids:
        logger.warning(f"No chunks found for UUID {stable_doc_uuid}. Deleting mapping entry only.")
        del doc_mapping_db[stable_doc_uuid]
        
    else:
        # 1. ลบ Chunks ออกจาก Vectorstore
        try:
            col_name = get_doc_type_collection_key(final_doc_type, final_enabler_from_entry)
            
            # 💡 ใช้ final_year_from_entry ในการเรียก get_vectorstore 
            # Note: final_year_from_entry จะเป็น None สำหรับ Global Docs
            vectorstore = get_vectorstore(col_name, tenant_clean, final_year_from_entry) 
            
            # Note: การลบใน ChromaDB ทำได้โดยใช้ ID
            vectorstore.delete(ids=chunk_uuids) 
            logger.info(f"✅ Deleted {len(chunk_uuids)} chunks for {stable_doc_uuid} from collection '{col_name}'.")
        except Exception as e:
            logger.error(f"❌ Failed to delete chunks from Vectorstore for {stable_doc_uuid}: {e}", exc_info=True)
            # ไม่ return False ถ้าลบไม่สำเร็จ แต่ให้ลบ entry ใน mapping DB ต่อไป
            
        # 🎯 FIX: ลบ Entry ออกจาก Mapping DB ที่โหลดมา
        del doc_mapping_db[stable_doc_uuid]
        
    # 📌 FIX: บันทึก Mapping DB คืน
    
    # 1. กรองเฉพาะ Entry ที่เป็นของบริบท Tenant/Year/Enabler นี้ (ที่โหลดมา)
    db_to_save: Dict[str, Dict[str, Any]] = {}
    
    # การเปรียบเทียบปีต้องทำกับค่าที่ Normalize แล้ว (str/None)
    final_year_for_map_str = str(final_year_for_map) if final_year_for_map is not None else None

    for s_uuid, entry_in_db in doc_mapping_db.items():
        # กรองเฉพาะ Entry ที่เป็นของไฟล์ Mapping ที่เรากำลังจะบันทึกคืน
        entry_year_str = str(entry_in_db.get("year")) if entry_in_db.get("year") is not None else None
        
        if (entry_in_db.get("doc_type", "").lower() == doc_type_lower and 
            str(entry_in_db.get("tenant")).lower() == tenant_clean and 
            entry_year_str == final_year_for_map_str and 
            entry_in_db.get("enabler") == final_enabler):
            
            db_to_save[s_uuid] = entry_in_db
            
    # 2. บันทึก Mapping DB คืน
    if db_to_save:
         save_doc_id_mapping(
            db_to_save, 
            doc_type_lower, 
            tenant_clean, 
            final_year_for_map, 
            enabler=final_enabler
        )
         logger.info(f"✅ Saved updated mapping DB for {doc_type_lower}/{final_enabler}/{final_year_for_map}. Entries remaining: {len(db_to_save)}.")
    else:
        # ลบไฟล์ mapping ถ้าไม่มี entry เหลืออยู่
        mapping_path = get_mapping_file_path(tenant_clean, final_year_for_map, final_enabler) 
        if os.path.exists(mapping_path):
            try:
                os.remove(mapping_path)
                logger.info(f"✅ Deleted (empty) Doc ID mapping file: {mapping_path}")
            except OSError as e:
                logger.error(f"❌ Error deleting mapping file: {e}")
        
    return True

# core/ingest.py: ฟังก์ชัน list_documents (แทนที่ของเดิมทั้งหมด)

def list_documents(
    doc_types: List[str],
    tenant: str = DEFAULT_TENANT,
    year: Optional[Union[str, int]] = None,
    enabler: Optional[str] = None,
    show_results: Literal["all", "missing", "ingested"] = "missing",
    skip_ext: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    List files and compare them with existing mapping database.
    (รวมการแก้ไข NFKC/NFD Path Matching อย่างถาวร)
    """
    logger.info("--- STARTING DOCUMENT LISTING ---")
    
    tenant_clean = unicodedata.normalize('NFKC', tenant.lower().replace(" ", "_"))
    
    load_contexts: Set[Tuple[str, Optional[str], Optional[Union[str, int]]]] = set()
    files_on_disk: List[Dict[str, Any]] = []

    # 1. Determine Contexts (Doc Type, Enabler, Year)
    for dt in doc_types:
        dt_lower = dt.lower()

        resolved_year, resolved_enabler = get_normalized_metadata(
            doc_type=dt_lower,
            year_input=year,
            enabler_input=enabler,
            default_enabler=DEFAULT_ENABLER,
        )
        
        # Evidence ต้องมีปี
        if dt_lower == EVIDENCE_DOC_TYPES.lower() and resolved_year is None:
            logger.error("Evidence requires --year. Skipping evidence listing.")
            continue
            
        load_contexts.add((dt_lower, resolved_enabler, resolved_year))
        
        root_path = get_document_source_dir(tenant_clean, resolved_year, resolved_enabler, dt_lower)
        logger.info(f" [SCAN] '{dt_lower}' Context: {dt_lower} / {resolved_enabler} / {resolved_year} | Path: {root_path}")

        # 2. Scan Files on Disk (Physical Path)
        if not os.path.exists(root_path):
            logger.warning(f"Directory not found: {root_path}")
            continue

        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in ['.DS_Store', '__pycache__', 'backup']]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if f.startswith('.') or ext not in SUPPORTED_TYPES:
                    continue
                if skip_ext and ext in skip_ext:
                    continue

                file_path_abs = os.path.join(root, f)
                
                # 📌 NEW: สร้าง UUID เสถียรจาก Physical Path
                stable_doc_uuid = create_stable_uuid_from_path(
                    file_path_abs,
                    tenant=tenant_clean,
                    year=resolved_year,
                    enabler=resolved_enabler
                )
                
                files_on_disk.append({
                    "doc_type": dt_lower,
                    "enabler": resolved_enabler,
                    "year": resolved_year,
                    "file_name": f,
                    "file_path_abs": file_path_abs,
                    "stable_doc_uuid": stable_doc_uuid,
                    "status": "MISSING",
                    "chunk_count": 0
                })

    if not files_on_disk:
        logger.warning("--- NO FILES FOUND ON DISK ---")
        return pd.DataFrame(columns=["Doc Type", "Enabler", "Year", "File Name", "Status", "Chunks"])

    # 3. Load Mappings (Saved Path)
    full_mapping: Dict[str, Dict] = {}
    # Key: Relative Key (tenant/data/...) -> Value: Stable UUID
    filepath_to_stable_uuid: Dict[str, str] = {} 
    
    for ctx in load_contexts:
        dt, ena, yr = ctx
        try:
            doc_mapping_db = load_doc_id_mapping(dt, tenant_clean, yr, ena)
            full_mapping.update(doc_mapping_db)
        except FileNotFoundError:
            continue
        
        for s_uuid, entry in doc_mapping_db.items():
            entry_context = (entry.get("doc_type", dt), entry.get("enabler", ena), entry.get("year", yr))
            
            # ตรวจสอบว่า entry นี้เป็นของ context ที่กำลังโหลดหรือไม่
            if entry_context in load_contexts: 
                 saved_filepath = entry["filepath"] # saved_filepath คือ relative path ที่ถูกเก็บไว้ใน mapping
                 
                 # 🟢 CRITICAL FIX: ใช้ get_mapping_key_from_physical_path เพื่อสร้าง Key จาก Saved Path
                 # Note: saved_filepath ที่ถูกเก็บไว้ใน mapping มักจะเป็น relative path ที่มีปัญหา NFD/NFKC
                 stable_lookup_key = get_mapping_key_from_physical_path(saved_filepath)
                 
                 if stable_lookup_key:
                     filepath_to_stable_uuid[stable_lookup_key] = s_uuid
                 else:
                     logger.warning(f"Could not create stable lookup key from saved path: {saved_filepath}")


    # 4. Compare Files on Disk with Mappings
    results: List[Dict] = []
    
    for info in files_on_disk:
        file_path_abs = info["file_path_abs"]
        
        # 🟢 CRITICAL FIX REVISED: Prepare lookup key from physical file (ใช้ Absolute Path ที่สแกนเจอ)
        # get_mapping_key_from_physical_path จะจัดการแปลง Path:
        # 1. Absolute Path
        # 2. NFKC Normalize
        # 3. Relative to DATA_STORE_ROOT
        # 4. Forward Slashes
        relative_key_candidate = get_mapping_key_from_physical_path(file_path_abs) 
        
        stable_doc_uuid = None
        if relative_key_candidate:
            # ค้นหา Stable UUID จาก Key ที่สร้างขึ้น (ซึ่งมาจาก saved_filepath ที่ถูก normalize แล้วใน Section 3)
            stable_doc_uuid = filepath_to_stable_uuid.get(relative_key_candidate)
        
        
        entry = None
        if stable_doc_uuid:
            entry = full_mapping.get(stable_doc_uuid)
        elif info["stable_doc_uuid"] in full_mapping:
             # Fallback: ตรวจสอบด้วย UUID ปกติ (กรณีไฟล์ไม่ถูก normalize ตอน ingest รอบเก่าๆ)
             entry = full_mapping.get(info["stable_doc_uuid"])

        if entry:
            info["status"] = entry.get("status", "Ingested")
            info["chunk_count"] = entry.get("chunk_count", 0)
            if info["chunk_count"] == 0:
                 info["status"] = "PENDING_REINGEST" # Chunk หาย ต้อง ingest ใหม่

        if show_results == "all" or (show_results == "missing" and info["status"] in ["MISSING", "PENDING_REINGEST"]) or (show_results == "ingested" and info["status"] == "Ingested"):
            results.append({
                "Doc Type": info["doc_type"].upper(),
                "Enabler": info["enabler"] or "-",
                "Year": info["year"] or "-",
                "File Name": info["file_name"],
                "Status": info["status"],
                "Chunks": info["chunk_count"],
                "UUID": info["stable_doc_uuid"]
            })

    logger.info(f"--- DOCUMENT LISTING COMPLETE | Total files found: {len(files_on_disk)} | Displayed results: {len(results)} ---")
    
    df = pd.DataFrame(results)
    if not df.empty:
        df.sort_values(by=["Doc Type", "Enabler", "Year", "File Name"], inplace=True)
    return df

# -------------------- Main Execution --------------------

if __name__ == "__main__":
    try:
        import argparse
        
        # -------------------- Argument Parser setup --------------------
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
        parser.add_argument("--yes", action="store_true", help="Bypass confirmation prompt for wiping (DANGER: use only when sure!).") 

        args = parser.parse_args()
        
        # -------------------- Pre-Command Setup & Validation --------------------
        
        # 1. Normalize doc_type
        # รับค่าแรกจาก list หรือใช้ DEFAULT_DOC_TYPES
        doc_type_for_ingest_wipe = args.doc_type[0].lower() if isinstance(args.doc_type, list) and args.doc_type else DEFAULT_DOC_TYPES[0].lower()
        
        # 2. Check Enabler for Evidence
        if doc_type_for_ingest_wipe == EVIDENCE_DOC_TYPES.lower() and (args.ingest or args.wipe or args.list) and not args.enabler:
            logger.error(f"When using '{EVIDENCE_DOC_TYPES.lower()}', you must specify --enabler.")
            sys.exit(1)

        logger.info(f"--- STARTING EXECUTION: Tenant={args.tenant}, Year={args.year}, DocType={args.doc_type}, Enabler={args.enabler} ---")
        
        # --- Handle all modes ---
        
        if args.ingest:
            logger.info("--- INGESTION MODE ACTIVATED ---")
            
            # 1. Prepare Normalized Year (int/None)
            year_to_use_ingest: Optional[Union[int, str]] = None
            try:
                year_to_use_ingest = int(args.year) if args.year and args.year.isdigit() and int(args.year) > 0 else None
            except ValueError:
                year_to_use_ingest = None
                
            # Global Doc Type ใช้ None สำหรับ year ถ้าไม่ได้ระบุปี
            if doc_type_for_ingest_wipe != EVIDENCE_DOC_TYPES.lower():
                 year_to_use_ingest = None 
            
            # 2. WIPE LOGIC (Optional)
            if not args.skip_wipe and not args.dry_run:
                logger.warning("⚠️ Wiping Vector Store before ingestion!!!")
                wipe_vectorstore(
                    doc_type_to_wipe=doc_type_for_ingest_wipe,
                    enabler=args.enabler, 
                    tenant=args.tenant, 
                    year=year_to_use_ingest # ใช้ปีที่ Normalize แล้ว
                )
            
            # 3. INGEST LOGIC
            # Note: ต้องมั่นใจว่า ingest_all_files รับ doc_type เป็น List[str]
            ingest_all_files(
                tenant=args.tenant,
                year=year_to_use_ingest,
                doc_types=args.doc_type, # ส่งเป็น List ที่รับมาจาก Argument
                enabler=args.enabler,
                subject=args.subject, 
                dry_run=args.dry_run,
                sequential=args.sequential
            )
            
        elif args.list:
            logger.info("--- LIST MODE ACTIVATED ---\n")
            
            # 3. LIST LOGIC
            list_documents(
                doc_types=[dt.lower() for dt in args.doc_type], 
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=args.year, # ส่งเป็น string, list_documents จัดการแปลง
                show_results=args.show_results 
            )
            
        elif args.wipe:
            logger.info("--- WIPE MODE ACTIVATED ---")
            logger.critical("⚠️ Wiping Vector Store and Mapping Files as requested!!!")
            
            # --- WIPE Confirmation ---
            if not args.yes:
                confirmation = input("Type 'YES' (all caps) to confirm deletion: ")
                if confirmation != "YES":
                    logger.info("Deletion cancelled.")
                    sys.exit(0)

            # 1. Prepare Normalized Year (int/None)
            year_to_use_wipe: Optional[Union[int, str]] = None
            try:
                year_to_use_wipe = int(args.year) if args.year and args.year.isdigit() and int(args.year) > 0 else None
            except ValueError:
                year_to_use_wipe = None
                
            if doc_type_for_ingest_wipe != EVIDENCE_DOC_TYPES.lower():
                year_to_use_wipe = None # Global Doc Type uses None for year
            
            # 2. Execute Vectorstore Wipe (ลบ Collection ภายใน Chroma)
            wipe_vectorstore(
                doc_type_to_wipe=doc_type_for_ingest_wipe,
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=year_to_use_wipe # ใช้ปีที่ Normalize แล้ว
            )
            
        else:
            print("\nUsage: Specify --ingest, --list, or --wipe mode.")
            parser.print_help()

        logger.info("Execution finished.")
        
    except ImportError:
         print("--- RUNNING SCRIPT STANDALONE FAILED: Missing necessary imports ---")
         
    except Exception as e:
         import traceback
         logger.critical(f"FATAL ERROR DURING MAIN EXECUTION: {e}", exc_info=True)
         print(f"--- FATAL ERROR: Check ingest.log for details... \n{traceback.format_exc()}")