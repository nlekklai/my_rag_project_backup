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
import numpy as np
from pydantic import ValidationError
from collections import defaultdict # 🟢 FIX 1: เพิ่ม defaultdict
import pandas as pd


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
    SUPPORTED_DOC_TYPES,
    MAX_PARALLEL_WORKERS
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
    get_mapping_key_from_physical_path,
    _update_doc_id_mapping
)
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("ingest.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)  # นี่คือตัวที่ทำให้เห็นบนจอ!
    ]
)

logger = logging.getLogger("IngestBatch")

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
                logger.warning(f"⚠️ Doc # {idx} from loader has no content (Empty/None). Skipping normalization for this document.")
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

        # เพิ่ม: พยายามได้ page_number จาก metadata ของ loader ก่อน
        page_from_meta = chunk.metadata.get("page")
        if page_from_meta is not None:
            try:
                page_val = int(page_from_meta) + 1  # สมมติว่า loader ใช้ 0-based
                chunk.metadata["page_number"] = page_val
                chunk.metadata["page"] = f"P{page_val}"
            except ValueError:
                pass

        # Detect sub_topic & page_number จาก text (override ถ้ามี)
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
    injected_metadata["original_stable_id"] = filename_doc_id_key[:32].lower()
    
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
    logger.info(f"================== START DEBUG INGESTION: {file_name} ==================")
    logger.info(f"🔍 DEBUG ID (stable_doc_uuid, 64-char Hash): {len(stable_doc_uuid)}-char: {stable_doc_uuid[:34]}...")
    logger.info(f"✅ FINAL ID TO STORE (34-char Ref ID): {len(filter_id_value)}-char: {filter_id_value[:34]}...")

    # 🎯 ส่ง Metadata ทั้งหมดผ่าน dict ไปให้ load_and_chunk_document
    chunks = load_and_chunk_document(
        file_path=file_path,
        stable_doc_uuid=stable_doc_uuid,
        doc_type=doc_type, 
        enabler=resolved_enabler, 
        subject=subject, 
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

# -------------------- [REVISED] Ingest all files --------------------
def ingest_all_files(
    tenant: str = DEFAULT_TENANT,
    year: Optional[Union[int, str]] = DEFAULT_YEAR,
    doc_types: List[str] = [EVIDENCE_DOC_TYPES],
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    dry_run: bool = False,
    sequential: bool = False,
    batch_size: int = 50, 
    ocr_pages: Optional[Iterable[int]] = None
) -> None:
    tenant_clean = unicodedata.normalize('NFKC', tenant.lower().replace(" ", "_"))

    logger.info("--- STARTING BATCH INGESTION ---")

    new_doc_id_entries: Dict[str, Dict[str, Any]] = {}

    total_chunks = 0
    total_docs = 0

    # 1. เก็บ Contexts ที่ต้อง Ingest
    load_contexts: Set[Tuple[str, Optional[str], Optional[int]]] = set()
    for dt in doc_types:
        dt_lower = dt.lower()
        
        # 🎯 FIX: ใช้ get_normalized_metadata เพื่อหา Resolved Year/Enabler
        resolved_year, resolved_enabler = get_normalized_metadata(
            doc_type=dt_lower,
            year_input=year,
            enabler_input=enabler,
            default_enabler=DEFAULT_ENABLER
        )
        
        if dt_lower == EVIDENCE_DOC_TYPES.lower() and resolved_year is None:
            logger.error(f"Skipping evidence doc_type: Year is required but none provided.")
            continue

        load_contexts.add((dt_lower, resolved_enabler, resolved_year))

    if not load_contexts:
        logger.warning("No valid contexts to ingest. Exiting.")
        return

    # 2. Scan ไฟล์ทั้งหมดจาก Disk ตาม Contexts
    files_to_ingest: List[Tuple[str, str, str, Optional[str], Optional[int], str]] = []  # (file_path, file_name, doc_type, enabler, year, stable_doc_uuid)

    for dt, ena, yr in load_contexts:
        root_path = get_document_source_dir(tenant_clean, yr, ena, dt)
        
        if not os.path.exists(root_path):
            logger.warning(f"Directory not found: {root_path}. Skipping context {dt}/{ena}/{yr}.")
            continue
        
        logger.info(f" [SCAN] {dt} | Enabler={ena} | Year={yr} | Path={root_path}")

        for root, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in ['.DS_Store', '__pycache__', 'backup']]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if f.startswith('.') or ext not in SUPPORTED_TYPES:
                    continue

                file_path_abs = os.path.join(root, f)
                
                # 🎯 FIX: ใช้ create_stable_uuid_from_path เพื่อสร้าง UUID ที่ stable กับ Tenant/Year/Enabler
                stable_doc_uuid = create_stable_uuid_from_path(file_path_abs, tenant=tenant_clean, year=yr, enabler=ena)

                files_to_ingest.append((file_path_abs, f, dt, ena, yr, stable_doc_uuid))

    if not files_to_ingest:
        logger.warning("--- NO FILES FOUND TO INGEST ---")
        return

    logger.info(f"Found {len(files_to_ingest)} files to ingest across all contexts.")

    # 3. Load existing mappings เพื่อกรองไฟล์ที่ ingested แล้ว (ใช้ UUID เป็น Key)
    existing_mappings: Dict[str, Dict] = {}
    for dt, ena, yr in load_contexts:
        try:
            mapping = load_doc_id_mapping(dt, tenant_clean, yr, ena)
            existing_mappings.update(mapping)
        except FileNotFoundError:
            pass
        
    # 4. กรองไฟล์ที่ยังไม่ ingested (ใช้ stable_doc_uuid เป็น Key)
    files_to_process = [
        (fp, fn, dt, ena, yr, s_uuid) for fp, fn, dt, ena, yr, s_uuid in files_to_ingest 
        if s_uuid not in existing_mappings or existing_mappings[s_uuid].get("status") != "Ingested" or existing_mappings[s_uuid].get("chunk_count", 0) == 0
    ]

    if not files_to_process:
        logger.info("All files already ingested. No action needed.")
        return

    logger.info(f"Filtered to {len(files_to_process)} new/pending files to process.")

    # 5. Ingest in batches
    def process_batch(batch: List[Tuple[str, str, str, Optional[str], Optional[int], str]]) -> Tuple[int, int, Dict[str, Dict[str, Any]]]:
        batch_chunks = 0
        batch_docs = 0
        batch_entries: Dict[str, Dict[str, Any]] = {}

        for file_path, file_name, dt, ena, yr, s_uuid in batch:
            try:
                chunks, stable_doc_uuid, doc_type = process_document(
                    file_path=file_path,
                    file_name=file_name,
                    stable_doc_uuid=s_uuid,
                    doc_type=dt,
                    enabler=ena,
                    subject=subject,
                    year=yr,
                    tenant=tenant_clean,
                    ocr_pages=ocr_pages
                )

                if not chunks:
                    logger.warning(f"Skipping {file_name}: No chunks generated.")
                    continue

                batch_chunks += len(chunks)
                batch_docs += 1

                # Prepare entry for mapping
                entry: Dict[str, Any] = {
                    "doc_id": stable_doc_uuid,
                    "file_name": file_name,
                    "filepath": get_mapping_key_from_physical_path(file_path), # 🎯 FIX: ใช้ relative key
                    "doc_type": doc_type,
                    "enabler": ena,
                    "year": yr,
                    "tenant": tenant_clean,
                    "upload_date": datetime.now(timezone.utc).isoformat(),
                    "chunk_count": len(chunks),
                    "status": "Ingested",
                    "size": os.path.getsize(file_path),
                    "chunk_uuids": [c.metadata["chunk_uuid"] for c in chunks if "chunk_uuid" in c.metadata]
                }

                batch_entries[stable_doc_uuid] = entry

                if dry_run:
                    logger.info(f"[DRY RUN] Processed {file_name} → {len(chunks)} chunks (not added to vectorstore)")
                    continue

                # Add to vectorstore
                col_name = get_doc_type_collection_key(doc_type, ena)
                vectorstore = get_vectorstore(col_name, tenant_clean, yr)
                vectorstore.add_documents(chunks)
                logger.info(f"Added {len(chunks)} chunks from {file_name} to collection '{col_name}'.")

            except Exception as e:
                logger.error(f"Error processing {file_name}: {e}", exc_info=True)
                continue

        return batch_docs, batch_chunks, batch_entries

    # 6. Execute ingestion (Sequential or Parallel)
    if sequential:
        processed_docs, processed_chunks, all_entries = process_batch(files_to_process)
        total_docs += processed_docs
        total_chunks += processed_chunks
        new_doc_id_entries.update(all_entries)
    else:
        with ThreadPoolExecutor(MAX_PARALLEL_WORKERS) as executor:
            futures = []
            for i in range(0, len(files_to_process), batch_size):
                batch = files_to_process[i:i + batch_size]
                futures.append(executor.submit(process_batch, batch))

            for future in as_completed(futures):
                batch_docs, batch_chunks, batch_entries = future.result()
                total_docs += batch_docs
                total_chunks += batch_chunks
                new_doc_id_entries.update(batch_entries)

    # 7. Save new entries to mappings (Group by context)
    grouped_entries: Dict[Tuple[str, Optional[str], Optional[int]], Dict[str, Dict[str, Any]]] = defaultdict(dict)

    for uuid, entry in new_doc_id_entries.items():
        dt = entry["doc_type"].lower()
        ena = entry.get("enabler")
        yr = entry.get("year")
        key = (dt, ena, yr)
        grouped_entries[key][uuid] = entry

    for (dt, ena, yr), entries in grouped_entries.items():
        _update_doc_id_mapping(entries, dt, tenant_clean, yr, ena)

    logger.info(f"--- INGESTION COMPLETE | Processed {total_docs} documents | Total chunks: {total_chunks} ---")

# -------------------- Wipe Vectorstore --------------------
# core/ingest.py → แทนที่ wipe_vectorstore ทั้งฟังก์ชันด้วยอันนี้เลย
def wipe_vectorstore(
    doc_type_to_wipe: str,
    enabler: Optional[str] = None,
    tenant: str = DEFAULT_TENANT,
    year: Optional[Union[int, str]] = None
) -> None:
    import shutil
    from utils.path_utils import (
        get_vectorstore_collection_path,
        get_mapping_file_path,
        get_vectorstore_tenant_root_path,
        get_mapping_tenant_root_path,
    )

    tenant_clean = unicodedata.normalize('NFKC', tenant.lower().replace(" ", "_"))
    dt = doc_type_to_wipe.lower()

    logger.warning(f"WIPE → {dt.upper()} | Year={year or 'Global'} | Enabler={enabler or 'None'}")

    # 1. ลบ vectorstore folder
    vec_path = get_vectorstore_collection_path(tenant_clean, year, dt, enabler)
    if os.path.exists(vec_path):
        shutil.rmtree(vec_path)
        logger.info(f"Deleted vectorstore folder: {vec_path}")

    # 2. ลบ mapping file — สำคัญมาก!
    mapping_path = get_mapping_file_path(dt, tenant_clean, year, enabler)
    if os.path.exists(mapping_path):
        os.remove(mapping_path)
        logger.info(f"Deleted mapping file: {mapping_path}")
    else:
        logger.debug(f"Mapping file not found (OK): {mapping_path}")

    # 3. ถ้าเป็น evidence → ลบ evidence mapping ด้วย
    if dt == EVIDENCE_DOC_TYPES.lower() and year is not None and enabler:
        ev_path = get_evidence_mapping_file_path(tenant_clean, year, enabler)
        if os.path.exists(ev_path):
            os.remove(ev_path)
            logger.info(f"Deleted evidence mapping: {ev_path}")

    # 4. ทำความสะอาดโฟลเดอร์ว่าง
    try:
        vec_root = get_vectorstore_tenant_root_path(tenant_clean)
        if os.path.isdir(vec_root) and not os.listdir(vec_root):
            shutil.rmtree(vec_root)
            logger.info(f"Cleaned empty vectorstore root")
    except: pass

    try:
        map_root = get_mapping_tenant_root_path(tenant_clean)
        if os.path.isdir(map_root):
            # ลบโฟลเดอร์ปีที่ว่าง (เช่น 2568)
            for item in os.listdir(map_root):
                item_path = os.path.join(map_root, item)
                if os.path.isdir(item_path) and not os.listdir(item_path):
                    shutil.rmtree(item_path)
            # ลบ root ถ้าว่างสนิท
            if len(os.listdir(map_root)) <= 1:  # เหลือแค่ .DS_Store
                shutil.rmtree(map_root)
                logger.info(f"Cleaned empty mapping root")
    except: pass

    logger.info(f"WIPE SUCCESS: {dt.upper()} context completely removed!")

# -------------------- [REVISED] Document Management Utilities --------------------
def delete_document_by_uuid(
    stable_doc_uuid: str, 
    tenant: str = "pea", 
    year: Union[int, str] = DEFAULT_YEAR, 
    collection_name: Optional[str] = None, 
    doc_type: Optional[str] = None, 
    enabler: Optional[str] = None, 
) -> bool:
    if not doc_type:
        logger.error(f"doc_type required for delete.")
        return False
        
    tenant_clean = unicodedata.normalize('NFKC', tenant.lower().replace(" ", "_"))
    
    doc_type_lower = doc_type.lower()
    
    year_int = int(year) if str(year).isdigit() else None
    
    final_year, final_enabler = get_normalized_metadata(doc_type_lower, year_int, enabler, DEFAULT_ENABLER)

    try:
        doc_mapping_db = load_doc_id_mapping(doc_type_lower, tenant_clean, final_year, final_enabler) 
    except FileNotFoundError:
        logger.warning(f"Mapping file not found.")
        return False

    entry = doc_mapping_db.get(stable_doc_uuid)
    if not entry:
        logger.warning(f"UUID not found.")
        return False

    final_doc_type = entry.get("doc_type", doc_type_lower)
    final_enabler_from_entry = entry.get("enabler", final_enabler)
    final_year_from_entry = entry.get("year", final_year)
    chunk_uuids = entry.get("chunk_uuids", [])

    if chunk_uuids:
        col_name = get_doc_type_collection_key(final_doc_type, final_enabler_from_entry)
        vectorstore = get_vectorstore(col_name, tenant_clean, final_year_from_entry) 
        vectorstore.delete(ids=chunk_uuids) 
        logger.info(f"Deleted {len(chunk_uuids)} chunks.")

    del doc_mapping_db[stable_doc_uuid]

    save_doc_id_mapping(doc_mapping_db, doc_type_lower, tenant_clean, final_year, final_enabler)
    logger.info(f"Updated mapping DB.")

    return True

def list_documents(
    doc_types: List[str],
    tenant: str = DEFAULT_TENANT,
    year: Optional[Union[str, int]] = None,
    enabler: Optional[str] = None,
    show_results: Literal["all", "missing", "ingested", "pending"] = "all", 
    skip_ext: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    logger.info("--- STARTING DOCUMENT LISTING ---")
    
    tenant_clean = unicodedata.normalize('NFKC', tenant.lower().replace(" ", "_"))
    
    load_contexts: Set[Tuple[str, Optional[str], Optional[Union[str, int]]]] = set()
    files_on_disk: List[Dict[str, Any]] = []

    for dt in doc_types:
        dt_lower = dt.lower()

        resolved_year, resolved_enabler = get_normalized_metadata(
            doc_type=dt_lower,
            year_input=year,
            enabler_input=enabler,
            default_enabler=DEFAULT_ENABLER,
        )
        
        if dt_lower == EVIDENCE_DOC_TYPES.lower() and resolved_year is None:
            logger.error("Evidence requires --year. Skipping evidence listing.")
            continue
            
        load_contexts.add((dt_lower, resolved_enabler, resolved_year))
        
        root_path = get_document_source_dir(tenant_clean, resolved_year, resolved_enabler, dt_lower)
        logger.info(f" [SCAN] '{dt_lower}' Context: {dt_lower} / {resolved_enabler} / {resolved_year} | Path: {root_path}")

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
        return []

    full_mapping: Dict[str, Dict] = {}
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
            
            if entry_context in load_contexts: 
                 saved_filepath = entry["filepath"] 
                 stable_lookup_key = get_mapping_key_from_physical_path(saved_filepath)
                 
                 if stable_lookup_key:
                     filepath_to_stable_uuid[stable_lookup_key] = s_uuid
                 else:
                     logger.warning(f"Could not create stable lookup key from saved path: {saved_filepath}")


    results: List[Dict[str, Any]] = []
    
    for info in files_on_disk:
        file_path_abs = info["file_path_abs"]
        relative_key_candidate = get_mapping_key_from_physical_path(file_path_abs) 
        
        stable_doc_uuid = None
        if relative_key_candidate:
            stable_doc_uuid = filepath_to_stable_uuid.get(relative_key_candidate)
        
        
        entry = None
        if stable_doc_uuid:
            entry = full_mapping.get(stable_doc_uuid)
        elif info["stable_doc_uuid"] in full_mapping:
             entry = full_mapping.get(info["stable_doc_uuid"])

        if entry:
            info["status"] = entry.get("status", "Ingested")
            info["chunk_count"] = entry.get("chunk_count", 0)
            if info["chunk_count"] == 0:
                 info["status"] = "PENDING_REINGEST" 

        status_to_display: List[str]
        if show_results == "all":
            status_to_display = ["MISSING", "PENDING_REINGEST", "Ingested"]
        elif show_results == "missing":
            status_to_display = ["MISSING"]
        elif show_results == "pending":
            status_to_display = ["PENDING_REINGEST"]
        elif show_results == "ingested":
            status_to_display = ["Ingested"]
        else:
             status_to_display = ["MISSING", "PENDING_REINGEST", "Ingested"]

        if info["status"] in status_to_display:
            results.append({
                "Doc Type": info["doc_type"].upper(),
                "Enabler": info["enabler"] or "-",
                "Year": info["year"] or "-",
                "File Name": info["file_name"],
                "Status": info["status"],
                "Chunks": info["chunk_count"],
                "UUID": info["stable_doc_uuid"]
            })

    logger.info(f"--- DOCUMENT LISTING COMPLETE | Total= {len(files_on_disk)} | Displayed= {len(results)} ---")
    
    if results:
        results = sorted(results, key=lambda x: (x["Doc Type"], x["Enabler"], x["Year"], x["File Name"]))
        
    return results

# -------------------- Main Execution --------------------

if __name__ == "__main__":
    try:
        import argparse
        
        parser = argparse.ArgumentParser(description="Multi-Tenant RAG Ingestion and Management Tool (SE-AM Ready)")
        
        parser.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant code (e.g., 'pea', 'pwa').")
        parser.add_argument("--year", type=str, default=str(DEFAULT_YEAR), help="Assessment year (e.g., '2568').")
        parser.add_argument("--doc-type", nargs='+', default=[EVIDENCE_DOC_TYPES], help="Document type(s) to process ('evidence', 'document', 'all').")
        parser.add_argument("--enabler", type=str, default=DEFAULT_ENABLER, help="Enabler code (e.g., 'KM', 'HCM').")
        parser.add_argument("--subject", type=str, default=None, help="Subject/Topic tag for documents (optional).")

        parser.add_argument("--ingest", action="store_true", help="Run ingestion mode.")
        parser.add_argument("--dry-run", action="store_true", help="Simulate ingestion without writing to Chroma/DB.")
        parser.add_argument("--sequential", action="store_true", help="Run ingestion in sequential mode (for debugging).")
        parser.add_argument("--skip-wipe", action="store_true", help="Skip the wiping of vector store before ingestion.")

        parser.add_argument("--list", action="store_true", help="Run document listing mode.")
        parser.add_argument("--show-results", type=str, default="all", choices=["all", "missing", "ingested", "pending"], help="Filter results for list mode.")
        
        parser.add_argument("--wipe", action="store_true", help="Wipe (delete) vector store and mapping files for the specified context.")
        parser.add_argument("--yes", action="store_true", help="Bypass confirmation prompt for wiping (DANGER: use only when sure!).") 

        args = parser.parse_args()
        
        has_evidence = any(dt.lower() == EVIDENCE_DOC_TYPES.lower() for dt in args.doc_type)
        if has_evidence and (args.ingest or args.wipe or args.list) and not args.enabler:
            logger.error(f"When using 'evidence', you must specify --enabler.")
            sys.exit(1)

        logger.info(f"--- STARTING EXECUTION: Tenant={args.tenant}, Year={args.year}, DocType={args.doc_type}, Enabler={args.enabler} ---")
        
        if args.ingest:
            logger.info("--- INGESTION MODE ACTIVATED ---")
            
            year_to_use_ingest = int(args.year) if args.year and args.year.isdigit() and int(args.year) > 0 else None
                
            if not has_evidence:
                 year_to_use_ingest = None 
            
            if not args.skip_wipe and not args.dry_run:
                logger.warning("⚠️ Wiping Vector Store before ingestion!!!")
                wipe_vectorstore(
                    doc_type_to_wipe=args.doc_type[0].lower() if args.doc_type else EVIDENCE_DOC_TYPES.lower(),
                    enabler=args.enabler, 
                    tenant=args.tenant, 
                    year=year_to_use_ingest
                )
            
            ingest_all_files(
                tenant=args.tenant,
                year=year_to_use_ingest,
                doc_types=args.doc_type,
                enabler=args.enabler,
                subject=args.subject, 
                dry_run=args.dry_run,
                sequential=args.sequential
            )
            
        elif args.list:
            logger.info("--- LIST MODE ACTIVATED ---\n")
            
            results = list_documents(
                doc_types=[dt.lower() for dt in args.doc_type], 
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=args.year,
                show_results=args.show_results 
            )
            
            if results:
                try:
                    from tabulate import tabulate
                    print("\n--- FOUND DOCUMENTS ---")
                    print(tabulate(results, headers="keys", tablefmt="simple"))
                except ImportError:
                    print("\nOptional dependency 'tabulate' not found. Falling back to plain table output.")
                    print(f"{'Doc Type':10} {'Enabler':8} {'Year':5} {'File Name':70} {'Status':8} {'Chunks':6} {'UUID'}")
                    print("-" * 180)
                    for r in results:
                        print(f"{r['Doc Type']:10} {r['Enabler']:8} {str(r['Year']):5} {r['File Name']:70} {r['Status']:8} {r['Chunks']:6} {r['UUID']}")
            else:
                print("No documents found.")
            
        elif args.wipe:
            logger.info("--- WIPE MODE ACTIVATED ---")
            logger.info("⚠️ Wiping Vector Store and Mapping Files as requested!!!")
            
            if not args.yes:
                confirmation = input("Type 'YES' (all caps) to confirm deletion: ")
                if confirmation != "YES":
                    logger.info("Deletion cancelled.")
                    sys.exit(0)

            year_to_use_wipe = int(args.year) if args.year and args.year.isdigit() and int(args.year) > 0 else None
                
            if not has_evidence:
                year_to_use_wipe = None 
            
            wipe_vectorstore(
                doc_type_to_wipe=args.doc_type[0].lower() if args.doc_type else EVIDENCE_DOC_TYPES.lower(),
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=year_to_use_wipe
            )
            
        else:
            print("\nUsage: Specify --ingest, --list, or --wipe mode.")
            parser.print_help()

        logger.info("Execution finished.")
        
    except ImportError:
         print("--- RUNNING SCRIPT STANDALONE FAILED: Missing necessary imports ---")
         
    except Exception as e:
         import traceback
         logger.info(f"FATAL ERROR DURING MAIN EXECUTION: {e}", exc_info=True)
         print(f"--- FATAL ERROR: Check ingest.log for details... \n{traceback.format_exc()}")