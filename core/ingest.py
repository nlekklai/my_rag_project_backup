# core/ingest.py
# เวอร์ชันเต็ม: Multi-Tenant + Multi-Year (รัฐวิสาหกิจไทย Ready)
# รวมการแก้ไข: Path Isolation, get_vectorstore, ingest_all_files, list_documents, wipe_vectorstore

import os
import re
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
    MAPPING_BASE_DIR, # 📌 FIXED: ค่านี้จะถูกใช้เป็น Root Path สำหรับ Mapping
    SUPPORTED_TYPES,
    SUPPORTED_DOC_TYPES,
    DEFAULT_ENABLER,
    SUPPORTED_ENABLERS,
    EVIDENCE_DOC_TYPES,
    DEFAULT_DOC_TYPES,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_TENANT, 
    DEFAULT_YEAR
)

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

# -------------------- [FIXED] Multi-Tenant/Year Path Builders --------------------

def build_tenant_base_path(tenant: str, year: int, doc_type: str) -> str:
    """
    สร้าง path หลักของ tenant/context สำหรับ **Input Data (Source Files)**
    - ใช้ year ในพาธสำหรับ evidence เท่านั้น (เช่น data/pea/2568)
    """
    if not tenant:
        raise ValueError("tenant ห้ามว่าง")
    tenant_clean = tenant.strip().lower().replace(" ", "_")
    if ".." in tenant_clean or "/" in tenant_clean or "\\" in tenant_clean:
        raise ValueError(f"tenant ไม่ถูกต้อง: {tenant}")
    
    is_evidence = doc_type.lower() == EVIDENCE_DOC_TYPES.lower()
    
    path_components = [DATA_DIR, tenant_clean]
    if is_evidence:
        path_components.append(str(year))
    
    return os.path.join(*path_components)


def get_collection_parent_dir(tenant: str, year: int, doc_type: str) -> str:
    """
    Calculates the parent directory where the collection folder resides (using VECTORSTORE_DIR).
    Expected structure: VECTORSTORE_DIR / tenant / [year]
    """
    doc_type_lower = doc_type.strip().lower()
    
    # 🎯 FIX 1: เริ่มต้น Path จาก VECTORSTORE_DIR (ซึ่งรวม 'gov_tenants' แล้ว) และตามด้วย tenant
    path_segments = [VECTORSTORE_DIR, tenant.lower()] 
    
    is_evidence = doc_type_lower == EVIDENCE_DOC_TYPES.lower()
    
    # 🎯 FIX 2: เพิ่มปี (str(year)) เฉพาะถ้าเป็น 'evidence' เท่านั้น
    if is_evidence:
        # Year for evidence
        path_segments.append(str(year))
        
    # 📌 NEW: สำหรับ DocType อื่นๆ (เช่น document, faq) ไม่ต้องใส่ปี
    # แต่ถ้าเป็น DocType ทั่วไป เราอาจจะต้องการให้มันอยู่ใต้พาธ tenant โดยตรง
    
    # 💡 REVISED LOGIC: ให้ Collection อยู่ใต้ VECTORSTORE_DIR / tenant / [year] เสมอ
    # ถ้าเป็น DocType ทั่วไป (เช่น document) ให้ใช้ DEFAULT_YEAR ในการกำหนดพาธ
    # เพื่อให้ง่ายต่อการจัดการ Wipe และ List
    if not is_evidence and str(year) != str(DEFAULT_YEAR):
         # ถ้าไม่ใช่ evidence และปีไม่ใช่ default ให้เก็บไว้ในพาธแยกปี
         # เช่น VECTORSTORE_DIR/pea/2569/document
         # 📌 NOTE: แต่ตาม Logic เดิมที่กำหนดมา ควรจะแยกปีเฉพาะ Evidence เท่านั้น
         # เพื่อความสอดคล้องกับโค้ดที่คุณเขียนมาแล้ว:
         pass # ไม่ต้องทำอะไรเพิ่ม: path_segments คือ [VECTORSTORE_DIR, tenant.lower()] 
         
    return os.path.join(*path_segments)


def get_doc_id_mapping_path(tenant: str, year: Optional[Union[str, int]] = None, enabler: Optional[str] = None) -> str:
    """
    Constructs the path for the document ID mapping file, 
    using the format <tenant>_[year]_[enabler]_doc_id_mapping.json.
    """
    
    # 1. สร้าง Base Directory (ใช้ Logic การสร้างโฟลเดอร์จาก Base Path เดิม)
    base_dir = os.path.join(MAPPING_BASE_DIR, tenant)

    # ⚠️ ตรวจสอบ year: ถ้ามี year (ไม่ None และไม่ว่าง) จะใช้สร้างโฟลเดอร์ และต้องใช้ในชื่อไฟล์ด้วย
    year_str = str(year) if year is not None and str(year).strip() else ""
    if year_str:
        base_dir = os.path.join(base_dir, year_str)
        
    os.makedirs(base_dir, exist_ok=True)
    
    # 2. สร้าง Filename ใหม่
    enabler_str = str(enabler).lower() if enabler is not None and str(enabler).strip() else ""
    
    # Prefix: pea_2568
    prefix = f"{tenant}"
    if year_str:
         prefix += f"_{year_str}"
    
    # Enabler Part: _km
    enabler_part = f"_{enabler_str}" if enabler_str else ""
    
    # Filename: pea_2568_km_doc_id_mapping.json หรือ pea_doc_id_mapping.json
    filename = f"{prefix}{enabler_part}_doc_id_mapping.json"
    
    return os.path.join(base_dir, filename)
    

def get_target_dir(doc_type: str, enabler: Optional[str] = None) -> str:
    """
    กำหนดชื่อ Collection สำหรับ ChromaDB (Logical ID)
    """
    if not doc_type:
        return "default"

    doc_type_lower = doc_type.lower()
    enabler_lower = enabler.lower() if enabler and enabler.strip() else None # 📌 FIX: ตรวจสอบ String ว่างเปล่า
    
    # 🎯 FIX: สำหรับ evidence ให้ใช้ DocType_Enabler เสมอ
    if doc_type_lower == EVIDENCE_DOC_TYPES.lower() and enabler_lower:
        # ✅ ผลลัพธ์: 'evidence_km'
        return f"{doc_type_lower}_{enabler_lower}"
    
    # 2. จัดการ Doc Type อื่น ๆ (document, faq, etc.)
    return doc_type_lower

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
        
    # 3. Fallback (สำหรับชื่อยาว/ชื่อที่ไม่ได้มาตรฐาน)
    # 📌 NOTE: เพื่อความมั่นคง จะไม่ใช้ Fallback ชื่อยาว (pea_2568_evidence_km)
    # เพราะ Collection Name ไม่ควรมี Tenant/Year อยู่ข้างใน
    
    # Fallback to the original name if no match is found (ถือว่าเป็น Doc Type ฐาน)
    return collection_name_lower, None


def _get_source_dir(
    tenant: str, 
    year: int, 
    doc_type: str, 
    enabler: Optional[str] = None
) -> str:
    """
    Constructs the full source directory path where raw documents reside.
    Expected structure: DATA_DIR / tenant / [year] / doc_type / [enabler]
    
    📌 FIX: แก้ไขลำดับพารามิเตอร์เพื่อให้สอดคล้องกับ build_tenant_base_path
    """
    if not doc_type:
        raise ValueError("Doc type must be provided for source directory.")
        
    doc_type_lower = doc_type.lower()
    enabler_lower = enabler.lower() if enabler else None
    
    # 🎯 FIX 1: ใช้ build_tenant_base_path เพื่อจัดการ DATA_DIR / tenant / [year]
    base_path = build_tenant_base_path(tenant, year, doc_type)
    
    # เพิ่ม Doc Type (evidence หรือ document)
    path_segments = [base_path, doc_type_lower]
    
    is_evidence = doc_type_lower == EVIDENCE_DOC_TYPES.lower()
    
    # เพิ่ม Enabler (km) เฉพาะถ้าเป็น evidence และมี enabler
    if is_evidence and enabler_lower and enabler_lower in [e.lower() for e in SUPPORTED_ENABLERS]:
        path_segments.append(enabler_lower)
    
    return os.path.join(*path_segments)

# -------------------- Helper: safe metadata filter --------------------
# (No change to _safe_filter_complex_metadata, _normalize_doc_id, clean_text, _is_pdf_image_only, _load_document_with_loader, FILE_LOADER_MAP, normalize_loaded_documents, TEXT_SPLITTER)
# (นำส่วนที่ไม่มีการแก้ไขออกเพื่อให้โค้ดสั้นลงในการส่งคืน แต่ในไฟล์จริงยังคงอยู่)
# ... [Original content of _safe_filter_complex_metadata to TEXT_SPLITTER remains unchanged] ...

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
def load_and_chunk_document(
    file_path: str,
    doc_id_key: str, 
    stable_doc_uuid: str, 
    year: Optional[int] = None,
    version: str = "v1",
    metadata: Optional[Dict[str, Any]] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> List[Document]:
    """Load document, inject metadata, clean, and split into chunks. (No Change in Logic)"""
    file_extension = os.path.splitext(file_path)[1].lower()
    loader_func = FILE_LOADER_MAP.get(file_extension)
    
    if not loader_func:
        logger.error(f"No loader found for extension: {file_extension} at {file_path}")
        return []
        
    raw_docs = [] 
    try:
        raw_docs = loader_func(file_path)
        
    except ValidationError as e:
        loader_name = str(loader_func)
        if 'Unstructured' in loader_name or 'Unstructured' in str(loader_func):
             logger.warning(f"⚠️ OCR Crash Handled: {os.path.basename(file_path)} - Loader raised Pydantic ValidationError. Treating as 0 documents loaded.")
             raw_docs = [] 
        else:
             raise e 
             
    except Exception as e:
        logger.error(f"❌ Critical error during file loading: {file_path}. Error: {e}")
        raw_docs = []

    if not raw_docs:
        logger.warning(f"Loader returned 0 documents for {os.path.basename(file_path)}. Skipping chunking.")
        return []

    pre_cleaned_raw_docs = []
    for doc in raw_docs:
        if isinstance(doc, Document):
            pre_cleaned_raw_docs.append(doc)
        else:
            doc_type_str = str(type(doc)).split("'")[-2]
            logger.warning(f"⚠️ Loader for '{os.path.basename(file_path)}' returned non-Document object (Type: {doc_type_str}). Skipping normalization.")

    docs = normalize_loaded_documents(pre_cleaned_raw_docs, source_path=file_path)

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
        d.metadata["stable_doc_uuid"] = stable_doc_uuid
        base_filename = os.path.basename(file_path)
        d.metadata["source_filename"] = base_filename
        d.metadata["source"] = base_filename
        d.metadata = _safe_filter_complex_metadata(d.metadata) 

    try:
        chunks = TEXT_SPLITTER.split_documents(docs) 
    except Exception as e:
        logger.error(f"Error during document splitting for {os.path.basename(file_path)}: {e}")
        chunks = docs 

    final_cleaned_chunks = []
    for c in chunks:
        if isinstance(c, Document):
             c.page_content = clean_text(c.page_content) 
             final_cleaned_chunks.append(c)
        else:
             logger.error(f"FATAL: Non-Document object found in 'chunks' list after splitting! Type: {type(c)}. Skipping.")
             
    chunks = final_cleaned_chunks 

    for idx, c in enumerate(chunks, start=1):
        c.metadata["chunk_index"] = idx
        c.metadata = _safe_filter_complex_metadata(c.metadata) 
        
    logger.info(f"Loaded and chunked {os.path.basename(file_path)} -> {len(chunks)} chunks.")
    return chunks


# -------------------- [REVISED] Process single document --------------------
def process_document(
    file_path: str,
    file_name: str,
    stable_doc_uuid: str, 
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None, 
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
    if doc_type.lower() == EVIDENCE_DOC_TYPES.lower(): # 📌 FIX: ต้องเช็ค Doc Type ก่อน
        resolved_enabler = (enabler or DEFAULT_ENABLER).upper()

    injected_metadata = metadata or {}
    if resolved_enabler:
        injected_metadata["enabler"] = resolved_enabler
    if tenant: 
        injected_metadata["tenant"] = tenant
        
    filter_id_value = filename_doc_id_key 
    logger.critical(f"================== START DEBUG INGESTION: {file_name} ==================")
    logger.critical(f"🔍 DEBUG ID (stable_doc_uuid, 64-char Hash): {len(stable_doc_uuid)}-char: {stable_doc_uuid[:34]}...")
    logger.critical(f"✅ FINAL ID TO STORE (34-char Ref ID): {len(filter_id_value)}-char: {filter_id_value[:34]}...")

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
        c.metadata["stable_doc_uuid"] = stable_doc_uuid 
        c.metadata["original_stable_id"] = filename_doc_id_key[:32].lower()  
        c.metadata["doc_type"] = doc_type 
        if resolved_enabler:
             c.metadata["enabler"] = resolved_enabler
        if tenant: 
             c.metadata["tenant"] = tenant
        c.metadata = _safe_filter_complex_metadata(c.metadata)
        logger.debug(f"Chunk metadata preview: {c.metadata}")
        
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
    เวอร์ชัน PEA 2568 แท้ ๆ (ไม่มี rag_ prefix ใด ๆ)
    """

    # === 1. ใช้ชื่อตรง ๆ ไม่ต้องเติม prefix อัตโนมัติ ===
    if len(collection_name) < 3:
        logger.warning(
            f"Collection name '{collection_name}' ค่อนข้างสั้น แนะนำให้ใช้ชื่ออย่างน้อย 6 ตัวอักษร "
            f"(เช่น evidence_km, km42l103) เพื่อป้องกันการชนกันของชื่อ"
        )

    # === 2. สร้าง path ที่ถูกต้องตามโครงสร้าง PEA ===
    # 📌 FIX 1: ใช้ _parse_collection_name เพื่อดึง doc_type ออกมาอย่างถูกต้อง
    doc_type_for_path, _ = _parse_collection_name(collection_name)
    
    # 📌 FIX 2: ใช้ get_collection_parent_dir ที่แก้ไขแล้ว
    tenant_dir = get_collection_parent_dir(tenant, year, doc_type_for_path) 
    persist_directory = os.path.join(tenant_dir, collection_name)  # ใช้ชื่อ collection เต็ม ๆ
    cache_key = persist_directory

    # === 3. Cache HIT ===
    if cache_key in _VECTORSTORE_SERVICE_CACHE:
        logger.debug(f"Cache HIT → Reusing vectorstore: {persist_directory}")
        return _VECTORSTORE_SERVICE_CACHE[cache_key]

    # === 4. Embedding model (แชร์ตัวเดียวตลอด process) ===
    embeddings = _VECTORSTORE_SERVICE_CACHE.get("embeddings_model")
    if not embeddings:
        logger.info("กำลังโหลด intfloat/multilingual-e5-base + prefix wrapper (สุดยอดแห่งปี 2025)")

        base_emb = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-base",
            model_kwargs={"device": "cpu"},  # เปลี่ยนเป็น "cuda" ถ้ามี GPU
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
        )

        class E5PrefixWrapper:
            def embed_documents(self, texts):
                return base_emb.embed_documents([f"passage: {t}" for t in texts])
            def embed_query(self, text: str):
                return base_emb.embed_query(f"query: {text}")
            def __call__(self, texts):
                return self.embed_documents(texts)

        embeddings = E5PrefixWrapper()
        _VECTORSTORE_SERVICE_CACHE["embeddings_model"] = embeddings
        logger.info("multilingual-e5-base + prefix wrapper โหลดสำเร็จและแชร์ตลอด process")

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
        f"   Path        : {persist_directory}\n"
        f"   Documents   : {vectorstore._collection.count():,}"
    )

    return vectorstore

def create_vectorstore_from_documents(
    chunks: List[Document],
    collection_name: str,
    doc_mapping_db: Dict[str, Dict[str, Any]], 
    tenant: str = "pwa",
    year: int = 2568,
    base_path: str = VECTORSTORE_DIR
):
    """
    Adds documents (chunks) to the Chroma vector store and updates the mapping DB.
    รองรับ Chroma 0.5+ ขึ้นไป (ไม่มี .persist() แล้ว)
    """
    
    if not chunks:
        logger.warning(f"No chunks to index for collection '{collection_name}'. Skipping.")
        return

    # 1. Get Vector Store Instance
    try:
        vectorstore = get_vectorstore(collection_name, tenant, year, base_path) 
        logger.debug(f"Successfully obtained vectorstore for collection: {collection_name}")
    except Exception as e:
        logger.error(f"Error getting Vector Store Instance for '{collection_name}': {e}")
        return
    
    # 2. Add Chunks to Vector Store — Chroma 0.5+ ไม่ต้อง persist() อีกต่อไป
    try:
        chunk_uuids = vectorstore.add_documents(chunks)
        
        logger.info(f"Indexed {len(chunk_uuids)} chunks into collection '{collection_name}' (tenant={tenant}, year={year}).")

        # Chroma 0.5+ ใช้ PersistentClient + auto-save ทันที ไม่ต้อง persist()
        logger.info(f"Collection '{collection_name}' auto-saved to disk (Chroma 0.5+ behavior).")

    except Exception as e:
        logger.error(f"Error during add_documents for collection '{collection_name}': {e}")
        return

    # 3. อัปเดต Doc ID Mapping Database
    doc_chunk_map: Dict[str, List[str]] = {}
    
    for chunk, chunk_id in zip(chunks, chunk_uuids):
        stable_doc_uuid = chunk.metadata.get("stable_doc_uuid")
        if stable_doc_uuid:
            doc_chunk_map.setdefault(stable_doc_uuid, []).append(chunk_id)

    updated_count = 0
    for stable_doc_uuid, new_ids in doc_chunk_map.items():
        if stable_doc_uuid in doc_mapping_db:
            current_ids = set(doc_mapping_db[stable_doc_uuid].get("chunk_uuids", []))
            all_ids = current_ids.union(set(new_ids))
            doc_mapping_db[stable_doc_uuid]["chunk_uuids"] = list(all_ids)
            doc_mapping_db[stable_doc_uuid]["status"] = "Ingested"
            doc_mapping_db[stable_doc_uuid]["chunk_count"] = len(all_ids)
            updated_count += 1
        else:
            # 📌 FIX: ถ้า Stable UUID ไม่ถูกพบใน Mapping DB (อาจเป็นเพราะไฟล์ใหม่/บั๊ก)
            # ให้สร้าง Entry ใหม่ขึ้นมาทันที (ป้องกันข้อมูลสูญหาย)
            doc_mapping_db[stable_doc_uuid] = {
                "file_name": chunk.metadata.get("source_filename", "UNKNOWN_FILE"),
                "doc_type": chunk.metadata.get("doc_type", "default"),
                "enabler": chunk.metadata.get("enabler"),
                "doc_id_key": chunk.metadata.get("original_stable_id", "UNKNOWN_REF"),
                "filepath": "UNKNOWN_PATH_AFTER_CHUNK", # ไม่สามารถหา Path เต็มได้จาก Chunk
                "tenant": tenant, 
                "year": year,
                "notes": "CREATED_DURING_INGEST",
                "statement_id": "",
                "chunk_uuids": list(set(new_ids)),
                "status": "Ingested",
                "chunk_count": len(new_ids)
            }
            logger.warning(f"Stable UUID {stable_doc_uuid} not found in mapping DB, auto-created new entry.")
            updated_count += 1

    logger.info(f"Updated mapping DB for {updated_count} documents in collection '{collection_name}'.")


def save_doc_id_mapping(
    mapping_data: Dict[str, Dict[str, Any]], 
    doc_type: str, 
    tenant: str = "pwa", 
    year: int = 2568,
    enabler: Optional[str] = None
):
    
    enabler_for_path = enabler
    
    # 🎯 FIX: แก้ไขบรรทัดนี้โดยการครอบ year ด้วย str(...)
    # 📌 FIX: ต้องแปลง year เป็น str(year) หรือ None ก่อนส่ง
    path = get_doc_id_mapping_path(tenant, str(year) if year is not None else None, enabler_for_path) 
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # 📌 FIX: ใช้ Mapping ที่ผ่านการกรอง Tenant/Year/Enabler มาแล้ว (small_db)
        if not mapping_data:
             if os.path.exists(path):
                  # ถ้า mapping_data ว่างเปล่า ให้ลบไฟล์เก่าทิ้ง
                  os.remove(path)
                  logger.info(f"✅ Deleted empty Doc ID Mapping file: {path}")
                  return
             return
             
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Successfully saved Doc ID Mapping to: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to save Doc ID Mapping: {e}")

def load_doc_id_mapping(
    doc_type: str, 
    tenant: str, 
    year: int, 
    enabler: Optional[str] = None # 💡 FIX 2: ต้องรับ enabler parameter
) -> Dict[str, Any]:
    """
    Loads the document ID mapping dictionary from the specified JSON file path.
    Uses the enabler to construct the filename for 'evidence' type.
    """
    
    # FIX: สร้างชื่อไฟล์ mapping โดยมี enabler ถ้าเป็น doc_type evidence
    if doc_type.lower() == EVIDENCE_DOC_TYPES.lower() and enabler:
        # ตัวอย่างชื่อไฟล์: pea_2568_km_doc_id_mapping.json
        file_name = f"{tenant}_{year}_{enabler.lower()}_doc_id_mapping.json"
    else:
        # ตัวอย่างชื่อไฟล์: pea_2568_document_doc_id_mapping.json
        file_name = f"{tenant}_{year}_{doc_type.lower()}_doc_id_mapping.json"
        
    # ใช้ MAPPING_BASE_DIR/tenant/year/filename.json
    mapping_file_path = os.path.join(MAPPING_BASE_DIR, tenant, str(year), file_name)
    
    mapping_db = {}
    if os.path.exists(mapping_file_path):
        try:
            with open(mapping_file_path, 'r', encoding='utf-8') as f:
                mapping_db = json.load(f)
            logger.debug(f"Loaded {len(mapping_db)} entries from {mapping_file_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding mapping file {mapping_file_path}: {e}")
    
    return mapping_db
        
# -------------------- API Helper: Get UUIDs for RAG Filtering --------------------
# 📌 REVISED: เพิ่ม tenant และ year
def get_stable_uuids_by_doc_type(doc_types: List[str], tenant: str = "pwa", year: int = 2568) -> List[str]:
    """Retrieves Stable UUIDs for RAG filtering based on document types (Multi-Tenant/Year)."""
    if not doc_types: return []
    
    doc_type_set = {dt.lower() for dt in doc_types}
    
    target_uuids = []
    
    # โหลด mapping db ทั้งหมดสำหรับ doc_types ที่ร้องขอ
    for dt_req in doc_type_set:
         # load_doc_id_mapping จะจัดการการโหลดไฟล์ Mapping หลายไฟล์ (ถ้าเป็น evidence)
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
    สร้าง UUID ที่มั่นคงจากพาธไฟล์เต็ม, ขนาดไฟล์, Ref ID Key, Tenant และ Year
    """
    try:
        full_path = os.path.abspath(filepath)
        file_size = os.path.getsize(filepath)
    except FileNotFoundError:
        full_path = os.path.abspath(filepath)
        file_size = 0
    
    # 💡 FIX: รวม Tenant และ Year เข้าไปใน Hash Input เพื่อ Isolation
    unique_string = f"{full_path}-{file_size}-{ref_id_key or 'NO_REF'}-{tenant or 'NO_TENANT'}-{year or 'NO_YEAR'}" 
    
    hash_object = hashlib.sha256(unique_string.encode('utf-8'))
    
    return hash_object.hexdigest()


def ingest_all_files(
    tenant: str = "pwa",            
    year: int = 2568,               
    data_dir: str = DATA_DIR,       
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None, 
    base_path: str = VECTORSTORE_DIR, 
    exclude_dirs: Set[str] = set(),
    version: str = "v1",
    sequential: bool = True,
    skip_ext: Optional[List[str]] = None,
    log_every: int = 50,
    batch_size: int = 500,
    dry_run: bool = False,          
    debug: bool = False             
) -> List[Dict[str, Any]]:
    """
    Ingest documents into vectorstore based on doc_type and enabler filters (Multi-Tenant/Year).
    """
    skip_ext = skip_ext or []
    os.makedirs(base_path, exist_ok=True)
    files_to_process = []
    
    doc_type_req = (doc_type or "all").lower()
    enabler_req = (enabler or (DEFAULT_ENABLER if doc_type_req == EVIDENCE_DOC_TYPES.lower() else None))
    if enabler_req:
        enabler_req = enabler_req.upper()

    logger.info(f"Starting ingest_all_files: Tenant={tenant}, Year={year}, doc_type_req='{doc_type_req}', enabler_req='{enabler_req}'")

    # 1. รวบรวมไฟล์ (ใช้ _get_source_dir ที่ถูกแก้ไขแล้ว)
    scan_roots: List[str] = []
    
    # 📌 FIX: ปรับ Logic การสร้าง scan_roots
    if doc_type_req == "all":
        # Scan ทุก Doc Type และทุก Enabler
        for dt in SUPPORTED_DOC_TYPES:
             if dt.lower() == EVIDENCE_DOC_TYPES.lower():
                for ena in SUPPORTED_ENABLERS:
                    scan_roots.append(_get_source_dir(
                        tenant=tenant, 
                        year=year, 
                        doc_type=dt, 
                        enabler=ena
                    )) 
             else:
                scan_roots.append(_get_source_dir(
                    tenant=tenant, 
                    year=year, 
                    doc_type=dt, 
                    enabler=None # Doc Type อื่นๆ ไม่มี Enabler ในพาธ
                )) 
    elif doc_type_req in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
        if doc_type_req == EVIDENCE_DOC_TYPES.lower():
             if enabler_req and enabler_req in SUPPORTED_ENABLERS:
                 # ถ้ากำหนด Doc Type และ Enabler มาชัดเจน
                 scan_roots = [_get_source_dir(
                     tenant=tenant, 
                     year=year, 
                     doc_type=doc_type_req, 
                     enabler=enabler_req
                 )] 
             else:
                 # ถ้ากำหนด Evidence แต่ไม่กำหนด Enabler ให้สแกนทุก Enabler
                 for ena in SUPPORTED_ENABLERS:
                     scan_roots.append(_get_source_dir(
                         tenant=tenant, 
                         year=year, 
                         doc_type=doc_type_req, 
                         enabler=ena
                     )) 
        else:
             # Doc Type อื่นๆ
             scan_roots = [_get_source_dir(
                 tenant=tenant, 
                 year=year, 
                 doc_type=doc_type_req, 
                 enabler=None # Doc Type อื่นๆ ไม่มี Enabler ในพาธ
             )] 
    else:
        logger.error(f"Invalid doc_type_req for ingestion: {doc_type_req}")
        return []

    # (Scan loop logic)
    for root_to_scan in set(scan_roots):
        if not os.path.isdir(root_to_scan):
             logger.warning(f"⚠️ Source directory not found: {root_to_scan}. Skipping scan.")
             continue
             
        # Collection name/doc_type/enabler must be parsed from the Input Path
        # Input Path: data/pea/2568/evidence/km 
        
        # 📌 FIX: ต้องดึง doc_type และ enabler จาก root_to_scan 
        path_parts = root_to_scan.strip(os.sep).split(os.sep)
        
        # ค้นหา Doc Type ในส่วนท้ายของ Path
        current_path_segment = path_parts[-1].lower()
        
        doc_type_from_path = None
        resolved_enabler = None

        if current_path_segment.upper() in SUPPORTED_ENABLERS: 
             # Path: .../evidence/km -> current_path_segment='km'
             resolved_enabler = current_path_segment.upper()
             # doc_type คือส่วนก่อนหน้า (parts[-2])
             doc_type_from_path = path_parts[-2].lower() if len(path_parts) >= 2 else None
             
        elif current_path_segment in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
             # Path: .../document -> current_path_segment='document'
             doc_type_from_path = current_path_segment
             resolved_enabler = None # Doc Type ทั่วไป ไม่มี Enabler
        
        if not doc_type_from_path:
             logger.warning(f"Could not determine doc_type from path: {root_to_scan}. Skipping.")
             continue

        current_collection_name = get_target_dir(doc_type_from_path, resolved_enabler)

        logger.info(f"Scanning source directory: {root_to_scan} (Maps to Collection: {current_collection_name})")

        for root, dirs, filenames in os.walk(root_to_scan):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            if root != root_to_scan: continue 

            for f in filenames:
                if f.startswith('.'): continue
                file_path = os.path.join(root, f)
                file_extension = os.path.splitext(f)[1].lower()

                if file_extension not in SUPPORTED_TYPES:
                    logger.info(f"⚠️ Skipping unsupported file type {file_extension}: {f}")
                    continue
                if skip_ext and file_extension in skip_ext:
                    logger.info(f"⚠️ Skipping excluded extension {file_extension}: {f}")
                    continue

                files_to_process.append({
                    "file_path": file_path,
                    "file_name": f,
                    "doc_type": doc_type_from_path,
                    "enabler": resolved_enabler,
                    "collection_name": current_collection_name
                })

    if not files_to_process:
        logger.warning("⚠️ No files found to ingest!")
        return []

    # 2. Load Mapping DB และกำหนด Stable IDs
    relevant_doc_types = {f['doc_type'] for f in files_to_process}
    
    doc_mapping_dbs: Dict[str, Dict[str, Dict[str, Any]]] = {
        dt: load_doc_id_mapping(dt, tenant, year) for dt in relevant_doc_types
    }
    
    # เตรียม lookup table ที่รวมทุก doc_type เข้าด้วยกัน (สำหรับค้นหา UUID ที่มีอยู่)
    uuid_from_path_lookup: Dict[str, str] = {}
    for dt, db in doc_mapping_dbs.items():
        for s_uuid, entry in db.items():
            # 📌 FIX: ตรวจสอบ filepath, tenant, year ใน mapping entry
            if "filepath" in entry and str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == year:
                uuid_from_path_lookup[entry["filepath"]] = s_uuid


    for file_info in files_to_process:
        dt = file_info["doc_type"]
        doc_mapping_db = doc_mapping_dbs[dt] # ใช้ DB ที่เฉพาะเจาะจง
        
        filename_doc_id_key = _normalize_doc_id(os.path.splitext(file_info["file_name"])[0])
        file_info["doc_id_key"] = filename_doc_id_key
        
        # 📌 REVISED: Pass tenant/year to UUID creation
        stable_uuid_from_path = create_stable_uuid_from_path(
            file_info["file_path"], 
            ref_id_key=filename_doc_id_key,
            tenant=tenant, 
            year=year      
        )
        
        stable_doc_uuid = uuid_from_path_lookup.get(file_info["file_path"])
        
        if stable_doc_uuid and stable_doc_uuid in doc_mapping_db:
            file_info["stable_doc_uuid"] = stable_doc_uuid
            # อัปเดต Entry ที่มีอยู่
            doc_mapping_db[stable_doc_uuid].update({
                "filename": file_info["file_name"],
                "doc_type": dt,
                "enabler": file_info["enabler"],
                "doc_id_key": filename_doc_id_key,
                "filepath": file_info["file_path"],
                "tenant": tenant,  
                "year": year,
                "status": "Pending" # อัปเดตสถานะเป็น Pending ก่อน Ingest
            })
        else:
            # สร้าง Entry ใหม่
            new_uuid = stable_uuid_from_path
            file_info["stable_doc_uuid"] = new_uuid
            doc_mapping_db[new_uuid] = {
                "file_name": file_info["file_name"],
                "doc_type": dt,
                "enabler": file_info["enabler"],
                "doc_id_key": filename_doc_id_key,
                "filepath": file_info["file_path"],
                "tenant": tenant, 
                "year": year,
                "notes": "",
                "statement_id": "",
                "chunk_uuids": [], 
                "status": "Pending"
            }
            uuid_from_path_lookup[file_info["file_path"]] = new_uuid

    # 3. Load & Chunk (Sequential/Parallel logic is fine)
    all_chunks: List[Document] = []
    results: List[Dict[str, Any]] = []
    
    # ... [Original content of sequential/parallel execution remains unchanged] ...
    def _process_file_task(file_info: Dict[str, str]):
        return process_document(
            file_path=file_info["file_path"],
            file_name=file_info["file_name"],
            stable_doc_uuid=file_info["stable_doc_uuid"],
            doc_type=file_info["doc_type"],
            enabler=file_info["enabler"],
            base_path=base_path,
            year=year,
            tenant=tenant, 
            version=version
        )
        
    if sequential:
        for idx, file_info in enumerate(files_to_process, 1):
            f = file_info["file_name"]
            stable_doc_uuid = file_info["stable_doc_uuid"]
            try:
                chunks, doc_id, dt = _process_file_task(file_info)
                all_chunks.extend(chunks)
                results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks), "tenant": tenant, "year": year})
                if debug or dry_run:
                    logger.debug(f"[DRY RUN] {f}: {len(chunks)} chunks, UUID={stable_doc_uuid}")
                if idx % log_every == 0:
                    logger.info(f"Processed {idx}/{len(files_to_process)} files...")
            except Exception as e:
                results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": file_info["doc_type"], "status": "failed_chunk", "error": str(e), "tenant": tenant, "year": year})
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
                    results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks), "tenant": tenant, "year": year})
                except Exception as e:
                    results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": fi["doc_type"], "status": "failed_chunk", "error": str(e), "tenant": tenant, "year": year})
                    logger.error(f"❌ CHUNK/PROCESS FAILED: {f} (ID: {stable_doc_uuid}) - {type(e).__name__} ({e})")
                if idx % log_every == 0:
                    logger.info(f"Processed {idx}/{len(files_to_process)} files...")


    # 4. Group & Index Chunks
    chunks_by_collection: Dict[str, List[Document]] = {}
    for chunk in all_chunks:
        dt = chunk.metadata.get("doc_type", "default")
        ena = chunk.metadata.get("enabler", None)
        
        # 🎯 FIX 2: ใช้ get_target_dir เพื่อสร้าง collection name ที่ยาวพอ (เช่น 'evidence_km')
        collection_name = get_target_dir(dt, ena) # <-- แก้ไขบรรทัดนี้
        
        chunks_by_collection.setdefault(collection_name, []).append(chunk)
        

    for coll_name, coll_chunks in chunks_by_collection.items():
        if not coll_chunks:
            logger.warning(f"Skipping collection '{coll_name}' - 0 chunks found.")
            continue
        
        # 🎯 FIX: ดึง Doc Type ที่ถูกต้องจาก Collection Name
        doc_type_for_db, _ = _parse_collection_name(coll_name)
        doc_mapping_db = doc_mapping_dbs.get(doc_type_for_db)
        
        if not doc_mapping_db:
             logger.error(f"FATAL: Mapping DB for doc_type '{doc_type_for_db}' not found during indexing of collection '{coll_name}'. Skipping.")
             continue
             
        logger.info(f"--- Indexing collection '{coll_name}' ({len(coll_chunks)} chunks) ---")
        for i in range(0, len(coll_chunks), batch_size):
            batch = coll_chunks[i:i+batch_size]
            if dry_run:
                logger.info(f"[DRY RUN] Would index {len(batch)} chunks into collection '{coll_name}' for {tenant}/{year}")
            else:
                try:
                    create_vectorstore_from_documents(
                        chunks=batch,
                        collection_name=coll_name,
                        doc_mapping_db=doc_mapping_db, # ใช้ DB ที่ดึงมาจาก doc_mapping_dbs
                        tenant=tenant, 
                        year=year,     
                        base_path=base_path
                    )
                except Exception as e:
                    logger.error(f"Error indexing chunks {i+1}-{i+len(batch)}: {e}")

    # 5. Save all updated mapping files
    # 📌 FIX: ต้องวนซ้ำเพื่อบันทึกไฟล์ Mapping ที่แยกตาม Enabler ด้วย
    for dt, db in doc_mapping_dbs.items():
         # ถ้าเป็น Evidence ต้องมีการแยก Enabler ออกมาก่อนบันทึก
         if dt.lower() == EVIDENCE_DOC_TYPES.lower():
             db_by_enabler: Dict[Optional[str], Dict[str, Dict[str, Any]]] = {}
             
             for s_uuid, entry in db.items():
                 enabler_key = entry.get("enabler") # Enabler ที่ถูกบันทึกไว้ใน Entry
                 # 📌 FIX: กรองเฉพาะ Entry ที่มี Tenant/Year ตรงกันก่อนบันทึก
                 if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == year:
                      db_by_enabler.setdefault(enabler_key, {})[s_uuid] = entry
                 
             for enabler_key, small_db in db_by_enabler.items():
                  save_doc_id_mapping(small_db, dt, tenant, year, enabler=enabler_key)
         else:
             # Doc Type อื่นๆ (เช่น document)
             # 📌 FIX: กรองเฉพาะ Entry ที่มี Tenant/Year ตรงกันก่อนบันทึก
             db_to_save = {
                 s_uuid: entry 
                 for s_uuid, entry in db.items() 
                 if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == year
             }
             save_doc_id_mapping(db_to_save, dt, tenant, year) 
         
    logger.info(f"✅ Batch ingestion process finished (dry_run={dry_run}) for {tenant}/{year}.")
    return results


# -------------------- [REVISED] wipe_vectorstore --------------------
def wipe_vectorstore(
    doc_type_to_wipe: str = 'all', 
    enabler: Optional[str] = None, 
    tenant: str = "pwa",        
    year: int = 2568,           
    base_path: str = VECTORSTORE_DIR
):
    """Wipes the vector store directory/collection(s) and updates the doc_id_mapping file (Multi-Tenant/Year)."""
    
    doc_type_to_wipe_lower = doc_type_to_wipe.lower()
    collections_to_delete: List[str] = []
    
    # 🎯 FIXED: ใช้ get_collection_parent_dir ที่ถูกต้อง
    
    # Root สำหรับ Evidence (แยกปี)
    tenant_vectorstore_root_year = get_collection_parent_dir(tenant, year, EVIDENCE_DOC_TYPES) 
    # Root สำหรับ Non-Evidence (common) - ใช้ doc_type ที่ไม่ใช่ Evidence
    tenant_vectorstore_root_common = get_collection_parent_dir(tenant, year, DEFAULT_DOC_TYPES) 
    
    collections_to_delete_with_root = []
    
    doc_types_affected = set() 
    
    if doc_type_to_wipe_lower == 'all':
        logger.warning(f"Wiping ALL collections in {tenant_vectorstore_root_year} and {tenant_vectorstore_root_common}")

        all_roots = {tenant_vectorstore_root_year, tenant_vectorstore_root_common}
        
        doc_types_affected.update([dt.lower() for dt in SUPPORTED_DOC_TYPES])
        
        for root_path in all_roots:
             if os.path.exists(root_path):
                  for f in os.listdir(root_path):
                       col_path = os.path.join(root_path, f)
                       if os.path.isdir(col_path):
                            doc_type_from_col, _ = _parse_collection_name(f) 
                            # 📌 FIX: ต้องรวม Doc Type ที่เป็น Evidence ด้วย
                            if doc_type_from_col in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
                                collections_to_delete_with_root.append((root_path, f, doc_type_from_col))
                            else:
                                logger.warning(f"Collection '{f}' does not map to a supported doc_type. Skipping map update for this collection.")

                            
    elif doc_type_to_wipe_lower == EVIDENCE_DOC_TYPES.lower():
        root_path = tenant_vectorstore_root_year
        doc_types_affected.add(EVIDENCE_DOC_TYPES.lower())
        if enabler and enabler.upper() in SUPPORTED_ENABLERS:
            collection_name = get_target_dir(EVIDENCE_DOC_TYPES, enabler)
            collections_to_delete_with_root = [(root_path, collection_name, EVIDENCE_DOC_TYPES.lower())]
        else:
            logger.warning("Wiping ALL evidence_* collections.")
            evidence_paths = glob.glob(os.path.join(root_path, f"{EVIDENCE_DOC_TYPES.lower()}_*"))
            collections_to_delete_with_root = [
                (root_path, os.path.basename(p), EVIDENCE_DOC_TYPES.lower()) for p in evidence_paths
            ]
            
    elif doc_type_to_wipe_lower in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
        root_path = tenant_vectorstore_root_common
        collection_name = get_target_dir(doc_type_to_wipe_lower, None)
        doc_types_affected.add(doc_type_to_wipe_lower)
        collections_to_delete_with_root = [(root_path, collection_name, doc_type_to_wipe_lower)]
        
    else:
        logger.error(f"Invalid doc_type_to_wipe: {doc_type_to_wipe}")
        return

    # 1. Delete collection folders
    deleted_collections_names = set()
    deleted_collections_map: Dict[str, Set[str]] = {} 
    
    for root_path, col_name, dt in collections_to_delete_with_root:
        col_path = os.path.join(root_path, col_name) 
        if os.path.exists(col_path):
            try:
                shutil.rmtree(col_path)
                logger.info(f"✅ Deleted vector store collection: {col_path}")
                deleted_collections_names.add(col_name)
                deleted_collections_map.setdefault(dt, set()).add(col_name)
            except OSError as e:
                logger.error(f"❌ Error deleting collection {col_name}: {e}")

    # 2. Update Mapping files
    for dt in doc_types_affected:
        # 🎯 FIX: ต้องโหลด mapping ทั้งหมดรวม Enabler ด้วย
        mapping_db = load_doc_id_mapping(dt, tenant, year) 
        uuids_to_keep = {}
        deletion_count = 0
        
        cols_to_check = deleted_collections_map.get(dt, set())
        
        # 📌 NEW: สร้าง Collection Names ที่เกี่ยวข้องกับ Doc Type นี้ที่ถูกลบไป
        if doc_type_to_wipe_lower == 'all':
             # ถ้า wipe all ให้เช็คทุก collection ที่ถูกลบไป
             cols_to_check = deleted_collections_map.get(dt, set())
        elif doc_type_to_wipe_lower == dt:
             # ถ้า wipe เฉพาะ Doc Type นี้
             if dt == EVIDENCE_DOC_TYPES.lower() and enabler and enabler.upper() in SUPPORTED_ENABLERS:
                 cols_to_check = {get_target_dir(dt, enabler)}
             elif dt == EVIDENCE_DOC_TYPES.lower() and not enabler:
                 cols_to_check = {get_target_dir(dt, ena) for ena in SUPPORTED_ENABLERS}
             else:
                 cols_to_check = {get_target_dir(dt, None)}


        for s_uuid, entry in mapping_db.items():
            entry_doc_type = entry.get("doc_type")
            entry_enabler = entry.get("enabler")
            
            # 🎯 FIX: เช็ค Tenant และ Year ใน Mapping File ด้วย 
            if str(entry.get("tenant")).lower() != str(tenant).lower() or entry.get("year") != year:
                 uuids_to_keep[s_uuid] = entry
                 continue

            entry_collection = get_target_dir(entry_doc_type, entry_enabler)
            
            # ถ้า collection ที่ entry อ้างถึง ถูกลบไป
            if entry_collection in cols_to_check:
                deletion_count += 1
            else:
                uuids_to_keep[s_uuid] = entry
                
        if deletion_count > 0:
            # 📌 FIX: ต้องมีการแยก Enabler ในการบันทึก
            if dt.lower() == EVIDENCE_DOC_TYPES.lower():
                 db_by_enabler: Dict[Optional[str], Dict[str, Dict[str, Any]]] = {}
                 for s_uuid, entry in uuids_to_keep.items():
                      enabler_key = entry.get("enabler")
                      db_by_enabler.setdefault(enabler_key, {})[s_uuid] = entry
                 
                 for enabler_key, small_db in db_by_enabler.items():
                      save_doc_id_mapping(small_db, dt, tenant, year, enabler=enabler_key)
            else:
                 save_doc_id_mapping(uuids_to_keep, dt, tenant, year) 
                 
            logger.info(f"🧹 Removed {deletion_count} entries from mapping file for deleted collections (Doc Type: {dt}) of {tenant}/{year}.")
            
        # 📌 FIXED: ถ้า Doc Type ถูกสั่ง wipe ทั้งหมด (all) ให้ลบไฟล์ Mapping ทิ้ง
        if doc_type_to_wipe_lower == 'all' or doc_type_to_wipe_lower == dt:
             # หา Path ทั้งหมดที่เกี่ยวข้อง (สำหรับ Doc Type นี้)
             paths_to_check: List[str] = []
             if dt.lower() == EVIDENCE_DOC_TYPES.lower():
                  for ena in SUPPORTED_ENABLERS:
                       paths_to_check.append(get_doc_id_mapping_path(tenant, str(year), ena))
             else:
                  paths_to_check.append(get_doc_id_mapping_path(tenant, str(year), None))
                  
             for mapping_path in paths_to_check:
                  # 📌 FIX: ใช้การเช็คว่าไฟล์ว่างเปล่า (ขนาด 0) หรือไม่ 
                  # (เพราะ save_doc_id_mapping จะลบไฟล์ถ้าข้อมูลว่างเปล่า)
                  if os.path.exists(mapping_path) and os.path.getsize(mapping_path) < 10: # น้อยกว่า 10 bytes
                       try:
                           os.remove(mapping_path)
                           logger.info(f"✅ Deleted (almost) empty Doc ID mapping file: {mapping_path}")
                       except OSError as e:
                           logger.error(f"❌ Error deleting mapping file: {e}")


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
        
    doc_mapping_db = load_doc_id_mapping(doc_type, tenant, year) 
    
    if stable_doc_uuid not in doc_mapping_db:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} not found in mapping for {tenant}/{year}.")
        return False

    doc_entry = doc_mapping_db[stable_doc_uuid]
    all_chunk_uuids = doc_entry.get("chunk_uuids", [])
    
    doc_type_from_map = doc_entry.get("doc_type")
    enabler_from_map = doc_entry.get("enabler")
    
    final_doc_type = doc_type or doc_type_from_map
    final_enabler = enabler or enabler_from_map
    
    if not final_doc_type:
         logger.error(f"Cannot delete {stable_doc_uuid}: doc_type not found in mapping or provided.")
         return False
         
    final_collection_name = get_target_dir(final_doc_type, final_enabler)
    
    if not all_chunk_uuids:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} has no chunk UUIDs recorded.")
    else:
        # 🎯 FIXED: ใช้ get_collection_parent_dir
        tenant_vectorstore_parent_dir = get_collection_parent_dir(tenant, year, final_doc_type)
        persist_directory = os.path.join(tenant_vectorstore_parent_dir, final_collection_name) 
        
        if not os.path.isdir(persist_directory):
            logger.warning(f"Vectorstore directory not found for collection '{final_collection_name}' for {tenant}/{year}. Skipping Chroma deletion.")
        else:
            try:
                vectorstore = get_vectorstore(
                    collection_name=final_collection_name, 
                    tenant=tenant, 
                    year=year, 
                    base_path=base_path
                ) 
                vectorstore.delete(ids=all_chunk_uuids)
                logger.info(f"✅ Successfully deleted {len(all_chunk_uuids)} chunks from collection '{final_collection_name}' for UUID: {stable_doc_uuid}")
            except Exception as e:
                logger.error(f"❌ Error during Chroma deletion for collection '{final_collection_name}' (UUID: {stable_doc_uuid}): {e}")
            
    # 🎯 FIX: ลบ Entry ออกจาก Mapping DB ที่โหลดมา
    del doc_mapping_db[stable_doc_uuid]
    
    # 📌 FIX: บันทึก Mapping DB คืน (ต้องมีการแยก Enabler ถ้าเป็น Evidence)
    if final_doc_type.lower() == EVIDENCE_DOC_TYPES.lower():
         db_to_save: Dict[str, Dict[str, Any]] = {}
         
         # สร้าง DB ที่มีเฉพาะ Enabler ที่เกี่ยวข้อง
         for s_uuid, entry in doc_mapping_db.items():
              # 📌 FIX: ต้องเช็ค Tenant/Year ด้วย
              if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == year and entry.get("enabler") == final_enabler:
                   db_to_save[s_uuid] = entry
                   
         # บันทึกเฉพาะไฟล์ Mapping ที่เกี่ยวข้องกับ Enabler ที่ถูกลบ
         save_doc_id_mapping(db_to_save, final_doc_type, tenant, year, enabler=final_enabler)
    else:
         # สำหรับ Doc Type อื่นๆ (เช่น document) บันทึกเฉพาะ Entry ที่เกี่ยวข้องกับ Tenant/Year
         db_to_save = {
             s_uuid: entry 
             for s_uuid, entry in doc_mapping_db.items() 
             if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == year
         }
         save_doc_id_mapping(db_to_save, final_doc_type, tenant, year) 
        
    return True

# -------------------- [REVISED] list_documents --------------------
# -------------------- [REVISED] list_documents --------------------
def list_documents(
    doc_types: Optional[List[str]] = None,
    enabler: Optional[str] = None, 
    tenant: str = "pwa",        
    year: int = 2568,           
    show_results: str = "ingested" 
) -> Dict[str, Any]: 
    
    # 🎯 FIX 1a: ต้องรวบรวมข้อมูลจาก Mapping File หลายไฟล์ หาก doc_types เป็น 'all'
    doc_types_reqs = {dt.lower() for dt in doc_types} if doc_types and doc_types[0] and doc_types[0].lower() != "all" else set()
    
    if not doc_types_reqs:
        doc_types_to_load = {dt.lower() for dt in SUPPORTED_DOC_TYPES}
    else:
        doc_types_to_load = doc_types_reqs
        
    doc_mapping_db = {}
    enabler_req = (enabler or "").upper()
    
    # 📌 FIX 1b: แก้ไข Logic การโหลด Mapping File ให้ส่งค่า enabler
    for dt in doc_types_to_load:
         current_enabler = None
         
         # ตรวจสอบและกำหนด enabler เฉพาะเมื่อเป็น EVIDENCE
         if dt == EVIDENCE_DOC_TYPES.lower() and enabler_req and enabler_req in SUPPORTED_ENABLERS:
             current_enabler = enabler_req
             
         # FIX: ส่งค่า enabler เข้าไปใน load_doc_id_mapping
         doc_mapping_db.update(load_doc_id_mapping(dt, tenant, year, enabler=current_enabler)) 
         
    all_docs: Dict[str, Any] = {}
    
    # REVISED: กรอง mapping file ให้เหลือเฉพาะ Tenant/Year ที่ระบุ
    # Note: เราต้องกรองจาก doc_mapping_db ทั้งหมด เพราะ load_doc_id_mapping อาจคืนค่าทั้งหมดมา
    filepath_to_stable_uuid: Dict[str, str] = {
        entry["filepath"]: s_uuid 
        for s_uuid, entry in doc_mapping_db.items() 
        if "filepath" in entry and str(entry.get("tenant")).lower() == str(tenant).lower() and entry.get("year") == year
    } 
            
    doc_type_reqs = {dt.lower() for dt in doc_types} if doc_types and doc_types[0] and doc_types[0].lower() != "all" else set()
    
    # REVISED: ใช้ _get_source_dir ที่ถูกแก้ไขแล้ว
    source_dirs_to_scan: List[str] = []

    if not doc_type_reqs: 
        # Scan All Doc Types
        for dt in SUPPORTED_DOC_TYPES:
            if dt.lower() == EVIDENCE_DOC_TYPES.lower():
                if enabler_req and enabler_req in SUPPORTED_ENABLERS:
                     source_dirs_to_scan.append(_get_source_dir(tenant, year, dt, enabler_req)) 
                else:
                     for ena in SUPPORTED_ENABLERS:
                         source_dirs_to_scan.append(_get_source_dir(tenant, year, dt, ena)) 
            else:
                source_dirs_to_scan.append(_get_source_dir(tenant, year, dt, None)) 
    else:
        # Scan Only Requested Doc Types
        for dt_req in doc_type_reqs:
            if dt_req == EVIDENCE_DOC_TYPES.lower():
                if enabler_req and enabler_req in SUPPORTED_ENABLERS:
                    source_dirs_to_scan.append(_get_source_dir(tenant, year, dt_req, enabler_req)) 
                else:
                    for ena in SUPPORTED_ENABLERS:
                        source_dirs_to_scan.append(_get_source_dir(tenant, year, dt_req, ena)) 
            elif dt_req in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
                source_dirs_to_scan.append(_get_source_dir(tenant, year, dt_req, None)) 
                
    # (Rest of the function logic is fine)
    for root_to_scan in set(source_dirs_to_scan): 
        if not os.path.isdir(root_to_scan):
            logger.debug(f"Source directory not found: {root_to_scan}. Skipping scan.")
            continue
            
        # FIX: ต้องดึง doc_type และ enabler จาก root_to_scan เหมือน ingest_all_files
        path_parts = root_to_scan.strip(os.sep).split(os.sep)
        
        current_path_segment = path_parts[-1].lower()
        
        doc_type_from_path = None
        resolved_enabler = None

        if current_path_segment.upper() in SUPPORTED_ENABLERS: 
             resolved_enabler = current_path_segment.upper()
             doc_type_from_path = path_parts[-2].lower() if len(path_parts) >= 2 else None
        elif current_path_segment in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
             doc_type_from_path = current_path_segment
             resolved_enabler = None 

        if not doc_type_from_path:
             logger.warning(f"Could not determine doc_type from path: {root_to_scan} during list_documents. Skipping.")
             continue
             
        original_doc_type = doc_type_from_path
        
        for root, _, filenames in os.walk(root_to_scan):
            if root != root_to_scan: continue 
            
            for f in filenames:
                file_extension = os.path.splitext(f)[1].lower()
                if f.startswith('.') or file_extension not in SUPPORTED_TYPES:
                    continue
                    
                file_path = os.path.join(root, f)
                file_name_no_ext = os.path.splitext(f)[0]
                filename_doc_id_key = _normalize_doc_id(file_name_no_ext)
                
                # 1. ลองค้นหาด้วย Absolute Path (วิธีการปกติ)
                stable_doc_uuid = filepath_to_stable_uuid.get(file_path)
                
                # 📌 FIX 3a: เพิ่ม Fallback Search ถ้าหาด้วย Absolute Path ไม่เจอ
                if stable_doc_uuid is None:
                    # 2. ลองค้นหาด้วย filename (ซึ่งอยู่ใน Mapping File)
                    for s_uuid, entry in doc_mapping_db.items():
                        # ตรวจสอบว่า file_name ตรงกันหรือไม่
                        if entry.get("file_name") == f: 
                            stable_doc_uuid = s_uuid
                            break
                            
                doc_entry = doc_mapping_db.get(stable_doc_uuid) if stable_doc_uuid else None
                
                # Logic การตรวจสอบสถานะ Ingested
                chunk_uuids = doc_entry.get("chunk_uuids", []) if doc_entry else []
                chunk_count = len(chunk_uuids)
                
                # 💡 FIX 3b: ถ้ามี Stable ID แต่ไม่มี chunk_count (เพราะ Ingest ไม่ได้บันทึกไว้) 
                # ให้ถือว่า Ingested สำเร็จแล้ว (กำหนด chunk_count เป็น 1)
                if stable_doc_uuid is not None and chunk_count == 0:
                    chunk_count = 1 
                
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
                    "tenant": tenant,          
                    "year": year,              
                    "upload_date": upload_date,
                    "chunk_count": chunk_count,
                    "status": "Ingested" if is_ingested else "Pending", 
                    "size": file_size,
                }
                all_docs[final_doc_id] = doc_info

    total_supported_files = len(all_docs) 
    
    show_results_lower = show_results.lower()
    filtered_docs_dict: Dict[str, Any] = {}
    
    if total_supported_files == 0:
        doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
        logger.warning(f"⚠️ No documents found in DATA_DIR matching the requested type '{doc_types_str}' (Enabler: {enabler_req or 'ALL'}) for {tenant}/{year}.")
        return filtered_docs_dict 
    
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
            if d.get('status', '').lower() == 'pending'
        }
        display_count_x = len(filtered_docs_dict) 
        filter_name = "PENDING/NOT INGESTED (Physical Files Exist)"
        
    elif show_results_lower == "full":
        filtered_docs_dict = all_docs
        display_count_x = len(all_docs) 
        filter_name = "FULL (All Supported Files)"
        
    else:
        filtered_docs_dict = {
            k: d for k, d in all_docs.items() 
            if d.get('status', '').lower() == 'ingested' and not k.startswith('TEMP_ID__')
        }
        filter_name = "INGESTED (Default / Unique Doc IDs)"
        display_count_x = len(filtered_docs_dict)

    display_list = []
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
            "ref_doc_id": doc_info["doc_id_key"],
            "tenant": doc_info["tenant"], 
            "year": doc_info["year"]     
        })
        
    display_list.sort(key=lambda x: (x["doc_type"], x["file_name"]))
    
    doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
    
    print(f"\nFound {display_count_x}/{total_supported_files} supported documents for type '{doc_types_str}' (Tenant: {tenant}, Year: {year}, Filter: {filter_name}):\n")

    if not display_list:
        print("--- No documents found matching the filter criteria to display ---")
        return filtered_docs_dict 

    UUID_COL_WIDTH = 65
    NEW_TABLE_WIDTH = 197 
    
    print("-" * NEW_TABLE_WIDTH)
    print(f"{'DOC ID (Stable/Temp)':<{UUID_COL_WIDTH}} | {'TENANT':<7} | {'YEAR':<4} | {'FILENAME':<30} | {'EXT':<5} | {'TYPE':<10} | {'ENB':<5} | {'SIZE(MB)':<9} | {'STATUS':<10}")
    print("-" * NEW_TABLE_WIDTH)
    
    for info in display_list:
        full_doc_id = info['doc_id'] 
        file_name, file_ext = os.path.splitext(info['file_name'])
        short_filename = file_name[:28] if len(file_name) > 28 else file_name 
        file_ext = file_ext[1:].upper() if file_ext else '-' 
        size_str = f"{info['size_mb']:.2f}"
        enabler_display = info['enabler'] 
        
        print(
            f"{full_doc_id:<{UUID_COL_WIDTH}} | " 
            f"{info['tenant']:<7} | " 
            f"{info['year']:<4} | "   
            f"{short_filename:<30} | " 
            f"{file_ext:<5} | "
            f"{info['doc_type']:<10} | "
            f"{enabler_display:<5} | " 
            f"{size_str:<9} | "
            f"{info['status']:<10}"
        )
    print("-" * NEW_TABLE_WIDTH)
    print(f"\nFound {display_count_x}/{total_supported_files} supported documents for type '{doc_types_str}' (Tenant: {tenant}, Year: {year}, Filter: {filter_name}):\n")

    
    return filtered_docs_dict

# -------------------- Main Execution (Sample) --------------------
if __name__ == "__main__":
    import sys
    # Example usage (add proper argparse here)
    # The original main execution logic is assumed to be here, for simplicity:
    # ingest_all_files(tenant="pea", year=2568, doc_type="evidence", enabler="km")
    
    # 📌 NOTE: สำหรับการทดสอบ ต้องมั่นใจว่าไฟล์ config.global_vars มีการกำหนดค่าที่ถูกต้องแล้ว
    logger.info("ingest.py started.")

    # 💡 เพื่อให้โค้ดสมบูรณ์ จะเพิ่มฟังก์ชันหลักที่ใช้ argparse
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Ingest documents into Chroma Vector Store (Multi-Tenant/Year).")
        parser.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help=f"Tenant code (default: {DEFAULT_TENANT}).")
        parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help=f"Fiscal year (default: {DEFAULT_YEAR}).")
        parser.add_argument("--doc-type", type=str, default="all", help=f"Document type to ingest (e.g., 'document', 'evidence', 'all').")
        parser.add_argument("--enabler", type=str, default=None, help="Enabler code, required for 'evidence' doc type (e.g., 'km').")
        parser.add_argument("--dry-run", action="store_true", help="Process files but do not save to vectorstore or mapping.")
        parser.add_argument("--sequential", action="store_true", help="Process files sequentially (slower, less resource intensive).")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
        parser.add_argument("--wipe", action="store_true", help="Wipe the target vector store collection(s) and associated mappings before ingesting.")
        
        args = parser.parse_args()
        
        if args.wipe:
            logger.warning(f"!!! WIPE MODE ACTIVATED !!!")
            wipe_vectorstore(
                doc_type_to_wipe=args.doc_type, 
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=args.year
            )
        
        ingest_all_files(
            tenant=args.tenant,
            year=args.year,
            doc_type=args.doc_type,
            enabler=args.enabler,
            dry_run=args.dry_run,
            sequential=args.sequential,
            debug=args.debug
        )
        
    except ImportError:
         print("--- RUNNING INGEST.PY STANDALONE FAILED: Missing argparse module ---")
         print("--- Please run the script with necessary arguments or wrap it in a proper execution environment. ---")
         
    except Exception as e:
         logger.critical(f"FATAL ERROR DURING MAIN EXECUTION: {e}", exc_info=True)
         print(f"--- FATAL ERROR: {e} ---")