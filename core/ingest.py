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
    DEFAULT_YEAR,
    EVIDENCE_MAPPING_FILENAME_SUFFIX,
    EMBEDDING_MODEL_NAME
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
    # 🎯 FIX: ลบ doc_id_key ออกไป เพราะไม่ใช้งานแล้ว
    stable_doc_uuid: str, 
    # 🟢 เพิ่มพารามิเตอร์ Metadata ที่จำเป็นเข้ามา
    doc_type: str, 
    enabler: Optional[str] = None,
    subject: Optional[str] = None,
    # ----------------------------------------------------
    year: Optional[int] = None,
    version: str = "v1",
    metadata: Optional[Dict[str, Any]] = None,
    ocr_pages: Optional[Iterable[int]] = None
) -> List[Document]:
    """
    Load document, inject metadata, clean, and split into chunks.
    ใช้ 64-char stable_doc_uuid เป็นตัวระบุเอกสารหลัก (Primary Document ID).
    """
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
        
        # 🟢 FIX: สร้างชุด Metadata จากพารามิเตอร์ใหม่ทั้งหมด
        doc_metadata = {}
        if metadata:
            # ใช้ metadata ที่ส่งมาจาก process_document เป็นฐาน
            doc_metadata.update(metadata) 
        
        # Metadata ใหม่/หลัก (override ถ้ามีใน metadata ที่ส่งมา)
        doc_metadata["doc_type"] = doc_type 
        if enabler: doc_metadata["enabler"] = enabler

        cleaned_subject = subject.strip() if subject else None
        if cleaned_subject: doc_metadata["subject"] = cleaned_subject

        # if subject: doc_metadata["subject"] = subject
        
        # Metadata เดิม
        if year: doc_metadata["year"] = year
        doc_metadata["version"] = version
        doc_metadata["stable_doc_uuid"] = stable_doc_uuid 
        
        base_filename = os.path.basename(file_path)
        doc_metadata["source_filename"] = base_filename
        doc_metadata["source"] = base_filename
        
        # Update Metadata เข้าไปใน Document
        try:
             d.metadata.update(doc_metadata)
             d.metadata = _safe_filter_complex_metadata(d.metadata)
        except Exception:
             d.metadata["injected_metadata_fail"] = str(doc_metadata) 
        
        # NOTE: หากต้องการให้ Metadata Field 'doc_id' เก็บค่า 64-char UUID ด้วย 
        # ต้องใช้ d.metadata["doc_id"] = stable_doc_uuid 
        
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
        unique_chunk_id = f"{stable_doc_uuid}_{idx}" 
        
        c.metadata["chunk_uuid"] = unique_chunk_id 
        c.metadata["chunk_index"] = idx
        c.metadata = _safe_filter_complex_metadata(c.metadata) 
        
    logger.info(f"Loaded and chunked {os.path.basename(file_path)} -> {len(chunks)} chunks.")
    return chunks

# -------------------- [REVISED] Process single document (Cleaned & Final) --------------------
def process_document(
    file_path: str,
    file_name: str,
    stable_doc_uuid: str, 
    doc_type: Optional[str] = None,
    enabler: Optional[str] = None, 
    subject: Optional[str] = None,  # 🟢 เพิ่ม subject
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
    if subject: # 🟢 ใส่ subject ถ้ามีการระบุมา (สำหรับ Global KM)
        injected_metadata["subject"] = subject
        
    filter_id_value = filename_doc_id_key 
    logger.critical(f"================== START DEBUG INGESTION: {file_name} ==================")
    logger.critical(f"🔍 DEBUG ID (stable_doc_uuid, 64-char Hash): {len(stable_doc_uuid)}-char: {stable_doc_uuid[:34]}...")
    logger.critical(f"✅ FINAL ID TO STORE (34-char Ref ID): {len(filter_id_value)}-char: {filter_id_value[:34]}...")

    # 🎯 FIX: ส่ง Metadata ทั้งหมดผ่าน dict ไปให้ load_and_chunk_document
    chunks = load_and_chunk_document(
        file_path=file_path,
        stable_doc_uuid=stable_doc_uuid,
        doc_type=doc_type, # 🟢 ส่ง Doc Type โดยตรง
        enabler=resolved_enabler, # 🟢 ส่ง Enabler โดยตรง
        subject=subject, # 🟢 ส่ง Subject โดยตรง
        year=year,
        version=version,
        metadata=injected_metadata, # 👈 metadata อื่น ๆ ที่เหลือ
        ocr_pages=ocr_pages
    )
    
    # 🔴 FIX: ลบการวนลูปซ้ำซ้อนด้านล่างนี้ทิ้งทั้งหมด
    
    if chunks:
         logger.debug(f"Chunk metadata preview: {chunks[0].metadata}")
        
    return chunks, stable_doc_uuid, doc_type


# -------------------- Vectorstore / Mapping Utilities --------------------
# 📌 Assumption: Chroma, HuggingFaceEmbeddings, os, logger, 
# EMBEDDING_MODEL_NAME, VECTORSTORE_DIR, _parse_collection_name, 
# และ get_collection_parent_dir ถูก import มาแล้ว

_VECTORSTORE_SERVICE_CACHE: dict = {}

def get_vectorstore(
    collection_name: str = "default",
    tenant: str = "pea",
    year: int = 2568,
    base_path: str = VECTORSTORE_DIR
) -> Chroma:
    """
    เวอร์ชัน PEA 2568 แท้ ๆ (ไม่มี rag_ prefix ใด ๆ) 
    ใช้ BAAI/bge-m3 สำหรับการฝังข้อมูล
    """

    # === 1. ใช้ชื่อตรง ๆ ไม่ต้องเติม prefix อัตโนมัติ ===
    if len(collection_name) < 3:
        logger.warning(
            f"Collection name '{collection_name}' ค่อนข้างสั้น แนะนำให้ใช้ชื่ออย่างน้อย 6 ตัวอักษร "
            f"(เช่น evidence_km, km42l103) เพื่อป้องกันการชนกันของชื่อ"
        )

    # === 2. สร้าง path ที่ถูกต้องตามโครงสร้าง PEA ===
    # 📌 ใช้ฟังก์ชันช่วยสร้าง path ที่ท่านได้กำหนดไว้
    try:
        doc_type_for_path, _ = _parse_collection_name(collection_name)
        tenant_dir = get_collection_parent_dir(tenant, year, doc_type_for_path) 
    except NameError:
        # Fallback สำหรับการทดสอบ หากไม่มีฟังก์ชันช่วย
        logger.error("Helper functions '_parse_collection_name' or 'get_collection_parent_dir' not found. Using default path.")
        tenant_dir = os.path.join(base_path, tenant, str(year), "km")
        
    persist_directory = os.path.join(tenant_dir, collection_name)  # ใช้ชื่อ collection เต็ม ๆ
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
        # 🎯 FIX 2A: ดึง Chunk UUID ที่สร้างไว้ใน metadata
        chunk_ids_to_add = [c.metadata.get("chunk_uuid") for c in chunks]
        
        # กรองค่า None ออก (หากมี)
        valid_chunks = [c for c, id in zip(chunks, chunk_ids_to_add) if id is not None]
        valid_ids = [id for id in chunk_ids_to_add if id is not None]
        
        if len(valid_ids) != len(chunks):
            logger.warning(f"Found {len(chunks) - len(valid_ids)} chunks without UUIDs. Skipping them.")
            chunks = valid_chunks
            chunk_ids_to_add = valid_ids
            
        if not chunks:
            logger.warning(f"No valid chunks with UUIDs found to index for collection '{collection_name}'. Skipping.")
            return

        # 🎯 FIX 2B: ส่ง Chunk IDs ที่สร้างไว้ไปเป็น Primary ID ของ ChromaDB
        chunk_uuids = vectorstore.add_documents(
            documents=chunks,
            ids=chunk_ids_to_add  # <-- แก้ไข: ใช้ IDs ที่เราสร้างขึ้น
        )
        
        logger.info(f"Indexed {len(chunk_uuids)} chunks into collection '{collection_name}' (tenant={tenant}, year={year}).")

        # Chroma 0.5+ ใช้ PersistentClient + auto-save ทันที ไม่ต้อง persist()
        logger.info(f"Collection '{collection_name}' auto-saved to disk (Chroma 0.5+ behavior).")

    except Exception as e:
        logger.error(f"Error during add_documents for collection '{collection_name}': {e}")
        return

    # 3. อัปเดต Doc ID Mapping Database
    doc_chunk_map: Dict[str, List[str]] = {}
    
    # 📌 Note: chunk_uuids ที่ได้จาก add_documents ตอนนี้คือ ID เดียวกันกับ chunk_ids_to_add
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
        
# =======================================================
# 1. get_doc_id_mapping_path (Utility)
# =======================================================
# โค้ดนี้ใช้ได้แล้ว ถูกแก้ไขให้รองรับ year/enabler เป็น None
def get_doc_id_mapping_path(tenant: str, year: Optional[Union[str, int]] = None, enabler: Optional[str] = None) -> str:
    """
    Constructs the path for the document ID mapping file.
    """
    
    # 1. สร้าง Base Directory
    base_dir = os.path.join(MAPPING_BASE_DIR, tenant)

    # ⚠️ ตรวจสอบ year: ถ้ามี year จะใช้สร้างโฟลเดอร์ และต้องใช้ในชื่อไฟล์ด้วย
    year_str = str(year).strip() if year is not None and str(year).strip() else ""
    if year_str:
        base_dir = os.path.join(base_dir, year_str)
        
    os.makedirs(base_dir, exist_ok=True)
    
    # 2. สร้าง Filename ใหม่
    enabler_str = str(enabler).lower().strip() if enabler is not None and str(enabler).strip() else ""
    
    # Prefix: pea หรือ pea_2568
    prefix = f"{tenant}"
    if year_str:
         prefix += f"_{year_str}"
    
    # Enabler Part: _km หรือว่างเปล่า
    enabler_part = f"_{enabler_str}" if enabler_str else ""
    
    # Filename: pea_2568_km_doc_id_mapping.json หรือ pea_doc_id_mapping.json
    filename = f"{prefix}{enabler_part}_doc_id_mapping.json"
    
    return os.path.join(base_dir, filename)

# =======================================================
# 2. save_doc_id_mapping
# =======================================================
# ต้องรับ year เป็น Optional[int] และส่งต่อไปอย่างถูกต้อง
def save_doc_id_mapping(
    mapping_data: Dict[str, Dict[str, Any]], 
    doc_type: str, 
    tenant: str = "pwa", 
    year: Optional[int] = None, # 💡 ต้องรับเป็น Optional
    enabler: Optional[str] = None
):
    
    # 🎯 FIX: แปลง year เป็น str(year) หรือ None ก่อนส่งให้ Path Utility
    year_for_path = str(year) if year is not None else None
    
    # ใช้ Path Utility ใหม่
    path = get_doc_id_mapping_path(
        tenant=tenant, 
        year=year_for_path, 
        enabler=enabler
    ) 
    
    try:
        # **Logic การสร้างโฟลเดอร์ถูกย้ายไปใน get_doc_id_mapping_path แล้ว**
        
        if not mapping_data:
             if os.path.exists(path):
                  os.remove(path)
                  logger.info(f"✅ Deleted empty Doc ID Mapping file: {path}")
                  return
             return
             
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping_data, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Successfully saved Doc ID Mapping to: {path}")
    except Exception as e:
        logger.error(f"❌ Failed to save Doc ID Mapping: {e}")

# =======================================================
# 3. load_doc_id_mapping
# =======================================================
# ต้องรับ year เป็น Optional[int] และส่งต่อไปอย่างถูกต้อง
def load_doc_id_mapping(
    doc_type: str, 
    tenant: str, 
    year: Optional[int] = None, # 💡 ต้องรับเป็น Optional
    enabler: Optional[str] = None
) -> Dict[str, Any]:
    """
    Loads the document ID mapping dictionary from the specified JSON file path.
    """
    
    # 🎯 FIX: แปลง year เป็น str(year) หรือ None ก่อนส่งให้ Path Utility
    year_for_path = str(year) if year is not None else None

    # ใช้ Path Utility ใหม่
    mapping_file_path = get_doc_id_mapping_path(
        tenant=tenant, 
        year=year_for_path, 
        enabler=enabler
    ) 
    
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
    subject: Optional[str] = None,
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

    # =================================================================
    # 🎯 FIX 1: OVERRIDE LOGIC สำหรับ Global Doc Types (เพื่อแก้ Path ผิด)
    # =================================================================
    DOC_TYPES_WITH_YEAR_AND_ENABLER = [EVIDENCE_DOC_TYPES.lower()]
    
    final_year: Optional[int] = year
    final_enabler: Optional[str] = enabler_req
    
    if doc_type_req.lower() not in DOC_TYPES_WITH_YEAR_AND_ENABLER:
        # ถ้าเป็น Doc Type Global (เช่น document) ให้บังคับใช้ final_year=None, final_enabler=None
        if year is not None:
             logger.warning(f"⚠️ Year ({year}) is ignored for Global Doc Type: {doc_type_req}. Setting year to None.")
        if enabler_req is not None:
             logger.warning(f"⚠️ Enabler ({enabler_req}) is ignored for Global Doc Type: {doc_type_req}. Setting enabler to None.")
             
        final_year = None
        final_enabler = None
        
    logger.info(f"Starting ingest_all_files: Tenant={tenant}, Year={final_year}, doc_type_req='{doc_type_req}', enabler_req='{final_enabler}', subject_req='{subject}'") 

    # 1. รวบรวมไฟล์ (ใช้ _get_source_dir ที่ถูกแก้ไขแล้ว)
    scan_roots: List[str] = []
    
    # 📌 FIX: ปรับ Logic การสร้าง scan_roots โดยใช้ final_year/final_enabler
    if doc_type_req == "all":
        # Scan ทุก Doc Type และทุก Enabler
        for dt in SUPPORTED_DOC_TYPES:
             if dt.lower() == EVIDENCE_DOC_TYPES.lower():
                for ena in SUPPORTED_ENABLERS:
                    scan_roots.append(_get_source_dir(
                        tenant=tenant, 
                        year=final_year, # 💡 ใช้ final_year
                        doc_type=dt, 
                        enabler=ena
                    )) 
             else:
                scan_roots.append(_get_source_dir(
                    tenant=tenant, 
                    year=final_year, # 💡 ใช้ final_year (จะเป็น None)
                    doc_type=dt, 
                    enabler=None 
                )) 
    elif doc_type_req in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
        if doc_type_req == EVIDENCE_DOC_TYPES.lower():
             if final_enabler and final_enabler in SUPPORTED_ENABLERS: # 💡 ใช้ final_enabler
                 # ถ้ากำหนด Doc Type และ Enabler มาชัดเจน
                 scan_roots = [_get_source_dir(
                     tenant=tenant, 
                     year=final_year, # 💡 ใช้ final_year
                     doc_type=doc_type_req, 
                     enabler=final_enabler # 💡 ใช้ final_enabler
                 )] 
             else:
                 # ถ้ากำหนด Evidence แต่ไม่กำหนด Enabler ให้สแกนทุก Enabler
                 for ena in SUPPORTED_ENABLERS:
                     scan_roots.append(_get_source_dir(
                         tenant=tenant, 
                         year=final_year, # 💡 ใช้ final_year
                         doc_type=doc_type_req, 
                         enabler=ena
                     )) 
        else:
             # Doc Type อื่นๆ final_year จะเป็น None แล้ว
             scan_roots = [_get_source_dir(
                 tenant=tenant, 
                 year=final_year, # 💡 ใช้ final_year (จะเป็น None)
                 doc_type=doc_type_req, 
                 enabler=None 
             )] 
    else:
        logger.error(f"Invalid doc_type_req for ingestion: {doc_type_req}")
        return []

    # (Scan loop logic)
    for root_to_scan in set(scan_roots):
        # ... (โค้ดส่วนการหา doc_type_from_path และ resolved_enabler จาก Path) ...
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
    
    # 📌 FIX: ใช้ final_year ในการโหลด Mapping DBs
    doc_mapping_dbs: Dict[str, Dict[str, Dict[str, Any]]] = {
        dt: load_doc_id_mapping(dt, tenant, final_year) for dt in relevant_doc_types
    }
    
    # เตรียม lookup table ที่รวมทุก doc_type เข้าด้วยกัน (สำหรับค้นหา UUID ที่มีอยู่)
    uuid_from_path_lookup: Dict[str, str] = {}
    for dt, db in doc_mapping_dbs.items():
        for s_uuid, entry in db.items():
            # 📌 FIX: ตรวจสอบ filepath, tenant, final_year ใน mapping entry
            if "filepath" in entry and str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year:
                uuid_from_path_lookup[entry["filepath"]] = s_uuid


    for file_info in files_to_process:
        dt = file_info["doc_type"]
        doc_mapping_db = doc_mapping_dbs[dt] # ใช้ DB ที่เฉพาะเจาะจง
        
        filename_doc_id_key = _normalize_doc_id(os.path.splitext(file_info["file_name"])[0])
        file_info["doc_id_key"] = filename_doc_id_key
        
        # 📌 FIX: ใช้ final_year ในการสร้าง UUID
        stable_uuid_from_path = create_stable_uuid_from_path(
            file_info["file_path"], 
            ref_id_key=filename_doc_id_key,
            tenant=tenant, 
            year=final_year      # 💡 ใช้ final_year
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
                "year": final_year, # 💡 ใช้ final_year
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
                "year": final_year, # 💡 ใช้ final_year
                "notes": "",
                "statement_id": "",
                "chunk_uuids": [], 
                "status": "Pending"
            }
            uuid_from_path_lookup[file_info["file_path"]] = new_uuid

    # 3. Load & Chunk (Sequential/Parallel logic is fine)
    all_chunks: List[Document] = []
    results: List[Dict[str, Any]] = []
    
    def _process_file_task(file_info: Dict[str, str]):
        return process_document(
            file_path=file_info["file_path"],
            file_name=file_info["file_name"],
            stable_doc_uuid=file_info["stable_doc_uuid"],
            doc_type=file_info["doc_type"],
            enabler=file_info["enabler"],
            subject=subject, 
            base_path=base_path,
            year=final_year, # 💡 ใช้ final_year
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
                results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks), "tenant": tenant, "year": final_year}) # 💡 ใช้ final_year
                if debug or dry_run:
                    logger.debug(f"[DRY RUN] {f}: {len(chunks)} chunks, UUID={stable_doc_uuid}")
                if idx % log_every == 0:
                    logger.info(f"Processed {idx}/{len(files_to_process)} files...")
            except Exception as e:
                results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": file_info["doc_type"], "status": "failed_chunk", "error": str(e), "tenant": tenant, "year": final_year}) # 💡 ใช้ final_year
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
                    results.append({"file": f, "doc_id": doc_id, "doc_type": dt, "status": "chunked", "chunks": len(chunks), "tenant": tenant, "year": final_year}) # 💡 ใช้ final_year
                except Exception as e:
                    results.append({"file": f, "doc_id": stable_doc_uuid, "doc_type": fi["doc_type"], "status": "failed_chunk", "error": str(e), "tenant": tenant, "year": final_year}) # 💡 ใช้ final_year
                    logger.error(f"❌ CHUNK/PROCESS FAILED: {f} (ID: {stable_doc_uuid}) - {type(e).__name__} ({e})")
                if idx % log_every == 0:
                    logger.info(f"Processed {idx}/{len(files_to_process)} files...")


    # 4. Group & Index Chunks
    chunks_by_collection: Dict[str, List[Document]] = {}
    for chunk in all_chunks:
        dt = chunk.metadata.get("doc_type", "default")
        ena = chunk.metadata.get("enabler", None)
        
        # 🎯 FIX 2: ใช้ get_target_dir เพื่อสร้าง collection name ที่ยาวพอ (เช่น 'evidence_km')
        collection_name = get_target_dir(dt, ena) 
        
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
                logger.info(f"[DRY RUN] Would index {len(batch)} chunks into collection '{coll_name}' for {tenant}/{final_year}") # 💡 ใช้ final_year
            else:
                try:
                    create_vectorstore_from_documents(
                        chunks=batch,
                        collection_name=coll_name,
                        doc_mapping_db=doc_mapping_db, 
                        tenant=tenant, 
                        year=final_year,     # 💡 ใช้ final_year
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
                 if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year: # 💡 ใช้ final_year
                      db_by_enabler.setdefault(enabler_key, {})[s_uuid] = entry
                 
             for enabler_key, small_db in db_by_enabler.items():
                  save_doc_id_mapping(small_db, dt, tenant, final_year, enabler=enabler_key) # 💡 ใช้ final_year
         else:
             # Doc Type อื่นๆ (เช่น document)
             # 📌 FIX: กรองเฉพาะ Entry ที่มี Tenant/Year ตรงกันก่อนบันทึก
             db_to_save = {
                 s_uuid: entry 
                 for s_uuid, entry in db.items() 
                 if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year # 💡 ใช้ final_year
             }
             # final_year ในกรณีนี้จะเป็น None และ save_doc_id_mapping จะสร้าง Path ที่ไม่มีปีให้
             save_doc_id_mapping(db_to_save, dt, tenant, final_year) # 💡 ใช้ final_year
         
    logger.info(f"✅ Batch ingestion process finished (dry_run={dry_run}) for {tenant}/{final_year}.") # 💡 ใช้ final_year
    return results

# -------------------- [REVISED] wipe_vectorstore (FINAL - UNCONDITIONAL MAPPING CLEANUP & EXPLICIT LOGS) --------------------
def wipe_vectorstore(
    doc_type_to_wipe: str = 'all', 
    enabler: Optional[str] = None, 
    tenant: str = "pwa",        
    year: int = 2568,           
    base_path: str = VECTORSTORE_DIR
):
    """Wipes the vector store directory/collection(s) and updates the doc_id_mapping file (Multi-Tenant/Year).
    
    🎯 Includes Logic Override: Sets year=None for Global Doc Types (e.g., 'document') 
    to correctly handle mapping path.
    """
    
    doc_type_to_wipe_lower = doc_type_to_wipe.lower()
    collections_to_delete: List[str] = []

    # =================================================================
    # 🎯 FIX 1: OVERRIDE LOGIC สำหรับ Global Doc Types (เพื่อแก้ Path ผิด)
    # =================================================================
    DOC_TYPES_WITH_YEAR_AND_ENABLER = [EVIDENCE_DOC_TYPES.lower()]
    
    final_year: Optional[int] = year
    
    if doc_type_to_wipe_lower != 'all' and doc_type_to_wipe_lower not in DOC_TYPES_WITH_YEAR_AND_ENABLER:
        logger.warning(f"⚠️ Year ({year}) is ignored for Global Doc Type wipe: {doc_type_to_wipe_lower}. Setting year to None for Mapping/Collection path.")
        final_year = None
    
    # -------------------- ส่วน 1: การกำหนด Collection ที่จะลบ (Vector Store) --------------------
    # Root สำหรับ Evidence (แยกปี) - ใช้ year เดิม
    tenant_vectorstore_root_year = get_collection_parent_dir(tenant, year, EVIDENCE_DOC_TYPES) 
    # Root สำหรับ Non-Evidence (common) - ใช้ final_year (ซึ่งจะเป็น None ถ้า doc_type เป็น document/policy)
    # Note: ในกรณี 'all' ตัวแปร final_year จะเป็น year แต่ root_common จะใช้ final_year=None เมื่อถึงจุดลบ Global
    
    # เราใช้ doc_type_to_wipe_lower ในการกำหนด parent dir
    vectorstore_root_to_use = final_year if doc_type_to_wipe_lower != EVIDENCE_DOC_TYPES.lower() else year
    tenant_vectorstore_root = get_collection_parent_dir(tenant, vectorstore_root_to_use, doc_type_to_wipe_lower)
    
    collections_to_delete_with_root = []
    
    doc_types_affected = set() 
    
    if doc_type_to_wipe_lower == 'all':
        # ในโหมด 'all' ต้องเช็คทั้ง Root ที่มีปี (Evidence) และ Root ที่ไม่มีปี (Global)
        tenant_vectorstore_root_common = get_collection_parent_dir(tenant, None, DEFAULT_DOC_TYPES)

        logger.warning(f"Wiping ALL collections in {tenant_vectorstore_root_year} and {tenant_vectorstore_root_common}")

        all_roots = {tenant_vectorstore_root_year, tenant_vectorstore_root_common}
        
        doc_types_affected.update([dt.lower() for dt in SUPPORTED_DOC_TYPES])
        
        for root_path in all_roots:
             if os.path.exists(root_path):
                  for f in os.listdir(root_path):
                       col_path = os.path.join(root_path, f)
                       if os.path.isdir(col_path):
                            doc_type_from_col, _ = _parse_collection_name(f) 
                            if doc_type_from_col in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
                                collections_to_delete_with_root.append((root_path, f, doc_type_from_col))
                            else:
                                logger.warning(f"Collection '{f}' does not map to a supported doc_type. Skipping map update for this collection.")

                            
    elif doc_type_to_wipe_lower == EVIDENCE_DOC_TYPES.lower():
        root_path = tenant_vectorstore_root_year # ใช้ year เดิม
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
        root_path = get_collection_parent_dir(tenant, final_year, doc_type_to_wipe_lower) # 💡 ใช้ final_year (จะเป็น None)
        collection_name = get_target_dir(doc_type_to_wipe_lower, None)
        doc_types_affected.add(doc_type_to_wipe_lower)
        collections_to_delete_with_root = [(root_path, collection_name, doc_type_to_wipe_lower)]
        
    else:
        logger.error(f"Invalid doc_type_to_wipe: {doc_type_to_wipe}")
        return
    
    # -------------------- 1. Delete collection folders (Vector Store) --------------------
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

    # -------------------- 2. Update Mapping files (Primary: doc_id_mapping.json) - Clean entries --------------------
    for dt in doc_types_affected:
        # 📌 FIX: กำหนด year ที่ใช้ในการโหลด/บันทึก Mapping
        map_year_to_use = year if dt == EVIDENCE_DOC_TYPES.lower() else final_year
        
        # 💡 ใช้ map_year_to_use ในการโหลด Mapping
        mapping_db = load_doc_id_mapping(dt, tenant, map_year_to_use) 
        uuids_to_keep = {}
        deletion_count = 0
        
        cols_to_check = deleted_collections_map.get(dt, set())
        
        # (Logic การสร้าง cols_to_check - ถูกต้องแล้ว)

        for s_uuid, entry in mapping_db.items():
            
            # 💡 ใช้ map_year_to_use ในการเช็คปี
            if str(entry.get("tenant")).lower() != str(tenant).lower() or entry.get("year") != map_year_to_use:
                 uuids_to_keep[s_uuid] = entry
                 continue
            
            entry_doc_type = entry.get("doc_type")
            entry_enabler = entry.get("enabler")
            entry_collection = get_target_dir(entry_doc_type, entry_enabler)
            
            # ถ้า collection ที่ entry อ้างถึง ถูกลบไป
            if entry_collection in cols_to_check:
                deletion_count += 1
            else:
                uuids_to_keep[s_uuid] = entry
                
        if deletion_count > 0:
            # ต้องมีการแยก Enabler ในการบันทึก
            if dt.lower() == EVIDENCE_DOC_TYPES.lower():
                 db_by_enabler: Dict[Optional[str], Dict[str, Dict[str, Any]]] = {}
                 for s_uuid, entry in uuids_to_keep.items():
                      enabler_key = entry.get("enabler")
                      db_by_enabler.setdefault(enabler_key, {})[s_uuid] = entry
                 
                 for enabler_key, small_db in db_by_enabler.items():
                      save_doc_id_mapping(small_db, dt, tenant, map_year_to_use, enabler=enabler_key) # 💡 ใช้ map_year_to_use
                      
                      # ลบไฟล์ Doc ID Mapping (Primary) ถ้าฐานข้อมูลว่างเปล่า 
                      if not small_db:
                           mapping_path = get_doc_id_mapping_path(tenant, map_year_to_use, enabler_key) # 💡 ใช้ map_year_to_use
                           if os.path.exists(mapping_path):
                               try:
                                   os.remove(mapping_path)
                                   logger.info(f"✅ Deleted (empty) Primary Doc ID mapping file: {mapping_path} (Via Step 2)")
                               except OSError as e:
                                   logger.error(f"❌ Error deleting mapping file: {e}")
            else:
                 save_doc_id_mapping(uuids_to_keep, dt, tenant, map_year_to_use) # 💡 ใช้ map_year_to_use
                 
                 # ลบไฟล์ Doc ID Mapping (Primary) ถ้าฐานข้อมูลว่างเปล่า 
                 if not uuids_to_keep:
                       mapping_path = get_doc_id_mapping_path(tenant, map_year_to_use, None) # 💡 ใช้ map_year_to_use
                       if os.path.exists(mapping_path):
                           try:
                               os.remove(mapping_path)
                               logger.info(f"✅ Deleted (empty) Primary Doc ID mapping file: {mapping_path} (Via Step 2)")
                           except OSError as e:
                               logger.error(f"❌ Error deleting mapping file: {e}")
                 
            logger.info(f"🧹 Removed {deletion_count} entries from mapping file for deleted collections (Doc Type: {dt}) of {tenant}/{map_year_to_use}.")
    
    # -------------------- 3. ลบ Mapping Files ที่เกี่ยวข้องกับคำสั่ง Wipe โดยตรง (บังคับลบ) --------------------
    doc_types_to_force_delete = set()
    if doc_type_to_wipe_lower == 'all':
        doc_types_to_force_delete.update([dt.lower() for dt in SUPPORTED_DOC_TYPES])
    else:
        doc_types_to_force_delete.add(doc_type_to_wipe_lower)
        
    for dt_force in doc_types_to_force_delete:
        map_year_to_use = year if dt_force == EVIDENCE_DOC_TYPES.lower() else final_year # 💡 กำหนดปีที่ใช้

        if dt_force == EVIDENCE_DOC_TYPES.lower():
            # Logic สำหรับ Evidence (ใช้ปีจริง)
            enablers_to_check = []
            if enabler and enabler.upper() in SUPPORTED_ENABLERS:
                enablers_to_check.append(enabler.upper())
            elif doc_type_to_wipe_lower == 'all' or (doc_type_to_wipe_lower == EVIDENCE_DOC_TYPES.lower() and not enabler):
                 enablers_to_check.extend(SUPPORTED_ENABLERS)
                 
            for ena_to_check in set(enablers_to_check):
                # --- A. Primary Doc ID Mapping File ---
                primary_mapping_path = get_doc_id_mapping_path(tenant, map_year_to_use, ena_to_check) # 💡 ใช้ map_year_to_use
                if os.path.exists(primary_mapping_path):
                     try:
                         os.remove(primary_mapping_path)
                         logger.info(f"✅ Deleted (FORCED) Primary Doc ID Mapping file: {primary_mapping_path}")
                     except OSError as e:
                         logger.error(f"❌ Error deleting Primary mapping file: {e}")

                # --- B. Secondary Evidence Mapping File ---
                secondary_mapping_filename = f"{tenant}_{year}_{ena_to_check.lower()}{EVIDENCE_MAPPING_FILENAME_SUFFIX}" 
                secondary_mapping_path = os.path.join(MAPPING_BASE_DIR, tenant, str(year), secondary_mapping_filename)
                
                if os.path.exists(secondary_mapping_path):
                     try:
                         os.remove(secondary_mapping_path)
                         logger.info(f"✅ Deleted secondary evidence mapping file: {secondary_mapping_path}")
                     except OSError as e:
                         logger.error(f"❌ Error deleting secondary mapping file: {e}")
                
                # --- C. Lock File ---
                lock_path = secondary_mapping_path + ".lock"
                if os.path.exists(lock_path):
                     try:
                         os.remove(lock_path)
                         logger.info(f"✅ Deleted lock file: {lock_path}")
                     except OSError as e:
                         logger.error(f"❌ Error deleting lock file: {e}")
                
        else:
             # --- Global Doc Type Mapping File (เช่น document) ---
             if dt_force in [dt.lower() for dt in SUPPORTED_DOC_TYPES] and dt_force != EVIDENCE_DOC_TYPES.lower():
                  primary_mapping_path = get_doc_id_mapping_path(tenant, map_year_to_use, None) # 💡 ใช้ map_year_to_use
                  if os.path.exists(primary_mapping_path):
                       try:
                           os.remove(primary_mapping_path)
                           logger.info(f"✅ Deleted (FORCED) Primary Doc ID Mapping file: {primary_mapping_path} (Global)")
                       except OSError as e:
                           logger.error(f"❌ Error deleting Primary mapping file: {e}")

    # -------------------- 4. Final cleanup: พยายามลบโฟลเดอร์ Mapping ถ้าว่างเปล่า --------------------
    try:
        # พยายามลบโฟลเดอร์ปี (2568) ที่ใช้เก็บ Evidence Mapping
        mapping_dir_year = os.path.join(MAPPING_BASE_DIR, tenant, str(year))
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
    DOC_TYPES_WITH_YEAR_AND_ENABLER = [EVIDENCE_DOC_TYPES.lower()]
    doc_type_lower = doc_type.lower()
    
    final_year_for_map: Optional[int] = year
    if doc_type_lower not in DOC_TYPES_WITH_YEAR_AND_ENABLER:
        logger.warning(f"⚠️ Year ({year}) is ignored for Global Doc Type deletion: {doc_type_lower}. Setting year to None for Mapping.")
        final_year_for_map = None
        
    # 💡 ใช้ final_year_for_map ในการโหลด Mapping
    doc_mapping_db = load_doc_id_mapping(doc_type, tenant, final_year_for_map) 
    
    if stable_doc_uuid not in doc_mapping_db:
        logger.warning(f"Deletion skipped: Stable Doc UUID {stable_doc_uuid} not found in mapping for {tenant}/{final_year_for_map}.")
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
        # 🎯 FIXED: ใช้ get_collection_parent_dir (ใช้ year จริงสำหรับ Evidence, None สำหรับ Global Doc Type)
        # เนื่องจาก vectorstore directory ยังคงแยกกันตามปีที่รับเข้ามา/หรือไม่แยกปีตาม _get_collection_parent_dir
        # เราต้องตรวจสอบว่า doc_type เป็น Evidence หรือไม่
        parent_dir_year = year if final_doc_type.lower() == EVIDENCE_DOC_TYPES.lower() else final_year_for_map

        tenant_vectorstore_parent_dir = get_collection_parent_dir(tenant, parent_dir_year, final_doc_type) # 💡 ใช้ parent_dir_year
        persist_directory = os.path.join(tenant_vectorstore_parent_dir, final_collection_name) 
        
        if not os.path.isdir(persist_directory):
            logger.warning(f"Vectorstore directory not found for collection '{final_collection_name}' for {tenant}/{parent_dir_year}. Skipping Chroma deletion.")
        else:
            try:
                vectorstore = get_vectorstore(
                    collection_name=final_collection_name, 
                    tenant=tenant, 
                    year=parent_dir_year, # 💡 ใช้ parent_dir_year
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
         # ... (Logic การบันทึก Evidence - โค้ดเดิม, ใช้ year ที่รับเข้ามา)
         # ในกรณีนี้ final_year_for_map จะเท่ากับ year
         db_to_save: Dict[str, Dict[str, Any]] = {}
         
         for s_uuid, entry in doc_mapping_db.items():
              # 📌 FIX: ต้องเช็ค Tenant/Year ด้วย
              if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year_for_map and entry.get("enabler") == final_enabler:
                   db_to_save[s_uuid] = entry
                   
         save_doc_id_mapping(db_to_save, final_doc_type, tenant, final_year_for_map, enabler=final_enabler)
    else:
         # สำหรับ Doc Type อื่นๆ (เช่น document) บันทึกเฉพาะ Entry ที่เกี่ยวข้องกับ Tenant/Year
         # ในกรณีนี้ final_year_for_map จะเป็น None
         db_to_save = {
             s_uuid: entry 
             for s_uuid, entry in doc_mapping_db.items() 
             if str(entry.get("tenant")).lower() == tenant.lower() and entry.get("year") == final_year_for_map
         }
         save_doc_id_mapping(db_to_save, final_doc_type, tenant, final_year_for_map) 
        
    return True

def list_documents(
    doc_types: Optional[List[str]] = None,
    enabler: Optional[str] = None, 
    tenant: str = "pwa",        
    year: Union[int, str] = 2568, # 💡 รองรับทั้ง int และ str ที่อาจมาจาก argparse
    show_results: str = "ingested" 
) -> Dict[str, Any]: 

    # --- 1. Preparation & Robust Year Conversion ---
    doc_mapping_db = {}
    enabler_req = (enabler or "").upper()
    
    # 🎯 FIX: Robustly convert the 'year' parameter to an integer for comparison
    year_int = 0
    try:
        if year is not None:
            # ใช้ str(year) เพื่อจัดการ Union[int, str]
            year_int = int(str(year)) 
    except (TypeError, ValueError):
        logger.error(f"Invalid year argument provided: {year}. Defaulting year_int to 0 (Global Context).")
        year_int = 0
    # --------------------------------------------------------------------------
    
    # Determine Doc Types to load
    doc_types_reqs = {dt.lower() for dt in doc_types} if doc_types and doc_types[0] and doc_types[0].lower() != "all" else set()
    if not doc_types_reqs:
        doc_types_to_load = {dt.lower() for dt in SUPPORTED_DOC_TYPES}
    else:
        doc_types_to_load = doc_types_reqs
        
    # Generate list of (doc_type, enabler, year) contexts to check
    load_contexts: Set[Tuple[str, Optional[str], Optional[int]]] = set() 

    for dt_lower in doc_types_to_load:
        
        # Case 1: Evidence (Requires Year and Enabler context)
        if dt_lower == EVIDENCE_DOC_TYPES.lower():
            # Evidence ต้องมีปีเสมอ (ปีต้อง > 0)
            if year_int > 0: # ใช้ year_int ที่แปลงแล้ว
                years_to_use = {year_int}
                
                if enabler_req and enabler_req in SUPPORTED_ENABLERS:
                    enablers_to_use = {enabler_req}
                else:
                    enablers_to_use = set(SUPPORTED_ENABLERS) 
                    
                for ena in enablers_to_use:
                    for yr in years_to_use:
                        load_contexts.add((dt_lower, ena, yr))
        
        # Case 2: Non-Evidence (document, faq, etc.) - Global Context
        else:
            # สำหรับ non-evidence จะไม่ใช้ Year และ Enabler ใน Context (year=None, enabler=None)
            load_contexts.add((dt_lower, None, None)) 

    # --- 2. Load Doc ID Mapping Files (using load_contexts) ---
    for dt, ena, yr in load_contexts:
        ena_code = ena
        # Note: load_doc_id_mapping จะใช้ yr=None ในการสร้าง Path Global
        try:
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
        
        doc_type_in_map = entry.get("doc_type").lower()
        if doc_type_in_map not in doc_types_to_load:
            continue
        
        doc_year_in_map = entry.get("year")
        
        is_context_match = False
        
        if doc_type_in_map == EVIDENCE_DOC_TYPES.lower():
            # Evidence: ต้องตรงกับปีที่ร้องขอ (year_int > 0)
            if year_int > 0 and doc_year_in_map == year_int: 
                 is_context_match = True
        else:
            # Non-Evidence (Global): ต้องไม่มีส่วนประกอบของปีใน mapping (year=None หรือ 0)
            if doc_year_in_map in [None, 0]:
                 is_context_match = True
            
        if is_context_match:
            filepath_to_stable_uuid[entry["filepath"]] = s_uuid


    # --- 4. Physical File Scan and Status Check (using load_contexts) ---
    all_docs: Dict[str, Any] = {}
    
    for dt_lower, resolved_enabler, resolved_year in load_contexts:
        
        # _get_source_dir จะใช้ resolved_year (ซึ่งอาจเป็น None) ในการสร้าง Path
        root_to_scan = _get_source_dir(tenant, resolved_year, dt_lower, resolved_enabler)
        
        if not os.path.isdir(root_to_scan):
            logger.debug(f"Source directory not found: {root_to_scan}. Skipping scan.")
            continue
            
        original_doc_type = dt_lower
        
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
                
                # 📌 Fallback Search ถ้าหาด้วย Absolute Path ไม่เจอ
                if stable_doc_uuid is None:
                    # 2. ลองค้นหาด้วย filename (ซึ่งอยู่ใน Mapping File)
                    for s_uuid, entry in doc_mapping_db.items():
                        # ตรวจสอบ file_name และ Context ที่ตรงกัน
                        if (entry.get("file_name") == f 
                            and str(entry.get("tenant")).lower() == str(tenant).lower() 
                            and entry.get("doc_type").lower() == original_doc_type
                            and entry.get("enabler", "").upper() == (resolved_enabler or "").upper()
                            # ตรวจสอบ Year: ถ้า resolved_year=None ให้ดูว่า year ใน mapping เป็น None/0
                            and (entry.get("year") == resolved_year if resolved_year is not None else entry.get("year") in [None, 0])):
                            stable_doc_uuid = s_uuid
                            break
                            
                doc_entry = doc_mapping_db.get(stable_doc_uuid) if stable_doc_uuid else None
                
                # Logic การตรวจสอบสถานะ Ingested
                chunk_uuids = doc_entry.get("chunk_uuids", []) if doc_entry else []
                chunk_count = len(chunk_uuids)
                
                # 💡 ถ้ามี Stable ID แต่ไม่มี chunk_count (เพราะ Ingest ไม่ได้บันทึกไว้) 
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
                    "year": resolved_year, # ใช้ resolved_year (int หรือ None)
                    "upload_date": upload_date,
                    "chunk_count": chunk_count,
                    "status": "Ingested" if is_ingested else "Pending", 
                    "size": file_size,
                }
                all_docs[final_doc_id] = doc_info

    # --- 5. Final Filtering and Display ---
    
    total_supported_files = len(all_docs) 
    
    show_results_lower = show_results.lower()
    filtered_docs_dict: Dict[str, Any] = {}
    
    # กำหนด String สำหรับแสดง Year
    year_request_str = str(year_int) if year_int > 0 else "Global"

    if total_supported_files == 0:
        doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
        logger.warning(f"⚠️ No documents found in DATA_DIR matching the requested type '{doc_types_str}' (Enabler: {enabler_req or 'ALL'}, Year: {year_request_str}) for {tenant}.")
        return filtered_docs_dict 
    
    # ... (Logic Filtering show_results) ...
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
        # แสดงปีเป็น '-' หากเป็น None หรือ 0
        year_display = str(doc_info['year']) if doc_info['year'] is not None and doc_info['year'] > 0 else '-'
        
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
            "year": year_display     
        })
        
    display_list.sort(key=lambda x: (x["doc_type"], x["file_name"]))
    
    doc_types_str = doc_types[0] if doc_types and doc_types[0] else "all"
    
    print(f"\nFound {display_count_x}/{total_supported_files} supported documents for type '{doc_types_str}' (Tenant: {tenant}, Year: {year_request_str}, Filter: {filter_name}):\n")

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
        year_display = info['year']
        
        print(
            f"{full_doc_id:<{UUID_COL_WIDTH}} | " 
            f"{info['tenant']:<7} | " 
            f"{year_display:<4} | "   
            f"{short_filename:<30} | " 
            f"{file_ext:<5} | "
            f"{info['doc_type']:<10} | "
            f"{enabler_display:<5} | " 
            f"{size_str:<9} | "
            f"{info['status']:<10}"
        )
    print("-" * NEW_TABLE_WIDTH)
    print(f"\nFound {display_count_x}/{total_supported_files} supported documents for type '{doc_types_str}' (Tenant: {tenant}, Year: {year_request_str}, Filter: {filter_name}):\n")

    
    return filtered_docs_dict

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    import sys
    
    # 📌 NOTE: Assume necessary global_vars and functions (logger, 
    # DEFAULT_TENANT, DEFAULT_YEAR, ingest_all_files, wipe_vectorstore, 
    # list_documents) are imported or defined.

    try:
        import argparse
        parser = argparse.ArgumentParser(description="Manage document ingestion and listing for Chroma Vector Store (Multi-Tenant/Year).")
        
        # --- Common Arguments ---
        parser.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help=f"Tenant code (default: {DEFAULT_TENANT}).")
        parser.add_argument("--year", type=str, default=str(DEFAULT_YEAR), help=f"Fiscal year (default: {DEFAULT_YEAR}). Use 'None' or '0' for global context.")
        parser.add_argument("--doc-type", nargs='+', type=str, default=["all"], help=f"Document type(s) to target (e.g., 'document', 'evidence', 'all').")
        parser.add_argument("--enabler", type=str, default=None, help="Enabler code, usually required for 'evidence' doc type (e.g., 'km').")
        
        # --- Ingest Arguments ---
        parser.add_argument("--ingest", action="store_true", help="Activate document ingestion mode.")
        parser.add_argument("--subject", type=str, default=None, help="Subject/Topic for Global Doc Types (e.g., 'HR Policy').") 
        parser.add_argument("--dry-run", action="store_true", help="Process files but do not save to vectorstore or mapping (Ingest mode only).")
        parser.add_argument("--sequential", action="store_true", help="Process files sequentially (Ingest mode only).")
        parser.add_argument("--wipe", action="store_true", help="Wipe the target vector store collection(s) and associated mappings before ingesting.")
        
        # --- List Arguments ---
        parser.add_argument("--list", action="store_true", help="Activate document listing mode.")
        parser.add_argument("--show-results", type=str, default="ingested", choices=["ingested", "failed", "full"], help="Filter results: 'ingested', 'failed' (pending), or 'full'. (List mode only)")

        parser.add_args = parser.parse_args()
        args = parser.parse_args()
        
        # Convert doc-type list back to single string or maintain list format based on target function needs
        doc_type_for_ingest_wipe = args.doc_type[0] if len(args.doc_type) == 1 else ",".join(args.doc_type)

        # -------------------- Execution Logic --------------------
        
        if args.ingest:
            logger.info("--- INGEST MODE ACTIVATED ---")
            
            # 1. WIPE LOGIC (ถ้ามีการสั่ง --wipe)
            if args.wipe:
                logger.warning(f"!!! WIPE MODE ACTIVATED for {doc_type_for_ingest_wipe} !!!")
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
            logger.info("--- LIST MODE ACTIVATED ---")
            
            # 3. LIST LOGIC
            list_documents(
                doc_types=args.doc_type, # list_documents รับเป็น list ของ doc_type ได้
                enabler=args.enabler, 
                tenant=args.tenant, 
                year=args.year, # ส่งเป็น string เพื่อให้ list_documents จัดการแปลงเป็น int/None เอง
                show_results=args.show_results 
            )
            
        else:
            print("\nUsage: Specify --ingest or --list mode.")
            parser.print_help()

        logger.info("Execution finished.")
        
    except ImportError:
         print("--- RUNNING SCRIPT STANDALONE FAILED: Missing argparse module ---")
         
    except Exception as e:
         logger.critical(f"FATAL ERROR DURING MAIN EXECUTION: {e}", exc_info=True)
         print(f"--- FATAL ERROR: {e} ---")