# utils/path_utils.py

import os
from typing import Dict, Any, Optional, Tuple, List
import json
import logging
import re

# -------------------- Import project modules --------------------
# 📌 ASSUMPTION: ค่าคงที่เหล่านี้ถูกกำหนดใน config.global_vars
from config.global_vars import (
    DATA_DIR, 
    MAPPING_BASE_DIR, 
    EVIDENCE_DOC_TYPES, 
    DOCUMENT_ID_MAPPING_FILENAME_SUFFIX,
    EVIDENCE_MAPPING_FILENAME_SUFFIX, 
    VECTORSTORE_DIR,
    RUBRIC_CONFIG_DIR, 
    RUBRIC_FILENAME_PATTERN,
    EXPORTS_DIR 
)
# ----------------------------------------------------------------

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# ## 1. Path Builders สำหรับ Source Data (เอกสารต้นฉบับ)
# ----------------------------------------------------------------

def _build_tenant_base_path(tenant: str, year: Optional[int], doc_type: str) -> str:
    """
    สร้าง path หลักของ tenant/context สำหรับ Input Data (Source Files)
    Logic: DATA_DIR / tenant / [year (เฉพาะ Evidence)]
    """
    tenant_clean = tenant.strip().lower().replace(" ", "_")
    
    if not tenant_clean or ".." in tenant_clean or "/" in tenant_clean or "\\" in tenant_clean:
        raise ValueError(f"tenant ไม่ถูกต้อง: {tenant}")

    is_evidence = doc_type.lower() == EVIDENCE_DOC_TYPES.lower()
    
    path_components = [DATA_DIR, tenant_clean]
    
    if is_evidence and year is not None:
        path_components.append(str(year))
    
    return os.path.join(*path_components)


def get_document_source_dir(
    tenant: str, 
    year: Optional[int], 
    enabler: Optional[str], 
    doc_type: str
) -> str:
    """
    สร้าง Path สมบูรณ์ไปยัง Source Document ที่ใช้ในการ Ingest
    Logic: _build_tenant_base_path / doc_type / [enabler (เฉพาะ Evidence)]
    """
    
    # 🟢 FIX: เพิ่มการตรวจสอบ doc_type เป็น None ก่อนเรียก .lower()
    if doc_type is None:
        # ควร Raise Error เพื่อให้รู้ว่า Caller ส่งค่าผิดมา
        raise ValueError(
            "doc_type cannot be None when calling get_document_source_dir. "
            "Check the caller (list_documents) logic."
        )

    doc_type_lower = doc_type.lower()
    enabler_lower = enabler.lower() if enabler else None
    
    base_path = _build_tenant_base_path(tenant, year, doc_type)
    path_segments = [base_path, doc_type_lower]
    
    is_evidence = doc_type_lower == EVIDENCE_DOC_TYPES.lower()
    
    if is_evidence and enabler_lower:
        path_segments.append(enabler_lower)
    
    return os.path.join(*path_segments)


def get_evidence_base_dir(tenant: str, year: int, enabler: str) -> str:
    """Helper สำหรับ Evidence Type โดยเฉพาะ (ใช้สำหรับ Source Files)"""
    return get_document_source_dir(tenant, year, enabler, doc_type=EVIDENCE_DOC_TYPES) 

# ----------------------------------------------------------------
# ## 2. Path Builders สำหรับ Vector Store (Chroma Collection)
# ----------------------------------------------------------------

def get_doc_type_collection_key(doc_type: str, enabler: Optional[str] = None) -> str:
    """กำหนดชื่อ Collection สำหรับ ChromaDB (Logical ID)"""
    doc_type_norm = doc_type.strip().lower()
    
    if doc_type_norm == EVIDENCE_DOC_TYPES.lower():
        enabler_norm = (enabler or "default").strip().lower() 
        return f"{doc_type_norm}_{enabler_norm}"
        
    return doc_type_norm


def get_vectorstore_collection_path(
    tenant: str, 
    year: Optional[int], 
    doc_type: str, 
    enabler: Optional[str] = None
) -> str:
    """
    สร้าง Path สมบูรณ์ไปยัง Vector Store Collection/Index
    Logic: VECTORSTORE_DIR / tenant / [year (เฉพาะ Evidence)] / collection_name
    """
    doc_type_lower = doc_type.lower()
    collection_name = get_doc_type_collection_key(doc_type, enabler)
    
    path_segments = [VECTORSTORE_DIR, tenant.lower()]
    
    if doc_type_lower == EVIDENCE_DOC_TYPES.lower() and year is not None:
        path_segments.append(str(year))
        
    path_segments.append(collection_name)
    
    return os.path.join(*path_segments)

def get_vectorstore_tenant_root_path(tenant: str) -> str:
    """Calculates the root path for a specific tenant within the vectorstore."""
    return os.path.join(VECTORSTORE_DIR, tenant.lower()) 


# ----------------------------------------------------------------
# ## 3. Path Builders สำหรับ Mapping File
# ----------------------------------------------------------------

def get_mapping_file_path(tenant: str, year: Optional[int], enabler: Optional[str]) -> str:
    """
    สร้าง Path สำหรับไฟล์ Document ID Mapping (รองรับ Legacy Mapping)
    """
    tenant_lower = tenant.lower()
    
    # Priority 1: รูปแบบใหม่ (แยกปี/Enabler)
    if year is not None and enabler:
        enabler_lower = enabler.lower()
        
        path_segments = [MAPPING_BASE_DIR, tenant_lower, str(year)]
        mapping_filename = f"{tenant_lower}_{year}_{enabler_lower}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
        
        path_segments.append(mapping_filename)
        return os.path.join(*path_segments)
    
    # Priority 2: รูปแบบรวม (Legacy/Fallback)
    else:
        path_segments = [MAPPING_BASE_DIR, tenant_lower]
        mapping_filename = f"{tenant_lower}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
        
        path_segments.append(mapping_filename)
        return os.path.join(*path_segments)
        
def get_evidence_mapping_file_path(tenant: str, year: int, enabler: str) -> str:
    """
    สร้าง Path สำหรับไฟล์ Evidence Statement Mapping (Persistent Map)
    Logic: MAPPING_BASE_DIR / tenant / year / {tenant}_{year}_{enabler}_evidence_mapping.json
    """
    tenant_lower = tenant.lower()
    enabler_lower = enabler.lower()
    
    path_segments = [MAPPING_BASE_DIR, tenant_lower, str(year)]
    
    mapping_filename = (
        f"{tenant_lower}_{year}_{enabler_lower}"
        f"{EVIDENCE_MAPPING_FILENAME_SUFFIX}"
    )

    path_segments.append(mapping_filename)
    return os.path.join(*path_segments)

# ----------------------------------------------------------------
# ## 3. Path Builders สำหรับ Mapping File (ต่อ)
# ----------------------------------------------------------------

def load_doc_id_mapping(
    doc_type: str, 
    tenant: str, 
    year: Optional[int], 
    enabler: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    โหลด Document ID Mapping จากไฟล์ โดยใช้บริบท (tenant, year, enabler) เพื่อกำหนด Path
    """
    # 📌 NOTE: ใช้ get_mapping_file_path ที่ถูกออกแบบมาเพื่อรองรับ year=None สำหรับ Global Doc Type
    map_path = get_mapping_file_path(tenant, year, enabler)
    
    # ถ้าไฟล์ไม่มีอยู่จริง ให้คืนค่า Dictionary เปล่า
    if not os.path.exists(map_path):
        # logger.info(f"Mapping file not found at {map_path}. Returning empty map.")
        return {}
        
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"❌ Error decoding JSON from mapping file: {map_path}. Returning empty map.")
        return {}
    except Exception as e:
        logger.error(f"❌ Error loading mapping file {map_path}: {e}")
        return {}


def save_doc_id_mapping(
    data: Dict[str, Dict[str, Any]], 
    doc_type: str, # ใช้สำหรับ Contextual Logger/Future Logic
    tenant: str, 
    year: Optional[int], 
    enabler: Optional[str] = None
) -> None:
    """
    บันทึก Document ID Mapping ลงในไฟล์ โดยใช้บริบท (tenant, year, enabler) เพื่อกำหนด Path
    """
    map_path = get_mapping_file_path(tenant, year, enabler)
    
    try:
        # สร้าง Directory หากยังไม่มี (รวมถึง dir สำหรับ year/enabler ถ้ามี)
        os.makedirs(os.path.dirname(map_path), exist_ok=True)
        
        with open(map_path, "w", encoding="utf-8") as f:
            # ใช้ indent 2 เพื่อให้อ่านง่าย
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"❌ FATAL: Failed to save mapping file to {map_path}: {e}")
        
# ----------------------------------------------------------------
# ## 4. Document File Path Resolver
# ----------------------------------------------------------------

def get_document_file_path(
    document_uuid: str, 
    tenant: str, 
    year: Optional[int], 
    enabler: Optional[str], 
    doc_type_name: str
) -> Optional[Dict[str, str]]:
    """
    แปลง document_uuid ไปเป็น path ของไฟล์จริง โดยระบุ doc_type_name
    (ต้องโหลด Mapping file ก่อน)
    """
    # 1. Load Mapping 
    try:
        doc_id_map_path = get_mapping_file_path(tenant, year, enabler)
        
        if not os.path.exists(doc_id_map_path): 
            logger.warning(f"Mapping file not found at {doc_id_map_path}")
            return None

        with open(doc_id_map_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
            original_filename = mapping_data.get(document_uuid, {}).get('file_name')
        
        if not original_filename: return None
    
    except Exception as e:
        logger.error(f"Error loading/decoding mapping file: {e}")
        return None

    # 2. Construct Base Document Store Path 
    BASE_DOCUMENT_STORE = get_document_source_dir(tenant, year, enabler, doc_type_name) 
    
    # 3. Construct Final File Path
    file_path = os.path.join(BASE_DOCUMENT_STORE, original_filename) 

    if not os.path.exists(file_path):
         logger.error(f"Original file not found on disk at {file_path}")
         return None
    
    return {
        "file_path": file_path,
        "original_filename": original_filename
    }


# ----------------------------------------------------------------
# ## 5. Path Builders สำหรับ Rubric และ Contextual Rules
# ----------------------------------------------------------------

def get_rubric_file_path(tenant: str, enabler: str) -> str:
    """
    สร้าง Path สำหรับไฟล์ Rubric หลัก
    Logic: RUBRIC_CONFIG_DIR / tenant / {tenant}_{enabler}_rubric.json
    """
    filename = RUBRIC_FILENAME_PATTERN.format(
        tenant=tenant.lower(), 
        enabler=enabler.upper() 
    )
    
    path = os.path.join(
        RUBRIC_CONFIG_DIR, 
        tenant.lower(), 
        filename
    )
    return path


def get_contextual_rules_file_path(tenant: str, enabler: str) -> str:
    """
    สร้าง Path สำหรับไฟล์ Contextual Rules
    Logic: RUBRIC_CONFIG_DIR / tenant / {tenant}_{enabler}_contextual_rules.json
    """
    filename = f"{tenant.lower()}_{enabler.lower()}_contextual_rules.json"
    
    path = os.path.join(
        RUBRIC_CONFIG_DIR, 
        tenant.lower(), 
        filename
    )
    return path


# ----------------------------------------------------------------
# ## 6. Path Builders สำหรับ Export Files
# ----------------------------------------------------------------

def get_export_dir(tenant: str, year: int, enabler: str) -> str:
    """
    สร้าง Path Root สำหรับ Export File (ผลลัพธ์การประเมิน)
    Logic: EXPORTS_DIR / tenant / year / enabler
    """
    path = os.path.join(
        EXPORTS_DIR,
        tenant.lower(),
        str(year),
        enabler.lower()
    )
    return path

def get_assessment_export_file_path(
    tenant: str, 
    year: int, 
    enabler: str, 
    suffix: str, 
    extension: str = "json"
) -> str:
    """
    สร้าง Full Path สำหรับไฟล์ Export ผลลัพธ์การประเมิน
    เช่น: EXPORTS_DIR / tenant / year / enabler / {tenant}_{year}_{enabler}_{suffix}.{ext}
    """
    base_dir = get_export_dir(tenant, year, enabler)
    
    filename = (
        f"{tenant.lower()}_{year}_{enabler.lower()}_{suffix}.{extension.lower()}"
    )
    
    return os.path.join(base_dir, filename)