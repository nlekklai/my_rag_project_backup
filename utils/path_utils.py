# utils/path_utils.py
# Production Final Version – 11 ธ.ค. 2568 (Path Matching Fixes)

import os
import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Union, List
import unicodedata # NEW: เพิ่ม import สำหรับการจัดการ Path/Filename encoding บน macOS

# 📌 ASSUME: config.global_vars มีการกำหนดค่าที่ถูกต้องตามที่ใช้
from config.global_vars import (
    DATA_STORE_ROOT,
    EVIDENCE_DOC_TYPES,
    DOCUMENT_ID_MAPPING_FILENAME_SUFFIX,
    EVIDENCE_MAPPING_FILENAME_SUFFIX,
    RUBRIC_FILENAME_PATTERN,
    EXPORTS_DIR,
    DEFAULT_TENANT,
    DEFAULT_YEAR,
    DEFAULT_ENABLER,
    PROJECT_NAMESPACE_UUID
)

logger = logging.getLogger(__name__)

# ==================== CORE HELPER ====================
def _n(s: Union[str, None]) -> str:
    """Normalize ทุก string ด้วย NFKC – แก้ macOS NFD bug ถาวร และแปลงเป็น clean key"""
    return unicodedata.normalize('NFKC', s.strip().lower().replace(" ", "_")) if isinstance(s, str) and s.strip() else ""

# ==================== INTERNAL BASE PATH ====================
def _build_tenant_base_path(tenant: str) -> str:
    tenant_clean = _n(tenant)
    return os.path.join(DATA_STORE_ROOT, tenant_clean, "data")

# ==================== 1. SOURCE PATHS ====================
def get_document_source_dir(
    tenant: str,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
    doc_type: str = "",
) -> str:
    if not doc_type:
        raise ValueError("doc_type is required")
    base = _build_tenant_base_path(tenant)
    base = os.path.join(base, _n(doc_type))
    if _n(doc_type) == EVIDENCE_DOC_TYPES.lower():
        if year is not None:
            base = os.path.join(base, str(year))
        if enabler:
            base = os.path.join(base, _n(enabler))
    return base

def get_evidence_base_dir(tenant: str, year: Union[int, str], enabler: str) -> str:
    return get_document_source_dir(tenant, year, enabler, EVIDENCE_DOC_TYPES)

# ==================== 2. VECTORSTORE PATHS ====================
def get_doc_type_collection_key(doc_type: str, enabler: Optional[str] = None) -> str:
    dt = _n(doc_type)
    if dt == EVIDENCE_DOC_TYPES.lower():
        return f"{dt}_{_n(enabler or 'default')}"
    return dt

def get_vectorstore_collection_path(
    tenant: str, year: Optional[Union[int, str]], doc_type: str, enabler: Optional[str] = None
) -> str:
    parts = [DATA_STORE_ROOT, _n(tenant), "vectorstore"]
    if _n(doc_type) == EVIDENCE_DOC_TYPES.lower() and year is not None:
        parts.append(str(year))
    parts.append(get_doc_type_collection_key(doc_type, enabler))
    return os.path.join(*parts)

def get_vectorstore_tenant_root_path(tenant: str) -> str:
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "vectorstore")

# ==================== 3. MAPPING FILES ====================
def get_mapping_file_path(
    doc_type: str,
    tenant: str,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None
) -> str:
    """
    สร้าง path ของ mapping file ตามมาตรฐานใหม่ (ธ.ค. 2568)

    - evidence → /mapping/{year}/pea_{year}_{enabler}_doc_id_mapping.json
    - global doc_type → /mapping/pea_{doc_type}_doc_id_mapping.json
      เช่น:
        pea_seam_doc_id_mapping.json
        pea_document_doc_id_mapping.json
        pea_faq_doc_id_mapping.json
        pea_policy_doc_id_mapping.json

    ไม่มี fallback ชื่อเก่า (pea_doc_id_mapping.json) อีกต่อไป → สะอาด 100%
    """
    base = get_mapping_tenant_root_path(tenant)

    # === 1. Evidence: ต้องมี year + enabler เท่านั้น ===
    if _n(doc_type) == EVIDENCE_DOC_TYPES.lower():
        if year is None:
            raise ValueError("Evidence doc_type ต้องระบุ year")
        if not enabler:
            raise ValueError("Evidence doc_type ต้องระบุ enabler")
        return os.path.join(
            base,
            str(year),
            f"{_n(tenant)}_{year}_{_n(enabler)}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
        )

    # === 2. Global doc_type: ใช้ชื่อเต็มเสมอ ไม่สน year/enabler ===
    # แม้จะส่ง year/enabler มาก็ตาม → ต้องแยกไฟล์ชัดเจน
    return os.path.join(
        base,
        f"{_n(tenant)}_{_n(doc_type)}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
    )


def get_mapping_file_path(
    doc_type: str,
    tenant: str,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None
) -> str:
    """
    สร้าง path ของ mapping file ตามมาตรฐานใหม่ (ธ.ค. 2568)

    - evidence → /mapping/{year}/pea_{year}_{enabler}_doc_id_mapping.json
    - global doc_type → /mapping/pea_{doc_type}_doc_id_mapping.json
      เช่น:
        pea_seam_doc_id_mapping.json
        pea_document_doc_id_mapping.json
        pea_faq_doc_id_mapping.json
        pea_policy_doc_id_mapping.json

    ไม่มี fallback ชื่อเก่า (pea_doc_id_mapping.json) อีกต่อไป → สะอาด 100%
    """
    base = get_mapping_tenant_root_path(tenant)

    # === 1. Evidence: ต้องมี year + enabler เท่านั้น ===
    if _n(doc_type) == EVIDENCE_DOC_TYPES.lower():
        if year is None:
            raise ValueError("Evidence doc_type ต้องระบุ year")
        if not enabler:
            raise ValueError("Evidence doc_type ต้องระบุ enabler")
        return os.path.join(
            base,
            str(year),
            f"{_n(tenant)}_{year}_{_n(enabler)}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
        )

    # === 2. Global doc_type: ใช้ชื่อเต็มเสมอ ไม่สน year/enabler ===
    # แม้จะส่ง year/enabler มาก็ตาม → ต้องแยกไฟล์ชัดเจน
    return os.path.join(
        base,
        f"{_n(tenant)}_{_n(doc_type)}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
    )
    

def get_evidence_mapping_file_path(tenant: str, year: Union[int, str], enabler: str) -> str:
    # 🟢 FIX: สร้าง Path โดยตรง โดยใช้ EVIDENCE_MAPPING_FILENAME_SUFFIX
    #        (แทนการเรียก get_mapping_file_path ที่สร้างชื่อไฟล์ผิด)
    
    # 1. ได้ Root Path ของ mapping/...
    base = get_mapping_tenant_root_path(tenant)
    
    # 2. สร้างชื่อไฟล์ที่ถูกต้อง: pea_2568_km_evidence_mapping.json
    filename = f"{_n(tenant)}_{year}_{_n(enabler)}{EVIDENCE_MAPPING_FILENAME_SUFFIX}"
    
    # 3. รวม Path: .../mapping/2568/pea_2568_km_evidence_mapping.json
    return os.path.join(
        base,
        str(year),
        filename
    )

def get_mapping_tenant_root_path(tenant: str) -> str:
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "mapping")

# ==================== 4. LOAD / SAVE MAPPING ====================
def load_doc_id_mapping(doc_type: str, tenant: str, year: Optional[Union[int, str]], enabler: Optional[str] = None) -> Dict:
    path = get_mapping_file_path(doc_type, tenant, year, enabler)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Load mapping failed {path}: {e}")
        return {}

def save_doc_id_mapping(data: Dict, doc_type: str, tenant: str, year: Optional[Union[int, str]], enabler: Optional[str] = None):
    path = get_mapping_file_path(doc_type, tenant, year, enabler)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Save mapping failed {path}: {e}")


# ==================== 6. PARSE COLLECTION NAME ====================
def parse_collection_name(collection_name: str) -> Tuple[str, Optional[str]]:
    name = _n(collection_name)
    if name.startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"):
        parts = name.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return name, None

# ==================== 7. DOCUMENT FILE PATH RESOLVER ====================
def get_document_file_path(
    document_uuid: str,
    tenant: str,
    year: Optional[Union[int, str]],
    enabler: Optional[str],
    doc_type_name: str
) -> Optional[Dict[str, str]]:
    
    try:
        mapping_path = get_mapping_file_path(doc_type_name, tenant, year, enabler)
        if not os.path.exists(mapping_path):
            logger.warning(f"Mapping file not found: {mapping_path}")
            return None

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        entry = mapping_data.get(document_uuid)
        if not entry:
            logger.warning(f"UUID {document_uuid} not found in mapping")
            return None

        original_filepath_relative = entry.get("filepath")
        if not original_filepath_relative:
            logger.warning(f"No relative filepath in mapping for UUID {document_uuid}")
            return None

        # 🟢 FIX: ใช้ os.path.join() เพื่อประกอบ Path ของไฟล์จริงจาก DATA_STORE_ROOT และ Path สัมพัทธ์
        file_path_abs = os.path.join(DATA_STORE_ROOT, original_filepath_relative)

        if not os.path.exists(file_path_abs):
            logger.error(f"File not found on disk: {file_path_abs}")
            return None

        return {
            "file_path": file_path_abs,
            "original_filename": entry.get("file_name", os.path.basename(file_path_abs))
        }
    except Exception as e:
        logger.error(f"Error resolving document path for UUID {document_uuid}: {e}")
        return None

# ==================== 8. OTHER PATHS ====================
def get_config_tenant_root_path(tenant: str) -> str:
    """Path สำหรับ Configuration Files ที่คงที่ เช่น Rubrics, Contextual Rules"""
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "config")

def get_rubric_file_path(tenant: str, enabler: str) -> str:
    return os.path.join(get_config_tenant_root_path(tenant),
                        RUBRIC_FILENAME_PATTERN.format(tenant=_n(tenant), enabler=_n(enabler)))

def get_contextual_rules_file_path(tenant: str, enabler: str) -> str:
    return os.path.join(get_config_tenant_root_path(tenant),
                        f"{_n(tenant)}_{_n(enabler)}_contextual_rules.json")

def get_export_dir(tenant: str, year: Union[int, str], enabler: str) -> str:
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "exports", str(year), _n(enabler))

def get_assessment_export_file_path(tenant: str, year: Union[int, str], enabler: str, suffix: str, ext: str = "json") -> str:
    return os.path.join(get_export_dir(tenant, year, enabler),
                        f"{_n(tenant)}_{year}_{_n(enabler)}_{suffix}.{ext.lower()}")

def get_normalized_metadata(doc_type: str, year_input=None, enabler_input=None, default_enabler=None):
    # Logic: Evidence ต้องมีปี/Enabler, Doc/Global ใช้ None
    return (None, None) if _n(doc_type) != EVIDENCE_DOC_TYPES.lower() else (year_input, enabler_input or default_enabler)

def resolve_filepath_to_absolute(path: str) -> str:
    """
    แปลง Path ให้เป็น Absolute Path และ Normalize (NFKC) เพื่อแก้ปัญหา macOS
    """
    # 1. ทำให้เป็น Absolute Path
    abs_path = os.path.abspath(path)
    # 2. Normalize เพื่อให้ Path ที่มีอักขระพิเศษ (ภาษาไทย) มีการเข้ารหัสที่ถูกต้อง
    return unicodedata.normalize('NFKC', abs_path)

# ==================== 9. EVIDENCE MAPPING ====================
def load_evidence_mapping(tenant=DEFAULT_TENANT, year=DEFAULT_YEAR, enabler=DEFAULT_ENABLER):
    path = get_evidence_mapping_file_path(tenant, year, enabler)
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

def save_evidence_mapping(data, tenant=DEFAULT_TENANT, year=DEFAULT_YEAR, enabler=DEFAULT_ENABLER):
    path = get_evidence_mapping_file_path(tenant, year, enabler)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(data, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=4)

# ==================== 10. UPDATE MAPPINGS ====================
def _update_doc_id_mapping(
    new_entries: Dict[str, Any],
    doc_type: str,
    tenant: str,
    year: Optional[Union[str, int]],
    enabler: Optional[str]
) -> None:
    try:
        existing_map = load_doc_id_mapping(doc_type, tenant, year, enabler)
    except FileNotFoundError:
        existing_map = {}
        
    existing_map.update(new_entries)
    save_doc_id_mapping(existing_map, doc_type, tenant, year, enabler)
    
    logger.info(f"Updated {len(new_entries)} entries in mapping for {doc_type} / {enabler or 'None'} / Year {year or 'None'}.")

def _update_evidence_mapping(
    new_entries: Dict[str, Any],
    tenant: str,
    year: Optional[Union[str, int]],
    enabler: Optional[str]
) -> None:
    _update_doc_id_mapping(
        new_entries=new_entries,
        doc_type=EVIDENCE_DOC_TYPES,
        tenant=tenant,
        year=year,
        enabler=enabler
    )

# ==================== 11. PATH KEY RESOLUTION (New Critical Logic) ====================
def get_mapping_key_from_physical_path(physical_path: str) -> str:
    """
    แปลง Physical Path (Absolute Path ที่สแกนเจอ) ให้เป็น Relative Key (format: tenant/data/doc_type/...) 
    ที่ใช้ในการค้นหาใน Doc ID Mapping
    
    ใช้ NFKC normalization และ forward slashes ('/').
    """
    if not physical_path:
        return ""
    
    # 📌 FIX 3: ถ้า Path ที่ส่งเข้ามาเป็น Path สัมพัทธ์อยู่แล้ว (เช่น Path จาก Mapping DB)
    if not os.path.isabs(physical_path):
        # ถ้าเป็น Path สัมพัทธ์ (เหมือนใน Mapping) ให้ Normalize และคืนค่าเลย
        relative_key = unicodedata.normalize('NFKC', physical_path).replace('\\', '/')
        return relative_key
        
    # 2. ถ้าเป็น Absolute Path (มาจาก os.walk หรือการสแกน)
    
    # 🟢 ใช้ resolve_filepath_to_absolute เพื่อให้แน่ใจว่าได้ NFKC-normalized Absolute Path
    normalized_abs_path = resolve_filepath_to_absolute(physical_path)
    
    # 3. Normalize DATA_STORE_ROOT
    normalized_abs_data_store_root = resolve_filepath_to_absolute(os.path.abspath(DATA_STORE_ROOT))

    # 4. Get Path relative to DATA_STORE_ROOT
    try:
        relative_path = os.path.relpath(normalized_abs_path, normalized_abs_data_store_root)
    except ValueError as e:
        logger.debug(f"Error getting relative path for {physical_path}: {e}")
        return ""

    # 5. Use forward slashes for the final key format and ensure it doesn't start with '..'
    relative_key = relative_path.replace('\\', '/')
    
    # Safety check: ถ้า Path อยู่นอก Root
    if relative_key.startswith('..'):
         logger.debug(f"File path is outside DATA_STORE_ROOT after relpath: {physical_path}")
         return ""
         
    return relative_key

# OTHER PATHS เ
def get_tenant_year_export_root(tenant: str, year: Union[int, str]) -> str:
    """คืนค่า Path ระดับปี (Root ของ exports) เพื่อใช้วนหาไฟล์ในทุก Enabler"""
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "exports", str(year))
# ==================== จบ utils/path_utils.py ====================