# utils/path_utils.py
# Production Final Version – 11 ธ.ค. 2568 (Path Matching Fixes)

"""
📌 PROJECT FILE STRUCTURE & MAPPING LOGIC (Updated: 2026)
-------------------------------------------------------
ระบบจัดการไฟล์แบ่งออกเป็น 2 รูปแบบหลัก ตามประเภทของ doc_type:

1. GLOBAL DOCUMENTS (document, faq, seam)
   - โครงสร้าง: data_store/{tenant}/data/{doc_type}/{filename}
   - การ Mapping: อยู่ที่ root ของ mapping folder เสมอ
   - ไฟล์ JSON: {tenant}_{doc_type}_doc_id_mapping.json 
   - หมายเหตุ: ไม่ใช้ year และ enabler ในการหาไฟล์

2. YEARLY EVIDENCE (evidence)
   - โครงสร้าง: data_store/{tenant}/data/evidence/{year}/{enabler}/{filename}
   - การ Mapping: แยกตามปีและกลุ่มข้อมูล
   - ไฟล์ JSON: mapping/{year}/{tenant}_{year}_{enabler}_doc_id_mapping.json
   - หมายเหตุ: ต้องมี year และ enabler ครบถ้วนในการระบุตำแหน่ง

การหาไฟล์ใช้ระบบ Fuzzy Scan (NFKC Normalization) เพื่อรองรับปัญหาชื่อไฟล์ภาษาไทย 
และการจัดเก็บไฟล์ที่อาจมีความต่างของ Case-sensitive บน macOS/Linux
"""

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
    base = get_mapping_tenant_root_path(tenant)
    dt = _n(doc_type)
    
    # ดึงค่า "evidence" จาก config มาเปรียบเทียบ
    from config.global_vars import EVIDENCE_DOC_TYPES
    evidence_type = _n(EVIDENCE_DOC_TYPES)

    # === 1. กรณีเป็น Evidence เท่านั้นที่จะไปหาใน Folder ปี (2568/...) ===
    if dt == evidence_type:
        # ป้องกัน error ถ้าลืมส่งปีหรือ enabler มาสำหรับ evidence
        safe_year = year if year else "default_year"
        safe_enabler = _n(enabler) if enabler else "default"
        
        return os.path.join(
            base,
            str(safe_year),
            f"{_n(tenant)}_{safe_year}_{safe_enabler}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
        )

    # === 2. อะไรก็ตามที่ไม่ใช่ evidence ให้ถือว่าเป็น Global Doc-Type ทั้งหมด ===
    # ตัด year และ enabler ทิ้งไปเลย เพื่อให้ได้ path: mapping/pea_document_doc_id_mapping.json
    return os.path.join(
        base,
        f"{_n(tenant)}_{dt}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
    )


def get_evidence_mapping_file_path(tenant: str, year: Optional[Union[int, str]], enabler: str) -> str:
    # 1. ได้ Root Path ของ mapping/
    base = get_mapping_tenant_root_path(tenant)
    
    # 2. เตรียมชื่อไฟล์: ถ้าไม่มีปี ก็ไม่ต้องใส่ปีในชื่อไฟล์
    year_prefix = f"{year}_" if year else ""
    filename = f"{_n(tenant)}_{year_prefix}{_n(enabler)}{EVIDENCE_MAPPING_FILENAME_SUFFIX}"
    
    # 3. รวม Path: ถ้าไม่มีปี ให้เอาวางไว้ที่ base เลย (data_store/pea/mapping/...)
    if year:
        return os.path.join(base, str(year), filename)
    else:
        return os.path.join(base, filename)

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

def save_doc_id_mapping(
    data: Dict,
    doc_type: str,
    tenant: str,
    year: Optional[Union[int, str]],
    enabler: Optional[str] = None
) -> None:
    path = get_mapping_file_path(doc_type, tenant, year, enabler)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Normalize path (macOS-safe)
    path = unicodedata.normalize("NFKC", path)
    tmp_path = f"{path}.tmp"

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Atomic replace
        os.replace(tmp_path, path)

    except Exception as e:
        logger.error(f"Failed to save doc_id_mapping: {path} | {e}")

        # Cleanup temp file if exists
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # IMPORTANT: propagate error to caller
        raise

# ==================== 6. PARSE COLLECTION NAME ====================
def parse_collection_name(collection_name: str) -> Tuple[str, Optional[str]]:
    name = _n(collection_name)
    if name.startswith(f"{EVIDENCE_DOC_TYPES.lower()}_"):
        parts = name.split("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return name, None

# ==================== 7. DOCUMENT FILE PATH RESOLVER (REVISED) ====================
def get_document_file_path(
    document_uuid: str,
    tenant: str,
    year: Optional[Union[int, str]],
    enabler: Optional[str],
    doc_type_name: str
) -> Optional[Dict[str, str]]:
    """
    ค้นหา Path ของไฟล์ PDF บน Disk โดยใช้ UUID 
    รองรับภาษาไทยแบบ Robust ทั้งระบบ Linux และการอัปโหลดจากเครื่อง Client
    """
    try:
        tenant_clean = _n(tenant)
        doc_type_clean = _n(doc_type_name).lower()
        
        # --- 1. โหลด Mapping Data ---
        mapping_path = get_mapping_file_path(doc_type_name, tenant, year, enabler)
        
        # Fallback: ลองหาในระดับ Global ถ้าตามปีไม่เจอ
        if not os.path.exists(mapping_path):
            mapping_path = get_mapping_file_path(doc_type_name, tenant, None, None)

        if not os.path.exists(mapping_path):
            logger.warning(f"❌ [Path Resolver] Mapping file not found: {mapping_path}")
            return None

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        entry = mapping_data.get(document_uuid)
        if not entry:
            logger.warning(f"❌ [Path Resolver] UUID {document_uuid} not found in mapping")
            return None

        # --- 2. ตรวจสอบ Direct Path (จากที่บันทึกไว้ใน DB) ---
        stored_path = entry.get("filepath", "")
        filename = entry.get("file_name") or entry.get("filename") or os.path.basename(stored_path)
        
        if stored_path:
            potential_path = stored_path if os.path.isabs(stored_path) else os.path.join(DATA_STORE_ROOT, stored_path)
            potential_path = resolve_filepath_to_absolute(potential_path)

            # ตรวจสอบไฟล์จริง (NFC)
            if os.path.exists(potential_path):
                logger.info(f"✅ [Path Resolver] Direct hit: {potential_path}")
                return {"file_path": potential_path, "original_filename": filename}
            
            # Fallback 1: ลองแบบ NFD (เผื่อไฟล์มาจาก Mac)
            nfd_path = unicodedata.normalize('NFD', potential_path)
            if os.path.exists(nfd_path):
                logger.info(f"✅ [Path Resolver] Direct hit (NFD): {nfd_path}")
                return {"file_path": nfd_path, "original_filename": filename}

        # --- 3. Fuzzy Scan (ถ้า Direct Path หาไม่เจอ หรือชื่อไฟล์ภาษาไทยไม่ตรงกัน) ---
        if doc_type_clean == "evidence":
            year_val = str(year) if year and str(year) != "None" else ""
            base_search_path = os.path.join(DATA_STORE_ROOT, tenant_clean, "data", "evidence", year_val)
        else:
            base_search_path = os.path.join(DATA_STORE_ROOT, tenant_clean, "data", doc_type_clean)

        if not os.path.exists(base_search_path):
            base_search_path = os.path.join(DATA_STORE_ROOT, tenant_clean, "data")

        logger.info(f"🔎 [Path Resolver] Scanning: {base_search_path} for: {filename}")
        
        # Normalize ชื่อไฟล์ที่ต้องการหาให้เป็นกลางที่สุด (ใช้ _n ที่คุณทำไว้)
        target_fn_norm = _n(filename) 
        
        for root, dirs, files in os.walk(base_search_path):
            for f in files:
                # 🟢 หัวใจสำคัญ: เทียบชื่อไฟล์โดยการ Normalize ทั้งคู่ก่อนเทียบ
                if _n(f) == target_fn_norm:
                    raw_full_path = os.path.join(root, f)
                    
                    # 🟢 แปลงเป็น NFC เพื่อให้ระบบ Linux/FastAPI Stream ไฟล์ได้
                    final_path = resolve_filepath_to_absolute(raw_full_path)
                    
                    if os.path.exists(final_path):
                        logger.info(f"✅ [Path Resolver] Fuzzy match found & Verified: {final_path}")
                        return {"file_path": final_path, "original_filename": f}
                    
                    # Last Resort: ถ้า OS บอกว่า NFC ไม่เจอ ให้ส่ง Path ดิบที่ os.walk เจอไป
                    if os.path.exists(raw_full_path):
                        logger.warning(f"⚠️ [Path Resolver] Found by walk but NFC check failed, using raw: {raw_full_path}")
                        return {"file_path": raw_full_path, "original_filename": f}

        logger.error(f"❌ [Path Resolver] File not found on disk: {filename}")
        return None

    except Exception as e:
        logger.error(f"🔴 [Path Resolver] Critical Error: {str(e)}", exc_info=True)
        return None

# ==================== 8. OTHER PATHS ====================
def get_config_tenant_root_path(tenant: str) -> str:
    """Path สำหรับ Configuration Files ที่คงที่ เช่น Rubrics, Contextual Rules"""
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "config")

def get_rubric_file_path(tenant: str, enabler: str) -> str:
    return os.path.join(get_config_tenant_root_path(tenant),
                        RUBRIC_FILENAME_PATTERN.format(tenant=_n(tenant), enabler=_n(enabler)))

def get_contextual_rules_file_path(tenant: str, enabler: str) -> str:
    """
    ปรับให้รองรับการหาไฟล์แบบลำดับชั้น (Tenant-Specific -> Global Default)
    """
    # 1. ลองหาใน Folder ของ Tenant ก่อน (เช่น data_store/pea/config/pea_km_contextual_rules.json)
    tenant_path = os.path.join(
        get_config_tenant_root_path(tenant),
        f"{_n(tenant)}_{_n(enabler)}_contextual_rules.json"
    )

    if os.path.exists(tenant_path):
        return tenant_path
        
    # 2. (Optional) ถ้าไม่เจอ ให้ Fallback ไปที่ Default Rules กลางของระบบ
    # เพื่อให้ Tenant ใหม่เริ่มใช้งานได้ทันทีโดยไม่ต้องเขียนกฎใหม่หมด
    return os.path.join(DATA_STORE_ROOT, "common", "config", f"default_{_n(enabler)}_rules.json")

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
    แปลง Path ให้เป็น Absolute Path และ Normalize เป็น NFC 
    ซึ่งเป็นมาตรฐานที่ Linux File System และ FastAPI (FileResponse) เข้าใจได้ดีที่สุด
    """
    if not path:
        return ""
    # 1. ทำให้เป็น Absolute Path
    abs_path = os.path.abspath(path)
    # 2. 🟢 เปลี่ยนจาก NFKC เป็น NFC เพื่อป้องกันสระไทย (เช่น สระอำ) ถูกแปลงรหัสจนไฟล์หาไม่เจอ
    return unicodedata.normalize('NFC', abs_path)

# ==================== 10. UPDATE MAPPINGS ====================
def _update_doc_id_mapping(
    new_entries: Dict[str, Any],
    doc_type: str,
    tenant: str,
    year: Optional[Union[str, int]],
    enabler: Optional[str]
) -> None:
    if not new_entries:
        logger.debug("No new entries to update in doc_id_mapping.")
        return

    try:
        existing_map = load_doc_id_mapping(doc_type, tenant, year, enabler) or {}
    except Exception as e:
        logger.error(
            f"Failed to load doc_id_mapping for "
            f"{doc_type} / {enabler or 'None'} / Year {year or 'None'} | {e}"
        )
        existing_map = {}

    before_count = len(existing_map)
    overwrite_keys = set(existing_map) & set(new_entries)

    if overwrite_keys:
        logger.warning(
            f"Overwriting {len(overwrite_keys)} existing doc_id keys "
            f"for {doc_type} / {enabler or 'None'} / Year {year or 'None'}"
        )

    existing_map.update(new_entries)

    save_doc_id_mapping(existing_map, doc_type, tenant, year, enabler)

    logger.info(
        f"Updated doc_id_mapping: +{len(new_entries)} entries "
        f"(overwrite {len(overwrite_keys)}) | "
        f"{doc_type} / {enabler or 'None'} / Year {year or 'None'} | "
        f"Total={before_count}→{len(existing_map)}"
    )


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

def get_tenant_year_report_root(
    tenant: str,
    year: Union[int, str],
    enabler: Optional[str] = None
) -> str:
    """
    สร้างและส่งคืน Path สำหรับเก็บไฟล์รายงาน (Reports)

    โครงสร้าง:
        <DATA_STORE_ROOT>/<tenant>/reports/<year>/<enabler?>
    """
    if not tenant:
        raise ValueError("tenant is required")
    if year is None:
        raise ValueError("year is required")

    parts = [
        DATA_STORE_ROOT,
        _n(tenant),
        "reports",
        str(year),
    ]

    if enabler:
        parts.append(_n(enabler))

    base_dir = os.path.join(*parts)

    # Normalize + ensure directory exists (safe & idempotent)
    base_dir = unicodedata.normalize("NFKC", base_dir)
    os.makedirs(base_dir, exist_ok=True)

    return base_dir


# ==================== 12. STABLE UUID V5 GENERATOR (Production Final) ====================

def create_stable_uuid_from_path(
    filepath: str,
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
) -> str:
    if not filepath:
        logger.error("Empty filepath provided")
        return str(uuid.uuid4())

    tenant_clean = _n(tenant or "")
    enabler_clean = _n(enabler or "")
    year_str = str(year) if year is not None else ""

    key_seed = None

    # Stat-based (preferred)
    try:
        filepath = resolve_filepath_to_absolute(filepath)
        st = os.stat(filepath)
        filename_norm = _n(os.path.basename(filepath))
        key_seed = f"{filename_norm}:{st.st_size}:{int(st.st_mtime)}:{tenant_clean}:{year_str}:{enabler_clean}"
    except Exception:
        pass

    # Path-based fallback
    if not key_seed:
        try:
            rel_key = get_mapping_key_from_physical_path(filepath)
            if rel_key:
                key_seed = f"{rel_key}:{tenant_clean}:{year_str}:{enabler_clean}"
        except Exception:
            pass

    if not key_seed:
        logger.error("Failed to create stable key, using random UUID4")
        return str(uuid.uuid4())

    # Namespace
    try:
        namespace = uuid.UUID(PROJECT_NAMESPACE_UUID) if isinstance(PROJECT_NAMESPACE_UUID, str) else PROJECT_NAMESPACE_UUID
    except Exception:
        namespace = uuid.NAMESPACE_DNS

    return str(uuid.uuid5(namespace, key_seed))

__all__ = [
    "_n",
    "get_document_source_dir",
    "get_evidence_base_dir",
    "get_doc_type_collection_key",
    "get_vectorstore_collection_path",
    "get_vectorstore_tenant_root_path",
    "get_mapping_file_path",
    "get_evidence_mapping_file_path",
    "get_mapping_tenant_root_path",
    "load_doc_id_mapping",
    "save_doc_id_mapping",
    "_update_doc_id_mapping",     # ✅ เพิ่ม
    "_update_evidence_mapping",   # ✅ เพิ่ม
    "parse_collection_name",
    "get_document_file_path",
    "get_config_tenant_root_path",
    "get_rubric_file_path",
    "get_contextual_rules_file_path",
    "get_export_dir",
    "get_assessment_export_file_path",
    "get_tenant_year_export_root",
    "get_tenant_year_report_root",
    "get_mapping_key_from_physical_path",
    "create_stable_uuid_from_path",
    "resolve_filepath_to_absolute" # ✅ เพิ่ม
]
