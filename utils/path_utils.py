# utils/path_utils.py
# Production Final Version – 10 ธ.ค. 2568 (แก้ Syntax Error แล้ว)
# ใช้ไฟล์นี้ทับของเดิมได้เลย – ทุกอย่างจะทำงานทันที

import os
import json
import logging
import hashlib
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, Union, List
import unicodedata

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
)

logger = logging.getLogger(__name__)

# ==================== CORE HELPER ====================
def _n(s: Union[str, None]) -> str:
    """Normalize ทุก string ด้วย NFKC – แก้ macOS NFD bug ถาวร"""
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
def get_mapping_file_path(tenant: str, year: Optional[Union[int, str]], enabler: Optional[str]) -> str:
    # แก้ตรงนี้: ลบ ) ซ้อนผิด
    base = get_mapping_tenant_root_path(tenant)
    if year is not None and enabler:
        return os.path.join(base, str(year), f"{_n(tenant)}_{year}_{_n(enabler)}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}")
    return os.path.join(base, f"{_n(tenant)}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}")

def get_evidence_mapping_file_path(tenant: str, year: Union[int, str], enabler: str) -> str:
    return os.path.join(get_mapping_tenant_root_path(tenant), str(year),
                         f"{_n(tenant)}_{year}_{_n(enabler)}{EVIDENCE_MAPPING_FILENAME_SUFFIX}")

def get_mapping_tenant_root_path(tenant: str) -> str:
    return os.path.join(DATA_STORE_ROOT, _n(tenant), "mapping")

# ==================== 4. LOAD / SAVE MAPPING ====================
def load_doc_id_mapping(doc_type: str, tenant: str, year: Optional[Union[int, str]], enabler: Optional[str] = None) -> Dict:
    path = get_mapping_file_path(tenant, year, enabler)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Load mapping failed {path}: {e}")
        return {}

def save_doc_id_mapping(data: Dict, doc_type: str, tenant: str, year: Optional[Union[int, str]], enabler: Optional[str] = None):
    path = get_mapping_file_path(tenant, year, enabler)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Save mapping failed {path}: {e}")

# ==================== 5. STABLE UUID ====================
def create_stable_uuid_from_path(
    filepath: str,
    tenant: Optional[str] = None,
    year: Optional[Union[int, str]] = None,
    enabler: Optional[str] = None,
) -> str:
    filepath = _n(filepath)
    tenant = _n(tenant or "")
    enabler = _n(enabler or "")

    if not os.path.exists(filepath):
        logger.warning(f"File not found for UUID: {filepath}")
        return str(uuid.uuid4())

    try:
        st = os.stat(filepath)
        key = f"{_n(os.path.basename(filepath))}:{st.st_size}:{int(st.st_mtime)}:{tenant}:{year or ''}:{enabler}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.error(f"Error creating stable UUID for {filepath}: {e}")
        return str(uuid.uuid4())

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
        mapping_path = get_mapping_file_path(tenant, year, enabler)
        if not os.path.exists(mapping_path):
            logger.warning(f"Mapping file not found: {mapping_path}")
            return None

        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)

        entry = mapping_data.get(document_uuid)
        if not entry:
            logger.warning(f"UUID {document_uuid} not found in mapping")
            return None

        original_filename = entry.get("file_name")
        if not original_filename:
            logger.warning(f"No filename in mapping for UUID {document_uuid}")
            return None

        base_dir = get_document_source_dir(tenant, year, enabler, doc_type_name)
        file_path = os.path.join(base_dir, original_filename)

        if not os.path.exists(file_path):
            logger.error(f"File not found on disk: {file_path}")
            return None

        return {
            "file_path": file_path,
            "original_filename": original_filename
        }
    except Exception as e:
        logger.error(f"Error resolving document path for UUID {document_uuid}: {e}")
        return None

# ==================== 8. OTHER PATHS ====================
def get_rubric_file_path(tenant: str, enabler: str) -> str:
    return os.path.join(get_mapping_tenant_root_path(tenant),
                        RUBRIC_FILENAME_PATTERN.format(tenant=_n(tenant), enabler=_n(enabler)))

def get_contextual_rules_file_path(tenant: str, enabler: str) -> str:
    return os.path.join(get_mapping_tenant_root_path(tenant),
                        f"{_n(tenant)}_{_n(enabler)}_contextual_rules.json")

def get_export_dir(tenant: str, year: Union[int, str], enabler: str) -> str:
    return os.path.join(EXPORTS_DIR, _n(tenant), str(year), _n(enabler))

def get_assessment_export_file_path(tenant: str, year: Union[int, str], enabler: str, suffix: str, ext: str = "json") -> str:
    return os.path.join(get_export_dir(tenant, year, enabler),
                        f"{_n(tenant)}_{year}_{_n(enabler)}_{suffix}.{ext.lower()}")

def get_normalized_metadata(doc_type: str, year_input=None, enabler_input=None, default_enabler=None):
    return (None, None) if _n(doc_type) != EVIDENCE_DOC_TYPES.lower() else (year_input, enabler_input or default_enabler)

def resolve_filepath_to_absolute(path: str) -> str:
    path = _n(path)
    return path if os.path.isabs(path) else os.path.join(DATA_STORE_ROOT, path)

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
    """ใช้ใน core/ingest.py"""
    _update_doc_id_mapping(
        new_entries=new_entries,
        doc_type=EVIDENCE_DOC_TYPES,
        tenant=tenant,
        year=year,
        enabler=enabler
    )

# ==================== จบไฟล์ – ใช้ทับได้เลย ====================