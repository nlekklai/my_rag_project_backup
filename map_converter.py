import re
import os
import json
import sys
from typing import Dict, List, Any
import datetime
import argparse 

# ----------------------------------------------------------------------
# 0. Path Setup for Global Import
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumes project root is one level up from BASE_DIR
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir)) 

# Add project root to path for module import
sys.path.append(PROJECT_ROOT) 
try:
    # Import Global Constants
    from config.global_vars import ( 
        EVIDENCE_MAPPING_FILENAME_SUFFIX, 
        DOCUMENT_ID_MAPPING_FILENAME_SUFFIX,
        MAPPING_BASE_DIR, 
        # ไม่จำเป็นต้องใช้ DEFAULT_TENANT/YEAR ใน Logic นี้แล้ว แต่เก็บไว้ใน block เพื่อความสมบูรณ์
        DEFAULT_TENANT,
        DEFAULT_YEAR
    )
except ImportError as e:
    print(f"FATAL: ไม่สามารถ Import ตัวแปร Global ได้: {e}", file=sys.stderr)
    print("โปรดตรวจสอบว่าไฟล์ global_vars.py และโครงสร้าง folder ถูกต้อง", file=sys.stderr)
    sys.exit(1)


# ----------------------------------------------------------------------
# 1. Configuration 
# ----------------------------------------------------------------------

# Regular Expression Pattern for KM Evidence Extraction from filename:
FILENAME_KM_PATTERN = re.compile(r'KM(\d+\.\d+)L(\d)\d*') 

# กำหนด Timestamp และ Mapper Type สำหรับรายการที่สร้างอัตโนมัติ
GENERATED_TIMESTAMP = datetime.datetime.now().isoformat(timespec='milliseconds')
GENERATED_MAPPER_TYPE = "AI_GENERATED"


# ----------------------------------------------------------------------
# 2. Core Functions
# ----------------------------------------------------------------------

def _load_full_doc_mapping(filepath: str) -> Dict[str, Any]:
    """
    Loads the Doc ID Mapping JSON file which stores the full 64-char Stable UUID.
    """
    print(f"อ่านไฟล์ต้นฉบับ (Doc ID Mapping): {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"FATAL: ไม่สามารถโหลดไฟล์ Doc ID Mapping ได้: {e}", file=sys.stderr)
        return {}

def generate_mapping(tenant: str, year: str, enabler: str, doc_map_file_path: str):
    """Main function to load doc mapping and generate the JSON mapping file in the correct format."""
    
    print("--- เริ่มต้นการสร้างไฟล์ Mapping จากไฟล์ Input ที่ระบุ (ใช้ Full UUID) ---")

    # 1. Load the reliable source of truth from the provided path
    full_mapping = _load_full_doc_mapping(doc_map_file_path)

    if not full_mapping:
        print("------------------------------------------------------------------")
        print("------------------------------------------------------------------")
        return
        
    # Dictionary to store the final grouped mapping (Target Format: {"1.1.L1": [...]})
    mapping_results: Dict[str, List[Dict[str, Any]]] = {}
    km_evidence_count = 0
    
    # ------------------------------------------------------
    # 3. Parsing and Grouping Logic 
    # ------------------------------------------------------
    
    for doc_id_64, info in full_mapping.items():
        
        # กรองเฉพาะเอกสาร 'evidence' สำหรับ Enabler ที่ระบุ และต้องมี chunk_uuids
        if (info.get('doc_type') != 'evidence' or 
            info.get('enabler', '').lower() != enabler.lower() or 
            not info.get('chunk_uuids') or 
            len(info.get('chunk_uuids', [])) == 0):
            continue

        full_filename = info.get('file_name', '') 
        
        # 2. Attempt to extract KM, Sub-ID, and Level from the FILENAME
        match_km = FILENAME_KM_PATTERN.search(full_filename)
        
        if match_km:
            sub_id_raw = match_km.group(1) 
            level_raw = match_km.group(2)  
            
            # Key Format: "1.1.L1"
            mapping_key = f"{sub_id_raw}.L{level_raw}"
            
            # สร้าง Evidence Dictionary ที่มี Field ครบ
            evidence_data = {
                "doc_id": doc_id_64, 
                "filename": full_filename, 
                "mapper_type": GENERATED_MAPPER_TYPE,
                "timestamp": GENERATED_TIMESTAMP
            }
            
            if mapping_key not in mapping_results:
                mapping_results[mapping_key] = []
            
            mapping_results[mapping_key].append(evidence_data)
            km_evidence_count += 1
        
    # ------------------------------------------------------
    # 4. Output Generation
    # ------------------------------------------------------
    if not mapping_results:
        print("------------------------------------------------------------------")
        print(f"⚠️ WARNING: ไม่พบ {enabler.upper()} Evidence ที่มีรูปแบบ KMX.XLX ที่ถูก Ingest แล้วในไฟล์ Input")
        print("------------------------------------------------------------------")
    else:
        # 1. Define Output Filename (e.g., pea_2568_km_evidence_mapping.json)
        output_filename = f"{tenant.lower()}_{year}_{enabler.lower()}{EVIDENCE_MAPPING_FILENAME_SUFFIX}"

        # 2. Define Output Directory (e.g., [Project Root]/config/mapping/pea/2568)
        output_dir = os.path.join(MAPPING_BASE_DIR, tenant, year)
        
        # 3. Define Final Output Path
        output_filepath = os.path.join(output_dir, output_filename)
        
        try:
            # สร้าง directory หากยังไม่มี
            os.makedirs(output_dir, exist_ok=True)
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(mapping_results, f, indent=4, ensure_ascii=False)
            
            print("------------------------------------------------------------------")
            print(f"✅ สร้างไฟล์ Mapping สำเร็จที่: {output_filepath}")
            print(f"สรุป: พบ {enabler.upper()} Evidence ที่จับคู่ได้: {km_evidence_count} รายการ (ใช้ Full 64-char UUID)")
            print("------------------------------------------------------------------")

        except Exception as e:
            print(f"ERROR: ไม่สามารถบันทึกไฟล์ JSON ได้: {e}", file=sys.stderr)


# ----------------------------------------------------------------------
# 5. Argument Parsing and Main Execution
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate evidence mapping JSON file from doc_id_mapping.json based on KM filename pattern.")
    parser.add_argument('--tenant', type=str, required=True, help="Tenant ID (e.g., pea).")
    parser.add_argument('--year', type=str, required=True, help="Assessment year (e.g., 2568).")
    parser.add_argument('--enabler', type=str, required=True, help="Enabler ID (e.g., km).")
    
    # 💡 FIX: ทำให้ --input_file เป็น Optional โดยมี Default เป็น None
    parser.add_argument('--input_file', type=str, 
                        default=None, 
                        help="Path to the source Doc ID Mapping JSON file (If omitted, uses tenant/year/enabler convention).")
    
    args = parser.parse_args()
    
    # Normalize inputs for consistency
    tenant = args.tenant.lower()
    enabler = args.enabler.lower()
    year = args.year
    
    # 💡 FIX: Logic การกำหนด Doc Map File Path
    if args.input_file is None:
        # 1. Construct the input filename using the standard suffix
        input_filename = f"{tenant}_{year}_{enabler}{DOCUMENT_ID_MAPPING_FILENAME_SUFFIX}"
        
        # 2. Construct the full path using MAPPING_BASE_DIR
        input_dir = os.path.join(MAPPING_BASE_DIR, tenant, year)
        doc_map_file_path = os.path.join(input_dir, input_filename)
        
        print(f"--- [Info] สร้าง Input Path อัตโนมัติ: {doc_map_file_path} ---")
    else:
        # Use the path provided by the user
        doc_map_file_path = args.input_file

    print(f"--- [Configuration] Tenant: {tenant}, Year: {year}, Enabler: {enabler.upper()} ---")
    
    # Call the core function
    generate_mapping(
        tenant=tenant, 
        year=year, 
        enabler=enabler,
        doc_map_file_path=os.path.abspath(doc_map_file_path), 
    )


if __name__ == "__main__":
    main()