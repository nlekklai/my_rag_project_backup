#map_converter.py
import re
import os
import json
import sys
from typing import Dict, List, Any
import datetime # <--- [NEW] นำเข้า datetime สำหรับสร้าง timestamp

# ----------------------------------------------------------------------
# 1. Configuration 
# ----------------------------------------------------------------------

# Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ✅ FINAL FIX: Path ชี้ไปที่ data/ ภายใน project root
DOC_ID_MAPPING_FILE = os.path.abspath(os.path.join(BASE_DIR, "data", "doc_id_mapping.json"))
OUTPUT_FILE_PATH = os.path.join("evidence_checklist", "km_evidence_mapping_new.json")

# Regular Expression Pattern for KM Evidence Extraction from filename:
FILENAME_KM_PATTERN = re.compile(r'KM(\d+\.\d+)L(\d)\d*') 

# [NEW] กำหนด Timestamp และ Mapper Type สำหรับรายการที่สร้างอัตโนมัติ
GENERATED_TIMESTAMP = datetime.datetime.now().isoformat(timespec='milliseconds')
GENERATED_MAPPER_TYPE = "AI_GENERATED"


# ----------------------------------------------------------------------
# 2. Core Functions
# ----------------------------------------------------------------------

def _load_full_doc_mapping(filepath: str) -> Dict[str, Any]:
    """
    Loads the internal doc_id_mapping.json which stores the full 64-char Stable UUID.
    """
    print(f"อ่านไฟล์ต้นฉบับ (doc_id_mapping.json): {filepath}")
    try:
        # Tries to handle different encodings if the default utf-8 fails
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return {}
    except Exception as e:
        # ในกรณีที่ไฟล์มีขนาด 2 ไบต์ หรือมีข้อมูลไม่สมบูรณ์
        print(f"FATAL: ไม่สามารถโหลด doc_id_mapping.json ได้ (อาจต้องรัน Ingest ก่อน)", file=sys.stderr)
        return {}

def generate_mapping():
    """Main function to load doc mapping and generate the JSON mapping file in the correct format."""
    
    print("--- เริ่มต้นการสร้างไฟล์ Mapping จาก doc_id_mapping.json (ใช้ Full UUID) ---")

    # 1. Load the reliable source of truth: doc_id_mapping.json
    full_mapping = _load_full_doc_mapping(DOC_ID_MAPPING_FILE)

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
        
        # กรองเฉพาะเอกสาร 'evidence' สำหรับ 'KM' ที่มี chunk_uuids
        if (info.get('doc_type') != 'evidence' or 
            info.get('enabler') != 'KM' or
            not info.get('chunk_uuids') or 
            len(info.get('chunk_uuids', [])) == 0):
            continue

        # ✅ FIX: ใช้ 'file_name'
        full_filename = info.get('file_name', '') 
        
        # 2. Attempt to extract KM, Sub-ID, and Level from the FILENAME
        match_km = FILENAME_KM_PATTERN.search(full_filename)
        
        if match_km:
            sub_id_raw = match_km.group(1) 
            level_raw = match_km.group(2)  
            
            # 🎯 [CHANGE 1] แก้ไข: Key Format เปลี่ยนจาก "1.1_L1" เป็น "1.1.L1"
            mapping_key = f"{sub_id_raw}.L{level_raw}"
            
            # 🎯 [CHANGE 3] สร้าง Evidence Dictionary ที่มี Field ครบตาม Format เป้าหมาย
            evidence_data = {
                "doc_id": doc_id_64, 
                "filename": full_filename, # 🎯 เปลี่ยนชื่อ Field เป็น 'filename'
                "mapper_type": GENERATED_MAPPER_TYPE,
                "timestamp": GENERATED_TIMESTAMP
            }
            
            # 🎯 [CHANGE 2] แก้ไข: เปลี่ยน Structure ให้เป็น List โดยตรงภายใต้ Key
            if mapping_key not in mapping_results:
                mapping_results[mapping_key] = []
            
            # เพิ่ม 1 Entry ต่อ 1 เอกสารที่ถูกค้นพบ (ผู้ใช้สามารถทำซ้ำเองได้หากต้องการความเสถียร 3 Entries)
            mapping_results[mapping_key].append(evidence_data)
            km_evidence_count += 1
        
    # ------------------------------------------------------
    # 4. Output Generation
    # ------------------------------------------------------
    if not mapping_results:
        print("------------------------------------------------------------------")
        print("⚠️ WARNING: ไม่พบ KM Evidence ที่มีรูปแบบ KMX.XLX ที่ถูก Ingest แล้ว")
        print("------------------------------------------------------------------")
    else:
        # Write to JSON file
        output_filepath = os.path.join(BASE_DIR, OUTPUT_FILE_PATH)
        try:
            # สร้าง directory หากยังไม่มี
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(mapping_results, f, indent=4, ensure_ascii=False)
            
            print("------------------------------------------------------------------")
            print(f"✅ สร้างไฟล์ Mapping สำเร็จที่: {output_filepath}")
            print(f"สรุป: พบ KM Evidence ที่จับคู่ได้: {km_evidence_count} รายการ (ใช้ Full 64-char UUID)")
            print("------------------------------------------------------------------")

        except Exception as e:
            print(f"ERROR: ไม่สามารถบันทึกไฟล์ JSON ได้: {e}", file=sys.stderr)


if __name__ == "__main__":
    generate_mapping()