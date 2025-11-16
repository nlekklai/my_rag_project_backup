#map_converter.py
import re
import os
import json
import sys
from typing import Dict, List, Any

# ----------------------------------------------------------------------
# 1. Configuration (Path Corrected)
# ----------------------------------------------------------------------

# Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ✅ FINAL FIX: Path ชี้ไปที่ data/ ภายใน project root
DOC_ID_MAPPING_FILE = os.path.abspath(os.path.join(BASE_DIR, "data", "doc_id_mapping.json"))
OUTPUT_FILE_PATH = os.path.join("evidence_checklist", "km_evidence_mapping_new.json")

# Regular Expression Pattern for KM Evidence Extraction from filename:
# KM(\d+\.\d+)L(\d)\d* เช่น KM3.1L101 จะได้เป็น 3.1_L1
FILENAME_KM_PATTERN = re.compile(r'KM(\d+\.\d+)L(\d)\d*') 


# ----------------------------------------------------------------------
# 2. Core Functions
# ----------------------------------------------------------------------

def _load_full_doc_mapping(filepath: str) -> Dict[str, Any]:
    """
    Loads the internal doc_id_mapping.json which stores the full 64-char Stable UUID 
    as the top-level key. This is the reliable source of truth.
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
    """Main function to load doc mapping and generate the JSON mapping file."""
    
    print("--- เริ่มต้นการสร้างไฟล์ Mapping จาก doc_id_mapping.json (ใช้ Full UUID) ---")

    # 1. Load the reliable source of truth: doc_id_mapping.json
    full_mapping = _load_full_doc_mapping(DOC_ID_MAPPING_FILE)

    if not full_mapping:
        print("------------------------------------------------------------------")
        # ข้อความนี้จะแสดงก็ต่อเมื่อ _load_full_doc_mapping พิมพ์ ERROR/FATAL ออกมาแล้ว
        print("------------------------------------------------------------------")
        return
        
    # Dictionary to store the final grouped mapping
    mapping_results: Dict[str, Any] = {}
    km_evidence_count = 0
    
    # ------------------------------------------------------
    # 3. Parsing and Grouping Logic (Iterate over the FULL UUID keys)
    # ------------------------------------------------------
    
    # doc_id_64 คือ Full 64-char Stable UUID
    for doc_id_64, info in full_mapping.items():
        
        # 🎯 FINAL ROBUST FIX: ตรวจสอบจาก chunk_uuids แทน status
        # กรองเฉพาะเอกสาร 'evidence' สำหรับ 'KM' ที่มี chunk_uuids (บ่งชี้ว่าถูก Ingest สำเร็จแล้ว)
        if (info.get('doc_type') != 'evidence' or 
            info.get('enabler') != 'KM' or
            # 🟢 NEW CONDITION: ต้องมี 'chunk_uuids' และต้องมีสมาชิก (array ไม่ว่างเปล่า)
            not info.get('chunk_uuids') or 
            len(info.get('chunk_uuids', [])) == 0):
            continue

        # ✅ FIX: ใช้ 'file_name' ตามตัวอย่าง JSON ล่าสุด
        full_filename = info.get('file_name', '') 
        
        # 2. Attempt to extract KM, Sub-ID, and Level from the FILENAME
        match_km = FILENAME_KM_PATTERN.search(full_filename)
        
        if match_km:
            sub_id_raw = match_km.group(1) 
            level_raw = match_km.group(2)  
            
            # Combine to form the unique mapping key (e.g., "1.1_L1")
            mapping_key = f"{sub_id_raw}_L{level_raw}"
            
            # Create the evidence dictionary
            evidence_data = {
                "doc_id": doc_id_64, # <-- ใช้ FULL 64-char UUID โดยตรง
                "file_name": full_filename,
                "notes": "Generated from doc_id_mapping.json (Full UUID, checked via chunk_uuids)."
            }
            
            # Initialize the key if it doesn't exist
            if mapping_key not in mapping_results:
                mapping_results[mapping_key] = {
                    "title": f"Mapping for Sub-Criteria {sub_id_raw} Level L{level_raw}",
                    "evidences": []
                }
            
            # Append the new evidence
            mapping_results[mapping_key]['evidences'].append(evidence_data)
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
                # Use ensure_ascii=False for proper Thai character encoding
                json.dump(mapping_results, f, indent=4, ensure_ascii=False)
            
            print("------------------------------------------------------------------")
            print(f"✅ สร้างไฟล์ Mapping สำเร็จที่: {output_filepath}")
            print(f"สรุป: พบ KM Evidence ที่จับคู่ได้: {km_evidence_count} รายการ (ใช้ Full 64-char UUID)")
            print("------------------------------------------------------------------")

        except Exception as e:
            print(f"ERROR: ไม่สามารถบันทึกไฟล์ JSON ได้: {e}", file=sys.stderr)


if __name__ == "__main__":
    generate_mapping()