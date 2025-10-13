import os
import json
import glob
from typing import Dict, List, Any
from pathlib import Path
import sys

# --- CONFIGURATION (ใช้ CWD เป็น Project Root) ---

# 1. กำหนด Project Root เป็น Current Working Directory (ตำแหน่งที่รัน python)
PROJECT_ROOT = Path(os.getcwd())

# 2. Path ไปยัง Checklist: CWD / evidence_checklist / km_evidence_statements_checklist.json
CHECKLIST_DIR = "evidence_checklist"
CHECKLIST_FILENAME = "km_evidence_statements_checklist.json"
CHECKLIST_PATH = PROJECT_ROOT / CHECKLIST_DIR / CHECKLIST_FILENAME

# 3. Path ไปยังโฟลเดอร์หลักฐาน: CWD / evidence 
#    *** Based on your ls -la output, your evidence folder might be in 'data/evidence' ***
#    *** ผมจะแก้ไข EVIDENCE_DIR ตามผลลัพธ์ที่คุณแสดง: data/evidence ***
EVIDENCE_DIR = PROJECT_ROOT / "vectorstore" / "evidence" 
# ถ้าไม่ใช่ 'data/evidence' และเป็น 'evidence' ให้ใช้: EVIDENCE_DIR = PROJECT_ROOT / "evidence"

# 4. Path สำหรับบันทึก Evidence Mapping JSON: CWD / core / km_evidence_mapping.json
MAPPING_OUTPUT_PATH = PROJECT_ROOT / "evidence_checklist" / "km_evidence_mapping.json"

# รายการ Enabler ทั้งหมด
ENABLER_ABBR_LIST = ["CG", "L", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"]

# --------------------------------------------------------

def load_checklist(path: Path) -> List[Dict[str, Any]]:
    """โหลดไฟล์ KM Checklist JSON."""
    try:
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Checklist file not found at {path.resolve()}")
        if not path.parent.is_dir():
             print(f"*** Debug Note: Directory '{path.parent}' does not exist. ***")
        return []
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in {path.resolve()}. Details: {e}")
        return []

def get_evidence_files_by_prefix(prefix: str, base_dir: Path) -> List[str]:
    """ใช้ glob เพื่อค้นหาชื่อไฟล์/โฟลเดอร์ใน EVIDENCE_DIR."""
    # แปลง Path object เป็น string สำหรับ glob
    search_pattern = str(base_dir / f"{prefix}*")
    
    found_paths = glob.glob(search_pattern)
    doc_ids = [os.path.basename(p) for p in found_paths]
    return doc_ids

def create_auto_mapping(target_enabler: str):
    """สร้าง Evidence Mapping โดยการสแกน Checklist และ Evidence Directory."""
    if target_enabler not in ENABLER_ABBR_LIST:
        print(f"❌ Error: Target Enabler '{target_enabler}' not in the defined list: {ENABLER_ABBR_LIST}")
        return

    print("-" * 50)
    print(f"      Starting Automated Evidence Mapping for ENABLER: {target_enabler}")
    print("-" * 50)
    print(f"*** Run Directory (CWD): {PROJECT_ROOT.resolve()} ***")
    print(f"🔎 Loading checklist from: {CHECKLIST_PATH.resolve()}")
    print(f"📂 Scanning evidence directory: {EVIDENCE_DIR.resolve()}")
    
    # 1. โหลด Checklist
    checklist = load_checklist(CHECKLIST_PATH)
    if not checklist:
        print("🛑 Cannot proceed with mapping. Checklist not loaded.")
        return # ออกจากฟังก์ชันเมื่อโหลดล้มเหลว

    # 2. สร้าง Mapping (กำหนดตัวแปรให้อยู่ใน Scope ที่ใช้งาน)
    evidence_mapping: Dict[str, Any] = {}
    mapped_count = 0
    
    for enabler in checklist:
        sub_criteria_id = enabler.get("Sub_Criteria_ID")
        
        if not sub_criteria_id:
            continue
            
        for level in range(1, 6):
            level_key = f"Level_{level}_Statements"
            statements: List[str] = enabler.get(level_key, [])
            
            if not statements:
                continue

            prefix = f"{target_enabler}{sub_criteria_id}L{level}"
            target_doc_ids = get_evidence_files_by_prefix(prefix, EVIDENCE_DIR)
            mapping_key = f"{sub_criteria_id}_L{level}"
            
            if target_doc_ids:
                evidence_mapping[mapping_key] = {
                    "enabler": target_enabler,
                    "filter_ids": target_doc_ids,
                    "notes": f"Auto-matched files/folders with prefix '{prefix}'.",
                    "statements_count": len(statements)
                }
                mapped_count += 1
                print(f"  -> {mapping_key} matched {len(target_doc_ids)} items (e.g., '{target_doc_ids[0][:20]}...')")
            else:
                print(f"  -> {mapping_key} found 0 items. Requires manual check.")


    # 3. บันทึกผลลัพธ์ลงในไฟล์ JSON (ตอนนี้ตัวแปร evidence_mapping อยู่ใน Scope แล้ว)
    try:
        MAPPING_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with MAPPING_OUTPUT_PATH.open('w', encoding='utf-8') as f:
            json.dump(evidence_mapping, f, indent=2, ensure_ascii=False)
        
        print("-" * 50)
        print(f"✨ Success! Total {mapped_count} mappings created for {target_enabler}.")
        print(f"Output saved to: {MAPPING_OUTPUT_PATH.resolve()}")
        print("-" * 50)
    except Exception as e:
        # หากเกิด Error ในการบันทึก ให้แสดงชื่อตัวแปรที่ใช้งานเพื่อ Debug
        print(f"❌ Failed to save mapping file to {MAPPING_OUTPUT_PATH.resolve()}: {e}")

if __name__ == "__main__":
    TARGET_ENABLER_SELECTION = "KM" 
    create_auto_mapping(TARGET_ENABLER_SELECTION)