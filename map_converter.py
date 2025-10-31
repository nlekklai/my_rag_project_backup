import re
import os
import json
import sys
from typing import Dict, List, Any

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

# Path Setup: Assumes log file is in the same or defined subdirectory
# ใช้ os.path.join() เพื่อให้รองรับระบบปฏิบัติการต่างๆ ได้ดีกว่า
LOG_FILE_PATH = os.path.join("evidence_checklist", "evidence_uuid.log")
OUTPUT_FILE_PATH = os.path.join("evidence_checklist", "km_evidence_mapping_new.json")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Regular Expression Pattern for Log Parsing:
# 1. (\S+)             -> doc_id (UUID)
# 2. \s*\|\s* -> Separator
# 3. (.*?)             -> full_filename (Non-greedy match, สำคัญมากสำหรับคอลัมน์ชื่อไฟล์)
# 4. \s*\|\s* -> Separator
# 5. (\S+)             -> EXT (Non-whitespace characters)
# 6. .*$               -> Consumes the rest of the line
LINE_PARSE_PATTERN = re.compile(
    r'^\s*(\S+)\s*\|\s*(.*?)\s*\|\s*(\S+).*$'
)

# Regular Expression Pattern for KM Evidence Extraction from filename:
# **แก้ไขแล้ว:**
# 1. KM                     -> Literal match "KM"
# 2. (\d+\.\d+)             -> sub_id (e.g., "1.1", "3.3") -> Group 1
# 3. L(\d)                  -> level (e.g., "1", "5") -> Group 2 (จับแค่ตัวเลขแรก)
# 4. \d* -> อนุญาตให้มีตัวเลขลำดับ (Sequence Number) ตามมาได้อีกกี่ตัวก็ได้
FILENAME_KM_PATTERN = re.compile(r'KM(\d+\.\d+)L(\d)\d*') 


# ----------------------------------------------------------------------
# 2. Core Functions
# ----------------------------------------------------------------------

def _get_log_filepath() -> str:
    """Resolve the log file path relative to the script's location."""
    if not os.path.isabs(LOG_FILE_PATH):
        # Assumes LOG_FILE_PATH is relative to the directory where this script resides
        return os.path.join(BASE_DIR, LOG_FILE_PATH)
    return LOG_FILE_PATH

def _read_and_clean_log(filepath: str) -> List[str]:
    """Reads the log file, removes header/footer lines, and returns clean lines."""
    try:
        # Tries to handle different encodings if the default utf-8 fails
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
             # ลองใช้ cp874 (Thai standard encoding) เป็น fallback
             with open(filepath, 'r', encoding='cp874') as f: 
                lines = f.readlines()

    except FileNotFoundError:
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return []
    
    # 1. Remove Header/Footer/Separator lines
    clean_lines = []
    separator_pattern = re.compile(r'^-+$')
    
    # ตัวแปรสำหรับระบุว่าเราเข้าสู่ส่วนข้อมูลหลักแล้ว
    in_data_section = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # ตรวจสอบบรรทัดคอลัมน์หัวตารางเพื่อเริ่มต้นการเก็บข้อมูล
        if line.startswith('DOC ID (Stable/Temp)'):
             in_data_section = True 
             continue # ข้ามบรรทัดหัวตาราง

        # ตรวจสอบบรรทัดคั่น (ถ้ายังไม่เจอหัวตาราง)
        if separator_pattern.match(line):
            in_data_section = True
            continue # ข้ามบรรทัดคั่น
        
        # เมื่อเข้าสู่ส่วนข้อมูลหลักแล้ว (in_data_section == True)
        if in_data_section:
            # ตรวจสอบว่าบรรทัดนี้มีตัวคั่น '|' หรือไม่ เพื่อกรองบรรทัดข้อมูลจริง
            if '|' in line and not line.startswith('DOC ID'): 
                clean_lines.append(line)
        
    
    return clean_lines

def generate_mapping():
    """Main function to parse the log and generate the JSON mapping file."""
    
    log_filepath = _get_log_filepath()
    print("--- เริ่มต้นการสร้างไฟล์ Mapping จาก Evidence List (ปรับแก้ Regex แล้ว) ---")
    print(f"อ่านไฟล์ต้นฉบับ (ตาราง Log): {log_filepath}")

    log_content_clean = _read_and_clean_log(log_filepath)
    
    if not log_content_clean:
        print("------------------------------------------------------------------")
        print("ERROR: ไม่พบข้อมูลที่สะอาดในไฟล์ Log (ไฟล์อาจว่างเปล่าหรือรูปแบบผิดพลาด)")
        print("------------------------------------------------------------------")
        return

    # Dictionary to store the final grouped mapping
    mapping_results: Dict[str, Any] = {}
    
    parsed_count = 0
    km_evidence_count = 0
    
    # ------------------------------------------------------
    # 3. Parsing and Grouping Logic
    # ------------------------------------------------------
    for line in log_content_clean:
        # 1. Attempt to parse the main log line (Doc ID, Filename, etc.)
        match_log = LINE_PARSE_PATTERN.match(line)
        
        if not match_log:
            # ไม่แสดง DEBUG หากบรรทัดที่ Parse ไม่สำเร็จมีจำนวนมาก
            if parsed_count % 100 == 0 and parsed_count > 0:
                 print(f"⚠️ DEBUG: บรรทัดที่ {parsed_count+1} Parse ไม่สำเร็จ (รูปแบบไม่ตรง): {line[:80]}...")
            parsed_count += 1
            continue
        
        # The new pattern captures 3 groups: (1) UUID, (2) FILENAME, (3) EXT
        captures = match_log.groups()
        
        # เราใช้แค่ Doc ID และ Full Filename
        doc_id = captures[0].strip()
        full_filename = captures[1].strip()
        
        # 2. Attempt to extract KM, Sub-ID, and Level from the FILENAME
        match_km = FILENAME_KM_PATTERN.search(full_filename)
        
        if match_km:
            sub_id_raw = match_km.group(1) # e.g., "1.1"
            level_raw = match_km.group(2)  # e.g., "1" หรือ "5" (แก้ไขแล้ว)
            
            # Combine to form the unique mapping key (e.g., "1.1_L5")
            mapping_key = f"{sub_id_raw}_L{level_raw}"
            
            # Create the evidence dictionary
            evidence_data = {
                "doc_id": doc_id,
                "file_name": full_filename,
                "notes": "Generated from FILENAME prefix. Please review and refine."
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
        
        parsed_count += 1

    # ------------------------------------------------------
    # 4. Output Generation
    # ------------------------------------------------------
    if not mapping_results:
        print("------------------------------------------------------------------")
        print("⚠️ WARNING: ไม่พบ KM Evidence ที่มีรูปแบบ KMX.XLX ในชื่อไฟล์เลย (อาจจะต้องปรับ FILENAME_KM_PATTERN)")
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
            print(f"สรุป: ประมวลผลไป {parsed_count} บรรทัด | พบ KM Evidence ที่จับคู่ได้: {km_evidence_count} รายการ")
            print("------------------------------------------------------------------")

        except Exception as e:
            print(f"ERROR: ไม่สามารถบันทึกไฟล์ JSON ได้: {e}", file=sys.stderr)


if __name__ == "__main__":
    # ----------------------------------------------------------
    # 5. Execution
    # ----------------------------------------------------------
    generate_mapping()
