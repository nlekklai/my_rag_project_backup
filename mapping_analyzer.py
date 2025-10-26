import json
from collections import defaultdict
import os
import re

# กำหนดชื่อไฟล์และโฟลเดอร์สำหรับ Export
OUTPUT_FILE_PATH = "evidence_checklist/km_mapping_by_level.json"
DEFAULT_STATEMENTS_COUNT = 3
MAPPING_THRESHOLD = 0.9500 # ใช้ค่า default ตามที่คุณเคยกำหนด

# ----------------------------------------------------------------------
# ฟังก์ชัน Utility สำหรับเตรียมข้อมูล
# ----------------------------------------------------------------------

def get_statement_level_key(statement_key: str) -> str:
    """แปลง statement_key (e.g., '1.1_L4_2') เป็น level key (e.g., '1.1_L4')"""
    parts = statement_key.split('_')
    # ใช้แค่ส่วน Subtopic และ Level
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return ""

def get_prefix_for_note(level_key: str) -> str:
    """สร้าง prefix สำหรับ field 'notes' (e.g., '1.1_L1' -> 'KM1.1L1')"""
    # แทนที่ underscore ด้วยไม่มีอะไรเพื่อให้ได้รูปแบบ KM1.1L1
    return f"KM{level_key}".replace('_', '')

def clean_doc_id(file_name: str) -> str:
    """
    ตัดส่วนขยายของไฟล์ที่อยู่ท้ายสุดออกเพียงชุดเดียวเท่านั้น (.pdf.pdf -> .pdf, .pdf -> ไม่มี .pdf)
    """
    if file_name.count('.') == 0:
        return file_name

    # แยกชื่อไฟล์ออกเป็น 2 ส่วนที่จุดสุดท้าย (rsplit(separator, maxsplit))
    parts = file_name.rsplit('.', 1)
    
    # ตรวจสอบว่าส่วนที่ตัดออกไปเป็นส่วนขยายหรือไม่ (มีตัวอักษรอย่างน้อย 1 ตัว และยาวไม่เกิน 5 ตัวอักษร)
    # เราตัดออกโดยมีเงื่อนไขสมมติว่า extension ยาวไม่เกิน 5 ตัวอักษร เพื่อไม่ให้ตัด Subtopic ID หรือตัวเลขยาวๆ ผิด
    # ตัวอย่าง: .pdf, .docx, .jpeg, .xlsx
    if len(parts) == 2 and 1 <= len(parts[1]) <= 5 and parts[1].isalnum():
        return parts[0]
        
    return file_name # คืนชื่อเดิมถ้าไม่ตรงตามเงื่อนไข (เช่น ไม่มี extension, หรือ extension แปลกๆ)


# ----------------------------------------------------------------------
# ฟังก์ชันหลักสำหรับการสร้าง Summary (CLI Display) - เก็บไว้ตามโครงสร้างเดิม
# ----------------------------------------------------------------------

def generate_mapping_summary(json_file_path: str, threshold: float = MAPPING_THRESHOLD):
    """
    อ่านไฟล์ JSON ผลลัพธ์ Evidence Mapping และสรุปผลโดยจัดกลุ่มตาม Statement Level 
    พร้อมแสดงรายชื่อไฟล์ Evidence ที่รองรับ (CLI Display)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: ไม่พบไฟล์ที่ระบุ: {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: รูปแบบไฟล์ JSON ไม่ถูกต้อง: {json_file_path}")
        return None

    # 1. รวบรวมข้อมูล: จาก Evidence File-Centric เป็น Statement-Centric
    statement_evidence_map = defaultdict(lambda: {
        'statement_text': '',
        'subtopic': '',
        'level': '',
        'stmt_num': '',
        'evidence_files': set()
    })

    for evidence_file, mappings in mapping_data.items():
        if isinstance(mappings, dict) and 'error' in mappings:
            continue
        
        # ทำความสะอาด Doc ID ก่อนนำไปเก็บ
        cleaned_evidence_id = clean_doc_id(evidence_file) 
        
        for mapping in mappings:
            if mapping['score'] >= threshold:
                key = mapping['statement_key']
                # ต้องตรวจสอบก่อนว่า split ได้ 3 ส่วนหรือไม่เพื่อป้องกัน Error
                try:
                    subtopic, level_num, stmt_num = key.split('_')
                except ValueError:
                    continue # ข้าม key ที่ format ไม่ถูกต้อง
                
                statement_evidence_map[key]['statement_text'] = mapping['statement_text']
                statement_evidence_map[key]['subtopic'] = subtopic
                statement_evidence_map[key]['level'] = level_num
                statement_evidence_map[key]['stmt_num'] = stmt_num
                # ใช้ cleaned_evidence_id ในการเก็บ
                statement_evidence_map[key]['evidence_files'].add(cleaned_evidence_id) 

    # 2. จัดกลุ่มและเตรียมข้อมูลสำหรับแสดงผล
    summary_list = []
    for key, data in statement_evidence_map.items():
        summary_list.append({
            'key': key,
            'Subtopic': data['subtopic'],
            'Level': data['level'],
            'Statement #': data['stmt_num'],
            'Statement Text': data['statement_text'],
            'Evidence Files Count': len(data['evidence_files']),
            'Evidence Files': ', '.join(sorted(list(data['evidence_files'])))
        })

    # 3. จัดเรียงข้อมูลตาม Subtopic, Level, และ Statement #
    summary_list.sort(key=lambda x: (x['Subtopic'], x['Level'], x['Statement #']))

    # 4. แสดงผลในรูปแบบตาราง (CLI friendly)
    print("\n" + "="*120)
    print(f"สรุป Evidence Mapping (Threshold >= {threshold})")
    print("="*120)

    col_width = [10, 8, 12, 50, 45]
    
    # พิมพ์ Header
    print(f"{'Subtopic':<{col_width[0]}} {'Level':<{col_width[1]}} {'Stmt. #':<{col_width[2]}} {'Statement Text':<{col_width[3]}} {'Evidence Files'}")
    print("-" * 120)

    for item in summary_list:
        subtopic_display = item['Subtopic']
        level_display = item['Level']
        stmt_num_display = item['Statement #']
        # ตัดข้อความ Statement Text เพื่อให้แสดงผลได้สวยงามใน CLI
        text_display = item['Statement Text'][:col_width[3]-3] + "..." if len(item['Statement Text']) > col_width[3] else item['Statement Text']
        
        evidence_files = item['Evidence Files']
        
        # ส่วนนี้เป็นการแสดงผลแบบพื้นฐาน อาจไม่ได้จัดตารางสวยงามเป๊ะสำหรับไฟล์ยาวมากๆ แต่เน้นให้ข้อมูลครบ
        print(f"{subtopic_display:<{col_width[0]}} {level_display:<{col_width[1]}} {stmt_num_display:<{col_width[2]}} {text_display:<{col_width[3]}} {evidence_files}")
    
    print("="*120)
    print(f"สรุป: พบ {len(summary_list)} Statements ที่มีการ Mapping")
    print("หมายเหตุ: ไฟล์ .jpg ถูกข้ามการประมวลผล (หากมี)")
    
    # ส่งข้อมูล mapping_data กลับไปใช้ในฟังก์ชัน Export
    return mapping_data 
    
    
# ----------------------------------------------------------------------
# ฟังก์ชันใหม่สำหรับการ Export JSON (ตาม format ที่ต้องการ)
# ----------------------------------------------------------------------

def export_to_level_json(
    mapping_data: dict, 
    output_json_path: str = OUTPUT_FILE_PATH, 
    threshold: float = MAPPING_THRESHOLD,
    default_statements_count: int = DEFAULT_STATEMENTS_COUNT
):
    """
    ประมวลผลข้อมูล mapping_data และแปลงให้เป็นรูปแบบจัดกลุ่มตาม Statement Level 
    พร้อมบันทึกเป็นไฟล์ JSON ใหม่ในโฟลเดอร์ results/
    """
    if not mapping_data:
        print("ไม่สามารถ Export JSON ได้ เนื่องจากไม่มีข้อมูล Mapping ที่จะประมวลผล")
        return

    print(f"\nเริ่มการสร้างไฟล์ JSON สำหรับ Export ไปที่ {output_json_path}...")

    # 1. รวบรวม Evidence File ตาม Statement Level (Subtopic_Level)
    statement_level_map = defaultdict(set) 

    for evidence_file, mappings in mapping_data.items():
        if isinstance(mappings, dict) and 'error' in mappings:
            continue
        
        # ทำความสะอาด Doc ID ก่อนนำไปเก็บ
        cleaned_evidence_id = clean_doc_id(evidence_file) 
        
        for mapping in mappings:
            if mapping['score'] >= threshold:
                statement_key = mapping['statement_key']
                level_key = get_statement_level_key(statement_key)
                    
                if level_key:
                    # ใช้ cleaned_evidence_id ในการเก็บ
                    statement_level_map[level_key].add(cleaned_evidence_id) 

    # 2. จัดรูปแบบข้อมูลให้เป็นตามที่ผู้ใช้ต้องการ
    final_output = {}
    sorted_level_keys = sorted(statement_level_map.keys())

    for level_key in sorted_level_keys:
        evidence_files_set = statement_level_map[level_key]
        sorted_filter_ids = sorted(list(evidence_files_set))
        prefix = get_prefix_for_note(level_key)
        
        final_output[level_key] = {
            "enabler": "KM",
            "filter_ids": sorted_filter_ids,
            "notes": f"Auto-matched files/folders with prefix '{prefix}'.",
            "statements_count": default_statements_count 
        }

    # 3. บันทึกไฟล์ JSON ใหม่
    try:
        # ตรวจสอบและสร้าง directory ถ้าไม่มี
        output_dir = os.path.dirname(output_json_path)
        # ตรวจสอบว่า output_dir ไม่ใช่ empty string (เช่นกรณีที่ output_json_path ไม่มี '/')
        if output_dir and not os.path.exists(output_dir): 
            os.makedirs(output_dir)
            
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ **Export JSON สำเร็จ!** บันทึกไฟล์ผลลัพธ์ใหม่เรียบร้อยแล้วที่: **{output_json_path}**")
        
    except Exception as e:
        print(f"❌ Error ในการบันทึกไฟล์: {e}")


# ----------------------------------------------------------------------
# การเรียกใช้ฟังก์ชันหลัก
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Path ไปยังไฟล์ Input ที่ต้องการวิเคราะห์
    INPUT_FILE_PATH = "results/km_evidence_mapping_09500_full_run_no_jpg.json" 
    
    # 1. รันการวิเคราะห์และแสดงผลใน CLI
    mapping_data = generate_mapping_summary(INPUT_FILE_PATH, MAPPING_THRESHOLD)
    
    # 2. ถ้ามีข้อมูล (ไม่เกิด Error) ให้ทำการ Export JSON
    if mapping_data:
        export_to_level_json(
            mapping_data=mapping_data, 
            output_json_path=OUTPUT_FILE_PATH,
            threshold=MAPPING_THRESHOLD
        )