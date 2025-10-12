import pandas as pd
import json
import numpy as np
import os

# 1. กำหนดชื่อไฟล์และ Sheet Name (คงตามที่คุณต้องการ)
EXCEL_FILE_PATH = 'data/faq/KM Scoring Guideline-Concept.xlsx' 
# ใช้ Sheet นี้เป็นแหล่งข้อมูลหลักและแหล่งเดียว
MAIN_CRITERIA_SHEET = 'การแบ่งเกณฑ์การให้คะแนน' 

# 2. Level Score Mapping (Fixed Score)
def get_level_score_map():
    """Returns the fixed score for each Maturity Level."""
    return {
        1: 0.16667, 2: 0.33333, 3: 0.50000, 4: 0.66667, 5: 0.83333
    }

# 3. ฟังก์ชันสำหรับอ่านและแปลงเกณฑ์หลัก (*** ปรับปรุง: แยก 3 ข้อย่อยออกจากกัน ***)
def parse_criteria_separated(file_path, sheet_name, score_map):
    """
    อ่าน Sheet การแบ่งเกณฑ์การให้คะแนน และดึง ID, Name, Weight และแยก Level Text 3 แถวออกจากกัน
    """
    print(f"Reading separated criteria details from Excel file: {file_path}, Sheet: {sheet_name}...")
    
    # *** ตรวจสอบ Path ของไฟล์ Excel ***
    if not os.path.exists(file_path):
         print(f"Error: Excel file not found at {file_path}")
         return []
         
    try:
        # ใช้ pd.read_excel ตามโค้ดเดิมของคุณ
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    except Exception as e:
        print(f"Error reading Excel sheet '{sheet_name}': {e}")
        return []

    # Column Index Mapping
    COL_MAP = {
        'SUB_CRITERIA_ID': 0,       # คอลัมน์ A (0)
        'SUB_CRITERIA_NAME': 1,     # คอลัมน์ B (1)
        'WEIGHT': 2,                # คอลัมน์ C (2)
        'L1_TEXT': 3, 'L2_TEXT': 5, 'L3_TEXT': 7, 'L4_TEXT': 9, 'L5_TEXT': 11,
    }

    criteria_list = []
    current_index = 3 # เริ่มอ่านข้อมูลเกณฑ์ย่อยจากแถวที่ 4 (Index 3)
    
    while current_index < len(df):
        row = df.iloc[current_index]
        sub_id_raw = str(row[COL_MAP['SUB_CRITERIA_ID']]).strip()
        
        if sub_id_raw.count('.') == 1 and sub_id_raw.split('.')[0].isdigit():
            
            sub_id = sub_id_raw
            
            # กรองเฉพาะ KM 1.x-6.x (ถ้าคุณไม่ต้องการ IM)
            if int(sub_id.split('.')[0]) > 6: break
            
            # ดึง Weight และ Name
            try:
                weight = float(row[COL_MAP['WEIGHT']]) if pd.notna(row[COL_MAP['WEIGHT']]) else 0.0
            except (ValueError, TypeError):
                weight = 0.0
            sub_name_raw = str(row[COL_MAP['SUB_CRITERIA_NAME']]).strip()
            
            criteria_obj = {
                "Enabler_ID": sub_id.split('.')[0], 
                "Sub_Criteria_ID": sub_id,          
                "Sub_Criteria_Name_TH": sub_name_raw if pd.notna(row[COL_MAP['SUB_CRITERIA_NAME']]) and sub_name_raw else f"Criteria {sub_id}",
                "Weight": weight, 
            }
            
            all_texts_empty = True
            
            # 4. วนลูป 5 Level และ *** แยก 3 ข้อย่อย ***
            for level in range(1, 6):
                col_index = COL_MAP[f'L{level}_TEXT']
                
                criteria_obj[f'Level_{level}_Score'] = score_map.get(level)
                
                # วน 3 แถว (เกณฑ์ย่อย 1, 2, 3) เพื่อแยกเก็บเป็น Field
                for i in range(3):
                    detail_field_name = f'Level_{level}_Detail_{i+1}_Text' # สร้าง Field แยก
                    text_chunk = ""
                    
                    if current_index + i < len(df):
                        row_to_check = df.iloc[current_index + i]
                        text_chunk_raw = str(row_to_check[col_index]).strip()
                        
                        if pd.notna(row_to_check[col_index]) and text_chunk_raw:
                            text_chunk = text_chunk_raw
                            all_texts_empty = False # มีข้อความอย่างน้อย 1 ตัว
                        
                    criteria_obj[detail_field_name] = text_chunk
            
            # 5. *** การกรอง: เพิ่มเฉพาะ Sub-Criteria ที่มี Level Text ไม่ว่าง ***
            if not all_texts_empty:
                criteria_list.append(criteria_obj)
            else:
                 print(f"Skipping incomplete criteria: {sub_id} - {sub_name_raw} (No Level Text found).")
                 
            current_index += 3 # ข้ามไป 3 แถว
        else:
            current_index += 1 # เลื่อนไปแถวถัดไป

    print(f"Successfully parsed {len(criteria_list)} complete Sub-Criteria.")
    return criteria_list

# 4. รันฟังก์ชันและรวมข้อมูล (Main Execution)
if __name__ == "__main__":
    
    score_map = get_level_score_map()

    # 1. นำเข้าข้อมูลเกณฑ์หลักและ "แยก" เกณฑ์ย่อย
    # *** เรียกใช้ฟังก์ชันที่แก้ไขแล้ว และใช้ตัวแปร Excel เดิม ***
    criteria_data_list = parse_criteria_separated(EXCEL_FILE_PATH, MAIN_CRITERIA_SHEET, score_map)

    # 2. บันทึกผลลัพธ์เป็นไฟล์ JSON
    JSON_OUTPUT_PATH = 'km_separated_details_metadata.json' # เปลี่ยนชื่อ Output ให้สอดคล้อง
    if criteria_data_list:
        with open(JSON_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(criteria_data_list, f, ensure_ascii=False, indent=4)
            
        print("\n--- Processing Complete ---")
        print(f"Total COMPLETE Sub-Criteria Parsed and Separated: {len(criteria_data_list)}")
        print(f"Final JSON structure saved to {JSON_OUTPUT_PATH}")
    else:
        print("\nProcess failed or no complete criteria found: Check the Excel file path and sheet name.")