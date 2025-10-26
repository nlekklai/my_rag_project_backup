import os
import json
from typing import Set

# 📍 พาธและชื่อไฟล์ที่ต้องตรวจสอบ
MAPPING_FILE_PATH = "evidence_checklist/km_evidence_mapping.json"

# **✅ FIX: ใช้พาธสัมพัทธ์ (Relative Path)**
VECTOR_STORE_BASE_PATH = "vectorstore/evidence" 

def load_mapping_doc_ids(file_path: str) -> Set[str]:
    """
    โหลดไฟล์ mapping และดึง Doc ID ทั้งหมดจาก 'filter_ids' โดยไม่มีการ Clean ชื่อ
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ ERROR: ไม่พบไฟล์ Mapping ที่พาธ: {os.path.abspath(file_path)}")
        return set()
    except json.JSONDecodeError:
        print(f"❌ ERROR: ไฟล์ Mapping ไม่ใช่ JSON ที่ถูกต้อง: {file_path}")
        return set()

    all_doc_ids = set()
    for key in mapping_data:
        filter_ids = mapping_data[key].get("filter_ids", [])
        if isinstance(filter_ids, list):
            all_doc_ids.update(set(filter_ids)) 
            
    return all_doc_ids

def check_doc_id_existence(doc_ids: Set[str], base_path: str) -> tuple[Set[str], Set[str]]:
    """
    ตรวจสอบว่า Doc ID (ชื่อเต็ม) มีโฟลเดอร์ Vector Store (ชื่อเต็ม) ตรงกันหรือไม่
    """
    missing_ids = set(doc_ids)
    found_ids = set()

    # ไม่ต้องแสดงพาธเต็มอีกแล้ว เนื่องจากใช้ Relative Path
    if not os.path.exists(base_path):
        print(f"❌ ERROR: ไม่พบโฟลเดอร์ Vector Store ที่พาธ: {os.path.abspath(base_path)}")
        return set(), doc_ids 

    # ดึงรายชื่อโฟลเดอร์ที่มีอยู่จริงทั้งหมด (ชื่อเต็ม)
    try:
        existing_folders = set([
            item for item in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, item))
        ])
    except Exception as e:
        print(f"❌ ERROR: ไม่สามารถอ่านโฟลเดอร์ {base_path} ได้: {e}")
        return set(), doc_ids
        
    # เปรียบเทียบ Doc ID (ชื่อเต็มจาก Mapping) กับชื่อโฟลเดอร์ (ชื่อเต็มจาก Vector Store)
    for doc_id in doc_ids:
        if doc_id in existing_folders:
            found_ids.add(doc_id)
            if doc_id in missing_ids:
                missing_ids.remove(doc_id) 

    return found_ids, missing_ids

def main():
    """
    ฟังก์ชันหลักในการรันการตรวจสอบ และแสดงเฉพาะรายการที่ขาดหายไป
    """
    # 1. โหลดและรวบรวม Doc ID
    all_doc_ids_to_check = load_mapping_doc_ids(MAPPING_FILE_PATH)
    total_ids = len(all_doc_ids_to_check)
    
    if total_ids == 0:
        print("ไม่พบ Doc ID ที่ต้องตรวจสอบ หรือเกิดข้อผิดพลาดในการโหลดไฟล์ Mapping")
        return

    # 2. ตรวจสอบการมีอยู่ของ Doc ID ใน Vector Store
    found_ids, missing_ids = check_doc_id_existence(all_doc_ids_to_check, VECTOR_STORE_BASE_PATH)
    
    # 3. แสดงผลลัพธ์
    if missing_ids:
        print("\n## 🔴 Doc ID ที่ **ขาดหายไป** ใน Vector Store (ต้องแก้ไข Mapping หรือ Ingest ใหม่):")
        sorted_missing = sorted(list(missing_ids))
        
        # แสดงผลรวมก่อน
        print(f"รวม Doc ID ที่ต้องตรวจสอบทั้งหมด: **{total_ids}**")
        print(f"จำนวน Doc ID ที่ **ไม่พบ** (Missing): **{len(missing_ids)}**")
        print("-" * 25)

        for doc_id in sorted_missing:
            # แสดงเฉพาะ 20 รายการแรก เพื่อไม่ให้ผลลัพธ์ยาวเกินไป
            if sorted_missing.index(doc_id) < 20: 
                 print(f"- {doc_id}")
            else:
                 print(f"และ Doc ID อื่นๆ ที่ขาดหายไปอีก {len(missing_ids) - 20} รายการ...")
                 break
    else:
        print("\n🎉 ยอดเยี่ยม! Doc ID ทั้งหมดใน Mapping File พบใน Vector Store ครับ.")

if __name__ == "__main__":
    main()