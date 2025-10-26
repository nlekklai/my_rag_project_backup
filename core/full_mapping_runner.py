# 🚀 Full Code: core/full_mapping_runner.py (ปรับปรุง Import และ Path)

import os
import json
import logging
import argparse
import datetime
from typing import Dict, List, Any, Set

# ----------------------------------------------------------------------
# [IMPORT FIX] - พยายามใช้ Relative Import ก่อน และ Fallback ไปหา Absolute Import
# ----------------------------------------------------------------------
try:
    # 1. ลองใช้ Relative Import (ทำงานเมื่อรันด้วย python -m)
    from .evidence_mapping_generator import EvidenceMappingGenerator
except ImportError:
    # 2. หากล้มเหลว (มักจะเกิดเมื่อรันด้วย python core/full_mapping_runner.py)
    # เราจะลองรันแบบ Absolute Import โดยคาดหวังว่า core/ ถูกเพิ่มใน PYTHONPATH หรือถูกรู้จัก
    try:
        from core.evidence_mapping_generator import EvidenceMappingGenerator
    except ImportError:
        # 3. หากยัง Import ไม่ได้ ให้แจ้ง Error ชัดเจนและออกจากโปรแกรม
        print("❌ Error: ไม่พบ evidence_mapping_generator. โปรดตรวจสอบโครงสร้าง Project และ core/__init__.py")
        print("💡 TIP: ลองรันด้วยคำสั่ง 'python -m core.full_mapping_runner' แทน")
        exit(1) # ออกจากโปรแกรมเพื่อไม่ให้เกิด NameError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ----------------------------------------------------------------------
# [CONFIGURATION & PATHS]
# ----------------------------------------------------------------------
# กำหนด PROJECT_ROOT ให้ย้อนกลับไป 1 ระดับจากโฟลเดอร์ core/ 
# ใช้ os.path.abspath(os.path.dirname(__file__)) เพื่อให้มั่นใจว่า path ถูกต้อง
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

DEFAULT_EVIDENCE_DIR = "data/evidence"
INPUT_MASTER_FILE_FORMAT = "{enabler}_evidence_mapping.json" 
OUTPUT_MAPPING_FOLDER = "results"
OUTPUT_MAPPING_FORMAT = "{enabler}_mapping_output_{suffix}.json" 
OUTPUT_MAPPING_SUFFIX = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_THRESHOLD = 0.5 
TOP_K_STATEMENTS = 3 

# ----------------------------------------------------------------------
# [Utility Functions]
# ----------------------------------------------------------------------

def load_master_list(enabler_id: str, file_path: str) -> List[str]:
    """
    โหลดรายการไฟล์หลักฐานทั้งหมดจาก Input Master JSON เพื่อกำหนด List ไฟล์
    ที่จะถูกนำเข้ากระบวนการ RAG Mapping
    """
    
    file_name_to_use = file_path
    if file_path == "": 
         # หากไม่ได้ระบุ path มา จะใช้ชื่อ default และคาดหวังว่าไฟล์จะอยู่ใน PROJECT_ROOT
         file_name_to_use = INPUT_MASTER_FILE_FORMAT.format(enabler=enabler_id.lower())
         full_path = os.path.join(PROJECT_ROOT, file_name_to_use)
    else:
         # หากมีการระบุ path มา (ซึ่งควรเป็น path สัมพัทธ์จาก Project Root)
         full_path = os.path.join(PROJECT_ROOT, file_path)
    
    if not os.path.exists(full_path):
        # บรรทัดนี้คือที่แจ้ง ERROR
        logger.error(f"❌ Input Master File not found at {full_path}") 
        return [] 
        
    all_evidence_files = set()
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for key_id, info in data.items():
            files = info.get("filter_ids", []) 
            all_evidence_files.update(files)
            
        logger.info(f"✅ โหลด {len(data)} SubCriteria_Levels และพบไฟล์หลักฐานที่ไม่ซ้ำกัน {len(all_evidence_files)} ไฟล์จาก {full_path}")
        return sorted(list(all_evidence_files))
        
    except Exception as e:
        logger.error(f"❌ เกิดข้อผิดพลาดในการโหลดหรือแยกวิเคราะห์ไฟล์ Master List: {e}")
        return []


def generate_full_mapping(enabler_id: str, 
                          evidence_dir: str, 
                          master_list_path: str,
                          threshold: float, 
                          top_k: int) -> Dict[str, Any]:
    """
    กระบวนการหลักในการวนลูปผ่านไฟล์หลักฐานและเรียกใช้ RAG Mapping
    """
    
    # 1. โหลด Master List 
    all_evidence_files = load_master_list(enabler_id, master_list_path)
    if not all_evidence_files:
        return {}

    # 2. เตรียม Generator 
    try:
        # NOTE: เมื่อรันมาถึงตรงนี้ EvidenceMappingGenerator จะต้องถูก Import สำเร็จแล้ว
        generator = EvidenceMappingGenerator(enabler_id=enabler_id) 
    except FileNotFoundError as e:
         logger.critical(f"❌ ข้อผิดพลาดร้ายแรง: Statement checklist โหลดไม่สำเร็จ: {e}")
         return {}
    except Exception as e:
        logger.critical(f"❌ ข้อผิดพลาดร้ายแรงระหว่างการเริ่มต้น Generator: {e}")
        return {}
        
    # 3. เตรียมโครงสร้าง Output ในรูปแบบ SubCriteria_Level (1.1_L1)
    
    # 3.1 สร้าง mapping จาก statement_key (1.1_L1_1) ไป SubCriteria_Level (1.1_L1)
    statement_to_level_map: Dict[str, str] = {}
    
    # 3.2 สร้างโครงสร้างผลลัพธ์ SubCriteria_Level และนับจำนวน Statement ในแต่ละ Level
    output_mapping_result: Dict[str, Any] = {}
    
    for block in generator.statement_data: 
        sub_criteria_id = block.get("Sub_Criteria_ID") # e.g., '1.1'
        for i in range(1, 6):
            level_key = f"Level_{i}_Statements"
            statements_list = block.get(level_key, [])
            sub_level_key = f"{sub_criteria_id}_L{i}" # e.g., '1.1_L1'
            
            # เตรียมโครงสร้าง Output
            if sub_level_key not in output_mapping_result:
                 output_mapping_result[sub_level_key] = {
                    "enabler": enabler_id.upper(),
                    "filter_ids": set(), 
                    "notes": f"Auto-matched files/folders with prefix '{sub_level_key}'.",
                    "statements_count": len(statements_list)
                 }

            # สร้าง mapping จาก statement_key ไป SubCriteria_Level
            for j in range(len(statements_list)):
                 statement_key = f"{sub_criteria_id}_L{i}_{j + 1}"
                 statement_to_level_map[statement_key] = sub_level_key
                 
    logger.info(f"💾 เตรียมโครงสร้าง Output สำหรับ {len(output_mapping_result)} SubCriteria_Levels")
    
    # 4. วนลูปและ Process ไฟล์หลักฐานทีละไฟล์
    processed_count = 0
    # ใช้ os.path.join เพื่อรวม PROJECT_ROOT กับ evidence_dir (ซึ่งเป็น data/evidence)
    absolute_evidence_dir = os.path.join(PROJECT_ROOT, evidence_dir) 
    
    for doc_id in all_evidence_files:
        # รวม absolute_evidence_dir กับ doc_id (ชื่อไฟล์)
        file_path = os.path.join(absolute_evidence_dir, doc_id) 

        if not os.path.exists(file_path):
             logger.warning(f"⚠️ ไม่พบไฟล์ (ข้ามไป): {doc_id} ที่ {file_path}")
             continue
        
        logger.info(f"กำลังประมวลผล ({processed_count + 1}/{len(all_evidence_files)}): {doc_id}")
        
        # 4.1. รัน RAG Mapping
        # NOTE: generator.process_and_suggest_mapping คาดหวัง path เต็มของไฟล์
        suggested_mappings = generator.process_and_suggest_mapping(
            file_path=file_path,
            doc_id=doc_id,
            top_k_statements=top_k,
            similarity_threshold=threshold
        )
        
        # 4.2. อัปเดตโครงสร้างผลลัพธ์ (กรุ๊ปตาม SubCriteria_Level)
        if suggested_mappings:
            mapped_levels: Set[str] = set() 
            for mapping in suggested_mappings:
                statement_key = mapping["statement_key"]
                
                if statement_key in statement_to_level_map:
                    sub_level_key = statement_to_level_map[statement_key]
                    
                    if sub_level_key in output_mapping_result and sub_level_key not in mapped_levels:
                        output_mapping_result[sub_level_key]["filter_ids"].add(doc_id)
                        mapped_levels.add(sub_level_key)
                        
        processed_count += 1
        
    # 5. แปลง Set ของ filter_ids เป็น List และจัดเรียง
    final_output: Dict[str, Any] = {}
    for sub_level_key, data in output_mapping_result.items():
        data["filter_ids"] = sorted(list(data["filter_ids"]))
        final_output[sub_level_key] = data
        
    return final_output

# ----------------------------------------------------------------------
# [Main Function]
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the full evidence-to-statement mapping process for all evidence files.")
    parser.add_argument("--enabler", type=str, required=True, help="Enabler ID (e.g., KM)")
    parser.add_argument(
        "--evidence_dir", 
        type=str, 
        default=DEFAULT_EVIDENCE_DIR, 
        help="Path to the directory containing all evidence PDF files (Relative to Project Root). Default: data/evidence"
    ) 
    parser.add_argument(
        "--master_list", 
        type=str, 
        default="", 
        help=f"Path to the master JSON list of evidence files (Relative to Project Root). Default: {{enabler}}_evidence_mapping.json (at Project Root)"
    )
    parser.add_argument("--output_file", type=str, default="", help="Output file path for the final JSON mapping (Relative to Project Root). Default: results/{enabler}_mapping_output_<timestamp>.json")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help=f"Minimum relevance score threshold for suggestion (Default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--top_k", type=int, default=TOP_K_STATEMENTS, help=f"Maximum number of top statements to map per evidence file (Default: {TOP_K_STATEMENTS})")
    
    args = parser.parse_args()

    # 1. จัดการชื่อ Output File แบบ Dynamic
    output_file_name = args.output_file
    if args.output_file == "":
        # สร้างชื่อ Output File ใน results/ folder
        output_file_name = os.path.join(
            OUTPUT_MAPPING_FOLDER,
            OUTPUT_MAPPING_FORMAT.format(
                enabler=args.enabler.lower(),
                suffix=OUTPUT_MAPPING_SUFFIX
            )
        )
        logger.info(f"ใช้ชื่อไฟล์ผลลัพธ์แบบ dynamic: {output_file_name}")

    logger.info(f"--- 🚀 เริ่มต้น Full Evidence Mapping สำหรับ Enabler: {args.enabler.upper()} ---")
    logger.warning("⚠️ โครงสร้าง Output ถูกปรับเป็นรูปแบบ SubCriteria_Level (e.g., 1.1_L1) เพื่อให้เข้ากับ run_assessmnent.py ของคุณ")

    final_mapping_result = generate_full_mapping(
        enabler_id=args.enabler,
        evidence_dir=args.evidence_dir,
        master_list_path=args.master_list,
        threshold=args.threshold,
        top_k=args.top_k
    )

    if final_mapping_result:
        try:
            full_output_path = os.path.join(PROJECT_ROOT, output_file_name)
            # ตรวจสอบและสร้าง Folder results/ หากยังไม่มี
            os.makedirs(os.path.dirname(full_output_path), exist_ok=True) 
            
            with open(full_output_path, 'w', encoding='utf-8') as f:
                json.dump(final_mapping_result, f, ensure_ascii=False, indent=4) 
            logger.info(f"\n✅ สำเร็จ! บันทึกผลลัพธ์ Mapping ฉบับเต็มไปยัง {full_output_path}")
            logger.info(f"ประมวลผล SubCriteria Levels ทั้งหมด: {len(final_mapping_result)}")
        except Exception as e:
            logger.error(f"❌ บันทึกไฟล์ผลลัพธ์ไม่สำเร็จไปยัง {full_output_path}: {e}")
    else:
        logger.error("❌ กระบวนการล้มเหลว ไม่มีการสร้างผลลัพธ์ Mapping")

if __name__ == "__main__":
    main()