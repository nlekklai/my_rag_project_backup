#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ====================================================================
# 🛠️ WARNING SUPPRESSION BLOCK (ถูกย้ายมาที่นี่เพื่อให้มีผลก่อน Import อื่นๆ)
# ====================================================================
import warnings
import os

# 1. ปิด FutureWarning (จัดการ TRANSFORMERS_CACHE)
warnings.filterwarnings("ignore", category=FutureWarning) 

# 2. ปิด DeprecationWarning (จัดการ LangChainDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# 3. ปิด UserWarning ทั่วไป (จัดการ 'No languages specified...' และ pypdf warnings)
warnings.filterwarnings("ignore", category=UserWarning)

# 4. ปิด RuntimeWarning (เผื่อกรณีมีข้อผิดพลาดที่ไม่คาดคิดในการรัน)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# 5. ปิด Hugging Face Tokenizer Parallelism Warning
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# ====================================================================

import json
import argparse
import sys
from typing import List, Dict, Any, Optional
# ต้องแน่ใจว่า import ถูกต้องตามโครงสร้างโปรเจกต์ของคุณ
from core.ingest import load_and_chunk_document 
from core.vectorstore import load_vectorstore, FINAL_K_RERANKED 
from langchain.schema import Document as LcDocument

import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EvidenceMappingGenerator:
    """
    Class สำหรับสร้าง Suggested Mappings ระหว่าง Evidence File
    กับ Statement ใน Vector Store โดยใช้ Logic การ Augment Query แบบ Dynamic
    """
    def __init__(self, enabler_id: str):
        self.enabler_id = enabler_id.lower()
        # ตรวจสอบ Path ของ Statement Checklist
        self.STATEMENTS_JSON_PATH = f"evidence_checklist/{self.enabler_id}_evidence_statements_checklist.json"

        # โหลด Retriever สำหรับ Statement Vector Store
        self.statement_retriever = load_vectorstore(
            doc_id=f"{self.enabler_id}_statements",
            doc_types="statement"
        )
        self.statement_data = self._load_statements_data()

    def _load_statements_data(self):
        if not os.path.exists(self.STATEMENTS_JSON_PATH):
            raise FileNotFoundError(f"❌ Statement checklist not found at {self.STATEMENTS_JSON_PATH}")
        with open(self.STATEMENTS_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"✅ Loaded statements data from {os.path.basename(self.STATEMENTS_JSON_PATH)}.")
            return data

    def _extract_sub_criteria_id(self, doc_id: str) -> Optional[str]:
        # Logic นี้ถูกคงไว้ แต่ถูก Bypass ใน _get_dynamic_augmentation
        match = re.search(r'(\d\.\d)L?', doc_id, re.IGNORECASE)
        if match:
             return match.group(1)
        match_alt = re.search(r'(\d\.\d)_L', doc_id, re.IGNORECASE)
        return match_alt.group(1) if match_alt else None

    # 🌟 HELPER METHOD: สำหรับสร้างข้อจำกัดให้กับ Query ตาม Level
    def _get_level_constraint_prompt(self, level: int) -> str:
        """
        สร้าง Prompt Constraint เพื่อบอก LLM/Vector Search ให้กรองหลักฐาน L3/L4/L5 ออก
        เพื่อให้ RAG ดึงหลักฐานที่เหมาะสมกับระดับวุฒิภาวะ
        """
        # หลักการ: ห้ามดึงหลักฐานที่มีระดับสูงกว่าระดับที่กำลังประเมิน
        if level == 1:
            # L1: ห้ามดึงหลักฐาน L3-L5 (การบูรณาการ, นวัตกรรม, การวัดผลระยะยาว)
            return "ข้อจำกัด: หลักฐานต้องเกี่ยวกับ 'การเริ่มต้น', 'การมีอยู่', หรือ 'การวางแผน' เท่านั้น ห้ามเกี่ยวข้องกับ 'การปรับปรุงอย่างต่อเนื่อง', 'การบูรณาการ', 'นวัตกรรม', หรือ 'การวัดผลลัพธ์ระยะยาว' (L1-Filter)"
        elif level == 2:
            # L2: อนุญาต L1/L2 เท่านั้น ห้ามดึงหลักฐาน L4-L5
            return "ข้อจำกัด: หลักฐานต้องเกี่ยวกับ 'การปฏิบัติ', 'การทำให้เป็นมาตรฐาน', หรือ 'การประเมินเบื้องต้น' ห้ามเกี่ยวข้องกับ 'การบูรณาการ', 'นวัตกรรม', หรือ 'การวัดผลลัพธ์ระยะยาว' (L2-Filter)"
        elif level == 3:
            # L3: มุ่งเน้นไปที่การควบคุมและการวัดผลระยะสั้น ห้าม L5
            return "ข้อจำกัด: หลักฐานควรเกี่ยวกับ 'การควบคุม', 'การกำกับดูแล', หรือ 'การวัดผลเบื้องต้น' ห้ามเกี่ยวข้องกับ 'นวัตกรรม', หรือ 'การสร้างคุณค่าทางธุรกิจระยะยาว' (L3-Filter)"
        elif level == 4:
            # L4: อนุญาตทุกอย่างยกเว้น L5 (เน้นการบูรณาการและการปรับปรุง)
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'การบูรณาการ' หรือ 'การปรับปรุงอย่างต่อเนื่อง' ห้ามเกี่ยวข้องกับการพิสูจน์ 'คุณค่าทางธุรกิจระยะยาว' (L4-Filter)"
        elif level == 5:
            # L5: ไม่จำกัด แต่เน้นเฉพาะคำสำคัญของ L5
            return "ข้อจำกัด: หลักฐานควรแสดงถึง 'นวัตกรรม', 'การสร้างคุณค่าทางธุรกิจ', หรือ 'ผลลัพธ์ระยะยาว' โดยชัดเจน (L5-Focus)"
        else:
            # Default หรือ Level ไม่ถูกต้อง ให้ใช้คำค้นหาสากล
            return "กรุณาหาหลักฐานที่สอดคล้องกับเกณฑ์ที่ต้องการ"
        
    def _get_dynamic_augmentation(self, doc_id: str, base_query_content: str) -> str:
        """
        🛠️ 2. ปรับปรุง: ใช้ Keywords ที่เฉพาะเจาะจงตาม Enabler ID (รองรับ 10 Enabler)
        """
        
        # 🟢 กำหนด Keywords สำหรับแต่ละ Enabler
        ENABLER_KEYWORDS = {
            "km": [
                "การจัดการความรู้", "Knowledge Management", "นโยบาย KM", 
                "แผนแม่บท KM", "การแลกเปลี่ยนเรียนรู้", "การเก็บความรู้"
            ],
            "hcm": [
                "การบริหารทุนมนุษย์", "Human Capital", "แผนกำลังคน", 
                "การสรรหา", "การพัฒนาบุคลากร", "Competency", "การประเมินผล"
            ],
            "sp": [
                "การวางแผนกลยุทธ์", "Strategy Planning", "วิสัยทัศน์", 
                "พันธกิจ", "เป้าประสงค์องค์กร", "KPI", "แผนปฏิบัติการ"
            ],
            "dt": [
                "เทคโนโลยีดิจิทัล", "Digital Transformation", "IT Governance", 
                "Cyber Security", "IT Roadmap", "ระบบ ERP", "เทคโนโลยีสารสนเทศ"
            ],
            "cg": [
                "บรรษัทภิบาล", "Corporate Governance", "จริยธรรม", 
                "การกำกับดูแลกิจการ", "ความโปร่งใส", "คณะกรรมการ"
            ],
            "l": [
                "กฎหมายและกฎระเบียบ", "Legal & Regulatory", "การปฏิบัติตามกฎหมาย", 
                "ข้อบังคับ", "สัญญา", "กฎหมายดิจิทัล"
            ],
            "rm&ic": [ # คงไว้เป็นกลุ่ม RM&IC ตามการกำหนดเดิม
                "บริหารความเสี่ยง", "Risk Management", "การควบคุมภายใน", 
                "Internal Control", "แผนบริหารความต่อเนื่อง", "ความเสี่ยงองค์กร"
            ],
            "scm": [
                "การจัดการห่วงโซ่อุปทาน", "Supply Chain", "การจัดซื้อจัดจ้าง", 
                "บริหารคลัง", "ผู้ขาย", "การส่งมอบสินค้า"
            ],
            "im": [
                "นวัตกรรม", "Innovation Management", "การวิจัยและพัฒนา", 
                "R&D", "ทรัพย์สินทางปัญญา", "การสร้างมูลค่า"
            ],
            "ia": [
                "การตรวจสอบภายใน", "Internal Audit", "การสอบทาน", 
                "ผลการตรวจสอบ", "รายงานการตรวจสอบ", "การประเมินความเพียงพอ"
            ],
        }
        
        enabler_key = self.enabler_id.lower()
        
        if enabler_key not in ENABLER_KEYWORDS:
             logger.warning(f"⚠️ Warning: Enabler '{enabler_key.upper()}' not defined, using KM as default.")
             keywords = ENABLER_KEYWORDS.get("km")
        else:
             keywords = ENABLER_KEYWORDS[enabler_key]
        
        logger.warning(f"⚠️ Warning: Using {enabler_key.upper()}-specific keywords for augmentation.")
        
        # 🟢 สร้าง String สำหรับ Augmentation
        return (
            f"หลักฐานที่เกี่ยวข้องโดยตรงกับการดำเนินงาน '{self.enabler_id.upper()}' "
            f"หรือ '{self.enabler_id.upper()}' ({', '.join(keywords)})"
        )
        
    def _get_statement_detail(self, content: str) -> Optional[Dict[str, str]]:
        # โค้ดส่วนนี้ยังคงใช้ statement_data ที่โหลดมาจาก JSON เพื่อดึงรายละเอียด
        for enabler_block in self.statement_data:
            sub_criteria_id = enabler_block.get("Sub_Criteria_ID")
            for i in range(1, 6):
                level_key = f"Level_{i}_Statements"
                statements_list = enabler_block.get(level_key, [])
                for j, statement_text in enumerate(statements_list):
                    clean_content = content.strip().lower()
                    clean_statement = statement_text.strip().lower()
                    
                    # 1. ตรวจสอบแบบมีเลขนำหน้า
                    if clean_statement and clean_statement[0].isdigit():
                        clean_statement_no_num = clean_statement.split(maxsplit=1)[-1].strip()
                        if clean_content == clean_statement_no_num.lower():
                            return {
                                "statement_key": f"{sub_criteria_id}_L{i}_{j + 1}",
                                "sub_level": f"{sub_criteria_id} Level {i} Statement {j + 1}",
                                "statement_text": statement_text
                            }
                            
                    # 2. ตรวจสอบแบบไม่มีเลขนำหน้า
                    if clean_content == clean_statement:
                        return {
                            "statement_key": f"{sub_criteria_id}_L{i}_{j + 1}",
                            "sub_level": f"{sub_criteria_id} Level {i} Statement {j + 1}",
                            "statement_text": statement_text
                        }
        return None

    def process_and_suggest_mapping(self,
                                    file_path: str,
                                    doc_id: Optional[str] = None,
                                    level: Optional[int] = None, # 👈 รับค่า Level เข้ามา
                                    top_k_statements: int = FINAL_K_RERANKED,
                                    similarity_threshold: float = 0.9900,
                                    suggestion_limit: int = 3) -> List[Dict[str, Any]]:
        """
        Return: List of suggested mappings (JSON serializable)
        """
        effective_doc_id = doc_id if doc_id is not None else os.path.splitext(os.path.basename(file_path))[0]
        
        docs = load_and_chunk_document(file_path=file_path, doc_id=effective_doc_id)
        if not docs:
            logger.error(f"❌ Failed to load or chunk the file: {file_path}")
            return []

        base_query = docs[0].page_content
        
        logger.info(f"Loaded and chunked {file_path} -> {len(docs)} chunks.")
        logger.info(f"Primary Chunk Content (Base Query): \n---START---\n{base_query}\n---END---")
        
        if len(re.sub(r'[\d\s\W]', '', base_query)) < len(base_query) * 0.1:
             logger.warning(f"⚠️ Warning: Base Query chunk from {effective_doc_id} appears to be mostly noise/numbers.")

        # ===================================================================================================
        # FINAL/BALANCED MODE: ใช้ Dynamic Augmentation (Enabler-specific Keywords) และ Level Constraint
        # ===================================================================================================
        
        # 🟢 1. สร้าง Enabler-specific Augmentation
        augmentation_keywords = self._get_dynamic_augmentation(effective_doc_id, base_query)
        
        # 🟢 2. สร้าง Level Constraint (ถ้า Level ถูกระบุ)
        level_constraint = ""
        if level is not None and 1 <= level <= 5:
            level_constraint = self._get_level_constraint_prompt(level)
            logger.info(f"Applying Level Constraint: {level_constraint}")
        
        # 🟢 3. รวม Prompt ทั้งหมด
        instruction_prompt = (
            f"โปรดจัดอันดับความเกี่ยวข้องของข้อความ Statement เหล่านี้กับหลักฐานที่แสดงถึง ({augmentation_keywords}). "
            f"โดยพิจารณาว่าหลักฐานนี้มีความชัดเจนในการสนับสนุนข้อความ Statement ในด้าน '{self.enabler_id.upper()}' หรือไม่. "
            f"{level_constraint}"
            f"หลักฐาน: "
        )
        
        query = f"{instruction_prompt}{base_query[:1000]}"
        logger.info(f"Using {self.enabler_id.upper()} Augmented Query for RAG: '{query[:120]}...'")

        # ===================================================================================================
        
        # 📌 Note: self.statement_retriever.invoke(query) จะทำการ Rerank ภายใน
        retrieved_statements: List[LcDocument] = self.statement_retriever.invoke(query)

        suggested_mappings = []
        for i, doc in enumerate(retrieved_statements):
            score = float(doc.metadata.get('relevance_score', 0.0))
            if score < similarity_threshold:
                # หยุดเมื่อคะแนนต่ำกว่า Threshold
                break
            
            # ดึงรายละเอียด Statement
            details = self._get_statement_detail(doc.page_content)
            
            if details:
                # นับเฉพาะ suggestion ที่ผ่าน threshold
                if len(suggested_mappings) < suggestion_limit:
                    suggested_mappings.append({
                        "suggestion_rank": i + 1,
                        "score": score,
                        "statement_key": details["statement_key"],
                        "sub_level": details["sub_level"],
                        "statement_text": details["statement_text"],
                        "justification": f"ความคล้ายทางความหมายสูง: เนื้อหาหลักฐานเกี่ยวข้องกับหัวข้อ: '{details['sub_level']}'"
                    })

        return suggested_mappings

    def process_directory(self,
                          directory: str,
                          output_file: str = "results/merged_results.json",
                          level: Optional[int] = None, # 👈 รับค่า Level เข้ามา
                          top_k: int = 7,
                          threshold: float = 0.9900,
                          suggestion_limit: int = 3):
        """
        Process all supported files in a directory and save merged results into a single JSON
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        merged_results = {}
        # 🛠️ ปรับปรุง: เพิ่ม .jpg, .jpeg, .png ในรายการที่รองรับ
        SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt", ".xlsx", ".pptx", ".md", ".csv", ".jpg", ".jpeg", ".png"]

        for filename in os.listdir(directory):
            ext = os.path.splitext(filename)[1].lower()
            
            # 💡 Skip directories and hidden files/temp files
            file_path = os.path.join(directory, filename)
            if os.path.isdir(file_path) or filename.startswith('.'):
                 continue

            if ext not in SUPPORTED_EXTENSIONS:
                logger.info(f"Skipping unsupported file type: {filename}")
                continue
            
            doc_id = os.path.splitext(filename)[0]
            
            logger.info(f"\n==================================================")
            logger.info(f"🚀 STARTING PROCESSING: {filename}")
            logger.info(f"==================================================")

            try:
                suggested = self.process_and_suggest_mapping(
                    file_path=file_path,
                    doc_id=doc_id,
                    level=level, # 👈 ส่ง Level เข้าไป
                    top_k_statements=top_k,
                    similarity_threshold=threshold,
                    suggestion_limit=suggestion_limit
                )
                merged_results[doc_id] = suggested
            except Exception as e:
                logger.error(f"❌ Failed to process {filename}: {e}")
                merged_results[doc_id] = {"error": str(e)}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ Merged results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evidence Mapping Generator CLI")
    # 🛠️ 3. ปรับปรุง: เพิ่ม choices ตามรายการ Enabler ที่กำหนด
    parser.add_argument("--enabler", 
                            type=str, 
                            default="KM",
                            choices=["CG", "L", "SP", "RM&IC", "SCM", "DT", "HCM", "KM", "IM", "IA"],
                            help="The core business enabler abbreviation (e.g., 'KM', 'SCM').")
    
    # 📌 4. เพิ่ม Argument สำหรับ Level เพื่อใช้ใน Constraint Prompt
    parser.add_argument("--level", 
                            type=int, 
                            choices=[1, 2, 3, 4, 5], 
                            default=None, 
                            help="Maturity Level constraint (1-5). Used to filter out higher level statements.")
                            
    parser.add_argument("--file_path", type=str, help="Path to a single evidence file")
    
    parser.add_argument("--doc_id", type=str, help="Optional Document ID to override the filename. (e.g., KM1.1L106)")
    
    parser.add_argument("--directory", type=str, help="Directory to process all evidence files")
    parser.add_argument("--output_file", type=str, default="results/merged_results.json", help="Output JSON file path")
    parser.add_argument("--top_k", type=int, default=7, help="Top K statements to retrieve")
    parser.add_argument("--threshold", type=float, default=0.9900, help="Similarity threshold") 
    parser.add_argument("--suggestion_limit", type=int, default=3, help="Max number of suggestions") 
    args = parser.parse_args()

    try:
        generator = EvidenceMappingGenerator(enabler_id=args.enabler)
    except FileNotFoundError as e:
        print(f"❌ Error during initialization: {e}")
        return
    except Exception as e:
         print(f"❌ General Error during initialization: {e}")
         return

    if args.file_path:
        # Process single file
        result = generator.process_and_suggest_mapping(
            file_path=args.file_path,
            doc_id=args.doc_id, 
            level=args.level, # 👈 ส่ง Level เข้าไป
            top_k_statements=args.top_k,
            similarity_threshold=args.threshold,
            suggestion_limit=args.suggestion_limit
        )
        if isinstance(result, list) and result:
             print("================================================================================")
             print(f"✅ Suggested Mappings for Evidence File '{os.path.basename(args.file_path)}' ({args.enabler}) [Level Filter: {args.level or 'None'}]:")
             print("================================================================================")
             for i, suggestion in enumerate(result):
                print(f"--- Suggestion {i + 1} (Score: {suggestion['score']:.4f}) ---")
                print(f"  Statement Key:   {suggestion['statement_key']}")
                print(f"  Sub/Level:       {suggestion['sub_level']}")
                print(f"  Statement Text:  {suggestion['statement_text']}")
                print(f"  Justification:   {suggestion['justification']}")
                print("----------------------------------------")
             print(f"Found {len(result)} suggestions (filtered by top {args.top_k}).")
        else:
             print("================================================================================")
             print(f"✅ Suggested Mappings for Evidence File '{os.path.basename(args.file_path)}' ({args.enabler}) [Level Filter: {args.level or 'None'}]:")
             print("================================================================================")
             print("❌ No suggested mappings found above the threshold, or an error occurred.")
    elif args.directory:
        # Process all files in directory
        generator.process_directory(
            directory=args.directory,
            output_file=args.output_file,
            level=args.level, # 👈 ส่ง Level เข้าไป
            top_k=args.top_k,
            threshold=args.threshold,
            suggestion_limit=args.suggestion_limit
        )
    else:
        print("❌ Please provide either --file_path or --directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
