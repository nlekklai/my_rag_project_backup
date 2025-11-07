import os
import json
import sys
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging

# --- LangChain/Loader Imports ---
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.prompts import PromptTemplate

# --- LLM Instance Loading และ CONFIG (คงเดิม) ---
try:
    from models.llm import llm as llm_instance
except Exception:
    llm_instance = None
    
try:
    current_dir = Path(__file__).parent.parent
    sys.path.append(str(current_dir)) 
    from config import global_vars 
    PROJECT_ROOT = Path(global_vars.PROJECT_ROOT)
    SUPPORTED_ENABLERS = global_vars.SUPPORTED_ENABLERS 
except (ImportError, AttributeError):
    PROJECT_ROOT = Path(os.getcwd())
    SUPPORTED_ENABLERS = ["KM"] 

logger = logging.getLogger(__name__)

# --- 1. DEFINING THE STRUCTURED OUTPUT (Pydantic Schema) ---
class SubCriteria(BaseModel):
    """โมเดลสำหรับเกณฑ์ย่อย (Sub-Criteria) เช่น KM 1.1"""
    Enabler_ID: str = Field(description="รหัส Enabler หลัก (เช่น '1', '2' หรือ 'KM')")
    Sub_Criteria_ID: str = Field(description="รหัสเกณฑ์ย่อย (เช่น '1.1', '2.1')")
    Sub_Criteria_Name_TH: str = Field(description="ชื่อเกณฑ์ย่อยเป็นภาษาไทย")
    Weight: Optional[float] = Field(description="ค่าน้ำหนักของเกณฑ์นี้ (ถ้าสกัดได้, หากไม่ได้ให้เป็น 0.0)")
    
    Level_1_Statements: List[str] = Field(description="รายการข้อความเกณฑ์การประเมินย่อยที่สมบูรณ์จากคู่มือ (ไม่ย่อ) สำหรับ Level 1")
    Level_2_Statements: List[str] = Field(description="รายการข้อความเกณฑ์การประเมินย่อยที่สมบูรณ์จากคู่มือ (ไม่ย่อ) สำหรับ Level 2")
    Level_3_Statements: List[str] = Field(description="รายการข้อความเกณฑ์การประเมินย่อยที่สมบูรณ์จากคู่มือ (ไม่ย่อ) สำหรับ Level 3")
    Level_4_Statements: List[str] = Field(description="รายการข้อความเกณฑ์การประเมินย่อยที่สมบูรณ์จากคู่มือ (ไม่ย่อ) สำหรับ Level 4")
    Level_5_Statements: List[str] = Field(description="รายการข้อความเกณฑ์การประเมินย่อยที่สมบูรณ์จากคู่มือ (ไม่ย่อ) สำหรับ Level 5")
    
class EnablerStatementList(BaseModel):
    """โมเดลสำหรับรายการ Statement ทั้งหมดของ Enabler หนึ่งตัว"""
    statement_list: List[SubCriteria] = Field(description="รายการเกณฑ์ประเมินทั้งหมดที่สกัดได้จากคู่มือ")


def extract_single_enabler_statements(target_enabler: str):
    """ดึงข้อมูล Statement ของ Enabler ตัวเดียวจาก PDF และจัดโครงสร้าง"""
    
    # --- DYNAMIC PATH GENERATION ---
    pdf_filename = f"SE-AM_{target_enabler}.pdf"
    pdf_path = PROJECT_ROOT / "data" / "seam" / pdf_filename
    output_enabler_name = target_enabler.lower().replace("&", "").replace("-", "")
    output_filename = f"official_{output_enabler_name}_statements.json"
    output_path = PROJECT_ROOT / "evidence_checklist" / output_filename
    
    # --- LOGGING AND VALIDATION ---
    logger.info("-" * 60)
    logger.info(f"  Starting Extraction for ENABLER: {target_enabler}")
    logger.info(f"  Source File: {pdf_path.name}")
    logger.info("-" * 60)

    if llm_instance is None:
        logger.error("🛑 Cannot proceed: LLM instance is not available.")
        return
    if not pdf_path.exists():
        logger.error(f"❌ Error: File not found at the expected path: {pdf_path.resolve()}")
        return

    # 1. โหลดเอกสารและสกัดข้อความดิบ
    try:
        loader = UnstructuredPDFLoader(str(pdf_path), mode="elements")
        docs = loader.load()
    except Exception as e:
        logger.error(f"❌ Error loading PDF {pdf_path.name}: {e}")
        return

    full_text = "\n\n".join([d.page_content for d in docs if len(d.page_content.strip()) > 10])

    if not full_text:
        logger.error(f"❌ Error: Extracted text from {pdf_path.name} is empty.")
        return
    
    logger.info(f"✅ Extracted {len(full_text)} characters. Sending to LLM...")

    # 3. กำหนด Prompt และเรียกใช้ LLM
    try:
        # 🟢 3.1 สร้าง Pydantic Output Parser
        parser = PydanticOutputParser(pydantic_object=EnablerStatementList)
        format_instructions = parser.get_format_instructions()
        
        # --- PROMPT INSTRUCTION (ปรับปรุงให้เน้นย้ำความสมบูรณ์) ---
        system_instruction = (
            f"คุณคือผู้เชี่ยวชาญด้านการประเมินผลรัฐวิสาหกิจ (SE-AM) ภารกิจของคุณคือการสกัดเกณฑ์การประเมิน "
            f"Enabler: {target_enabler} **ทั้งหมดโดยสมบูรณ์** จากข้อความดิบ "
            "โดยเฉพาะอย่างยิ่งต้องสกัดให้ครบถ้วนทุก Sub-Criteria (เช่น KM 1.1, 2.1, 3.1, 4.1, 5.1, 6.1) "
            "Statement ต้องมีความสมบูรณ์ที่สุดจากคู่มือ (ไม่ถูกย่อ) และ **ต้องจัดโครงสร้างให้เป็น JSON ตาม Schema ที่กำหนดไว้อย่างเคร่งครัดด้านล่างนี้:** \n\n"
        )
        
        # 🟢 3.2 สร้าง Prompt Template
        prompt = PromptTemplate(
            template="{system_instruction}{format_instructions}\n\n[RAW TEXT]:\n{raw_text}",
            input_variables=["raw_text"],
            partial_variables={
                "system_instruction": system_instruction,
                "format_instructions": format_instructions
            }
        )
        
        # 🟢 3.3 สร้าง Chain (ใช้ OutputFixingParser)
        fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm_instance)
        
        # 🟢 3.4 เรียกใช้ LLM และ Parse ผลลัพธ์
        full_prompt_text = prompt.format(raw_text=full_text)
        
        llm_output = llm_instance.invoke(full_prompt_text)
        
        # 🟢 3.5 Parse และตรวจสอบผลลัพธ์
        result_pydantic = fixing_parser.parse(llm_output)
        
        # 4. บันทึกผลลัพธ์
        final_json_data = result_pydantic.model_dump() 
        
        # 5. จัดรูปแบบให้เป็น JSON ที่อ่านง่าย
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            # ใช้ indent=2 ใน json.dump()
            json.dump(final_json_data['statement_list'], f, indent=2, ensure_ascii=False) 

        logger.info("-" * 60)
        logger.info(f"✨ Success! Extracted {len(final_json_data['statement_list'])} sub-criteria saved to: {output_path.resolve()}")
        logger.info("-" * 60)
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed during LLM processing for {target_enabler}: {e}", exc_info=True)
        logger.error("*** NOTE: LLM is likely truncating the output due to large context. Check Ollama context size (num_ctx) or use a larger model. ***")
        return False
        
        
def extract_all_enabler_statements():
    """วนลูปเพื่อดึงข้อมูล Statement ของ Enabler ทั้งหมดที่รองรับ"""
    
    if not SUPPORTED_ENABLERS:
        logger.error("🛑 SUPPORTED_ENABLERS list is empty. Cannot proceed.")
        return

    logger.info(f"Starting batch extraction for Enablers: {SUPPORTED_ENABLERS}")
    
    for enabler in SUPPORTED_ENABLERS:
        extract_single_enabler_statements(enabler)
        
    logger.info("Batch extraction process complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description="Extracts SE-AM Statements from PDF to JSON format."
    )
    parser.add_argument(
        '--enabler', 
        type=str, 
        nargs='?', 
        default=None, 
        help='Specify a single Enabler (e.g., KM, CG, HCM) to process. If omitted, all supported Enablers will be processed.'
    )
    
    args = parser.parse_args()
    
    if args.enabler:
        target_enabler = args.enabler.upper()
        if target_enabler in SUPPORTED_ENABLERS:
            extract_single_enabler_statements(target_enabler)
        else:
            logger.error(f"❌ Enabler '{target_enabler}' is not supported in the configuration.")
    else:
        extract_all_enabler_statements()