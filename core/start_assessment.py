# -*- coding: utf-8 -*-
# core/start_assessment.py - Optimized for Mac & GPU Server

import os
import sys
import logging
import argparse
import time
import uuid
import multiprocessing
from typing import Optional, Dict, Any

# -------------------- PATH SETUP --------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# -------------------- IMPORT CORE --------------------
from models.llm import create_llm_instance
from database import init_db, db_create_task, db_finish_task, db_update_task_status 
from config.global_vars import (
    EVIDENCE_DOC_TYPES, DEFAULT_ENABLER, DEFAULT_LLM_MODEL_NAME, 
    DEFAULT_TENANT, DEFAULT_YEAR
) 

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- IMPORT CORE --------------------
from models.llm import create_llm_instance
from database import init_db, db_create_task, db_finish_task, db_update_task_status 

# ✅ นำเข้าฟังก์ชันโหลด mapping จาก path_utils โดยตรง
try:
    from utils.path_utils import load_doc_id_mapping as load_document_map
    logger.info("✅ Successfully linked 'load_doc_id_mapping' from path_utils")
except ImportError:
    # Fallback กรณีหาไม่เจอจริงๆ (ไม่ควรเกิดขึ้นถ้า path_utils อยู่ถูกที่)
    def load_document_map(*args, **kwargs): return {}
    print("⚠️ Warning: No document mapping function found in utils.path_utils")

try:
    from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig 
    from core.vectorstore import load_all_vectorstores
    # ลบส่วน Import load_document_map เก่าๆ ทิ้งไปได้เลย
except Exception as e:
    print(f"❌ FATAL: Missing critical modules: {e}", file=sys.stderr)
    raise

# -------------------- ARGUMENT PARSING --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEAM PDCA Assessment Runner (Production Mode)")
    p.add_argument("--sub", type=str, default="all", help="Sub-Criteria ID (e.g., 1.1)")
    p.add_argument("--enabler", type=str, default=DEFAULT_ENABLER, help="Enabler (KM/IT/...)")
    p.add_argument("--target_level", type=int, default=5, help="Max target maturity level")
    p.add_argument("--export", action="store_true", help="Save results to JSON")
    p.add_argument("--mock", choices=["none", "random", "control"], default="none", help="Mock mode")
    p.add_argument("--sequential", action="store_true", help="Force sequential execution")
    p.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant ID")
    p.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year")
    p.add_argument("--min-retry-score", type=float, default=0.65)
    p.add_argument("--max-retrieval-attempts", type=int, default=3)
    p.add_argument("--record-id", type=str, default=None, help="Custom record ID")
    return p.parse_args()

# -------------------- MAIN EXECUTION --------------------
def main():
    args = parse_args()
    
    # 1. Initialize Record ID
    record_id = args.record_id if args.record_id else uuid.uuid4().hex[:12]
    run_mode = "Sequential" if args.sequential else "Parallel"
    logger.info(f"🚀 Runner Started | ID: {record_id} | Mode: {run_mode}")
    start_ts = time.time()

    # 2. Database Initialization
    try:
        init_db()
        db_create_task(
            record_id=record_id, tenant=args.tenant, year=str(args.year),
            enabler=args.enabler, sub_criteria=args.sub, user_id="CLI_SYSTEM"
        )
        logger.info(f"✅ Database Task Registered: {record_id}")
    except Exception as e:
        logger.warning(f"⚠️ DB Registration Warning: {e}")

    # 3. Resource Loading (Vector Store & Document Map)
    vsm = None
    if args.mock == "none":
        try:
            vsm = load_all_vectorstores(
                tenant=args.tenant, year=str(args.year),
                doc_types=EVIDENCE_DOC_TYPES, enabler_filter=args.enabler
            )
        except Exception as e:
            logger.error(f"❌ VSM Load failed: {e}")
            sys.exit(1)

    document_map = {}
    try:
        raw_map = load_document_map(EVIDENCE_DOC_TYPES, args.tenant, str(args.year), args.enabler)
        if isinstance(raw_map, dict) and raw_map:
            # ตรวจสอบว่าเป็น dict ซ้อน dict หรือไม่
            sample_val = next(iter(raw_map.values()))
            if isinstance(sample_val, dict):
                document_map = {k: v.get("file_name", k) for k, v in raw_map.items()}
            else:
                document_map = raw_map
        logger.info(f"🎯 Loaded {len(document_map)} document mappings.")
    except Exception as e:
        logger.warning(f"⚠️ Document map warning: {e}")

    # Initialize LLM
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)

    # 4. Engine Configuration
    config = AssessmentConfig(
        enabler=args.enabler, tenant=args.tenant, year=args.year,
        target_level=args.target_level, mock_mode=args.mock,
        force_sequential=args.sequential, model_name=DEFAULT_LLM_MODEL_NAME,
        min_retry_score=args.min_retry_score, max_retrieval_attempts=args.max_retrieval_attempts
    )
    
    engine = SEAMPDCAEngine(
        config=config, llm_instance=llm, logger_instance=logger,             
        doc_type=EVIDENCE_DOC_TYPES, vectorstore_manager=vsm, 
        document_map=document_map, record_id=record_id
    )

    # 5. Run Assessment
    final_results = None
    try:
        final_results = engine.run_assessment(
            target_sub_id=args.sub, export=args.export, 
            vectorstore_manager=vsm, sequential=args.sequential,
            document_map=document_map, record_id=record_id
        )

        # Persistence: บันทึกผลลง DB (เพิ่ม Safe Check)
        if final_results is not None:
            db_finish_task(record_id, final_results)
            logger.info(f"💾 Results saved to database: {record_id}")
        else:
            logger.error("❌ run_assessment returned None")

    except Exception as e:
        logger.exception(f"❌ Engine execution failed: {e}")
        db_update_task_status(record_id, 0, f"Error: {str(e)}", status="FAILED")
        sys.exit(1)

    # ---------------------------------------------------------------
    # 6. Final Summary Extraction (CLI Display Logic) - REVISED v2026
    # ---------------------------------------------------------------
    duration_s = time.time() - start_ts
    
    # ค่าเริ่มต้นป้องกัน Error
    summary_display = {
        "level": "L0",
        "score": 0.0,
        "path": "N/A"
    }

    if isinstance(final_results, dict):
        # 🛡️ Step 1: Unwrap ชั้นข้อมูล (รองรับหลายรูปแบบ Payload)
        data = final_results.get("result") or final_results.get("assessment_result") or final_results
        res_summary = data.get("result_summary") or data.get("summary") or {}
        sub_details = data.get("sub_criteria_results") or data.get("sub_criteria_details") or []
        
        # 🛡️ Step 2: Extract Level (Maturity)
        # ลองดึงจาก Summary รวมก่อน ถ้าไม่เจอ (กรณีรัน --sub) ให้ดึงจากหัวข้อแรกที่ประเมิน
        lvl = res_summary.get("maturity_level")
        if lvl is None:
            lvl = data.get("highest_full_level")
        if lvl is None and sub_details:
            lvl = sub_details[0].get("highest_full_level") or sub_details[0].get("maturity_level")
        
        # ประกันว่าต้องเป็นตัวเลข (Default 0)
        try:
            lvl_int = int(lvl) if lvl is not None else 0
            summary_display["level"] = f"L{lvl_int}"
        except:
            summary_display["level"] = str(lvl or "L0")

        # 🛡️ Step 3: Extract Score (Weighted Score)
        # ตรรกะ: ถ้าคะแนนรวมเป็น 0 (อาจเพราะรันข้อเดียว) ให้ใช้คะแนนของ Sub-criteria นั้นๆ แทน
        score = res_summary.get("total_weighted_score") or data.get("weighted_score")
        
        if not score or float(score) == 0:
            if sub_details:
                # ใช้คะแนนของข้อแรกที่เจอใน List (กรณีระบุ --sub)
                score = sub_details[0].get("weighted_score") or sub_details[0].get("score")

        summary_display["score"] = float(score or 0.0)
        summary_display["path"] = data.get("export_path") or data.get("export_path_used") or "N/A"

    # ---------------------------------------------------------------
    # 🏁 Display Summary UI (Console Output)
    # ---------------------------------------------------------------
    print("\n" + "═"*65)
    print(f" 🏁  ASSESSMENT COMPLETE | ID: {record_id}")
    print("═"*65)
    print(f" [MODE]        : {run_mode}")
    print(f" [RESULT]      : {summary_display['level']}") # จะแสดงเป็น L5 แทน L0
    
    # ถ้าคะแนนคือ 20.00 และ Weight คือ 4.0 จะหมายถึงได้ Level 5 เต็มในหัวข้อนั้น
    print(f" [SCORE]       : {summary_display['score']:.2f}") 
    print(f" [DURATION]    : {duration_s:.2f} seconds")
    print("-" * 65)
    if args.export:
        print(f" 💾 Exported to: {summary_display['path']}")
    print("═"*65 + "\n")

if __name__ == "__main__":
    # สำคัญมากสำหรับการรันบน Mac (ARM) และป้องกันปัญหาตอนทำ Multiprocessing
    multiprocessing.freeze_support()
    main()