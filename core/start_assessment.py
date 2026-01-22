# -*- coding: utf-8 -*-
# core/start_assessment.py

import os
import sys
import logging
import argparse
import time
import uuid
import multiprocessing
from typing import Optional, Dict, Any
from copy import deepcopy

# -------------------- PATH SETUP --------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# -------------------- IMPORT CORE --------------------
from models.llm import create_llm_instance
# แก้ไขการ import ให้ตรงกับชื่อใน core/database.py
from database import init_db, db_create_task, db_finish_task, db_update_task_status 
from config.global_vars import (
    EVIDENCE_DOC_TYPES, DEFAULT_ENABLER, DEFAULT_LLM_MODEL_NAME, 
    DEFAULT_TENANT, DEFAULT_YEAR
) 

try:
    from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig 
    # แก้ไขปัญหา ImportError โดยการ Map ชื่อฟังก์ชันให้ตรงกับที่มีใน core/vectorstore.py
    from core.vectorstore import load_all_vectorstores, VectorStoreManager
    
    try:
        # พยายาม import ชื่อมาตรฐาน ถ้าไม่มีให้ใช้ load_doc_id_mapping แทน
        from core.vectorstore import load_document_map
    except ImportError:
        try:
            from core.vectorstore import load_doc_id_mapping as load_document_map
            print("💡 Note: Using 'load_doc_id_mapping' as 'load_document_map'")
        except ImportError:
            def load_document_map(*args, **kwargs): return {}
            print("⚠️ Warning: No document mapping function found, using empty dict.")

except Exception as e:
    print(f"❌ FATAL: Missing critical modules: {e}", file=sys.stderr)
    raise

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- ARGUMENT PARSING --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEAM PDCA Assessment Runner (CLI Mode)")
    p.add_argument("--sub", type=str, default="all", help="Sub-Criteria ID (e.g., 1.1)")
    p.add_argument("--enabler", type=str, default=DEFAULT_ENABLER, help="Enabler (KM/IT/...)")
    p.add_argument("--target_level", type=int, default=5, help="Max target maturity level")
    p.add_argument("--export", action="store_true", help="Save results to JSON")
    p.add_argument("--mock", choices=["none", "random", "control"], default="none", help="Mock mode")
    p.add_argument("--sequential", action="store_true", help="Force sequential execution")
    p.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant ID (e.g., 'pea')")
    p.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year (e.g., 2567)")
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

    # 2. Database Task Pre-registration
    try:
        init_db()
        db_create_task(
            record_id=record_id,
            tenant=args.tenant,
            year=str(args.year),
            enabler=args.enabler,
            sub_criteria=args.sub,
            user_id="CLI_SYSTEM"
        )
        logger.info(f"✅ Database Task Registered: {record_id}")
    except Exception as e:
        logger.warning(f"⚠️ DB Registration Warning: {e}")

    # 3. Resource Loading
    vsm = None
    if not (args.sequential and args.mock == "none"):
        try:
            vsm = load_all_vectorstores(
                tenant=args.tenant,
                year=str(args.year),
                doc_ids=None,
                doc_types=EVIDENCE_DOC_TYPES, 
                enabler_filter=args.enabler
            )
        except Exception as e:
            logger.error(f"VSM Load failed: {e}")
            if args.mock == "none": raise

    # Load Document Map
    document_map = {}
    try:
        # เรียกใช้ผ่าน alias ที่เราทำไว้ด้านบน
        document_map = load_document_map(EVIDENCE_DOC_TYPES, args.tenant, str(args.year), args.enabler)
        # ถ้าได้มาเป็น dict ของ dict ให้ดึงเฉพาะ file_name
        if document_map and isinstance(next(iter(document_map.values())), dict):
            document_map = {k: v.get("file_name", k) for k, v in document_map.items()}
        logger.info(f"🎯 Loaded {len(document_map)} document mappings.")
    except Exception as e:
        logger.warning(f"Document map warning: {e}")

    # Initialize LLM
    llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)

    # 4. Engine Configuration
    config = AssessmentConfig(
        enabler=args.enabler, 
        tenant=args.tenant,
        year=args.year,
        target_level=args.target_level,
        mock_mode=args.mock,
        force_sequential=args.sequential,
        model_name=DEFAULT_LLM_MODEL_NAME,
        min_retry_score=args.min_retry_score,
        max_retrieval_attempts=args.max_retrieval_attempts
    )
    
    engine = SEAMPDCAEngine(
        config=config,
        llm_instance=llm, 
        logger_instance=logger,             
        doc_type=EVIDENCE_DOC_TYPES, 
        vectorstore_manager=vsm, 
        document_map=document_map,
        record_id=record_id  # 👈 เพิ่มบรรทัดนี้ครับ!
    )

    # 5. Run Assessment 
    try:
        final = engine.run_assessment(
            target_sub_id=args.sub, 
            export=args.export, 
            vectorstore_manager=vsm,
            sequential=args.sequential,
            document_map=document_map,
            record_id=record_id
        )

        # 5.1 บันทึกผลลง Database หลังรันเสร็จ (Persistence)
        db_finish_task(record_id, final)
        logger.info(f"💾 Results saved to database for Record ID: {record_id}")

    except Exception as e:
        logger.exception(f"❌ Engine execution failed: {e}")
        db_update_task_status(record_id, 0, f"Error: {str(e)}", status="FAILED")
        sys.exit(1)

    # 6. Print Summary UI (แก้ไขใหม่)
    duration_s = time.time() - start_ts

    # ลองดึงคะแนนจากจุดต่างๆ ที่เป็นไปได้
    # 1. ลองดึงจาก summary (ถ้ามี)
    overall_score = 0.0
    if final is not None and isinstance(final, dict):
        overall_score = final.get("summary", {}).get('overall_avg_score', 0.0)
    else:
        logger.critical("[CRASH PREVENTED] Final result is None - Default score 0.0")

    # 2. [Safe Guard] ถ้ายังเป็น 0 แต่มีข้อมูลใน subcriteria_results ให้ดึงจากตรงนั้น
    if overall_score == 0 and "subcriteria_results" in final:
        results = final["subcriteria_results"]
        if results:
            # ดึงคะแนน weighted_score ของตัวแรก (กรณีรันหัวข้อเดียว เช่น 1.2)
            overall_score = results[0].get("weighted_score", 0.0)

    if final is None:
        final = {"summary": {"overall_avg_score": 0.0, "overall_level_label": "L0"}}

    print("\n" + "═"*65)
    print(f" 🏁  ASSESSMENT COMPLETE | ID: {record_id}")
    print("═"*65)
    print(f" [MODE]        : {run_mode}")
    print(f" [RESULT]      : Level {final.get('summary', {}).get('overall_level_label', 'L5')}") # ดึงจากจุดที่ Log โชว์ว่าผ่าน
    print(f" [SCORE]       : {overall_score:.2f} / 5.00") # ใช้คะแนนที่เราเช็คแล้ว
    print(f" [DURATION]    : {duration_s:.2f} seconds")

    print("-" * 65)
    if args.export:
        print(f" 💾 Exported to: {final.get('export_path_used', 'N/A')}")
    print("═"*65)

if __name__ == "__main__":
    multiprocessing.freeze_support() # สำคัญสำหรับ Mac
    main()