# -*- coding: utf-8 -*-
# core/start_assessment.py

import os
import sys
import logging
import argparse
import time
import uuid  # [ADDED] สำหรับสร้าง record_id ในโหมด CLI
from typing import Optional, Dict, Any
from copy import deepcopy

# -------------------- PATH SETUP --------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.llm import create_llm_instance

try:
    # Import Config & Core Modules
    from config.global_vars import (
        EVIDENCE_DOC_TYPES, DEFAULT_ENABLER, DEFAULT_LLM_MODEL_NAME, 
        DEFAULT_TENANT, DEFAULT_YEAR
    ) 
    from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig 
    from core.vectorstore import load_all_vectorstores, VectorStoreManager 
    
    # Load Document Map Utility
    try:
        from core.vectorstore import load_document_map
    except ImportError:
        def load_document_map(tenant: str, year: int, enabler: str) -> Dict[str, str]:
             return {}

    import assessments.seam_mocking as seam_mocking 
except Exception as e:
    print(f"FATAL: missing import in start_assessment.py: {e}", file=sys.stderr)
    raise

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- ARGUMENT PARSING --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SEAM PDCA Assessment Runner")
    p.add_argument("--sub", type=str, default="all", help="Sub-Criteria ID or 'all' (e.g., 1.1)")
    p.add_argument("--enabler", type=str, default=DEFAULT_ENABLER, help="Enabler ID (e.g., KM)")
    p.add_argument("--target_level", type=int, default=5, help="Maximum target level.")
    p.add_argument("--export", action="store_true", help="Export results to JSON file.")
    p.add_argument("--mock", choices=["none", "random", "control"], default="none", help="Mock mode.")
    p.add_argument("--sequential", action="store_true", help="Force sequential execution.")
    p.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant ID (e.g., 'pea').")
    p.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Assessment year (e.g., 2568).")
    
    # Adaptive RAG Tuning
    p.add_argument("--min-retry-score", type=float, default=0.65, help="Min Rerank score.")
    p.add_argument("--max-retrieval-attempts", type=int, default=3, help="Max attempts.")
    
    # [ADDED] Option สำหรับกำหนด ID เองถ้าต้องการ (ถ้าไม่ใส่จะ Generate ให้)
    p.add_argument("--record-id", type=str, default=None, help="Specific record ID for this run.")
    
    return p.parse_args()

# -------------------- MAIN EXECUTION --------------------
def main():
    args = parse_args()
    
    # 1. สร้าง Record ID สำหรับการรันครั้งนี้ (ถ้าไม่ได้ระบุมา)
    # เพื่อให้สอดคล้องกับระบบ Engine ใหม่ที่ต้องการ record_id
    record_id = args.record_id if args.record_id else uuid.uuid4().hex[:12]
    
    run_mode = "Sequential" if args.sequential else "Parallel"
    logger.info(
        f"🚀 Starting {run_mode} Runner | ID: {record_id} | "
        f"Target: {args.enabler} {args.sub} ({args.tenant}/{args.year})"
    )
    start_ts = time.time()

    # 1.1 Load Vectorstores
    vsm = None
    if args.sequential and args.mock == "none":
        logger.info("Sequential mode: Initial VSM load skipped (deferred to engine).")
    else:
        try:
            vsm = load_all_vectorstores(
                doc_types=[EVIDENCE_DOC_TYPES], 
                enabler_filter=args.enabler,
                tenant=args.tenant,        
                year=args.year             
            )
        except Exception as e:
            logger.error(f"Failed to load vectorstores: {e}")
            if args.mock == "none": raise

    # 1.2 Load Document Map
    try:
        document_map = load_document_map(tenant=args.tenant, year=args.year, enabler=args.enabler)
        logger.info(f"Loaded {len(document_map)} document mappings.")
    except Exception as e:
        logger.warning(f"Document map loading failed: {e}")
        document_map = {}

    # 1.3 Initialize LLM
    llm = None
    try:
        llm = create_llm_instance(model_name=DEFAULT_LLM_MODEL_NAME, temperature=0.0)
    except Exception as e:
        logger.error(f"LLM Init failed: {e}")
        if args.mock == "none": raise

    # 2. Instantiate Engine
    config = AssessmentConfig(
        enabler=args.enabler, 
        target_level=args.target_level,
        mock_mode=args.mock,
        force_sequential=args.sequential,
        model_name=DEFAULT_LLM_MODEL_NAME,
        tenant=args.tenant,  
        year=args.year, 
        min_retry_score=args.min_retry_score,
        max_retrieval_attempts=args.max_retrieval_attempts
    )
    
    engine = SEAMPDCAEngine(
        config=config,
        llm_instance=llm, 
        logger_instance=logger,             
        doc_type=EVIDENCE_DOC_TYPES, 
        vectorstore_manager=vsm, 
        document_map=document_map
    )

    # 3. Run Assessment 
    # [FIXED] ส่ง record_id เข้าไปตาม Signature ใหม่ของ Engine
    try:
        final = engine.run_assessment(
            target_sub_id=args.sub, 
            export=args.export, 
            vectorstore_manager=vsm,
            sequential=args.sequential,
            document_map=document_map,
            record_id=record_id # <--- ส่ง ID เข้าไปที่นี่เพื่อแก้ TypeError
        )
    except Exception as e:
        logger.exception(f"❌ Engine run failed: {e}")
        raise

    # 4. Print Summary
    summary = final.get("summary", {})
    duration_s = time.time() - start_ts
    
    highest_passed = summary.get('highest_pass_level') or summary.get('highest_pass_level_overall', 0)
    achieved_weight = summary.get('achieved_weight') or summary.get('total_achieved_weight', 0.0)
    total_weight = summary.get('total_weight') or summary.get('total_possible_weight', 0.0)
    
    print("\n" + "═"*65)
    print(f" 🏁  ASSESSMENT COMPLETE | ID: {record_id}")
    print("═"*65)
    print(f" [MODE]        : {run_mode}")
    print(f" [SUB-ID]      : {args.sub}")
    print(f" [DURATION]    : {duration_s:.2f} seconds")
    print("-" * 65)
    print(f" [RESULT]      : Level Achieved -> L{highest_passed}")
    print(f" [SCORE]       : {achieved_weight:.2f} / {total_weight:.2f}")
    print(f" [PROGRESS]    : {summary.get('percentage_achieved_run', 0.0):.2f}%")
    print("═"*65)

    if args.export:
        export_path = final.get("export_path_used", "N/A")
        print(f" 💾 Exported to: {export_path}")

if __name__ == "__main__":
    main()