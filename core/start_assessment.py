# core/start_assessment.py
"""
CLI runner that:
 - parses args (--sub, --enabler, --export, --mock, --sequential) 
 - loads central evidence vectorstore (via core.vectorstore.load_all_vectorstores)
 - instantiates SEAMPDCAEngine and runs assessment
 - prints summary and optionally detailed output and exports files
"""

import os
import sys
import logging
import argparse
import time
from typing import Optional, Dict, Any


# -------------------- PATH SETUP --------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from models.llm import create_llm_instance

try:
    # Import Config & Core Modules
    # 📌 เพิ่ม DEFAULT_TENANT และ DEFAULT_YEAR
    from config.global_vars import (
        EVIDENCE_DOC_TYPES, DEFAULT_ENABLER, DEFAULT_LLM_MODEL_NAME, 
        DEFAULT_TENANT, DEFAULT_YEAR
    ) 
    from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig 
    # 🟢 FIX: ต้องเพิ่ม load_document_map
    from core.vectorstore import load_all_vectorstores, VectorStoreManager 
    
    # 🟢 ASSUMPTION: load_document_map is available in core.vectorstore
    try:
        from core.vectorstore import load_document_map
    except ImportError:
        def load_document_map(tenant: str, year: int, enabler: str) -> Dict[str, str]:
             """MOCK: Returns an empty dictionary if the real function is not imported."""
             return {}

    import assessments.seam_mocking as seam_mocking 
except Exception as e:
    print(f"FATAL: missing import in start_assessment.py: {e}", file=sys.stderr)
    raise

from config.global_vars import EVIDENCE_DOC_TYPES

# -------------------- LOGGING SETUP --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------- ARGUMENT PARSING --------------------
def parse_args() -> argparse.Namespace:
    """Parses command line arguments for the assessment runner."""
    p = argparse.ArgumentParser(description="SEAM PDCA Assessment Runner")
    p.add_argument("--sub", type=str, default="all", help="Sub-Criteria ID or 'all' (e.g., 1.1)")
    p.add_argument("--enabler", type=str, default=DEFAULT_ENABLER, help="Enabler ID (e.g., KM)")
    p.add_argument("--target_level", type=int, default=5, help="Maximum target level for sequential assessment.")
    p.add_argument("--export", action="store_true", help="Export results to JSON file.")
    p.add_argument("--mock", choices=["none", "random", "control"], default="none", help="Mock mode ('none', 'random', 'control').")
    p.add_argument("--sequential", action="store_true", help="Force sequential execution, even when assessing all sub-criteria (recommended for low-resource machines).")
    # 🟢 Arguments for Evidence Mapping scope
    p.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant ID for mapping file scope (e.g., 'EGAT').")
    p.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Assessment year for mapping file scope (e.g., 2024).")
    
    # 📌 NEW FIX: Arguments for Adaptive RAG Tuning
    # ใช้ค่า default จาก AssessmentConfig เพื่อให้สอดคล้อง
    p.add_argument("--min-retry-score", type=float, default=0.65, help="Min Rerank score to stop adaptive retrieval loop.")
    p.add_argument("--max-retrieval-attempts", type=int, default=3, help="Max attempts for adaptive RAG retrieval loop.")
    
    return p.parse_args()

# -------------------- MAIN EXECUTION --------------------
def main():
    args = parse_args()
    
    # 🟢 เพิ่มการแสดงผล Mode ใน Log (และ Tenant/Year)
    run_mode = "Sequential" if args.sequential else "Parallel"
    logger.info(
        f"Starting {run_mode} assessment runner "
        f"(enabler={args.enabler}, sub={args.sub}, tenant={args.tenant}, year={args.year}, "
        f"mock={args.mock}, target_level={args.target_level})"
    )
    start_ts = time.time()

    # 1. Load Vectorstores and Document Map (โหลดเพียงครั้งเดียวใน Process หลัก)
    vsm: Optional[VectorStoreManager] = None
    document_map: Optional[Dict[str, str]] = None 
    
    # 🟢 FIX: Skip VSM loading if running in Sequential Mode 
    if args.sequential and args.mock == "none":
        logger.info("Sequential mode (non-mock): Skipping initial VSM load in main process. VSM will be loaded one time inside the Engine for robustness.")
        # vsm remains None, forcing the load in SEAMPDCAEngine
    else:
        try:
            logger.info("Loading central evidence vectorstore(s)...")
            # 🎯 FIX: เพิ่ม tenant และ year ในการเรียกฟังก์ชัน load_all_vectorstores 
            vsm = load_all_vectorstores(
                doc_types=[EVIDENCE_DOC_TYPES], 
                evidence_enabler=args.enabler,
                tenant=args.tenant,        
                year=args.year             
            )
        except Exception as e:
            logger.error(f"Failed to load vectorstores: {e}")
            if args.mock == "none":
                 logger.error("Non-mock mode requires VectorStoreManager to load successfully. Raising fatal error.")
                 raise

    # 1.3 Load Document Map (สำหรับ mapping doc_id -> filename)
    try:
        logger.info("Loading document ID to filename map...")
        # 🎯 NEW FIX: โหลด Document Map
        document_map = load_document_map(
            tenant=args.tenant, 
            year=args.year,
            enabler=args.enabler
        )
        logger.info(f"Loaded {len(document_map)} document mappings.")
    except Exception as e:
        logger.warning(f"Failed to load document map: {e}. Assessment will continue, but filenames in results may be limited.")
        document_map = {} # Ensure it's an empty dictionary if failed
        

    # 1.5. Initialize LLM for Classification & Evaluation
    llm_for_classification = None
    try:
        llm_for_classification = create_llm_instance(
            model_name=DEFAULT_LLM_MODEL_NAME, 
            temperature=0.0
        )
        if not llm_for_classification:
             raise RuntimeError("LLM Factory returned None.")

        logger.info("✅ LLM Instance initialized for Engine injection.")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Inference Engine: {e}")
        if args.mock == "none":
            raise

    # 2. Instantiate Engine
    # 📌 FIX: สร้าง Config และส่งค่า Adaptive RAG ใหม่เข้าไป
    config = AssessmentConfig(
        enabler=args.enabler, 
        target_level=args.target_level,
        mock_mode=args.mock,
        # 🟢 Pass the core arguments
        force_sequential=args.sequential,
        model_name=DEFAULT_LLM_MODEL_NAME,
        temperature=0.0, 
        tenant=args.tenant,  
        year=args.year, 
        # 🟢 NEW: RAG Adaptive Tuning
        min_retry_score=args.min_retry_score,           # ⬅️ New Config
        max_retrieval_attempts=args.max_retrieval_attempts # ⬅️ New Config
    )
    engine = SEAMPDCAEngine(
        config=config,
        llm_instance=llm_for_classification, 
        logger_instance=logger,             
        doc_type=EVIDENCE_DOC_TYPES, 
        vectorstore_manager=vsm, 
        document_map=document_map, # 🟢 FIX: ส่ง Document Map ไปให้ Engine
    )

    # 3. Run Assessment
    try:
        final = engine.run_assessment(
            target_sub_id=args.sub, 
            export=args.export, 
            vectorstore_manager=vsm,
            sequential=args.sequential,
            document_map=document_map # 🟢 FIX: ส่ง Document Map ไปให้ run_assessment (สำหรับส่งต่อไปยัง Workers)
        )
    except Exception as e:
        logger.exception(f"Engine run failed: {e}")
        raise

    # 4. Print Summary
    summary = final.get("summary", {})
    duration_s = time.time() - start_ts
    
    print("\n" + "="*60)
    print(f"ASSESSMENT COMPLETE - ENABLER: {args.enabler}")
    print(f"RUN MODE: {run_mode}")
    print("="*60)
    print(f"Tenant/Year: {args.tenant}/{args.year}")
    print(f"Target Level: {summary.get('target_level', config.target_level)}")
    print(f"RAG Config (Score/Attempts): {config.min_retry_score}/{config.max_retrieval_attempts}") # ⬅️ Show new config
    print(f"Total sub-criteria run: {summary.get('total_subcriteria', 0)}")
    print(f"Percentage Achieved: {summary.get('percentage_achieved_run', 0.0):.3f}%")
    print(f"Duration (s): {duration_s:.2f}")
    print("="*60)

    # 5. Detailed print if single sub requested
    if args.sub and args.sub.lower() != "all":
        # engine.print_detailed_results(target_sub_id=args.sub)
        pass 

    if args.export:
        print("\nReport export status logged (see INFO logs for path).")

    logger.info(f"Full runner execution completed in {duration_s:.2f}s")

if __name__ == "__main__":
    main()