# core/start_assessment.py
"""
CLI runner that:
 - parses args (--sub, --enabler, --export, --mock)
 - loads central evidence vectorstore (via core.vectorstore.load_all_vectorstores)
 - instantiates SEAMPDCAEngine and runs assessment
 - prints summary and optionally detailed output and exports files
"""

import os
import sys
import logging
import argparse
import time
from typing import Optional

# -------------------- PATH SETUP --------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Import Config & Core Modules
    from config.global_vars import EVIDENCE_DOC_TYPES, DEFAULT_ENABLER
    from core.seam_assessment import SEAMPDCAEngine, AssessmentConfig
    from core.vectorstore import load_all_vectorstores, VectorStoreManager
    import assessments.seam_mocking as seam_mocking 
except Exception as e:
    # This block catches the import error first, which was the previous issue
    print(f"FATAL: missing import in start_assessment.py: {e}", file=sys.stderr)
    raise

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
    return p.parse_args()

# -------------------- MAIN EXECUTION --------------------
def main():
    args = parse_args()
    logger.info(f"Starting assessment runner (enabler={args.enabler}, sub={args.sub}, mock={args.mock}, target_level={args.target_level})")
    start_ts = time.time()

    # 1. Load Vectorstores
    vsm: Optional[VectorStoreManager] = None
    try:
        logger.info("Loading central evidence vectorstore(s)...")
        # Note: EVIDENCE_DOC_TYPES is a string, load_all_vectorstores expects a list of types
        # Assuming the function can handle a single string or the intent is a list containing the evidence type
        vsm = load_all_vectorstores(doc_types=[EVIDENCE_DOC_TYPES], evidence_enabler=args.enabler)
    except Exception as e:
        logger.error(f"Failed to load vectorstores: {e}")
        # Only raise error if VSM load is critical (i.e., not in mock mode)
        if args.mock == "none":
             logger.error("Non-mock mode requires VectorStoreManager to load successfully. Raising fatal error.")
             raise

    # 2. Instantiate Engine
    config = AssessmentConfig(
        enabler=args.enabler, 
        target_level=args.target_level,
        mock_mode=args.mock
    )
    engine = SEAMPDCAEngine(config=config)

    # 3. Run Assessment
    try:
        final = engine.run_assessment(
            target_sub_id=args.sub, 
            export=args.export, 
            vectorstore_manager=vsm
        )
    except Exception as e:
        logger.exception(f"Engine run failed: {e}")
        raise

    # 4. Print Summary
    summary = final.get("summary", {})
    duration_s = time.time() - start_ts
    
    print("\n" + "="*60)
    print(f"ASSESSMENT COMPLETE - ENABLER: {args.enabler}")
    print("="*60)
    print(f"Target Level: {summary.get('target_level', config.target_level)}")
    print(f"Total sub-criteria run: {summary.get('total_subcriteria', 0)}")
    print(f"Average weighted score: {summary.get('avg_weighted_score', 0.0):.3f}%")
    print(f"Duration (s): {duration_s:.2f}")
    print("="*60)

    # 5. Detailed print if single sub requested
    if args.sub and args.sub.lower() != "all":
        engine.print_detailed_results(target_sub_id=args.sub)

    if args.export:
        print("\nReport export status logged (see INFO logs for path).")

    logger.info(f"Full runner execution completed in {duration_s:.2f}s")

if __name__ == "__main__":
    main()