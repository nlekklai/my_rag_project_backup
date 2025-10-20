# process_assessment.py
import argparse
import os
import sys
import time
import json
import logging
from typing import Dict, Any, List

# --- Core Imports ---
# 🚨 CRITICAL: เปลี่ยนการ Import ให้ชี้ไปที่ core/run_assessment.py
try:
    from core.run_assessment import run_assessment_process 
except ImportError as e:
    print(f"Error importing run_assessment_process: {e}")
    print("Please ensure core/run_assessment.py exists and the project root is in Python path.")
    sys.exit(1)

# -----------------------------
# --- Logging Setup ---
# -----------------------------
# ใช้ basicConfig เพื่อให้ logging ทำงานใน CLI script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("process_assessment")
logger.setLevel(logging.INFO)

# -----------------------------
# --- Main CLI Entry Point ---
# -----------------------------

if __name__ == "__main__":
    
    # Start global timer
    start_time_global = time.perf_counter()
    
    # 1. Setup Argparse (ใช้ Logic เดิมในการรับ Argument)
    parser = argparse.ArgumentParser(description="Automated Enabler Maturity Assessment System (CLI Runner).")
    parser.add_argument("--mode", choices=["mock", "random", "real"], default="real", help="Assessment mode ('mock', 'random', 'real').")
    parser.add_argument("--enabler", type=str, default="KM", help="Enabler abbreviation (e.g., 'KM', 'SCM').")
    parser.add_argument("--sub", type=str, default="all", help="Filter to a specific Sub-Criteria ID (e.g., '1.1') or 'all'.")
    parser.add_argument("--filter", action="store_true", help="Enable strict metadata filtering based on mapping data for RAG.")
    parser.add_argument("--export", action="store_true", help="Export the final summary results to a JSON file in the 'results' directory.")
    args = parser.parse_args()
    
    logger.info(f"CLI: Starting Assessment (Enabler: {args.enabler}, Sub: {args.sub}, Mode: {args.mode})")

    try:
        # 🚨 CRITICAL CHANGE: เรียกใช้ฟังก์ชันจากไฟล์ใหม่ (core/run_assessment.py)
        final_summary = run_assessment_process(
            enabler=args.enabler,
            sub_criteria_id=args.sub,
            mode=args.mode,
            filter_mode=args.filter,
            export=args.export
        )
        
        # 2. Output Summary (Print Final JSON to terminal)
        print("\n\n=============================================")
        print(f"===== FINAL ASSESSMENT SUMMARY ({args.enabler}/{args.sub}) =====")
        print("=============================================")
        
        # แสดงผลลัพธ์ในรูปแบบ JSON ที่อ่านง่าย
        print(json.dumps(final_summary, indent=4, ensure_ascii=False))

        # 3. Output Detailed Results (ลบการเรียกใช้ print_detailed_results ออก)
        # ผลลัพธ์โดยละเอียดถูกจัดการโดย run_assessment_process/print_detailed_results ภายในแล้ว
        
        # 4. Output Export Path (ถ้ามีการ Export)
        if args.export and 'export_path_used' in final_summary:
            logger.info(f"Report exported to: {final_summary['export_path_used']}")


    except Exception as e:
        logger.critical(f"A fatal error occurred during the assessment process: {e}")
        # Exit with error code
        sys.exit(1)


    # Stop global timer and print total
    end_time_global = time.perf_counter()
    global_duration = end_time_global - start_time_global
    
    # ดึงเวลา Execution Time ที่คำนวณใน core/run_assessment.py มาใช้
    if 'Execution_Time' in final_summary:
        global_duration = final_summary['Execution_Time'].get('total', global_duration)
        
    print(f"\n[⏱️ TOTAL PROCESS TIME] All processes completed in: {global_duration:.2f} seconds.")