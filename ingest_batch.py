# ingest_batch.py (เวอร์ชันแก้ไขล่าสุด: แก้ไขปัญหา list command และ Default Year)

import argparse
import logging
import sys
import os
import shutil
from typing import Final, List, Dict, Any

# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------- Import project modules --------------------
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # ต้องมั่นใจว่าไฟล์ config.global_vars มีการกำหนดค่าเหล่านี้
    from config.global_vars import (
        DATA_DIR,
        VECTORSTORE_DIR,
        SUPPORTED_DOC_TYPES,
        SUPPORTED_ENABLERS,
        EVIDENCE_DOC_TYPES,
        DEFAULT_TENANT,  
        DEFAULT_YEAR,    
    )
    
    # ต้องมั่นใจว่าไฟล์ core/ingest.py มีฟังก์ชันเหล่านี้
    from core.ingest import (
        ingest_all_files,
        list_documents,
        wipe_vectorstore,
        delete_document_by_uuid,
        get_target_dir # สำหรับการคำนวณพาธแสดงผล
    )

except ImportError as e:
    logger.critical(f"Cannot import core modules: {e}")
    if 'config.global_vars' in str(e):
        logger.critical("HINT: Ensure config/global_vars.py exists and defines DEFAULT_TENANT, DEFAULT_YEAR, etc.")
    elif 'core.ingest' in str(e):
        logger.critical("HINT: Ensure core/ingest.py exists and is correctly defined.")
    sys.exit(1)


# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser(description="RAG Batch Ingestion & Vectorstore Management")
subparsers = parser.add_subparsers(dest="command", required=True)


# --- 1. ingest ---
ingest_parser = subparsers.add_parser("ingest", help="Ingest files into vectorstore")
ingest_parser.add_argument(
    "--tenant", type=str, default=DEFAULT_TENANT,
    help=f"Specify the tenant (e.g., pea, pwa). Default: {DEFAULT_TENANT}"
)
ingest_parser.add_argument(
    # NOTE: ใช้ type=str เพื่อรับค่า 2568, 2569 ได้ แต่ต้องแปลงเป็น int ก่อนส่งให้ core/ingest
    "--year", type=str, default=DEFAULT_YEAR,
    help=f"Specify the year (e.g., 2567, 2568). Default: {DEFAULT_YEAR} (Only applies to 'evidence')."
)
ingest_parser.add_argument(
    "--doc_type", type=str, default="all",
    help=f"Document type to ingest. Supported: {', '.join(SUPPORTED_DOC_TYPES + ['all'])}. Default: all"
)
ingest_parser.add_argument(
    "--enabler", type=str,
    help=f"Enabler to ingest (Required for doc_type='evidence'). Supported: {', '.join(SUPPORTED_ENABLERS)}."
)
ingest_parser.add_argument(
    "--subject", type=str, default=None, # 🟢 เพิ่ม subject argument
    help="Subject/Topic for Global Doc Types (e.g., 'HR Policy')."
)
ingest_parser.add_argument(
    "--skip_ext", type=str, nargs='+', default=[],
    help="File extensions to skip (e.g., .jpg .png)."
)
ingest_parser.add_argument(
    "--sequential", action="store_true",
    help="Ingest files sequentially (single-threaded) for easier debugging."
)
ingest_parser.add_argument(
    "--dry_run", action="store_true",
    help="Only scan and log, do not perform ingestion."
)
ingest_parser.add_argument(
    "--log_every", type=int, default=100,
    help="Log progress every N files."
)
ingest_parser.add_argument(
    "--debug", action="store_true", 
    help="Enable debug logging and stable document ID creation."
)


# --- 2. list ---
list_parser = subparsers.add_parser("list", help="List all documents in vectorstore collection")
list_parser.add_argument(
    "--tenant", type=str, default=DEFAULT_TENANT,
    help=f"Specify the tenant. Default: {DEFAULT_TENANT}"
)
list_parser.add_argument(
    # 🟢 FIX 1: เปลี่ยน default เป็น None เพื่อไม่ให้ใช้ year filter โดยไม่จำเป็น
    "--year", type=str, default=None,
    help="Specify the year. Default: None (If doc_type is NOT evidence, year is ignored/not required)."
)
list_parser.add_argument(
    "--doc_type", type=str, required=True,
    help=f"Document type to list. Supported: {', '.join(SUPPORTED_DOC_TYPES)}."
)
list_parser.add_argument(
    "--enabler", type=str,
    help=f"Enabler to list (Required for doc_type='evidence'). Supported: {', '.join(SUPPORTED_ENABLERS)}."
)
list_parser.add_argument(
    "--debug", action="store_true", 
    help="Enable debug logging and stable document ID creation."
)


# --- 3. wipe ---
wipe_parser = subparsers.add_parser("wipe", help="Wipe (Delete) vectorstore collection or files")
wipe_parser.add_argument(
    "--tenant", type=str, default=DEFAULT_TENANT,
    help=f"Specify the tenant. Default: {DEFAULT_TENANT}"
)
wipe_parser.add_argument(
    "--year", type=str, default=DEFAULT_YEAR,
    help=f"Specify the year. Default: {DEFAULT_YEAR}"
)
wipe_parser.add_argument(
    "--doc_type", type=str, required=True,
    help=f"Document type to wipe. Supported: {', '.join(SUPPORTED_DOC_TYPES)}."
)
wipe_parser.add_argument(
    "--enabler", type=str,
    help=f"Enabler to wipe (Required for doc_type='evidence'). Supported: {', '.join(SUPPORTED_ENABLERS)}."
)
wipe_parser.add_argument(
    "--yes",
    action="store_true",
    help="Bypass confirmation prompt for wiping (DANGER: use only when sure!)",
)
wipe_parser.add_argument(
    "--debug", action="store_true", 
    help="Enable debug logging and stable document ID creation."
)


# --- 4. delete ---
delete_parser = subparsers.add_parser("delete", help="Delete a specific document by its UUID from the vectorstore")
delete_parser.add_argument(
    "--tenant", type=str, default=DEFAULT_TENANT,
    help=f"Specify the tenant. Default: {DEFAULT_TENANT}"
)
delete_parser.add_argument(
    "--year", type=str, default=DEFAULT_YEAR,
    help=f"Specify the year. Default: {DEFAULT_YEAR}"
)
delete_parser.add_argument(
    "--doc_type", type=str, required=True,
    help=f"Document type containing the document. Supported: {', '.join(SUPPORTED_DOC_TYPES)}."
)
delete_parser.add_argument(
    "--enabler", type=str,
    help=f"Enabler containing the document (Required for doc_type='evidence'). Supported: {', '.join(SUPPORTED_ENABLERS)}."
)
delete_parser.add_argument(
    "doc_uuid", type=str,
    help="The full 64-character Stable Document UUID to delete."
)
delete_parser.add_argument(
    "--debug", action="store_true", 
    help="Enable debug logging and stable document ID creation."
)


# -------------------- Main Execution --------------------

args = parser.parse_args()

# --- Pre-Command Validation ---
doc_type_input = args.doc_type.lower() if hasattr(args, 'doc_type') else None
if doc_type_input and doc_type_input != "all" and doc_type_input not in [dt.lower() for dt in SUPPORTED_DOC_TYPES]:
    logger.error(f"Invalid doc_type: {doc_type_input}. Supported: {SUPPORTED_DOC_TYPES}")
    sys.exit(1)

# Check enabler for 'evidence' type (applies to ingest, list, wipe, delete)
if doc_type_input == EVIDENCE_DOC_TYPES.lower() and args.command in ["ingest", "list", "wipe", "delete"] and not args.enabler:
    logger.error(f"When using '{EVIDENCE_DOC_TYPES.lower()}', you must specify --enabler {', '.join(SUPPORTED_ENABLERS)}.")
    sys.exit(1)

# ตั้งค่า Logging Level ถ้าใช้ --debug
if hasattr(args, 'debug') and args.debug:
    logger.setLevel(logging.DEBUG)


# -------------------- COMMAND: list --------------------
if args.command == "list":
    
    # 🟢 FIX 2: กำหนด Year ที่จะใช้กรอง (ใช้ DEFAULT_YEAR ก็ต่อเมื่อ doc_type เป็น evidence และไม่มีการระบุปีมา)
    year_to_filter = args.year
    if doc_type_input == EVIDENCE_DOC_TYPES.lower() and not args.year:
        year_to_filter = DEFAULT_YEAR

    list_documents(
        tenant=args.tenant,
        # 🟢 ส่ง year_to_filter ที่ถูกจัดการแล้ว (จะเป็น None หากไม่ใช่ evidence และไม่ได้ถูกระบุ)
        year=year_to_filter, 
        doc_types=[doc_type_input],
        enabler=args.enabler,
    )
    sys.exit(0)


# -------------------- COMMAND: delete --------------------
elif args.command == "delete":
    final_enabler = args.enabler if doc_type_input == EVIDENCE_DOC_TYPES.lower() else None
    
    delete_document_by_uuid(
        tenant=args.tenant,
        year=args.year,
        doc_type=doc_type_input,
        enabler=final_enabler,
        doc_uuid_to_delete=args.doc_uuid,
        base_path=VECTORSTORE_DIR
    )
    sys.exit(0)


# -------------------- COMMAND: wipe --------------------
elif args.command == "wipe":
    logger.warning("!!! WARNING: You are about to wipe the entire Vector Store Collection !!!")
    
    # คำนวณพาธสำหรับแสดงผล (อิงจากตรรกะใน core/ingest.py)
    doc_type_key = get_target_dir(doc_type_input, args.enabler)
    tenant_clean = args.tenant.lower().replace(" ", "_")
    
    # กำหนดส่วนของ Year หรือ Common
    year_or_common = str(args.year)
    if doc_type_input != EVIDENCE_DOC_TYPES.lower():
         year_or_common = "common" 
    
    # ✅ แก้ไข: ลบ "gov_tenants" ออก เนื่องจาก VECTORSTORE_DIR มีอยู่แล้ว
    wipe_path_display = os.path.join(VECTORSTORE_DIR, tenant_clean, year_or_common, doc_type_key)
    
    logger.warning(f"Target Collection Path (based on arguments): {wipe_path_display}")
    
    if not args.yes:
        confirmation = input("Type 'YES' (all caps) to confirm deletion: ")
        if confirmation != "YES":
            logger.info("Deletion cancelled.")
            sys.exit(0)

    # รัน Wipe จริง
    logger.info("Starting actual deletion...")
    wipe_vectorstore(
        tenant=args.tenant, 
        year=args.year,
        doc_type_to_wipe=doc_type_input,
        enabler=args.enabler,
        base_path=VECTORSTORE_DIR,
    )
    logger.info("✅ Wipe completed.")
    
    # 💡 Cleanup: พยายามลบโฟลเดอร์เปล่าที่เหลืออยู่ (ถ้า wipe all)
    if doc_type_input == 'all':
        try:
            # ✅ แก้ไข: ลบ "gov_tenants" ออก
            target_cleanup_dir = os.path.join(VECTORSTORE_DIR, tenant_clean, year_or_common)
            shutil.rmtree(target_cleanup_dir, ignore_errors=True)
            logger.info(f"Cleaned up empty directory: {target_cleanup_dir}")
        except Exception:
             pass 

    sys.exit(0)

# -------------------- COMMAND: ingest --------------------
elif args.command == "ingest":
    # 🎯 NOTE: ต้องมั่นใจว่าใน core/ingest.py มีการตรวจสอบ args.year ก่อนแปลงเป็น int
    if args.doc_type.lower() != EVIDENCE_DOC_TYPES.lower() and args.year and args.year != DEFAULT_YEAR:
        logger.warning(f"⚠️ Warning: Year '{args.year}' provided for doc_type='{doc_type_input}'. Year is usually ignored for non-evidence types.")
    
    logger.info(f"Starting ingestion → tenant: {args.tenant}, year: {args.year}, type: {doc_type_input}, enabler: {args.enabler or 'ALL'}, subject: {args.subject or 'None'}") # 🟢 Log subject
    logger.info(f"Dry run: {args.dry_run} | Sequential: {args.sequential} | Debug: {args.debug}")

    # 🟢 ตรวจสอบและแปลงปีเป็น int เมื่อมีค่า
    year_to_ingest = int(args.year) if args.year else None

    results: List[Dict[str, Any]] = ingest_all_files( # กำหนด Type Hint ให้ชัดเจนว่าเป็น List
        tenant=args.tenant,
        year=year_to_ingest, 
        doc_type=None if doc_type_input == "all" else doc_type_input,
        enabler=args.enabler,
        subject=args.subject, # 🟢 ส่ง subject ที่ถูกรับเข้ามา
        data_dir=DATA_DIR,
        base_path=VECTORSTORE_DIR,
        skip_ext=args.skip_ext,
        sequential=args.sequential,
        log_every=args.log_every,
        dry_run=args.dry_run,
        debug=args.debug,
    )

    total = len(results)
    success = 0
    failed = 0
    
    # 🎯 FINAL FIX: ปรับ Logic การนับให้วนซ้ำใน List of Dictionaries
    if isinstance(results, list):
        # 🟢 นับจำนวนรายการที่สถานะเป็น 'chunked'
        success = sum(1 for status_dict in results if status_dict.get('status') == 'chunked')
        failed = total - success
    else:
        # ❌ จัดการกรณีที่ผลลัพธ์ไม่ใช่ List (ไม่ควรเกิดขึ้นแล้ว)
        logger.error(f"❌ Cannot calculate summary: 'results' expected list, got {type(results)}. Assuming 0 successes.")
        failed = total # ถ้า total > 0
        
    logger.info("-" * 50)
    logger.info(f"🔥 INGESTION SUMMARY: {doc_type_input.upper()} ({args.enabler or 'ALL'})")
    logger.info(f"Tenant/Year: {args.tenant.upper()}/{args.year or 'N/A'}")
    logger.info(f"Total files scanned: {total}")
    logger.info(f"✅ Successfully chunked: {success}")
    logger.info(f"❌ Failed or skipped chunking: {failed}")
    logger.info("-" * 50)
    
    if failed > 0:
        logger.error("Some files failed to chunk/process. Please review the logs above.")
    
    sys.exit(0)

else:
    parser.print_help()
    sys.exit(1)