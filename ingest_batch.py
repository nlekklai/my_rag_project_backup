# ingest_batch.py (เวอร์ชันแก้ไขล่าสุด: ปรับการลบ Mapping สำหรับ wipe all ให้เหลือเฉพาะ doc_id_mapping และ Vectorstore)

import argparse
import logging
import sys
import os
import shutil
from typing import Final, List, Dict, Any, Union

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
        DATA_STORE_ROOT,
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
    )
    
    # NEW: Import Path Utility สำหรับการแสดงผล Path/Key
    from utils.path_utils import (
        get_doc_type_collection_key,
        load_doc_id_mapping,
        save_doc_id_mapping,
        get_mapping_file_path # ✅ เพิ่ม get_mapping_file_path เพื่อใช้ในการ cleanup
    )


except ImportError as e:
    logger.critical(f"Cannot import core modules: {e}")
    if 'config.global_vars' in str(e):
        logger.critical("HINT: Ensure config/global_vars.py exists and defines DEFAULT_TENANT, DEFAULT_YEAR, etc.")
    elif 'core.ingest' in str(e):
        logger.critical("HINT: Ensure core/ingest.py exists and is correctly defined.")
    elif 'utils.path_utils' in str(e):
         logger.critical("HINT: Ensure utils/path_utils.py exists and is correctly defined.")
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
    "--year", type=str, default=str(DEFAULT_YEAR), 
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
    "--subject", type=str, default=None, 
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
    "--log_every", type=int, default=100, # NOTE: Argument นี้ถูกลบออกจากการเรียกใช้ ingest_all_files แล้ว
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
    "--year", type=str, default=str(DEFAULT_YEAR),
    help=f"Specify the year. Default: {DEFAULT_YEAR}"
)
wipe_parser.add_argument(
    "--doc_type", type=str, required=True,
    help=f"Document type to wipe. Supported: {', '.join(SUPPORTED_DOC_TYPES + ['all'])}."
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
    "--year", type=str, default=str(DEFAULT_YEAR),
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
    
    # กำหนด Year ที่จะใช้กรอง (ใช้ DEFAULT_YEAR ก็ต่อเมื่อ doc_type เป็น evidence และไม่มีการระบุปีมา)
    year_to_filter: Union[str, int, None] = args.year
    if doc_type_input == EVIDENCE_DOC_TYPES.lower() and not args.year:
        year_to_filter = DEFAULT_YEAR

    list_documents(
        tenant=args.tenant,
        # ส่ง year_to_filter ที่ถูกจัดการแล้ว (จะเป็น None หากไม่ใช่ evidence และไม่ได้ถูกระบุ)
        year=year_to_filter, 
        doc_types=[doc_type_input],
        enabler=args.enabler,
    )
    sys.exit(0)


# -------------------- COMMAND: delete --------------------
elif args.command == "delete":
    final_enabler = args.enabler if doc_type_input == EVIDENCE_DOC_TYPES.lower() else None
    
    # กำหนดปีเป็น None สำหรับ Global Doc Types
    if doc_type_input == EVIDENCE_DOC_TYPES.lower():
        # สำหรับ evidence ให้ใช้ year ที่ระบุ หรือ DEFAULT_YEAR
        year_to_delete = int(args.year) if args.year and str(args.year).isdigit() else DEFAULT_YEAR
    else:
        # สำหรับ doc_type อื่น ๆ (document, policy, manual) ให้ใช้ None
        year_to_delete = None 

    delete_document_by_uuid(
        tenant=args.tenant,
        year=year_to_delete, # ส่งเป็น None หรือ int
        doc_type=doc_type_input,
        enabler=final_enabler,
        stable_doc_uuid=args.doc_uuid,
        base_path=DATA_STORE_ROOT
    )
    sys.exit(0)


# -------------------- COMMAND: wipe --------------------
elif args.command == "wipe":
    logger.warning("!!! WARNING: You are about to wipe the entire Vector Store Collection !!!")
    
    # คำนวณ Year ที่จะใช้จริงสำหรับ WIPE และการแสดงผล
    tenant_clean = args.tenant.lower().replace(" ", "_")
    if doc_type_input == EVIDENCE_DOC_TYPES.lower():
        # สำหรับ evidence ให้ใช้ year ที่ระบุ หรือ DEFAULT_YEAR
        year_to_use: Union[int, None] = int(args.year) if args.year and args.year.isdigit() else DEFAULT_YEAR
        year_to_display = str(year_to_use)
    else:
        # สำหรับ doc_type อื่น ๆ ให้ใช้ None เพื่อระบุ Global/Common Collection
        year_to_use = None
        year_to_display = "Global" 

    # ใช้ get_doc_type_collection_key เพื่อคำนวณชื่อ Collection Key สำหรับการแสดงผล
    doc_type_key = get_doc_type_collection_key(doc_type_input, args.enabler)
    
    # เปลี่ยนการแสดงผลให้ชัดเจนขึ้นโดยใช้ Key
    wipe_path_display = f"Collection Key: {doc_type_key} (Tenant: {tenant_clean}, Year Context: {year_to_display})"
    
    logger.warning(f"Target: {wipe_path_display}")
    
    if not args.yes:
        confirmation = input("Type 'YES' (all caps) to confirm deletion: ")
        if confirmation != "YES":
            logger.info("Deletion cancelled.")
            sys.exit(0)

    # รัน Wipe จริง
    logger.info("Starting actual deletion...")

    wipe_vectorstore(
        doc_type_to_wipe=doc_type_input,
        enabler=args.enabler, 
        tenant=args.tenant, 
        year=year_to_use, # ส่ง year_to_use ที่เป็น None หรือ int
        base_path=DATA_STORE_ROOT,
    )
    logger.info("✅ Wipe completed.")
    
    # 🎯 FIX: ปรับ Logic Cleanup สำหรับ wipe all
    if doc_type_input == 'all':
        try:
            # 1. ลบโฟลเดอร์ Physical Data/Vector Store ของ Tenant/Year Context ทั้งหมด
            if year_to_use:
                # Target: DATA_STORE_ROOT/pea/2568
                target_cleanup_dir = os.path.join(DATA_STORE_ROOT, tenant_clean, str(year_to_use))
                shutil.rmtree(target_cleanup_dir, ignore_errors=True)
                logger.info(f"🗑️ Cleaned up physical data directory: {target_cleanup_dir}")
            
            # 2. ลบไฟล์ Doc ID Mapping ที่เกี่ยวข้อง
            
            # 📌 ลบ Mapping สำหรับ Evidence Doc Types (ถ้ามี Enabler และ Year)
            if args.enabler and year_to_use:
                mapping_file_path = get_mapping_file_path(
                    tenant=args.tenant, year=year_to_use, enabler=args.enabler
                )
                if os.path.exists(mapping_file_path):
                    os.remove(mapping_file_path)
                    logger.info(f"🗑️ Removed Evidence Mapping file: {os.path.basename(mapping_file_path)}")
            
            # 📌 ลบ Mapping สำหรับ Doc Types ทั่วไป/Global (ถ้า year_to_use เป็น None)
            elif year_to_use is None:
                # ลองดึง Mapping Path สำหรับ Global Doc ID (ไม่มี Enabler, ไม่มี Year)
                mapping_file_path = get_mapping_file_path(
                    tenant=args.tenant, year=None, enabler=None 
                )
                if os.path.exists(mapping_file_path):
                    os.remove(mapping_file_path)
                    logger.info(f"🗑️ Removed Global Doc ID Mapping file: {os.path.basename(mapping_file_path)}")
                    
            
        except Exception as e:
            logger.error(f"Error during post-wipe all cleanup: {e}")
            pass 
    else:
        # ข้อความเตือนสำหรับ doc_type อื่นๆ ที่ไฟล์ Mapping ร่วมยังคงอยู่
        logger.info(f"ℹ️ Mapping file remains. It is used for other Global Doc Types. Run 'wipe --doc_type all' to clean up all physical files/mappings for the specified tenant/year.")

    sys.exit(0)

# -------------------- COMMAND: ingest --------------------
elif args.command == "ingest":
    
    if args.doc_type.lower() != EVIDENCE_DOC_TYPES.lower() and args.year and args.year != str(DEFAULT_YEAR):
        logger.warning(f"⚠️ Warning: Year '{args.year}' provided for doc_type='{doc_type_input}'. Year is usually ignored for non-evidence types.")

    logger.info(f"Starting ingestion → tenant: {args.tenant}, year: {args.year}, type: {doc_type_input}, enabler: {args.enabler or 'ALL'}, subject: {args.subject or 'None'}") 
    logger.info(f"Dry run: {args.dry_run} | Sequential: {args.sequential} | Debug: {args.debug}")

    # ตรวจสอบและแปลงปีเป็น int เมื่อมีค่า
    year_to_ingest: Union[int, None] = int(args.year) if args.year and str(args.year).isdigit() else None
    
    # สำหรับ Global Doc Type ให้ year เป็น None
    if doc_type_input != EVIDENCE_DOC_TYPES.lower():
         year_to_ingest = None

    # 🎯 FIX 1: สร้าง List ของ Document Types ที่จะ Ingest
    # ใช้ SUPPORTED_DOC_TYPES ถ้า doc_type เป็น "all" ไม่เช่นนั้นให้ใช้ doc_type ที่ระบุเป็น List
    doc_types_to_ingest = SUPPORTED_DOC_TYPES if doc_type_input == "all" else [doc_type_input]

    # ลบ Argument ที่เกินมา 3 ตัว (data_dir, base_path, debug)
    results: List[Dict[str, Any]] = ingest_all_files( 
        doc_types=doc_types_to_ingest, # 🟢 FIX: ใช้ doc_types แทน doc_type
        tenant=args.tenant,
        year=year_to_ingest, 
        enabler=args.enabler,
        subject=args.subject, 
        skip_ext=args.skip_ext,
        sequential=args.sequential,
        # 🔴 ลบ log_every=args.log_every ออกไป
        dry_run=args.dry_run,
    )

    total = len(results)
    success = 0
    failed = 0
    
    if isinstance(results, list):
        # NOTE: การนับผลลัพธ์ควรปรับตามโครงสร้างผลลัพธ์จริงของ ingest_all_files
        success = sum(1 for status_dict in results if 'chunks' in status_dict and status_dict.get('chunks', 0) > 0)
        # เนื่องจากโค้ดของคุณไม่ได้มี 'status' == 'chunked' ผมจะใช้ 'chunks' > 0
        failed = total - success
    else:
        logger.error(f"❌ Cannot calculate summary: 'results' expected list, got {type(results)}. Assuming 0 successes.")
        # total ในที่นี้อาจเป็นจำนวนไฟล์ที่ถูกสแกนทั้งหมด
        failed = total 
        
    logger.info("-" * 50)
    logger.info(f"🔥 INGESTION SUMMARY: {doc_type_input.upper()} ({args.enabler or 'ALL'})")
    logger.info(f"Tenant/Year: {args.tenant.upper()}/{args.year or 'N/A'}")
    logger.info(f"Total files scanned: {total}") # NOTE: ต้องมั่นใจว่า results มีจำนวนเท่ากับ files_to_ingest
    logger.info(f"✅ Successfully chunked: {success}")
    logger.info(f"❌ Failed or skipped chunking: {failed}")
    logger.info("-" * 50)
    
    if failed > 0:
        logger.error("Some files failed to chunk/process. Please review the logs above.")
    
    sys.exit(0)

else:
    parser.print_help()
    sys.exit(1)