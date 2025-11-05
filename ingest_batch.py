#ingest_batch.py
import argparse
import logging
import sys
from typing import List, Dict, Any, Optional

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------- Import global vars and core functions --------------------
try:
    # Path fix
    import os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    # -------------------- Config --------------------
    from config.global_vars import (
        VECTORSTORE_DIR,
        SUPPORTED_DOC_TYPES,
        SUPPORTED_ENABLERS
    )

    # -------------------- Core logic --------------------
    from core.ingest import (
        ingest_all_files,
        list_documents,
        wipe_vectorstore,
        delete_document_by_uuid,
        get_vectorstore,
        get_target_dir
    )

except ImportError as e:
    logger.critical(f"Cannot import core modules: {e}")
    sys.exit(1)


# -------------------- Argument Parsing --------------------
parser = argparse.ArgumentParser(description="RAG Batch Ingestion & Vectorstore Management")
subparsers = parser.add_subparsers(dest="command", required=True)

# ingest
ingest_parser = subparsers.add_parser("ingest", help="Ingest files into vectorstore")
ingest_parser.add_argument(
    "doc_type",
    nargs='?',
    default="all",
    help=f"Document type to ingest (supported: {SUPPORTED_DOC_TYPES} or 'all')",
)
ingest_parser.add_argument(
    "--enabler",
    type=str,
    default=None,
    help=f"Specific enabler code for 'evidence' type (supported: {SUPPORTED_ENABLERS})",
)
ingest_parser.add_argument(
    "--skip_ext",
    nargs="*",
    default=None,
    help="Skip files with these extensions (e.g., .txt .csv)",
)
ingest_parser.add_argument(
    "--sequential",
    action="store_true",
    help="Run ingestion sequentially (no multiprocessing)",
)
ingest_parser.add_argument(
    "--log_every",
    type=int,
    default=50,
    help="Log progress every N files",
)

# wipe
wipe_parser = subparsers.add_parser("wipe", help="Wipe vectorstore collection(s)")
wipe_parser.add_argument(
    "doc_type",
    nargs='?',
    default="all",
    help=f"Document type to wipe (supported: {SUPPORTED_DOC_TYPES} or 'all')",
)
wipe_parser.add_argument(
    "--enabler",
    type=str,
    default=None,
    help=f"Specific enabler code for 'evidence' type (e.g., KM, L). Required if wiping a specific evidence collection.",
)

# list
list_parser = subparsers.add_parser("list", help="List all indexed documents")
list_parser.add_argument(
    "doc_type",
    nargs='?',
    default="all",
    help=f"Document type to list (supported: {SUPPORTED_DOC_TYPES} or 'all')",
)
list_parser.add_argument(
    "--enabler",
    type=str,
    default=None,
    help="Specific enabler code for 'evidence' type.",
)
list_parser.add_argument(
    "--show-results",
    type=str,
    default="ingested", 
    choices=["full", "ingested", "failed"],
    help="Filter documents to show: 'full' (all), 'ingested' (successful), or 'failed' (not ingested)",
)


# delete
delete_parser = subparsers.add_parser("delete", help="Delete a specific document by its Stable UUID")
delete_parser.add_argument(
    "stable_doc_uuid",
    type=str,
    help="Stable UUID of the document to delete (found via 'list' command)",
)
delete_parser.add_argument(
    "doc_type", # required to locate the correct vectorstore collection
    type=str,
    help=f"Document type where the UUID resides (supported: {SUPPORTED_DOC_TYPES})",
)
delete_parser.add_argument(
    "--enabler",
    type=str,
    default=None,
    help="Enabler code for 'evidence' type (e.g., KM, L). Required if doc_type is evidence.",
)

# -------------------- Main Logic --------------------
args = parser.parse_args()

if args.command == "list":
    doc_type_req = (args.doc_type.lower() if args.doc_type != "all" else None)
    doc_types = [doc_type_req] if doc_type_req else None
    
    logger.info(f"--- Listing documents for type '{args.doc_type}' (Enabler: {args.enabler or 'ALL'}) (Show: {args.show_results.upper()}) ---")
    
    list_documents(
        doc_types=doc_types, 
        enabler=args.enabler,
        show_results=args.show_results 
    )
    sys.exit(0)

elif args.command == "wipe":
    target = args.doc_type.lower()
    
    if target not in SUPPORTED_DOC_TYPES and target != "all":
        # Check if the user tried to pass a full collection name like 'evidence_km'
        if target.startswith('evidence_') and len(target.split('_')) == 2:
            base_type, enabler_code = target.split('_')
            if base_type == 'evidence':
                 logger.error(f"❌ Invalid doc_type '{target}'. Please use 'evidence' as the doc_type and provide '--enabler {enabler_code.upper()}' instead.")
                 logger.error(f"   Example: python ingest_batch.py wipe evidence --enabler {enabler_code.upper()}")
                 sys.exit(1)
        
        # Original error message
        logger.error(f"❌ Invalid doc_type '{target}'. Choose from ['all', {', '.join(SUPPORTED_DOC_TYPES)}]")
        sys.exit(1)

    logger.warning(f"⚠️ You are about to wipe collection(s) matching '{target}' (Enabler: {args.enabler or 'ALL'})")
    
    if input("Type 'YES' to confirm: ") == "YES":
        # Pass enabler argument to wipe_vectorstore
        wipe_vectorstore(target, args.enabler)
        logger.info("✅ Wipe finished")
    else:
        logger.info("Cancelled")
    sys.exit(0)

elif args.command == "delete":
    doc_type = args.doc_type.lower()
    uuid = args.stable_doc_uuid
    enabler = args.enabler
    
    if doc_type == 'evidence' and not enabler:
        logger.error(f"❌ Cannot delete evidence document. Must specify --enabler argument for doc_type '{doc_type}'")
        sys.exit(1)
        
    logger.warning(f"⚠️ You are about to delete document {uuid} in {doc_type} (Enabler: {enabler or '-'})")
    if input("Type 'YES' to confirm: ") == "YES":
        # Pass doc_type and enabler argument to delete_document_by_uuid
        success = delete_document_by_uuid(uuid, doc_type=doc_type, enabler=enabler)
        if success:
            logger.info(f"✅ Deleted {uuid}")
        else:
            logger.error("❌ Document not found or deletion failed")
    else:
        logger.info("Cancelled")
    sys.exit(0)

elif args.command == "ingest":
    doc_type = args.doc_type.lower()
    enabler = args.enabler
    
    # 🟢 FIX 1: บังคับให้ระบุ --enabler หาก doc_type เป็น 'evidence'
    if doc_type == 'evidence' and not enabler:
        logger.error(f"❌ Cannot ingest 'evidence' without specifying an enabler. Please use --enabler argument (e.g., --enabler KM).")
        logger.error(f"   Supported enablers: {SUPPORTED_ENABLERS}")
        sys.exit(1)
        
    # 🟢 FIX 2: ตรวจสอบความถูกต้องของ enabler ที่ป้อนมา (เผื่อไว้)
    if enabler and enabler.upper() not in SUPPORTED_ENABLERS:
         logger.error(f"❌ Invalid enabler code '{enabler}'. Supported enablers: {SUPPORTED_ENABLERS}")
         sys.exit(1)
    
    logger.info(f"--- Starting ingestion for '{doc_type}' (Enabler: {enabler or 'ALL'}) ---")
    
    # 🔹 กำหนดชื่อ Collection ที่ถูกต้องสำหรับการ Pre-load
    target_coll_name = "document"
    if doc_type != "all" and doc_type in SUPPORTED_DOC_TYPES:
        try:
            # 🟢 FIX 3: ใช้ get_target_dir เพื่อหาชื่อ Collection ที่ถูกต้อง (e.g., evidence_km)
            # ต้องมั่นใจว่า get_target_dir ถูก import และทำงานถูกต้องตามที่เราแก้ไขไปก่อนหน้า
            target_coll_name = get_target_dir(doc_type, enabler) 
        except ValueError as e:
            logger.error(f"❌ Cannot determine target collection name for pre-load: {e}")
            sys.exit(1) 

    # 🔹 Pre-load vectorstore (embedding model/service)
    try:
        # 🟢 FIX 4: ใช้ชื่อ Collection ที่คำนวณแล้วสำหรับ Pre-load
        target_coll = get_vectorstore(target_coll_name) 
    except Exception as e:
        # Log แสดงว่ากำลังโหลด 'evidence' แทนที่จะเป็น 'evidence_km' (Log ก่อนหน้า) จะหายไป
        logger.warning(f"Cannot preload vectorstore service for '{target_coll_name}': {e}")
        # Continue even if pre-load fails, as it will be loaded later

    # 🔹 Ingest
    results = ingest_all_files(
        doc_type=(None if doc_type == "all" else doc_type),
        enabler=enabler, # Pass enabler argument
        base_path=VECTORSTORE_DIR,
        skip_ext=args.skip_ext,
        sequential=args.sequential,
        log_every=args.log_every
    )
    
    logger.info("Ingestion process completed. Check ingest.log for details.")