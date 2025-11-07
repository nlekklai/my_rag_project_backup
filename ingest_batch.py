import argparse
import logging
import sys
import os
from typing import Optional
from pathlib import Path as SysPath

# -------------------- Logging --------------------
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

    from config.global_vars import (
        DATA_DIR,
        VECTORSTORE_DIR,
        SUPPORTED_DOC_TYPES,
        SUPPORTED_ENABLERS,
    )

    from core.ingest import (
        ingest_all_files,
        list_documents,
        wipe_vectorstore,
        delete_document_by_uuid,
        get_vectorstore,
        get_target_dir,
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
    nargs="?",
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
    nargs="?",
    default="all",
    help=f"Document type to wipe (supported: {SUPPORTED_DOC_TYPES} or 'all')",
)
wipe_parser.add_argument(
    "--enabler",
    type=str,
    default=None,
    help=f"Specific enabler code for 'evidence' type (e.g., KM, L).",
)

# list
list_parser = subparsers.add_parser("list", help="List all indexed documents")
list_parser.add_argument(
    "doc_type",
    nargs="?",
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
    help="Filter documents to show: 'full', 'ingested', or 'failed'",
)

# delete
delete_parser = subparsers.add_parser("delete", help="Delete a specific document by its Stable UUID")
delete_parser.add_argument(
    "stable_doc_uuid",
    type=str,
    help="Stable UUID of the document to delete (found via 'list' command)",
)
delete_parser.add_argument(
    "doc_type",
    type=str,
    help=f"Document type where the UUID resides (supported: {SUPPORTED_DOC_TYPES})",
)
delete_parser.add_argument(
    "--enabler",
    type=str,
    default=None,
    help="Enabler code for 'evidence' type (required if doc_type is evidence).",
)

# -------------------- Main Logic --------------------
args = parser.parse_args()

# -------------------- LIST --------------------
if args.command == "list":
    doc_type_req = (args.doc_type.lower() if args.doc_type != "all" else None)
    doc_types = [doc_type_req] if doc_type_req else None
    logger.info(f"--- Listing documents for type '{args.doc_type}' (Enabler: {args.enabler or 'ALL'}) ---")
    list_documents(
        doc_types=doc_types,
        enabler=args.enabler,
        show_results=args.show_results
    )
    sys.exit(0)

# -------------------- WIPE --------------------
elif args.command == "wipe":
    target = args.doc_type.lower()
    if target not in SUPPORTED_DOC_TYPES and target != "all":
        logger.error(f"❌ Invalid doc_type '{target}'. Choose from ['all', {', '.join(SUPPORTED_DOC_TYPES)}]")
        sys.exit(1)

    logger.warning(f"⚠️ You are about to wipe collection(s) matching '{target}' (Enabler: {args.enabler or 'ALL'})")
    if input("Type 'YES' to confirm: ") == "YES":
        wipe_vectorstore(target, args.enabler)
        logger.info("✅ Wipe finished")
    else:
        logger.info("Cancelled")
    sys.exit(0)

# -------------------- DELETE --------------------
elif args.command == "delete":
    doc_type = args.doc_type.lower()
    uuid = args.stable_doc_uuid
    enabler = args.enabler

    if doc_type == "evidence" and not enabler:
        logger.error("❌ Must specify --enabler for evidence deletion")
        sys.exit(1)

    logger.warning(f"⚠️ Deleting document {uuid} in {doc_type} (Enabler: {enabler or '-'})")
    if input("Type 'YES' to confirm: ") == "YES":
        success = delete_document_by_uuid(uuid, doc_type=doc_type, enabler=enabler)
        if success:
            logger.info(f"✅ Deleted {uuid}")
        else:
            logger.error("❌ Document not found or deletion failed")
    else:
        logger.info("Cancelled")
    sys.exit(0)

# -------------------- INGEST --------------------
elif args.command == "ingest":
    doc_type = args.doc_type.lower()
    enabler = args.enabler

    # 📌 FIX: ตรวจสอบเฉพาะกรณีที่เป็น doc_type 'evidence' และไม่ได้เลือก 'all'
    if doc_type == "evidence" and not enabler:
        logger.error("❌ Must specify --enabler for evidence ingestion")
        sys.exit(1)

    logger.info(f"--- Starting Ingestion: doc_type='{doc_type}' (Enabler: {enabler or 'ALL'}) ---")

    # 🟢 FIX: เรียก ingest_all_files เพียงครั้งเดียว 
    # โดยฟังก์ชันจะจัดการการสแกนโฟลเดอร์ย่อยทั้งหมดภายใน DATA_DIR เอง
    ingest_all_files(
        doc_type=doc_type,
        enabler=enabler,
        data_dir=DATA_DIR,          # ⬅️ Source Files Base Path
        base_path=VECTORSTORE_DIR,  # ⬅️ Vector Store Base Path (แก้ไขปัญหา Vector Store Path)
        skip_ext=args.skip_ext,
        sequential=args.sequential,
        log_every=args.log_every
    )

    logger.info("✅ Ingestion process completed. Check ingest.log for details.")
    sys.exit(0)