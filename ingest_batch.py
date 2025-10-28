#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# -------------------- Import core functions --------------------
try:
    from core.ingest import (
        ingest_all_files,
        list_documents,
        wipe_vectorstore,
        delete_document_by_uuid,
        VECTORSTORE_DIR,
        get_vectorstore,
        SUPPORTED_DOC_TYPES,
    )
except ImportError as e:
    logger.critical(f"Cannot import core.ingest: {e}")
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
    help=f"Document type to ingest (supported: {SUPPORTED_DOC_TYPES})",
)

# list
subparsers.add_parser("list", help="List all documents and status")

# wipe
wipe_parser = subparsers.add_parser("wipe", help="Wipe vectorstore")
wipe_parser.add_argument(
    "doc_type",
    nargs='?',
    default="all",
    help="Document type to wipe ('all' for all types)",
)

# delete
delete_parser = subparsers.add_parser("delete", help="Delete document by UUID")
delete_parser.add_argument("doc_type", help="Document collection")
delete_parser.add_argument("stable_doc_uuid", help="Stable UUID of document to delete")

args = parser.parse_args()

# -------------------- Command Handling --------------------
if args.command == "list":
    list_documents()
    sys.exit(0)

elif args.command == "wipe":
    target = args.doc_type.lower()
    logger.warning(f"⚠️ You are about to wipe '{target}' collection(s)")
    if input("Type 'YES' to confirm: ") == "YES":
        wipe_vectorstore(target)
        logger.info("✅ Wipe finished")
    else:
        logger.info("Cancelled")
    sys.exit(0)

elif args.command == "delete":
    doc_type = args.doc_type.lower()
    uuid = args.stable_doc_uuid
    logger.warning(f"⚠️ You are about to delete document {uuid} in {doc_type}")
    if input("Type 'YES' to confirm: ") == "YES":
        success = delete_document_by_uuid(uuid, doc_type, VECTORSTORE_DIR)
        if success:
            logger.info(f"✅ Deleted {uuid}")
        else:
            logger.error("❌ Document not found or deletion failed")
    else:
        logger.info("Cancelled")
    sys.exit(0)

elif args.command == "ingest":
    doc_type = args.doc_type.lower()
    logger.info(f"--- Starting ingestion for '{doc_type}' ---")
    
    # 🔹 Pre-load vectorstore (embedding model)
    try:
        vector_service = get_vectorstore()  # Prepare model/service
    except Exception as e:
        logger.warning(f"Cannot preload vectorstore service: {e}")
        vector_service = None
    
    ingest_all_files(doc_type=(None if doc_type=="all" else doc_type), base_path=VECTORSTORE_DIR)
    
    if vector_service:
        try:
            vector_service.close()
        except Exception:
            pass
    
    logger.info("--- Ingestion finished ---")
    sys.exit(0)
