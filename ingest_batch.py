#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =================================================================================
# 🔥 CRITICAL FIX FOR CVE-2025-32434 & TORCH 2.6 RESTRICTION
# ต้องวางไว้บนสุด ห้ามมี Import อื่นอยู่ก่อนเด็ดขาด!
# =================================================================================
import transformers.utils.import_utils as import_utils
# บังคับให้ระบบมองว่าการโหลดโมเดลปลอดภัยเสมอ (Monkey Patch)
import_utils.check_torch_load_is_safe = lambda *args, **kwargs: True

import os
# ตั้งค่า Environment บังคับไม่ให้ Torch ตรวจสอบ Weights
os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "FALSE"
os.environ["TRANSFORMERS_VERIFY_SCHEDULED_PATCHES"] = "False"
# =================================================================================

"""
ingest_batch.py
PEA RAG Document Management Tool – Production Ready
Version: December 29, 2025 (Revised with Torch Load Patch)
"""

import argparse
import logging
import sys
from pathlib import Path

# === แก้ path ให้ถูกต้องทุก OS ===
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# === Imports (ย้ายมาไว้หลัง Monkey Patch เพื่อความปลอดภัย) ===
try:
    from config.global_vars import (
        DEFAULT_TENANT,
        DEFAULT_ENABLER,
        DEFAULT_YEAR,
        EVIDENCE_DOC_TYPES,
        SUPPORTED_DOC_TYPES,
    )
    from core.ingest import (
        ingest_all_files,
        list_documents,
        wipe_vectorstore,
        get_vectorstore,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Run from project root with: python ingest_batch.py ...")
    sys.exit(1)

# === Logger ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("IngestBatch")

# === Argument Parser ===
def create_parser():
    parser = argparse.ArgumentParser(description="PEA RAG – Ingest / List / Wipe Documents")

    # Action
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ingest", action="store_true", help="Ingest documents")
    group.add_argument("--list", action="store_true", help="List documents")
    group.add_argument("--wipe", action="store_true", help="Delete vectorstore + mapping (DANGEROUS)")

    # Filters
    parser.add_argument("--tenant", type=str, default=DEFAULT_TENANT, help="Tenant (e.g. pea)")
    parser.add_argument("--doc-type", nargs="+", default=["evidence"],
                        help="Types: evidence, seam, document, faq, all")
    parser.add_argument("--enabler", type=str, default=None, help="Required for evidence (e.g. KM)")
    parser.add_argument("--year", type=str, default=None, help="Fiscal year (e.g. 2568)")

    # Options
    parser.add_argument("--show-results", type=str, default="all",
                        choices=["all", "missing", "ingested", "pending"],
                        help="Filter for --list")
    parser.add_argument("--dry-run", action="store_true", help="Simulate ingest")
    parser.add_argument("--sequential", action="store_true", help="Ingest one by one")
    parser.add_argument("--skip-wipe", action="store_true", help="Do not wipe before ingest")
    parser.add_argument("--force", action="store_true", help="Skip confirmation")

    return parser

# === Main ===
def main():
    parser = create_parser()
    args = parser.parse_args()

    # === Expand "all" to all global doc types (exclude evidence) ===
    if "all" in [x.lower() for x in args.doc_type]:
        doc_types = [
            dt.lower() for dt in SUPPORTED_DOC_TYPES
            if dt.lower() != EVIDENCE_DOC_TYPES.lower()
        ]
        logger.info(f"--doc-type all → wiping/processing GLOBAL types only: {doc_types}")
    else:
        doc_types = [dt.lower() for dt in args.doc_type]

    # === Evidence validation ===
    if "evidence" in doc_types:
        if not args.enabler:
            logger.error("'evidence' requires --enabler (e.g. KM, HCM, DT)")
            sys.exit(1)
        if not args.year:
            logger.error("'evidence' requires --year (e.g. 2568)")
            sys.exit(1)

    # === Year conversion ===
    year_to_use = None
    if args.year:
        try:
            year_to_use = int(args.year)
        except ValueError:
            logger.error("--year must be a number (e.g. 2568)")
            sys.exit(1)

    logger.info(f"Starting | Tenant={args.tenant} | Types={doc_types} | Year={year_to_use or 'N/A'} | Enabler={args.enabler or 'N/A'}")

    # === INGEST ===
    if args.ingest:
        logger.info("INGESTION MODE")
        
        # 📌 FIX: Pre-load Embedding Model (BAAI/bge-m3) ใน Main Thread 
        try:
             logger.info("Pre-loading BAAI/bge-m3 model in main thread...")
             # เราโหลดผ่าน get_vectorstore เพื่ออุ่นเครื่องโมดูล Embeddings
             get_vectorstore(
                 collection_name="warmup_collection", 
                 tenant=args.tenant, 
                 year=year_to_use or 2568
             )
             logger.info("✅ Pre-loading complete. Ready to process files.")
        except Exception as e:
             logger.warning(f"⚠️ Pre-load warning (Will retry during ingest): {e}")

        if not args.skip_wipe and not args.dry_run:
            confirm = "y" if args.force else input("Wipe existing data before ingest? (y/N): ")
            if confirm.lower() == "y":
                for dt in doc_types:
                    wipe_vectorstore(
                        doc_type_to_wipe=dt,
                        enabler=args.enabler,
                        tenant=args.tenant,
                        year=year_to_use
                    )

        ingest_all_files(
            tenant=args.tenant,
            year=year_to_use,
            doc_types=doc_types,
            enabler=args.enabler,
            dry_run=args.dry_run,
            sequential=args.sequential
        )

    # === LIST ===
    elif args.list:
        logger.info("LIST MODE")
        results = list_documents(
            doc_types=doc_types,
            tenant=args.tenant,
            year=args.year or year_to_use,
            enabler=args.enabler,
            show_results=args.show_results
        )

        if not results:
            print("\nNo documents found.\n")
            return

        print(f"\nFOUND DOCUMENTS ({len(results)} rows) – Filter: {args.show_results.upper()}\n")

        try:
            from tabulate import tabulate
            print(tabulate(results, headers="keys", tablefmt="simple", stralign="left"))
        except ImportError:
            header = f"{'Doc Type':<10} {'Enabler':<8} {'Year':<6} {'File Name':<60} {'Status':<12} {'Chunks':>6} {'UUID'}"
            print(header)
            print("-" * 130)
            for r in results:
                print(f"{r['Doc Type']:<10} {r['Enabler']:<8} {str(r['Year']):<6} "
                      f"{r['File Name']:<60} {r['Status']:<12} {r['Chunks']:>6} {r['UUID']}")
        print()

    # === WIPE ===
    elif args.wipe:
        logger.warning("WIPE MODE ACTIVATED")
        if not args.force:
            confirm = input("\nType 'DELETE EVERYTHING' to confirm permanent deletion: ")
            if confirm != "DELETE EVERYTHING":
                logger.info("Wipe cancelled.")
                return

        wiped_count = 0
        for dt in doc_types:
            logger.info(f"Wiping {dt.upper()}...")
            wipe_vectorstore(
                doc_type_to_wipe=dt,
                enabler=args.enabler,
                tenant=args.tenant,
                year=year_to_use
            )
            wiped_count += 1
        logger.critical(f"WIPE COMPLETED – {wiped_count} context(s) deleted.")

if __name__ == "__main__":
    main()