import os
import shutil
import logging
import sys
import glob
from typing import Dict, Set

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# -------------------------------------------------

SUPPORTED_EXTENSIONS = ('.pdf', '.docx', '.png', '.jpg', '.xlsx')


def copy_not_ingested_files_by_fullname(
    status_filename: str = "export_ingest_status.txt",
    source_root_dir: str = "data/evidence_km/",
    target_dir: str = "data/evidence_test/"
):
    EXPECTED_INGESTED_COUNT = 109

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        logger.info(f"✅ Created target directory: {target_dir}")

    # --- 1. Gather all source file fullnames ---
    source_full_names_to_paths: Dict[str, str] = {}
    for full_path in glob.glob(os.path.join(source_root_dir, '**', '*'), recursive=True):
        if os.path.isdir(full_path):
            continue
        fname = os.path.basename(full_path)
        if fname.lower().endswith(SUPPORTED_EXTENSIONS):
            source_full_names_to_paths[fname] = full_path

    source_full_names = set(source_full_names_to_paths.keys())
    logger.info(f"Prepared **{len(source_full_names)}** unique full filenames from ALL source files.")

    # --- 2. Parse status file ---
    ingested_full_names: Set[str] = set()
    ingested_count = 0

    STATUS_INDEX = 6
    FILENAME_INDEX = 1
    EXT_INDEX = 2

    try:
        with open(status_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data_lines = lines[6:]  # skip headers (line 1-6)
        for i, line in enumerate(data_lines, start=7):
            line = line.strip()
            if not line:
                continue

            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) < STATUS_INDEX + 1:
                continue

            status_raw = parts[STATUS_INDEX].upper()
            if status_raw == 'INGESTED':
                filename_base = parts[FILENAME_INDEX]
                extension = parts[EXT_INDEX]
                full_filename = f"{filename_base}.{extension}"
                ingested_full_names.add(full_filename)
                ingested_count += 1
            elif i < 12:
                logger.warning(f"DEBUG: Line {i} Status Read: '{status_raw}'")

    except FileNotFoundError:
        logger.critical(f"❌ Error: Status file '{status_filename}' not found.")
        sys.exit(1)

    logger.info(f"Found **{len(ingested_full_names)}** unique full filenames based on 'Ingested' status. (Total records checked: {ingested_count})")

    # --- 3. Diff ---
    if len(ingested_full_names) != EXPECTED_INGESTED_COUNT:
        logger.warning("-" * 80)
        logger.warning(f"⚠️ SAFETY CHECK SKIPPED: Expected {EXPECTED_INGESTED_COUNT}, but found {len(ingested_full_names)}.")
        logger.warning("-" * 80)

    not_ingested = source_full_names - ingested_full_names
    logger.info(f"--- Calculated **{len(not_ingested)}** files NOT Ingested. ---")

    # --- 4. Copy ---
    copied_count = 0
    for fn in not_ingested:
        src = source_full_names_to_paths.get(fn)
        if not src:
            continue
        dst = os.path.join(target_dir, fn)
        try:
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                copied_count += 1
        except Exception as e:
            logger.error(f"❌ Failed to copy {fn}: {e}")

    logger.info("-" * 80)
    logger.info(f"Summary: Successfully copied **{copied_count}** files (Not Ingested) to '{target_dir}'.")
    logger.info("-" * 80)


if __name__ == "__main__":
    copy_not_ingested_files_by_fullname()
