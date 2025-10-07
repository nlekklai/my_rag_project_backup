# utils/ingest_by_type.py
import os
import sys

# เพิ่ม root project ให้ Python หา core ได้
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ingest import ingest_all_files, DATA_DIR

def main():
    folder_type = "document"
    folder_path = os.path.join(DATA_DIR, folder_type)

    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        return

    print(f"📂 Ingesting files from folder: {folder_path}")

    results = ingest_all_files(
        data_dir=folder_path,
        exclude_dirs=set(),
        sequential=True,
        version="v1"
    )

    print("\n✅ Ingestion results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    main()
