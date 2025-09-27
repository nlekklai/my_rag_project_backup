import os
import shutil
from core.ingest import process_document

VECTORSTORE_DIR = "vectorstore"
DATA_DIR = "data"

def clear_vectorstore():
    if os.path.exists(VECTORSTORE_DIR):
        for item in os.listdir(VECTORSTORE_DIR):
            item_path = os.path.join(VECTORSTORE_DIR, item)
            if os.path.isdir(item_path):
                try:
                    print(f"Deleting: {item_path}")
                    shutil.rmtree(item_path)
                except Exception as e:
                    print(f"⚠️ Failed to delete {item_path}: {e}")
    else:
        os.makedirs(VECTORSTORE_DIR)
    print("✅ Vectorstore cleared.")

def rebuild_vectorstore():
    if not os.path.exists(DATA_DIR):
        print(f"❌ No data folder found at '{DATA_DIR}'. Aborting.")
        return

    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    if not files:
        print("❌ No files found in data folder. Aborting.")
        return

    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        try:
            doc_id = process_document(file_path, file)
            print(f"✅ Processed {file} -> doc_id: {doc_id}")
        except Exception as e:
            print(f"⚠️ Error processing {file}: {e}")

if __name__ == "__main__":
    clear_vectorstore()
    rebuild_vectorstore()
    print("🎉 Vectorstore rebuild complete!")
