#check_vectorstores.py
import os

VECTORSTORE_DIR = "vectorstore"

def check_vectorstore(vectorstore_dir=VECTORSTORE_DIR):
    summary = {}

    # วน doc_type แต่ละ folder
    for doc_type in os.listdir(vectorstore_dir):
        doc_type_path = os.path.join(vectorstore_dir, doc_type)
        if not os.path.isdir(doc_type_path):
            continue
        
        vector_folders = [f for f in os.listdir(doc_type_path) 
                          if os.path.isdir(os.path.join(doc_type_path, f))]
        total = len(vector_folders)
        ready = 0
        missing = []

        for doc_id in vector_folders:
            doc_path = os.path.join(doc_type_path, doc_id)
            # ตรวจสอบไฟล์สำคัญ เช่น chroma.sqlite3 หรือ data_level0.bin
            chroma_file = os.path.join(doc_path, "chroma.sqlite3")
            data_file = os.path.join(doc_path, "data_level0.bin")
            if os.path.exists(chroma_file) or os.path.exists(data_file):
                ready += 1
            else:
                missing.append(doc_id)
        
        summary[doc_type] = {
            "total_vectors": total,
            "ready": ready,
            "missing_or_broken": len(missing),
            "missing_ids": missing
        }

    return summary

if __name__ == "__main__":
    result = check_vectorstore()
    for doc_type, stats in result.items():
        print(f"Doc Type: {doc_type}")
        print(f"  Total Vectors: {stats['total_vectors']}")
        print(f"  Ready: {stats['ready']}")
        print(f"  Missing/Broken: {stats['missing_or_broken']}")
        if stats['missing_ids']:
            print(f"    IDs: {stats['missing_ids']}")
        print("-" * 40)
