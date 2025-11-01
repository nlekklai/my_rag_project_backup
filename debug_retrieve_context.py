import logging
from core.vectorstore import get_vectorstore_manager # ต้องมั่นใจว่า vectorstore_manager.py อยู่ใน PATH ที่ถูกค้นพบ

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("--- 🔍 Debug Chroma Collection Status ---")
    
    # NOTE: ต้องมั่นใจว่าไฟล์ vectorstore_manager.py สามารถเข้าถึง get_vectorstore_manager ได้
    try:
        manager = get_vectorstore_manager()
    except Exception as e:
        print(f"❌ ERROR: Failed to get vectorstore manager instance. Check imports/paths. Error: {e}")
        exit()
        
    print("\n📦 All Available Collections:")
    try:
        all_colls = manager.get_all_collection_names()
        for c in all_colls:
            print(" -", c)
    except Exception as e:
        print("⚠️ Failed to list collections:", e)

    try:
        vs = manager._load_chroma_instance("evidence_km")
        print("\n✅ evidence_km loaded successfully")
        coll = vs._collection
        sample = coll.peek()
        
        # แสดงคีย์ Metadata ที่เราสนใจ
        if sample.get("metadatas") and len(sample["metadatas"]) > 0:
            sample_metadata = sample["metadatas"][0]
            print("🧩 Sample metadata keys:", list(sample_metadata.keys()))
            print(f"🧩 stable_doc_uuid sample: {sample_metadata.get('stable_doc_uuid')}")
        else:
            print("⚠️ Collection is empty or peek failed.")
            
    except Exception as e:
        print("❌ ERROR loading evidence_km:", e)