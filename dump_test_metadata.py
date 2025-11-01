from core.vectorstore import VectorStoreManager
import logging
import json

# ตั้งค่า Logger ชั่วคราวเพื่อให้เห็นผลลัพธ์
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print("---------------------------------------------------------")
print("🔥 Running Metadata Dump Test on 'evidence_km' collection")
print("---------------------------------------------------------")

try:
    manager = VectorStoreManager()
    
    # 1. โหลด Chroma Instance
    vectordb = manager._load_chroma_instance("evidence_km")
    
    if vectordb is None:
        raise ConnectionError("Failed to load Chroma instance for 'evidence_km'.")

    collection = vectordb._collection
    
    # 2. ดึง ID ของ Chunk แรกสุด
    # NOTE: เราต้องเรียก collection.get() ก่อน เพื่อให้มั่นใจว่ามี ID
    all_ids = collection.get(limit=1)["ids"]
    
    if not all_ids:
        print("❌ ERROR: Collection 'evidence_km' appears to be empty (0 chunks found).")
    else:
        first_chunk_id = all_ids[0]

        # 3. ดึง Chunk แรก พร้อม Metadata
        sample = collection.get(
            ids=[first_chunk_id], 
            include=["metadatas"]
        )
        
        # 4. แสดงผลลัพธ์
        if sample.get("metadatas") and sample["metadatas"][0]:
            print(f"✅ SUCCESS: Found {collection.count()} total chunks. Dumping metadata of first chunk (ID: {first_chunk_id}).")
            
            metadata = sample["metadatas"][0]
            print("\n================= METADATA DUMP RESULT ==================")
            print(json.dumps(metadata, indent=2, ensure_ascii=False))
            print("=========================================================\n")
            
            # 5. วิเคราะห์ค่า ID ในเบื้องต้น
            id_keys = ["stable_doc_uuid", "original_stable_id", "doc_id", "source_uuid"]
            print("🔍 Potential ID Keys & Lengths:")
            for key in id_keys:
                if key in metadata:
                    value = metadata[key]
                    print(f"   - {key}: {value[:32]}... ({len(value)} chars)")
                
        else:
            print(f"❌ ERROR: Could not retrieve metadata for chunk ID: {first_chunk_id}.")

except ConnectionError as e:
    print(f"🛑 CRITICAL ERROR: {e}")
except Exception as e:
    print(f"🛑 UNEXPECTED ERROR during metadata dump: {e}")

print("---------------------------------------------------------")