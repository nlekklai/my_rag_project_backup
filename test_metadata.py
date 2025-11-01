import chromadb
# สมมติว่า Chroma DB ถูกรันบนเครื่องเดียวกันและพอร์ตมาตรฐาน
CHROMA_PATH = "chroma_db" 
COLLECTION_NAME = "evidence_km" 
TARGET_DOC_ID = "8c27ae78b7dbdc2b6c94f63a33463e1fed5bc2" # ⬅️ **เปลี่ยน ID นี้**

# เชื่อมต่อกับ Chroma Client
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)

# ค้นหา Chunk ที่มี Stable Document ID นี้ โดยใช้ทุกคีย์ที่เป็นไปได้
# (เราจะใช้การ Query แบบทั่วไปก่อน เพื่อดึง Chunk แรกของเอกสารนี้มา)
# เนื่องจากเราไม่รู้ชื่อคีย์ที่ถูกต้อง เราจะใช้วิธี Query โดยไม่มี Filter ก่อน
print(f"Attempting to query Chroma for documents belonging to ID: {TARGET_DOC_ID}")

# [สำคัญมาก: เราต้องใช้การกรองที่ใช้คีย์ที่ถูกบันทึกจริง]
# เนื่องจากเราไม่รู้คีย์ที่ถูกบันทึกจริง ให้ลองใช้ 'source' ซึ่งเป็นค่าเริ่มต้นของ LangChain
try:
    results = collection.get(
        where={"source": TARGET_DOC_ID},
        limit=1,
        include=["metadatas"]
    )
    if results and results["metadatas"]:
        print("✅ SUCCESS: Found Chunk metadata using key 'source'!")
        print("-" * 50)
        print("--- ACTUAL METADATA KEYS ---")
        import pprint
        pprint.pprint(results["metadatas"][0])
    else:
        # หากไม่พบด้วย key 'source'
        print("❌ 'source' key failed. Attempting to get ALL documents and inspect the first one...")
        
        # NOTE: การดึงเอกสารทั้งหมดมาดูนั้นอันตรายต่อประสิทธิภาพ
        # แต่เพื่อการดีบั๊ก ให้ดึงมาแค่ 10 รายการแรก แล้วค้นหา metadata ด้วยสายตา
        all_results = collection.peek(limit=10, include=["metadatas"])
        
        print("-" * 50)
        print("--- FIRST 10 CHUNK METADATA ---")
        for metadata in all_results["metadatas"]:
            pprint.pprint(metadata)
            
        print("-" * 50)
        print("🔎 โปรดตรวจสอบรายการ Metadata ด้านบนเพื่อหาคีย์ที่มีค่าเป็น UUIDs ยาวๆ")

except Exception as e:
    print(f"An error occurred during Chroma client connection or query: {e}")