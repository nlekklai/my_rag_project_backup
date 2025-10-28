#test_vectorstore.py
# from core.vectorstore import VectorStoreManager, load_all_vectorstores

# # 1️⃣ ตรวจสอบ collection ที่ระบบเห็น
# manager = VectorStoreManager()
# print(manager.get_all_collection_names())
# # 👉 ควรเห็น ['default', 'document']

# # 2️⃣ โหลด retriever รวมทุก collection
# retriever = load_all_vectorstores()

# # 3️⃣ ทดสอบ query
# docs = retriever.invoke("นโยบายด้านการจัดการความรู้ขององค์กรคืออะไร")
# print(len(docs), "documents found")
# for d in docs[:3]:
#     print(d.metadata.get("doc_type"), d.metadata.get("relevance_score"), d.page_content[:100])


import json
with open("data/doc_id_mapping.json") as f:
    mapping = json.load(f)
print(mapping.keys())

# import os
# import logging
# from core.ingest import get_vectorstore

# logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
# logger = logging.getLogger(__name__)

# VECTORSTORE_DIR = "vectorstore"  # หรือ path ตาม config ของคุณ

# def main():
#     # โหลด instance ของ vectorstore
#     try:
#         vs = get_vectorstore(base_path=VECTORSTORE_DIR)
#     except Exception as e:
#         logger.error(f"ไม่สามารถโหลด vectorstore: {e}")
#         return

#     # ตรวจสอบ collections
#     try:
#         collections = vs.list_collections()
#         logger.info(f"Collections ที่มีอยู่ใน vectorstore: {collections}")
#     except Exception as e:
#         logger.error(f"ไม่สามารถดึง collections: {e}")
#         return

#     # ตรวจสอบ UUID ของแต่ละ collection
#     for col_name in collections:
#         try:
#             docs = vs.get_all_documents(collection_name=col_name)
#             uuids = [doc['id'] for doc in docs]
#             logger.info(f"Collection '{col_name}' มี {len(uuids)} documents, UUIDs: {uuids}")
#         except Exception as e:
#             logger.warning(f"ไม่สามารถดึง documents ของ collection '{col_name}': {e}")

# if __name__ == "__main__":
#     main()

