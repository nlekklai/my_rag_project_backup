from core.vectorstore import load_vectorstore

# โหลด retriever โดยตรง
seam_retriever = load_vectorstore("seam")

# query ว่างเพื่อดึงเอกสารทั้งหมด (top_k=100)
# แก้ไข: ใช้ .invoke() แทน .get_relevant_documents() เพื่อเลี่ยง Deprecation Warning
docs = seam_retriever.invoke("")

print(f"Retrieved {len(docs)} documents\n")

for i, doc in enumerate(docs):
    print(f"--- Document {i+1} ---")
    print("Content:", doc.page_content[:500])  # preview 500 chars
    print("Metadata:", doc.metadata)
    print("-" * 50)
