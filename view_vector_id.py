import os
from langchain_community.vectorstores.chroma import Chroma
from langchain.schema import Document
from core.vectorstore import get_hf_embeddings, VECTORSTORE_DIR

doc_id = "seam"
doc_type = "document"

# path ของ vectorstore
vectordb_path = os.path.join(VECTORSTORE_DIR, doc_type, doc_id)

output_file = "output.txt"  # ไฟล์ที่จะบันทึกผล

if not os.path.exists(vectordb_path):
    msg = f"Vectorstore path does not exist: {vectordb_path}"
    print(msg)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(msg + "\n")
else:
    embeddings = get_hf_embeddings()
    vectordb = Chroma(persist_directory=vectordb_path, embedding_function=embeddings)

    # ดึงทุก documents
    all_docs = vectordb._collection.get(include=["metadatas", "documents"])
    docs = all_docs["documents"]
    metadatas = all_docs["metadatas"]

    header = f"Total chunks in '{doc_id}': {len(docs)}\n"
    print(header)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for i, (text, meta) in enumerate(zip(docs, metadatas), start=1):
            chunk_info = (
                f"Chunk {i}:\n"
                f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}\n"
                f"  Metadata: {meta}\n"
                + "-" * 60 + "\n"
            )
            print(chunk_info)
            f.write(chunk_info)
