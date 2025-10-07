# from core.vectorstore import load_vectorstore
# retriever = load_vectorstore("2566-PEA")
# docs = retriever.get_relevant_documents("KM (Knowledge Management)")
# print(docs)

# import os
# from core.vectorstore import vectorstore_exists

# path = "vectorstore/document"
# doc_id = "2566-PEA"
# print(vectorstore_exists(doc_id, base_path=path))

# from core.vectorstore import vectorstore_exists
# import os

# VECTORSTORE_DIR = "vectorstore"
# doc_types = ["document", "faq"]
# doc_ids = ["2566-PEA", "2567-PEA"]

# for dtype in doc_types:
#     for doc_id in doc_ids:
#         base_path = os.path.join(VECTORSTORE_DIR, dtype)
#         exists = vectorstore_exists(doc_id, base_path=base_path)
#         print(f"Vectorstore exists? doc_type={dtype}, doc_id={doc_id} -> {exists}")


# # # test.py
# from core.vectorstore import load_all_vectorstores

# def main():
#     doc_types = ["document", "faq"]
#     retrievers = load_all_vectorstores(doc_type=doc_types, top_k=10)

#     print("\nLoaded retrievers:")
#     for r in retrievers.retrievers_list:
#         print(f"- doc_id={r.doc_id}, doc_type={r.doc_type}")

#     query = "KM (Knowledge Management)"
#     results = retrievers.get_relevant_documents(query)

# if __name__ == "__main__":
#     main()

# from core.vectorstore import vectorstore_exists
# import os

# base_dir = "vectorstore"
# doc_ids = ["2566-PEA", "2567-PEA"]
# doc_types = ["document", "faq"]

# for doc_id in doc_ids:
#     for dt in doc_types:
#         base_path = os.path.join(base_dir, dt)
#         exists = vectorstore_exists(doc_id, base_path=base_path)
#         print(f"Vectorstore exists for {doc_id} in {dt}: {exists}")






# from core.vectorstore import load_vectorstore

# retriever = load_vectorstore(doc_id="seam", top_k=5)  # top_k มากขึ้นเพื่อดึง chunk
# docs = retriever.get_relevant_documents("สรุปเอกสาร")
# print(f"Retrieved {len(docs)} chunks")
# for d in docs:
#     print(d.page_content[:200])

from langchain_community.document_loaders import UnstructuredExcelLoader
loader = UnstructuredExcelLoader("data/faq/seam_faq.xlsx")
docs = loader.load()
print(len(docs))
print(docs[0].page_content[:100])
