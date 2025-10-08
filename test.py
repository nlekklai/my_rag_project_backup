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

# from langchain_community.document_loaders import UnstructuredExcelLoader
# loader = UnstructuredExcelLoader("data/faq/seam_faq.xlsx")
# docs = loader.load()
# print(len(docs))
# print(docs[0].page_content[:100])


import json
from collections import defaultdict
import pandas as pd  # Optional: for clean tabular display

def summarize_mapping(json_path: str):
    """Summarize mapping by organization/section."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary = defaultdict(lambda: {"questions": 0, "valid_evidence_total": 0})

    for entry in data:
        org = entry.get("organization", "Unknown")
        section = entry.get("section", "Unknown")
        key = (org, section)
        summary[key]["questions"] += 1
        summary[key]["valid_evidence_total"] += entry.get("valid_evidence_count", 0)

    # Convert to rows
    rows = []
    for (org, section), stats in summary.items():
        avg = stats["valid_evidence_total"] / stats["questions"] if stats["questions"] > 0 else 0
        rows.append({
            "organization": org,
            "section": section,
            "number_of_questions": stats["questions"],
            "average_evidence_per_question": round(avg, 2),
            "total_valid_evidence": stats["valid_evidence_total"]
        })

    # Output as table (pandas optional)
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))
    return df

if __name__ == "__main__":
    summarize_mapping("output/mappings_pea_v3_clean.json")
