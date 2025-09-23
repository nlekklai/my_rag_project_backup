
from vectorstore import load_vectorstore
vs = load_vectorstore("2567-PEA")
docs = vs.similarity_search("สรุปหัวข้อหลัก", k=3)
for d in docs:
    print(d.page_content)
