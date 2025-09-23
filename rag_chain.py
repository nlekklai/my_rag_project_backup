# rag_chain.py

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from vectorstore import load_vectorstore
from models.llm import get_llm
import difflib

# ---- Prompt ภาษาไทย ----
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
คุณคือผู้ช่วยวิเคราะห์เอกสารและสรุปเนื้อหา
- อ่าน context อย่างละเอียด
- ตอบคำถามโดยสรุปหัวข้อหลักและเนื้อหาสำคัญ
- แยกหัวข้อด้วย bullet points หรือ numbered list

Context:
{context}

คำถาม:
{question}

คำตอบ:
"""
)

# ---- สร้าง RAG chain สำหรับ doc_id ใดก็ได้ ----
def create_rag_chain(doc_id: str):
    """
    สร้าง RetrievalQA chain สำหรับเอกสาร doc_id
    """
    vs = load_vectorstore(doc_id)
    retriever = vs.as_retriever(search_kwargs={"k": 3})

    chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT, "document_variable_name": "context"}
    )
    return chain

# ---- ตอบคำถามจากเอกสาร ----
def answer_question(question: str, doc_id: str):
    """
    ตอบคำถามจากเอกสาร doc_id
    """
    chain = create_rag_chain(doc_id)
    return chain.run(question)

# ---- เปรียบเทียบเอกสาร 2 ฉบับ ----
def compare_documents(doc1: str, doc2: str):
    """
    เปรียบเทียบหัวข้อหลักของสองเอกสาร
    """
    chain1 = create_rag_chain(doc1)
    chain2 = create_rag_chain(doc2)

    # ดึงหัวข้อและเนื้อหาหลักจากเอกสาร
    content1 = chain1.run(
        "สรุปหัวข้อหลักและเนื้อหาสำคัญสั้น ๆ ของเอกสารนี้ โดยแยกบรรทัดและใส่ bullet (*) สำหรับแต่ละหัวข้อย่อย"
    )
    content2 = chain2.run(
        "สรุปหัวข้อหลักและเนื้อหาสำคัญสั้น ๆ ของเอกสารนี้ โดยแยกบรรทัดและใส่ bullet (*) สำหรับแต่ละหัวข้อย่อย"
    )

    # แยกเป็นบรรทัดเพื่อเปรียบเทียบง่าย
    lines1 = content1.splitlines()
    lines2 = content2.splitlines()

    # สร้าง diff
    diff = difflib.unified_diff(lines1, lines2, lineterm='', n=0)
    diff_text = '\n'.join(diff)

    return {
        "metrics": [
            {
                "metric": "หัวข้อหลักและเนื้อหาสำคัญ",
                "doc1": content1,
                "doc2": content2,
                "delta": diff_text,
                "remark": "ความแตกต่าง, การเพิ่มหรือลบ จะแสดงใน delta"
            }
        ]
    }
