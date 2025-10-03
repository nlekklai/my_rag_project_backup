# faq_generator.py
from typing import List, Dict
from core.vectorstore import load_all_vectorstores
from models.llm import get_llm
from core.rag_prompts import QA_PROMPT

def generate_faq_from_vectors(doc_ids: List[str], top_k: int = 5) -> Dict:
    """
    สร้าง FAQ ครอบคลุม 8 Enablers x 10 ข้อ ต่อ Enabler จาก vectorstores หลายตัว
    คืนค่าเป็น JSON พร้อมใช้งาน
    """
    # 1. โหลด retriever แบบรวมจากหลาย vectorstore
    retriever = load_all_vectorstores(doc_ids=doc_ids, top_k=top_k)
    
    # 2. สร้าง LLM
    llm = get_llm()
    
    # 3. เตรียมคำถามสำหรับแต่ละ Enabler
    enablers = [
        f"Enabler {i+1}" for i in range(8)
    ]
    
    faq_result = {}
    
    for enabler in enablers:
        faq_result[enabler] = []
        for q_num in range(1, 11):
            user_question = f"สร้าง FAQ ข้อที่ {q_num} สำหรับ {enabler} โดยอ้างอิงจาก context ของเอกสารทั้งหมด"
            
            # 4. ดึง context ที่เกี่ยวข้องจาก retriever
            relevant_docs = retriever.get_relevant_documents(user_question)
            context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # 5. สร้าง prompt
            prompt_text = QA_PROMPT.format(context=context_text, question=user_question)
            
            # 6. เรียก LLM
            answer = llm(prompt_text)
            
            # 7. เพิ่มลง JSON
            faq_result[enabler].append({
                "question": user_question,
                "answer": answer
            })
    
    return faq_result

if __name__ == "__main__":
    doc_ids = ["seam", "seam2", "seam2567"]
    faq_json = generate_faq_from_vectors(doc_ids=doc_ids)
    
    import json
    print(json.dumps(faq_json, ensure_ascii=False, indent=2))
