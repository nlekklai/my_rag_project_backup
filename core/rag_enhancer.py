# core/rag_enhancer.py

def enhance_query_for_statement(statement_id: str, enabler_id: str, statement_text: str, focus_hint: str) -> str:
    """
    ปรับปรุง Query โดยเพิ่มน้ำหนักคำสำคัญตาม Statement และ Enabler 
    เพื่อแก้ไขปัญหา RAG Quality ที่สังเกตพบ.
    """

    # Query มาตรฐาน (Default)
    # RAG Query ที่ดีที่สุดเริ่มต้นด้วยข้อความ Statement และ Constraint/Focus Hint เท่านั้น
    default_query = f"{statement_text} {focus_hint}" 
    
    # --- Logic การเพิ่มน้ำหนักคำค้นหาเฉพาะจุด (Heuristic Patch) ---

    # 1. กรองตาม Enabler หลัก (KM Project)
    if enabler_id == "KM": 
        
        # 2. กรองตาม Statement ที่มีปัญหา (1.1 L1 FAIL)
        if statement_id == "1.1":
            # คำสำคัญที่ถูกวิเคราะห์ว่าจำเป็นสำหรับ 1.1 L1
            l1_keywords_for_1_1 = "นโยบาย วิสัยทัศน์ ทิศทางกลยุทธ์ แผนกลยุทธ์ ความมุ่งมั่น"
            
            # ⭐ Logic การเพิ่มน้ำหนักคำค้นหา (Boosting)
            # เพิ่มคำซ้ำเพื่อให้ Boost พลังงาน และใส่ Keywords เฉพาะของ 1.1 เข้าไป
            boosted_keywords = "นโยบาย นโยบาย นโยบาย วิสัยทัศน์ วิสัยทัศน์ " + l1_keywords_for_1_1
            
            # คืนค่า Query ที่มีการ Boost (รวม Query มาตรฐาน + คำที่ Boost)
            return f"{default_query} {boosted_keywords}"

    # คืนค่า Query มาตรฐาน (Statement Text + Focus Hint) 
    # สำหรับทุก Statement/Enabler ที่ไม่ต้องการ Boost
    return default_query