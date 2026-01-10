# database.py
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging  # เพิ่มบรรทัดนี้ถ้ายังไม่มี

# เพิ่มบรรทัดนี้เพื่อประกาศ logger
logger = logging.getLogger(__name__)


DATABASE_URL = "sqlite:///./seam_pro.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Models ---

class UserTable(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String) # ในโปรดักชั่นควรใช้ passlib hash
    full_name = Column(String)
    tenant = Column(String, index=True)
    role = Column(String, default="user")
    enablers = Column(Text) # JSON string: ["KM", "IT"]
    created_at = Column(DateTime, default=datetime.utcnow)

class AssessmentTaskTable(Base):
    __tablename__ = "assessment_tasks"
    record_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    tenant = Column(String, index=True)
    year = Column(String)
    enabler = Column(String)
    sub_criteria = Column(String)
    status = Column(String, default="RUNNING") # RUNNING, COMPLETED, FAILED
    progress_percent = Column(Integer, default=0)
    progress_message = Column(Text)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class AssessmentResultTable(Base):
    __tablename__ = "assessment_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, ForeignKey("assessment_tasks.record_id"), unique=True)
    overall_level = Column(String)
    overall_score = Column(Float)
    full_result_json = Column(Text) # เก็บ JSON จาก _transform_result_for_ui
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Helpers ---
def init_db():
    """คำสั่งสร้างไฟล์ .db และสร้าง Tables ตาม Models"""
    Base.metadata.create_all(bind=engine)
    print("📂 Database Tables Created Successfully!")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ฟังก์ชันอัปเดตสถานะสำหรับ Engine
def db_update_task_status(record_id: str, progress: int, message: str, status: str = "RUNNING", error: str = None):
    db = SessionLocal()
    task = db.query(AssessmentTaskTable).filter(AssessmentTaskTable.record_id == record_id).first()
    if task:
        task.progress_percent = progress
        task.progress_message = message
        task.status = status
        if error: task.error_message = error
        db.commit()
    db.close()

def db_finish_task(record_id: str, result_data: dict):
    """
    บันทึกสถานะงานเป็น COMPLETED และเก็บผลลัพธ์ JSON ลงในตาราง AssessmentResult
    """
    db = SessionLocal()
    try:
        # 1. อัปเดตสถานะในตาราง Task
        task = db.query(AssessmentTaskTable).filter(
            AssessmentTaskTable.record_id == record_id
        ).first()
        
        if task:
            task.status = "COMPLETED"
            task.progress_percent = 100
            task.progress_message = "การประเมินเสร็จสมบูรณ์"
            task.completed_at = datetime.now()

        # 2. บันทึกผลลัพธ์ลงตาราง Result
        # ตรวจสอบว่าเคยมีผลลัพธ์ของ record_id นี้หรือยัง (ป้องกัน Duplicate)
        existing_result = db.query(AssessmentResultTable).filter(
            AssessmentResultTable.record_id == record_id
        ).first()
        
        result_json_str = json.dumps(result_data, ensure_ascii=False)
        
        if existing_result:
            existing_result.result_json = result_json_str
        else:
            new_result = AssessmentResultTable(
                record_id=record_id,
                result_json=result_json_str
            )
            db.add(new_result)
        
        db.commit()
        if 'logger' in globals():
            logger.info(f"✅ [DB] Task {record_id} finished and result saved.")
            
    except Exception as e:
        db.rollback()
        if 'logger' in globals():
            logger.error(f"❌ [DB] Error finishing task {record_id}: {str(e)}")
        else:
            print(f"❌ Error finishing task: {e}")
    finally:
        db.close()