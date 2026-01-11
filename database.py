# core/database.py
import os
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # ตั้ง level ให้เห็น info/debug

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./seam_pro.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Models ---

class UserTable(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)  # ใน production ควรใช้ hash
    full_name = Column(String)
    tenant = Column(String, index=True)
    role = Column(String, default="user")
    enablers = Column(Text)  # JSON string: ["KM", "IT"]
    created_at = Column(DateTime, default=datetime.utcnow)

class AssessmentTaskTable(Base):
    __tablename__ = "assessment_tasks"
    record_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    tenant = Column(String, index=True)
    year = Column(String)
    enabler = Column(String)
    sub_criteria = Column(String)
    status = Column(String, default="RUNNING")  # RUNNING, COMPLETED, FAILED
    progress_percent = Column(Integer, default=0)
    progress_message = Column(Text)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)  # เพิ่ม column นี้

class AssessmentResultTable(Base):
    __tablename__ = "assessment_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    record_id = Column(String, ForeignKey("assessment_tasks.record_id"), unique=True, index=True)  # เพิ่ม index
    overall_level = Column(String)
    overall_score = Column(Float)
    full_result_json = Column(Text)  # เก็บ JSON จาก _transform_result_for_ui
    created_at = Column(DateTime, default=datetime.utcnow)

# --- Helpers ---

def init_db():
    """สร้าง Tables ตาม Models"""
    Base.metadata.create_all(bind=engine)
    logger.info("📂 Database Tables Created Successfully!")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def db_update_task_status(record_id: str, progress: int, message: str, status: str = "RUNNING", error: str = None):
    """อัปเดตสถานะ task ด้วย transaction ปลอดภัย"""
    db = SessionLocal()
    try:
        task = db.query(AssessmentTaskTable).filter(AssessmentTaskTable.record_id == record_id).first()
        if task:
            task.progress_percent = progress
            task.progress_message = message
            task.status = status
            if error:
                task.error_message = error
            db.commit()
            logger.debug(f"[DB] Updated task {record_id}: {progress}% - {message}")
        else:
            logger.warning(f"[DB] Task {record_id} not found")
    except Exception as e:
        db.rollback()
        logger.error(f"[DB] Update failed for {record_id}: {str(e)}")
    finally:
        db.close()

def db_finish_task(record_id: str, result_data: dict):
    """บันทึกสถานะ COMPLETED และผลลัพธ์ลงใน Database"""
    db = SessionLocal()
    try:
        # 1. อัปเดตสถานะ Task
        task = db.query(AssessmentTaskTable).filter(AssessmentTaskTable.record_id == record_id).first()
        if task:
            task.status = "COMPLETED"
            task.progress_percent = 100
            task.progress_message = "การประเมินเสร็จสมบูรณ์"
            task.completed_at = datetime.utcnow()

        # 2. ดึงข้อมูลสรุปจากผลลัพธ์ (จาก total_stats ใน engine)
        # ปกติ result_data จะมี key 'summary' ที่เราทำไว้ใน _calculate_overall_stats
        summary = result_data.get('summary', {})
        overall_level = summary.get('overall_level_label', 'N/A')
        overall_score = summary.get('overall_avg_score', 0.0)

        # 3. บันทึกหรืออัปเดตผลลัพธ์
        existing_result = db.query(AssessmentResultTable).filter(
            AssessmentResultTable.record_id == record_id
        ).first()

        result_json_str = json.dumps(result_data, ensure_ascii=False)

        if existing_result:
            existing_result.overall_level = overall_level
            existing_result.overall_score = overall_score
            existing_result.full_result_json = result_json_str
        else:
            new_result = AssessmentResultTable(
                record_id=record_id,
                overall_level=overall_level,
                overall_score=overall_score,
                full_result_json=result_json_str
            )
            db.add(new_result)

        db.commit()
        logger.info(f"[DB] Task {record_id} completed. Result: {overall_level} ({overall_score})")
    except Exception as e:
        db.rollback()
        logger.error(f"[DB] Error finishing task {record_id}: {str(e)}")
    finally:
        db.close()

def db_create_task(record_id: str, tenant: str, year: str, enabler: str, sub_criteria: str, user_id: str = "SYSTEM"):
    """สร้าง record ใหม่ใน assessment_tasks (ใช้ได้ทั้ง API และ CLI)"""
    db = SessionLocal()
    try:
        new_task = AssessmentTaskTable(
            record_id=record_id,
            user_id=user_id,
            tenant=tenant,
            year=str(year),
            enabler=enabler.upper(),
            sub_criteria=sub_criteria,
            status="RUNNING",
            progress_percent=0,
            progress_message="Initializing..."
        )
        db.add(new_task)
        db.commit()
        logger.info(f"[DB] Created new task record: {record_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"[DB] Failed to create task {record_id}: {str(e)}")
    finally:
        db.close()