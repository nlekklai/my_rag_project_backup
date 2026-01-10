# -*- coding: utf-8 -*-
# app.py
import sys
import os
import logging
import time
from contextlib import asynccontextmanager

# 🟢 FIX: เพิ่ม Root Project ลงใน Python Path ก่อนการ Import โมดูลย่อย
# เพื่อให้มั่นใจว่าโครงสร้างโปรเจคถูกมองเห็นเป็น module เดียวกัน
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 🎯 นำเข้า Database Components (ตามสถาปัตยกรรมใหม่)
from database import init_db
from auth_service import create_initial_admin

# -----------------------------
# Environment & AI Model Setup
# -----------------------------
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("SEAM-INSIGHT-API")

# เปิด Log พิเศษสำหรับส่วนประเมินเพื่อใช้ Debug ในระดับ Commercial
logging.getLogger("routers.assessment_router").setLevel(logging.INFO)

# -----------------------------
# Lifespan Management (Startup/Shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    logger.info("🚀 SEAM Insight API is starting up...")
    
    # 🎯 สร้าง Database Table และ Initial Users อัตโนมัติ
    try:
        logger.info("📂 Initializing Persistence Database...")
        init_db() 
        
        logger.info("👤 Checking & Creating Initial Admin Users...")
        create_initial_admin()
        logger.info("✅ Database & Auth System are ready.")
    except Exception as e:
        logger.error(f"💥 Critical Failure during DB Init: {e}")
    
    yield
    # --- Shutdown Logic ---
    logger.info("🛑 SEAM Insight API is shutting down...")

# -----------------------------
# FastAPI App Instance
# -----------------------------
app = FastAPI(
    title="SEAM Insight API",
    description="ระบบประเมินวุฒิภาวะการจัดการความรู้และนวัตกรรม (AI-Powered)",
    version="1.1.0",
    lifespan=lifespan
)

# -----------------------------
# Middleware: CORS Configuration (ห้ามตัด)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        # 1. Lovable & Development Tools
        "https://lovable.dev",
        "https://lovable.app",
        
        # 2. Localhost Development
        "http://localhost:8080",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173",
        
        # 3. Server Test / Production IP
        "http://192.168.19.41:8080",
        "http://192.168.19.41:5173",
        "http://192.168.19.41", 
    ],
    # 4. Regex สำหรับ Subdomain ของ Lovable และ Ngrok (สำคัญมากสำหรับ Remote Dev)
    allow_origin_regex=r"https://.*\.lovableproject\.com|https://.*\.lovable\.app|https://.*\.ngrok-free\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# -----------------------------
# Import & Include Routers
# -----------------------------
from routers.upload_router import upload_router
from routers.llm_router import llm_router
from routers.assessment_router import assessment_router   
from routers.auth_router import auth_router   

# รวม Routers เข้าสู่แอปพลิเคชัน
app.include_router(auth_router)        # นำ Auth ขึ้นก่อนเพื่อความปลอดภัย
app.include_router(upload_router)
app.include_router(llm_router)
app.include_router(assessment_router)

# -----------------------------
# Health Check & Status Endpoints
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/api/status")
async def api_status():
    return {
        "status": "online", 
        "message": "SEAM Insight API is running with SQLite Persistence",
        "version": "1.1.0"
    }

# 🎯 ย้าย Startup Logic จาก @app.on_event มาไว้ที่นี่เพื่อความชัวร์ (ถ้าไม่ใช้ lifespan)
# แต่แนะนำให้ใช้ lifespan เป็นหลักตามโค้ดด้านบนครับ
@app.on_event("startup")
async def legacy_startup():
    # กันเหนียว: เผื่อบาง environment ไม่รองรับ lifespan สมบูรณ์
    init_db()
    create_initial_admin()

# -----------------------------
# Development Server Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)