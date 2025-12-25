# app.py
import sys
import os # 🟢 เพิ่ม os และ sys เข้ามาจัดการ Path

# 🟢 FIX: เพิ่ม Root Project ลงใน Python Path ก่อนการ Import อื่นๆ
# ช่วยให้สามารถ Import โมดูลย่อย เช่น 'utils' ได้
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
# -------------------------------------------------------------

import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Environment setup
# -----------------------------
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# -----------------------------
# Logging config
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("KM-RAG-API")

# เปิด log ให้เห็น assessment ด้วย (แนะนำ)
logging.getLogger("routers.assessment_router").setLevel(logging.INFO)

# -----------------------------
# Import Routers 
# -----------------------------
from routers.upload_router import upload_router
from routers.llm_router import llm_router
# ✅ เพิ่ม assessment_router และ auth_router
from routers.assessment_router import assessment_router   
from routers.auth_router import auth_router   

# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("KM-RAG API starting up...")
    yield
    logger.info("KM-RAG API shutting down...")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="SEAM Insight API",
    description="ระบบประเมินวุฒิภาวะการจัดการความรู้ด้วย AI",
    version="1.0.0",
    lifespan=lifespan
)


# -----------------------------
# Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Routers 
# -----------------------------
# app.include_router(upload_router)
app.include_router(upload_router, prefix="/api/upload")  # ดักพวก POST upload
app.include_router(upload_router, prefix="/api/uploads") # ดักพวก GET list
app.include_router(llm_router)
# ✅ รวม assessment_router และ auth_router เข้าสู่แอปพลิเคชัน
app.include_router(assessment_router)   
app.include_router(auth_router)  

# -----------------------------
# Health check endpoints
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/status")
async def api_status():
    return {"status": "ok", "message": "SEAM Insight API is running"}