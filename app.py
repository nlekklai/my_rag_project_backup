# app.py
import os
import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Environment setup
# -----------------------------
os.environ.pop("TRANSFORMERS_CACHE", None)  # ลบตัวแปรเก่า
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

# -----------------------------
# Logging config
# -----------------------------
logging.basicConfig(
    level=logging.INFO,  # DEBUG สำหรับรายละเอียดมากขึ้น
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("KM-RAG-API")

# เพิ่ม level ให้ router module ชัดเจน
logging.getLogger("routers.llm_router").setLevel(logging.INFO)
logging.getLogger("routers.upload_router").setLevel(logging.INFO)
# logging.getLogger("routers.assessment_router").setLevel(logging.INFO)

# -----------------------------
# Import Routers
# -----------------------------
from routers.upload_router import upload_router
from routers.llm_router import llm_router
# from routers.assessment_router import assessment_router

# -----------------------------
# Lifespan
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 KM-RAG API starting up...")
    yield
    logger.info("🛑 KM-RAG API shutting down...")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(
    title="KM RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# -----------------------------
# Middleware
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ปรับเป็น domain ที่ต้องการได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Routers
# -----------------------------
app.include_router(upload_router)
app.include_router(llm_router)
# app.include_router(assessment_router)

# -----------------------------
# Health check endpoints
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/status")
async def api_status():
    return {"status": "ok", "message": "KM RAG API is running"}
