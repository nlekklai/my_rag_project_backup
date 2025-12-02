# app.py
import os
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
# Import Routers ← เพิ่มบรรทัดนี้!!!
# -----------------------------
from routers.upload_router import upload_router
from routers.llm_router import llm_router
from routers.assessment_router import assessment_router   # เพิ่มบรรทัดนี้!

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
# Routers ← เพิ่มบรรทัดนี้!!!
# -----------------------------
app.include_router(upload_router)
app.include_router(llm_router)
app.include_router(assessment_router)   # เพิ่มบรรทัดนี้!

# -----------------------------
# Health check endpoints
# -----------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/status")
async def api_status():
    return {"status": "ok", "message": "SEAM Insight API is running"}