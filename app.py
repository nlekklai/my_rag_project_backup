#app.py
import os
os.environ.pop("TRANSFORMERS_CACHE", None)  # ลบตัวแปรเก่าออก
os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import logging

from routers.upload_router import upload_router
from routers.llm_router import llm_router
from routers.assessment_router import assessment_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 App starting up...")
    yield
    logger.info("🛑 App shutting down...")

app = FastAPI(
    title="KM RAG API",
    version="1.0.0",
    lifespan=lifespan
)

# ✅ Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # หรือระบุ domain เฉพาะถ้าต้องการจำกัด
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Routers
app.include_router(upload_router)
app.include_router(llm_router)
app.include_router(assessment_router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
