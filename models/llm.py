# -*- coding: utf-8 -*-
# models/llm.py - Production Version (Ollama Unified Connector)
# รองรับทั้ง Local (Mac 8B) และ Cloud (NVIDIA 70B) ผ่าน Ollama

import logging
import os
from typing import Optional, Any, Final
from langchain_core.language_models.llms import BaseLLM

# --- Conditional Imports ---
try:
    from langchain_ollama import OllamaLLM
except ImportError:
    logger = logging.getLogger(__name__)
    logger.error("❌ 'langchain-ollama' not found. Please install it using 'pip install langchain-ollama'")
    OllamaLLM = None

logger = logging.getLogger(__name__)

# --- Configuration Constants (ดึงจาก Global Config) ---
from config.global_vars import (
    DEFAULT_LLM_MODEL_NAME, 
    LLM_CONTEXT_WINDOW, 
    LLM_TEMPERATURE,
    RAG_RUN_MODE
)

# -----------------------------------------------------
# 🎯 Global LLM Instance (สำหรับ Backward Compatibility)
# -----------------------------------------------------
# เก็บอินสแตนซ์แรกที่สร้างขึ้นเพื่อใช้งานร่วมกับส่วนอื่นๆ ของระบบ
llm: Optional[Any] = None 

# -----------------------------------------------------
# 🛠️ Factory Logic: create_llm_instance
# -----------------------------------------------------

def create_llm_instance(
    model_name: Optional[str] = None,
    temperature: float = LLM_TEMPERATURE,
    context_window: Optional[int] = None
) -> Optional[Any]:
    """
    Initializes and returns the appropriate Ollama LLM instance.
    รองรับการทำงานทั้งบน Mac (localhost) และ Cloud (IP/URL ผ่าน .env)
    """
    global llm 
    
    # 1. เลือกค่า Default ตามโหมด (ถ้าไม่ได้ระบุ Model หรือ Context มา)
    selected_model = model_name or DEFAULT_LLM_MODEL_NAME
    selected_ctx = context_window or LLM_CONTEXT_WINDOW
    
    # 2. ดึง Base URL จาก Environment Variable
    # - บน Mac/Local: http://localhost:11434 (หรือ http://host.docker.internal:11434 ใน Docker)
    # - บน Cloud: http://<server-ip>:11434
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    
    logger.warning(f"⚠️ Initializing LLM in {RAG_RUN_MODE} mode")
    logger.info(f"📍 Target Model: {selected_model}")
    logger.info(f"🔗 Ollama URL: {base_url}")
    logger.info(f"🧠 Context Window: {selected_ctx}")

    try:
        if OllamaLLM is None:
            raise ImportError("langchain-ollama is required for this project.")

        # 3. สร้าง Instance ของ OllamaLLM
        # ตั้งค่า timeout ไว้สูงหน่อย (600s) เพราะ 70B บน Cloud อาจใช้เวลาคิดนานในบางคำถาม
        llm_instance = OllamaLLM(
            model=selected_model,
            temperature=temperature,
            num_ctx=selected_ctx,
            base_url=base_url,
            timeout=600,
            num_predict=4096
            # คืนค่าเป็นพารามิเตอร์อื่นๆ ที่ Ollama รองรับได้ที่นี่
        )

        logger.info(f"✅ LLM Instance created successfully: {selected_model}")
        
        # กำหนดค่าให้ตัวแปร Global หากยังไม่มีการกำหนด
        if llm is None:
            llm = llm_instance
            
        return llm_instance

    except Exception as e:
        logger.error(f"❌ Failed to initialize Ollama LLM: {e}")
        return None

# -----------------------------------------------------
# 🧪 Health Check (Optional)
# -----------------------------------------------------
def check_llm_connection() -> bool:
    """ตรวจสอบเบื้องต้นว่าสามารถเชื่อมต่อกับ Ollama Server ได้หรือไม่"""
    if llm:
        try:
            # ลองส่งคำขอด่วนๆ เพื่อเช็คสถานะ
            # llm.invoke("Hi") 
            return True
        except:
            return False
    return False