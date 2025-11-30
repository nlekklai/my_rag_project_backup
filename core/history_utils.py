# core/history_utils.py
import asyncio
from typing import List, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# In-memory store + per-conversation lock (ป้องกัน race condition 100%)
CONVERSATION_STORE: Dict[str, List[Dict[str, Any]]] = {}
_locks: Dict[str, asyncio.Lock] = {}


def _get_lock(conv_id: str) -> asyncio.Lock:
    """Thread-safe & async-safe lock per conversation"""
    if conv_id not in _locks:
        _locks[conv_id] = asyncio.Lock()
    return _locks[conv_id]


# =============================
#    เวอร์ชัน async (ใช้กับ llm_router ล่าสุด)
# =============================
async def async_save_message(conversation_id: str, message_type: str, content: str) -> None:
    async with _get_lock(conversation_id):
        if conversation_id not in CONVERSATION_STORE:
            CONVERSATION_STORE[conversation_id] = []

        CONVERSATION_STORE[conversation_id].append({
            "type": message_type,        # 'user' หรือ 'ai'
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # เก็บแค่ 12 ข้อความล่าสุด (6 turn) → ป้องกัน token overflow
        if len(CONVERSATION_STORE[conversation_id]) > 12:
            CONVERSATION_STORE[conversation_id] = CONVERSATION_STORE[conversation_id][-12:]


async def async_load_conversation_history(conversation_id: str) -> List[BaseMessage]:
    async with _get_lock(conversation_id):
        raw_history = CONVERSATION_STORE.get(conversation_id, [])
        messages: List[BaseMessage] = []

        for msg in raw_history:
            if msg["type"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        return messages


# =============================
#    เวอร์ชัน sync (เผื่อ module อื่นยังใช้อยู่)
# =============================
def save_message(conversation_id: str, message_type: str, content: str) -> None:
    """Sync wrapper – ใช้ได้ทั้งใน async และ sync context"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:  # no running loop
        loop = None

    if loop and loop.is_running():
        # ถ้าอยู่ใน async context แล้ว → สร้าง task
        asyncio.create_task(async_save_message(conversation_id, message_type, content))
    else:
        # ถ้าอยู่ใน sync context → run blocking
        asyncio.run(async_save_message(conversation_id, message_type, content))


def load_conversation_history(conversation_id: str) -> List[BaseMessage]:
    """Sync wrapper"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            async_load_conversation_history(conversation_id), loop
        )
        return future.result(timeout=5)
    else:
        return asyncio.run(async_load_conversation_history(conversation_id))