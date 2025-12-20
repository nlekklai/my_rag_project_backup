import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ================= Store =================
CONVERSATION_STORE: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
_locks: Dict[str, Dict[str, asyncio.Lock]] = {}

# ================= Lock Helper =================
def _get_lock(user_id: str, conv_id: str) -> asyncio.Lock:
    if user_id not in _locks:
        _locks[user_id] = {}
    if conv_id not in _locks[user_id]:
        _locks[user_id][conv_id] = asyncio.Lock()
    return _locks[user_id][conv_id]

# ================= Async Save =================
async def async_save_message(
    user_id: str,
    conversation_id: str,
    message_type: str,
    content: str,
    intent: Optional[dict] = None,
    enabler: Optional[str] = None,
    sub_topic: Optional[str] = None
) -> None:
    async with _get_lock(user_id, conversation_id):
        if user_id not in CONVERSATION_STORE:
            CONVERSATION_STORE[user_id] = {}
        if conversation_id not in CONVERSATION_STORE[user_id]:
            CONVERSATION_STORE[user_id][conversation_id] = []

        CONVERSATION_STORE[user_id][conversation_id].append({
            "type": message_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "enabler": enabler,
            "sub_topic": sub_topic
        })

        # เก็บแค่ 12 ข้อความล่าสุด
        if len(CONVERSATION_STORE[user_id][conversation_id]) > 12:
            CONVERSATION_STORE[user_id][conversation_id] = \
                CONVERSATION_STORE[user_id][conversation_id][-12:]

# ================= Async Load =================
async def async_load_conversation_history(
    user_id: str,
    conversation_id: str
) -> List[BaseMessage]:
    async with _get_lock(user_id, conversation_id):
        raw_history = CONVERSATION_STORE.get(user_id, {}).get(conversation_id, [])
        messages: List[BaseMessage] = []

        for msg in raw_history:
            if msg["type"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "ai":
                messages.append(AIMessage(content=msg["content"]))

        return messages

# ================= Sync Wrappers =================
def save_message(
    user_id: str,
    conversation_id: str,
    message_type: str,
    content: str,
    intent: Optional[dict] = None,
    enabler: Optional[str] = None,
    sub_topic: Optional[str] = None
) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        asyncio.create_task(
            async_save_message(user_id, conversation_id, message_type, content, intent, enabler, sub_topic)
        )
    else:
        asyncio.run(
            async_save_message(user_id, conversation_id, message_type, content, intent, enabler, sub_topic)
        )

def load_conversation_history(
    user_id: str,
    conversation_id: str
) -> List[BaseMessage]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            async_load_conversation_history(user_id, conversation_id), loop
        )
        return future.result(timeout=5)
    else:
        return asyncio.run(async_load_conversation_history(user_id, conversation_id))
