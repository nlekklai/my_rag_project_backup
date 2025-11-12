# core/history_utils.py (Placeholder for conversation state management)

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# A simple in-memory store for demonstration purposes (DO NOT USE IN PRODUCTION)
# Key: conversation_id, Value: List of messages
CONVERSATION_STORE: Dict[str, List[Dict[str, Any]]] = {}

def load_conversation_history(conversation_id: str) -> List[BaseMessage]:
    """
    Loads conversation history from the store and converts it to LangChain BaseMessage format.
    
    Returns:
        List[BaseMessage]: A list of previous messages in the conversation.
    """
    history_data = CONVERSATION_STORE.get(conversation_id, [])
    
    messages: List[BaseMessage] = []
    # Note: Only include the last few messages to prevent token overflow.
    # The LangChain objects (HumanMessage, AIMessage) are necessary for the LLM chain.
    for msg in history_data:
        if msg.get('type') == 'user':
            messages.append(HumanMessage(content=msg.get('content')))
        elif msg.get('type') == 'ai':
            messages.append(AIMessage(content=msg.get('content')))
    
    return messages

def save_message(conversation_id: str, message_type: str, content: str):
    """
    Saves a new message to the conversation history store.
    """
    if conversation_id not in CONVERSATION_STORE:
        CONVERSATION_STORE[conversation_id] = []
        
    CONVERSATION_STORE[conversation_id].append({
        'type': message_type,
        'content': content,
        'timestamp': len(CONVERSATION_STORE[conversation_id])
    })
    
    # Keep history size manageable (last 10 messages = 5 user turns + 5 ai turns)
    CONVERSATION_STORE[conversation_id] = CONVERSATION_STORE[conversation_id][-10:]