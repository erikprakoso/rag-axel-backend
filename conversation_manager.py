import uuid
from typing import Dict, List, Optional
from datetime import datetime, timedelta


class ConversationManager:
    def __init__(self, max_history_per_conversation: int = 10, conversation_ttl: int = 3600):
        self.conversations: Dict[str, Dict] = {}
        self.max_history = max_history_per_conversation
        self.conversation_ttl = conversation_ttl

    def create_conversation(self) -> str:
        """Create new conversation"""
        conversation_id = str(uuid.uuid4())
        self.conversations[conversation_id] = {
            "id": conversation_id,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "messages": []
        }
        return conversation_id

    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)

    def add_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add message to conversation"""
        if conversation_id not in self.conversations:
            return  # Conversation tidak ditemukan

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }

        self.conversations[conversation_id]["messages"].append(message)
        self.conversations[conversation_id]["updated_at"] = datetime.now()

        # Trim history jika melebihi batas
        if len(self.conversations[conversation_id]["messages"]) > self.max_history:
            self.conversations[conversation_id]["messages"] = self.conversations[conversation_id]["messages"][-self.max_history:]

    def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history (internal use only)"""
        if conversation_id not in self.conversations:
            return []

        messages = self.conversations[conversation_id]["messages"]
        if limit:
            return messages[-limit:]
        return messages

    def get_recent_context(self, conversation_id: str, max_messages: int = 3) -> List[Dict]:
        """Get recent messages for context (internal use only)"""
        return self.get_conversation_history(conversation_id, max_messages)

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if conversation exists"""
        return conversation_id in self.conversations

    def cleanup_expired_conversations(self):
        """Clean up expired conversations"""
        now = datetime.now()
        expired_ids = []

        for conv_id, conv_data in self.conversations.items():
            if now - conv_data["updated_at"] > timedelta(seconds=self.conversation_ttl):
                expired_ids.append(conv_id)

        for conv_id in expired_ids:
            del self.conversations[conv_id]

        return len(expired_ids)


# Global instance
conversation_manager = ConversationManager()
