from db.db import engine
from sqlmodel import SQLModel, Field, Session, select
import json
from datetime import datetime
from typing import Optional


class Conversation(SQLModel, table=True):
    __tablename__ = "conversations"
    conversation_id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int
    timestamp: str
    messages: str

def create_conversation(user_id, first_message):
    timestamp = datetime.utcnow().isoformat()
    messages = json.dumps([{"role": "user", "content": first_message, "timestamp": timestamp}])
    with Session(engine) as session:
        conversation = Conversation(user_id=user_id, timestamp=timestamp, messages=messages)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return {
            "conversation_id": conversation.conversation_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "messages": json.loads(messages)
        }

def get_conversation(conversation_id, user_id):
    with Session(engine) as session:
        statement = select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user_id
        )
        result = session.exec(statement).first()
        if result:
            return {
                "conversation_id": result.conversation_id,
                "user_id": result.user_id,
                "timestamp": result.timestamp,
                "messages": json.loads(result.messages)
            }
        return None

def update_conversation(conversation_id, user_id, messages):
    """Update conversation with new messages"""
    with Session(engine) as session:
        statement = select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user_id
        )
        conversation = session.exec(statement).first()
        if not conversation:
            return None
        
        conversation.messages = json.dumps(messages)
        conversation.timestamp = datetime.utcnow().isoformat()
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        
        return {
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "timestamp": conversation.timestamp,
            "messages": json.loads(conversation.messages)
        }

def get_conversations_by_user(user_id):
    """Get all conversations for a user"""
    with Session(engine) as session:
        statement = select(Conversation).where(Conversation.user_id == user_id)
        results = session.exec(statement).all()
        conversations = []
        for result in results:
            messages = json.loads(result.messages)
            first_message = messages[0]["content"] if messages else "No messages"
            conversations.append({
                "conversation_id": result.conversation_id,
                "user_id": result.user_id,
                "timestamp": result.timestamp,
                "first_message": first_message[:100] + ("..." if len(first_message) > 100 else "")
            })
        return conversations

def delete_conversation(conversation_id, user_id) -> bool:
    with Session(engine) as session:
        statement = select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user_id
        )
        conversation = session.exec(statement).first()
        if not conversation:
            return False
        session.delete(conversation)
        session.commit()
        return True