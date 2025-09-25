from fastapi import HTTPException, Depends
from tasksapi.utils.utils import get_current_user
from tasksapi.crud.conversations import create_conversation, get_conversation, delete_conversation
from pydantic import BaseModel
from datetime import datetime
from db.db import engine
from sqlmodel import Session, select
from tasksapi.crud.conversations import Conversation as ConversationModel
from utils.logger import logger
from typing import List, Optional

class ConversationResponse(BaseModel):
    conversation_id: int
    title: str
    created_at: str
    messages: List[dict]

class ConversationController:
    """Handles conversation management operations"""
    
    @staticmethod
    async def get_user_conversations(current_user: dict = Depends(get_current_user)):
        """Get all conversations for the current user"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            user_id = current_user["user_id"]
            
            with Session(engine) as session:
                conversations = session.exec(
                    select(ConversationModel).where(ConversationModel.user_id == user_id)
                ).all()
                
                conversation_list = []
                for conv in conversations:
                    # Parse messages to get first message for display
                    try:
                        import json
                        messages = json.loads(conv.messages) if conv.messages else []
                        first_message = messages[0]["content"] if messages else "No messages"
                        first_message = first_message[:100] + ("..." if len(first_message) > 100 else "")
                    except (json.JSONDecodeError, IndexError, KeyError):
                        first_message = "No messages"
                    
                    conversation_list.append({
                        "conversation_id": conv.conversation_id,
                        "user_id": conv.user_id,
                        "timestamp": conv.timestamp,
                        "first_message": first_message
                    })
                
                logger.info(f"Retrieved {len(conversation_list)} conversations for user {user_id}")
                return conversation_list
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error retrieving conversations: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve conversations")

    @staticmethod
    async def start_conversation(
        message: str, 
        current_user: dict = Depends(get_current_user)
    ):
        """Start a new conversation"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            if not message or not message.strip():
                raise HTTPException(status_code=400, detail="Message cannot be empty")
            
            user_id = current_user["user_id"]
            
            # Create conversation using existing function
            result = create_conversation(user_id, message.strip())
            
            logger.info(f"Started new conversation {result['conversation_id']} for user {user_id}")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error starting conversation: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to start conversation")

    @staticmethod
    async def read_conversation(
        conversation_id: int, 
        current_user: dict = Depends(get_current_user)
    ):
        """Get a specific conversation"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            result = get_conversation(conversation_id, current_user["user_id"])
            
            if not result:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            logger.info(f"Retrieved conversation {conversation_id} for user {current_user['user_id']}")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error reading conversation {conversation_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to retrieve conversation")

    @staticmethod
    async def delete_conversation_endpoint(
        conversation_id: int, 
        current_user: dict = Depends(get_current_user)
    ):
        """Delete a specific conversation"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            # Verify conversation exists and belongs to user
            conversation = get_conversation(conversation_id, current_user["user_id"])
            if not conversation:
                raise HTTPException(status_code=404, detail="Conversation not found")
            
            # Delete the conversation
            success = delete_conversation(conversation_id, current_user["user_id"])
            
            if success:
                logger.info(f"Deleted conversation {conversation_id} for user {current_user['user_id']}")
                return {"message": "Conversation deleted successfully"}
            else:
                raise HTTPException(status_code=500, detail="Failed to delete conversation")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error deleting conversation {conversation_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to delete conversation")