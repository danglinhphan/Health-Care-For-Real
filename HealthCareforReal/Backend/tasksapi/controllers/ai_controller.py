from fastapi import HTTPException, Depends
from fastapi.responses import StreamingResponse
from tasksapi.utils.utils import get_current_user
from tasksapi.crud.conversations import get_conversation, update_conversation
from pydantic import BaseModel
import httpx
import json
from datetime import datetime
from config import settings
from utils.logger import logger
import traceback

class MessageRequest(BaseModel):
    content: str

class AIController:
    """Handles AI integration and message processing"""
    
    def __init__(self):
        # Use settings from config
        self.model = "qwen-lora"
        self.api_base_url = settings.custom_api_url
        # HTTP client for custom API - increased timeout for slow model inference
        self.http_client = httpx.AsyncClient(timeout=120.0)  # 2 minutes timeout

    async def call_custom_api_simple(self, message: str):
        """Call the simplified custom API endpoint"""
        try:
            # Validate input - reject empty or whitespace-only messages
            if not message or not message.strip():
                return "Please provide a message to get a response."
                
            payload = {"message": message}
            endpoint = f"{self.api_base_url}/chat"
            
            logger.debug(f"Calling API endpoint: {endpoint}")
            logger.debug(f"Payload: {payload}")
            
            response = await self.http_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info("API response received successfully", extra={"response_length": len(str(result))})
            return result.get("response", "No response generated")
                
        except Exception as e:
            logger.error(f"Custom API error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Custom API error: {str(e)}")

    async def call_custom_api_conversation(self, messages_list: list):
        """Call the conversation endpoint with message history"""
        try:
            if not messages_list:
                raise HTTPException(status_code=400, detail="No messages provided")

            payload = {"messages": messages_list}
            endpoint = f"{self.api_base_url}/chat/conversation"
            
            logger.debug(f"Calling conversation API endpoint: {endpoint}")
            logger.debug(f"Conversation payload: {payload}")
            
            response = await self.http_client.post(endpoint, json=payload)
            response.raise_for_status()
            
            result = response.json()
            logger.info("Conversation API response received", extra={"response_length": len(str(result))})
            
            return result.get("response", "No response generated")
            
        except Exception as e:
            logger.error(f"Custom API conversation error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Custom API conversation error: {str(e)}")

    async def add_message_to_conversation(
        self, 
        conversation_id: int, 
        request: MessageRequest, 
        current_user: dict = Depends(get_current_user)
    ):
        """Add a message to conversation (non-streaming)"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")

            if not request.content.strip():
                raise HTTPException(status_code=400, detail="Message content cannot be empty")

            # Get the conversation
            conv = get_conversation(conversation_id, current_user["user_id"])
            if not conv:
                raise HTTPException(status_code=404, detail="Conversation not found")

            # Create messages list for AI API
            messages_for_api = []
            
            # Add existing messages
            for msg in conv["messages"]:
                messages_for_api.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Add new user message
            messages_for_api.append({
                "role": "user",
                "content": request.content.strip()
            })

            # Get AI response
            ai_response = await self.call_custom_api_conversation(messages_for_api)

            # Update conversation with both messages
            new_messages = conv["messages"].copy()
            
            # Add user message
            new_messages.append({
                "role": "user",
                "content": request.content.strip(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Add AI response
            new_messages.append({
                "role": "assistant", 
                "content": ai_response,
                "timestamp": datetime.now().isoformat()
            })

            # Update in database
            update_conversation(conversation_id, current_user["user_id"], new_messages)

            logger.info(f"Added message to conversation {conversation_id} for user {current_user['user_id']}")
            
            return {
                "conversation_id": conversation_id,
                "messages": new_messages
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error adding message to conversation {conversation_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}")

    async def add_message_to_conversation_stream(
        self, 
        conversation_id: int, 
        request: MessageRequest, 
        current_user: dict = Depends(get_current_user)
    ):
        """Add a message to conversation with streaming response"""
        
        logger.info(f"Streaming endpoint called for conversation {conversation_id}")

        async def generate_stream():
            try:
                if not current_user:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Not authenticated'})}\n\n"
                    return

                logger.debug(f"User message received", extra={"content_length": len(request.content)})

                # Get the conversation
                conv = get_conversation(conversation_id, current_user["user_id"])
                if not conv:
                    yield f"data: {json.dumps({'type': 'error', 'error': 'Conversation not found'})}\n\n"
                    return

                logger.debug(f"Conversation loaded with {len(conv['messages'])} messages")

                # Create messages list for AI API
                messages_for_api = []
                
                # Add existing messages
                for msg in conv["messages"]:
                    messages_for_api.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                
                # Add new user message
                messages_for_api.append({
                    "role": "user",
                    "content": request.content.strip()
                })

                # Send start signal
                yield f"data: {json.dumps({'type': 'start'})}\n\n"

                logger.info("Starting streaming generation")

                # Get streaming response from AI
                full_content = ""
                chunk_count = 0

                logger.debug("Calling Custom API with streaming")

                # For now, we'll call the non-streaming endpoint and simulate streaming
                ai_response = await self.call_custom_api_conversation(messages_for_api)
                
                logger.debug("Starting to process chunks from Custom API")
                
                # Simulate streaming by sending chunks
                words = ai_response.split()
                for i, word in enumerate(words):
                    chunk_text = word + " "
                    chunk_count += 1
                    
                    logger.debug(f"Processing chunk {chunk_count}")
                    
                    full_content += chunk_text
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk_text})}\n\n"
                    
                    # Small delay to simulate real streaming
                    import asyncio
                    await asyncio.sleep(0.05)

                logger.info(f"Completed processing {chunk_count} chunks", extra={"content_length": len(full_content)})

                # Update conversation with both messages
                new_messages = conv["messages"].copy()
                
                # Add user message
                new_messages.append({
                    "role": "user",
                    "content": request.content.strip(),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Add AI response
                new_messages.append({
                    "role": "assistant", 
                    "content": full_content.strip(),
                    "timestamp": datetime.now().isoformat()
                })

                logger.debug("Saving to database")

                # Update in database
                update_conversation(conversation_id, current_user["user_id"], new_messages)

                logger.debug("Database save complete, sending completion signals")

                # Send completion signals
                yield f"data: {json.dumps({'type': 'complete', 'content': full_content.strip()})}\n\n"
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                
                logger.info("Streaming complete")

            except Exception as e:
                logger.error(f"Exception in streaming function: {str(e)}", exc_info=True)
                import traceback
                yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )