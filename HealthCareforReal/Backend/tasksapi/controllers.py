from fastapi import APIRouter, HTTPException, status, Request, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from tasksapi.crud.user import create_user, verify_user_login, get_user_by_username, UserCreate, UserLogin, save_user_token, clear_user_token
from tasksapi.utils.utils import create_access_token, get_current_user
from pydantic import BaseModel
from tasksapi.crud.conversations import create_conversation, get_conversation, delete_conversation
import httpx
import json
from datetime import datetime
from db.db import engine
from sqlmodel import Session, select
from tasksapi.crud.conversations import Conversation as ConversationModel
from fastapi import Path
import json
import asyncio
import os
from config import settings
from utils.logger import logger
from middleware.rate_limiter import rate_limit_auth, login_rate_limiter, register_rate_limiter

router = APIRouter()

# Use settings from config
MODEL = "qwen-lora"
API_BASE_URL = settings.custom_api_url

# HTTP client for custom API - increased timeout for slow model inference
http_client = httpx.AsyncClient(timeout=120.0)  # 2 minutes timeout

async def call_custom_api_simple(message: str):
    """Call the simplified custom API endpoint"""
    try:
        # Validate input - reject empty or whitespace-only messages
        if not message or not message.strip():
            return "Please provide a message to get a response."
            
        payload = {"message": message}
        endpoint = f"{API_BASE_URL}/chat"
        
        logger.debug(f"Calling API endpoint: {endpoint}")
        logger.debug(f"Payload: {payload}")
        
        response = await http_client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info("API response received successfully", extra={"response_length": len(str(result))})
        return result.get("response", "No response generated")
            
    except Exception as e:
        logger.error(f"Custom API error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Custom API error: {str(e)}")

async def call_custom_api_conversation(messages_list: list):
    """Call the conversation endpoint with message history"""
    try:
        # Prepare messages for the conversation endpoint
        api_messages = []
        for msg in messages_list:
            # Convert role format: assistant -> model for your API
            role = "model" if msg["role"] == "assistant" else msg["role"]
            api_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        payload = {"messages": api_messages}
        endpoint = f"{API_BASE_URL}/chat/conversation"
        
        logger.debug(f"Calling conversation API endpoint: {endpoint}")
        logger.debug(f"Conversation payload: {payload}")
        
        response = await http_client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Conversation API response received", extra={"response_length": len(str(result))})
        return result.get("response", "No response generated")
            
    except Exception as e:
        logger.error(f"Custom API conversation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Custom API conversation error: {str(e)}")

async def call_custom_api(messages_list: list = None, single_message: str = None, stream: bool = False):
    """Call the custom API - simplified version"""
    try:
        if messages_list:
            # Use the conversation endpoint for better context handling
            return await call_custom_api_conversation(messages_list)
        elif single_message:
            return await call_custom_api_simple(single_message)
        else:
            raise ValueError("Either messages_list or single_message must be provided")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom API error: {str(e)}")

async def stream_custom_api(messages_list: list):
    """Stream response from custom API"""
    try:
        # Use the conversation API for better context
        response_text = await call_custom_api_conversation(messages_list)
        
        # Simulate streaming by sending words in chunks
        words = response_text.split()
        chunk_size = 3  # Send 3 words at a time
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if i + chunk_size < len(words):
                chunk_text += " "
            yield chunk_text
            await asyncio.sleep(0.1)  # Small delay to simulate streaming
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom API streaming error: {str(e)}")


async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """Get current user information"""
    try:
        user_info = get_user_by_username(current_user)
        if user_info:
            # Remove password from response but keep token
            user_info_dict = {
                "user_id": user_info['user_id'],
                "username": user_info['username'],
                "emailaddress": user_info['emailaddress']
            }
            return {"user": user_info_dict}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def register_user(user_data: UserCreate, request: Request):
    # Apply rate limiting
    await rate_limit_auth(request, register_rate_limiter)
    
    try:
        # Basic validation
        if len(user_data.username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters long")
        
        if len(user_data.password) < 6:
            raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
        
        # Check if user already exists
        existing_user = get_user_by_username(user_data.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        result = create_user(user_data)
        
        if result:
            return {"message": "User registered successfully", "user": result}
        else:
            raise HTTPException(status_code=500, detail="Failed to register user")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def login_user(login_data: UserLogin, request: Request):
    # Apply rate limiting
    await rate_limit_auth(request, login_rate_limiter)
    
    try:
        result = verify_user_login(login_data.username, login_data.password)
        
        if result:
            # Create access token
            access_token = create_access_token(
                data={"sub": result["username"], "user_id": result["user_id"]}
            )
            
            # Save token to database
            save_user_token(result["user_id"], access_token)
            
            return {
                "message": "Login successful",
                "user": result,
                "access_token": access_token,
                "token_type": "bearer"
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid username or password")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def logout_user(current_user: str = Depends(get_current_user)):
    """Logout user by clearing their token"""
    try:
        user_info = get_user_by_username(current_user)
        if user_info:
            # Clear token from database
            clear_user_token(user_info["user_id"])
            return {"message": "Logout successful"}
        else:
            raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ConversationCreateRequest(BaseModel):
    first_message: str

class ConversationResponse(BaseModel):
    conversation_id: int
    user_id: int
    timestamp: str
    messages: list

@router.get("/conversations")
async def get_user_conversations(
    current_username: str = Depends(get_current_user)
):
    user = get_user_by_username(current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        with Session(engine) as session:
            statement = select(ConversationModel).where(
                ConversationModel.user_id == user["user_id"]
            ).order_by(ConversationModel.timestamp.desc())
            conversations = session.exec(statement).all()
            
            result = []
            for conv in conversations:
                # Parse messages to get first message
                messages = json.loads(conv.messages) if conv.messages else []
                first_message = ""
                if messages:
                    # Find first user message
                    for msg in messages:
                        if msg.get("role") == "user":
                            first_message = msg.get("content", "")
                            break
                
                result.append({
                    "conversation_id": conv.conversation_id,
                    "user_id": conv.user_id,
                    "timestamp": conv.timestamp.isoformat(),
                    "first_message": first_message
                })
            
            return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")

@router.post("/conversations", response_model = ConversationResponse)
async def start_conversation(
    request: ConversationCreateRequest,
    current_username: str = Depends(get_current_user)
):
    user = get_user_by_username(current_username)
    if not user:
        raise HTTPException(status_code = 404, detail = "User not found")

    conv = create_conversation(user_id = user["user_id"], first_message = request.first_message)

    try:
        response_text = await call_custom_api(single_message=request.first_message)

        conv["messages"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        })

        with Session(engine) as session:
            statement = select(ConversationModel).where(
                ConversationModel.conversation_id == conv["conversation_id"],
                ConversationModel.user_id == user["user_id"]
            )
            db_conv = session.exec(statement).first()
            if db_conv:
                db_conv.messages = json.dumps(conv["messages"])
                session.add(db_conv)
                session.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom API error: {str(e)}")

    return conv

@router.get("/conversations/{conversation_id}", response_model = ConversationResponse)
async def read_conversation(
    conversation_id: int,
    current_username: str = Depends(get_current_user)
):
    user = get_user_by_username(current_username)
    if not user:
        raise HTTPException(status_code = 404, detail = "User not found")
    
    conv = get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code = 404, detail = "Conversation not found")
    
    return conv

@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: int,
    current_username: str = Depends(get_current_user)
):

    user = get_user_by_username(current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    success = delete_conversation(conversation_id, user["user_id"])
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found or cannot delete")

    return {"message": "Conversation deleted successfully"}


class MessageRequest(BaseModel):
    content: str

@router.post("/conversations/{conversation_id}/messages", response_model=ConversationResponse)
async def add_message_to_conversation(
    conversation_id: int = Path(...),
    request: MessageRequest = None,
    current_username: str = Depends(get_current_user)
):
    user = get_user_by_username(current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate message content
    if not request or not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    conv = get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv["messages"].append({
        "role": "user",
        "content": request.content.strip(),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    try:
        response_text = await call_custom_api(messages_list=conv["messages"])

        conv["messages"].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        })

        with Session(engine) as session:
            statement = select(ConversationModel).where(
                ConversationModel.conversation_id == conversation_id,
                ConversationModel.user_id == user["user_id"]
            )
            db_conv = session.exec(statement).first()
            if db_conv:
                db_conv.messages = json.dumps(conv["messages"])
                session.add(db_conv)
                session.commit()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom API error: {str(e)}")

    return conv

@router.post("/conversations/{conversation_id}/messages/stream")
async def add_message_to_conversation_stream(
    conversation_id: int = Path(...),
    request: MessageRequest = None,
    current_username: str = Depends(get_current_user)
):
    logger.info(f"Streaming endpoint called for conversation {conversation_id}")
    user = get_user_by_username(current_username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Validate message content
    if not request or not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    conv = get_conversation(conversation_id, user["user_id"])
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    logger.debug(f"User message received", extra={"content_length": len(request.content)})
    # Add user message to conversation
    user_message = {
        "role": "user",
        "content": request.content.strip(),
        "timestamp": datetime.utcnow().isoformat()
    }
    conv["messages"].append(user_message)

    # Prepare message history for Custom API
    logger.debug(f"Conversation loaded with {len(conv['messages'])} messages")

    async def generate_stream():
        try:
            logger.info("Starting streaming generation")
            # Send user message first
            yield f"data: {json.dumps({'type': 'user_message', 'message': user_message})}\n\n"
            
            # Start assistant message
            assistant_message = {
                "role": "assistant",
                "content": "",
                "timestamp": datetime.utcnow().isoformat()
            }
            yield f"data: {json.dumps({'type': 'assistant_start', 'message': assistant_message})}\n\n"
            
            logger.debug("Calling Custom API with streaming")
            # Generate streaming response from Custom API
            full_content = ""
            chunk_count = 0
            
            logger.debug("Starting to process chunks from Custom API")
            async for chunk_text in stream_custom_api(conv["messages"]):
                chunk_count += 1
                logger.debug(f"Processing chunk {chunk_count}")
                full_content += chunk_text
                yield f"data: {json.dumps({'type': 'assistant_chunk', 'content': chunk_text})}\n\n"
            
            logger.info(f"Completed processing {chunk_count} chunks", extra={"content_length": len(full_content)})
            
            # Complete assistant message
            assistant_message["content"] = full_content
            conv["messages"].append(assistant_message)
            
            logger.debug("Saving to database")
            # Save to database
            with Session(engine) as session:
                statement = select(ConversationModel).where(
                    ConversationModel.conversation_id == conversation_id,
                    ConversationModel.user_id == user["user_id"]
                )
                db_conv = session.exec(statement).first()
                if db_conv:
                    db_conv.messages = json.dumps(conv["messages"])
                    session.add(db_conv)
                    session.commit()
            
            logger.debug("Database save complete, sending completion signals")
            # Send completion signal
            yield f"data: {json.dumps({'type': 'assistant_complete', 'message': assistant_message})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
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
            "Access-Control-Allow-Headers": "*",
        }
    )
