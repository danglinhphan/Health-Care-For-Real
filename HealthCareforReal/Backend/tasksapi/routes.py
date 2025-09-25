from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from tasksapi.controllers.auth_controller import AuthController
from tasksapi.controllers.conversation_controller import ConversationController
from tasksapi.controllers.ai_controller import AIController, MessageRequest
from tasksapi.crud.user import UserCreate, UserLogin
from tasksapi.utils.utils import get_current_user

router = APIRouter()
auth_controller = AuthController()
conversation_controller = ConversationController()
ai_controller = AIController()

# Authentication routes
@router.post("/register")
async def register_user(user_data: UserCreate, request: Request):
    return await auth_controller.register_user(user_data, request)

@router.post("/login")
async def login_user(login_data: UserLogin, request: Request):
    return await auth_controller.login_user(login_data, request)

@router.post("/logout")
async def logout_user(request: Request, current_user: dict = Depends(get_current_user)):
    return await auth_controller.logout_user(request, current_user)

@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return await auth_controller.get_current_user_info(current_user)

# Conversation routes
@router.get("/conversations")
async def get_user_conversations(current_user: dict = Depends(get_current_user)):
    return await conversation_controller.get_user_conversations(current_user)

class ConversationRequest(BaseModel):
    first_message: str

@router.post("/conversations")
async def start_conversation(request: ConversationRequest, current_user: dict = Depends(get_current_user)):
    return await conversation_controller.start_conversation(request.first_message, current_user)

@router.get("/conversations/{conversation_id}")
async def read_conversation(conversation_id: int, current_user: dict = Depends(get_current_user)):
    return await conversation_controller.read_conversation(conversation_id, current_user)

@router.delete("/conversations/{conversation_id}")
async def delete_conversation_endpoint(conversation_id: int, current_user: dict = Depends(get_current_user)):
    return await conversation_controller.delete_conversation_endpoint(conversation_id, current_user)

# AI/Message routes
@router.post("/conversations/{conversation_id}/messages")
async def add_message_to_conversation(conversation_id: int, request: MessageRequest, current_user: dict = Depends(get_current_user)):
    return await ai_controller.add_message_to_conversation(conversation_id, request, current_user)

@router.post("/conversations/{conversation_id}/messages/stream")
async def add_message_to_conversation_stream(conversation_id: int, request: MessageRequest, current_user: dict = Depends(get_current_user)):
    return await ai_controller.add_message_to_conversation_stream(conversation_id, request, current_user)

# JWT token management routes
class RefreshTokenRequest(BaseModel):
    refresh_token: str

@router.post("/auth/refresh")
async def refresh_access_token(request: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    return await auth_controller.refresh_token(request.refresh_token)

@router.post("/auth/revoke")
async def revoke_refresh_token(request: RefreshTokenRequest):
    """Revoke a refresh token"""
    return await auth_controller.revoke_refresh_token(request.refresh_token)
