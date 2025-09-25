from fastapi import HTTPException, Request, Depends
from tasksapi.crud.user import create_user, verify_user_login, get_user_by_username, UserCreate, UserLogin, save_user_token, clear_user_token
from tasksapi.utils.utils import get_current_user
from utils.logger import logger
from utils.jwt_security import jwt_security
from middleware.rate_limiter import rate_limit_auth, login_rate_limiter, register_rate_limiter

class AuthController:
    """Handles user authentication operations with enhanced JWT security"""
    
    @staticmethod
    async def refresh_token(refresh_token: str):
        """Refresh access token using refresh token"""
        try:
            result = jwt_security.refresh_access_token(refresh_token)
            
            if not result:
                raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
            
            logger.info("Access token refreshed successfully")
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token refresh error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Token refresh failed")
    
    @staticmethod
    async def revoke_refresh_token(refresh_token: str):
        """Revoke a refresh token"""
        try:
            success = jwt_security.revoke_refresh_token(refresh_token)
            
            if not success:
                raise HTTPException(status_code=400, detail="Invalid refresh token")
            
            logger.info("Refresh token revoked successfully")
            return {"message": "Refresh token revoked successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token revocation error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Token revocation failed")
    
    @staticmethod
    async def register_user(user_data: UserCreate, request: Request):
        """Register a new user with rate limiting"""
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
                logger.info(f"User registered successfully: {user_data.username}")
                return {"message": "User registered successfully", "user": result}
            else:
                raise HTTPException(status_code=500, detail="Failed to register user")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Registration error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def login_user(login_data: UserLogin, request: Request):
        """Authenticate user with rate limiting"""
        # Apply rate limiting
        await rate_limit_auth(request, login_rate_limiter)
        
        try:
            result = verify_user_login(login_data.username, login_data.password)
            
            if result:
                # Create access and refresh tokens using enhanced security
                access_token = jwt_security.create_access_token(data={"sub": str(result["user_id"])})
                refresh_token = jwt_security.create_refresh_token(str(result["user_id"]))
                
                # Save access token to database (for compatibility)
                save_user_token(result["user_id"], access_token)
                
                logger.info(f"User logged in successfully: {login_data.username}")
                return {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "token_type": "bearer",
                    "user": result,
                    "message": "Login successful"
                }
            else:
                raise HTTPException(status_code=401, detail="Invalid username or password")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Login error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def logout_user(request: Request, current_user: dict = Depends(get_current_user)):
        """Logout user and invalidate tokens"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            # Extract token from request header
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                
                # Blacklist the current access token
                jwt_security.blacklist_token(token)
            
            # Clear user token from database
            clear_user_token(current_user["user_id"])
            
            logger.info(f"User logged out: {current_user.get('username', 'unknown')}")
            return {"message": "Successfully logged out"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Logout error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_current_user_info(current_user: dict = Depends(get_current_user)):
        """Get current authenticated user information"""
        try:
            if not current_user:
                raise HTTPException(status_code=401, detail="Not authenticated")
            
            # Remove sensitive information
            safe_user = {
                "user_id": current_user.get("user_id"),
                "username": current_user.get("username"),
                "emailaddress": current_user.get("emailaddress")
            }
            
            return {"user": safe_user}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Get user info error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=404, detail="User not found")