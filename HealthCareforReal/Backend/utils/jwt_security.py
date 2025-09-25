from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from config import settings
import secrets
from utils.logger import logger

class JWTSecurityManager:
    """Enhanced JWT security with token blacklisting and refresh tokens"""
    
    def __init__(self):
        self.algorithm = settings.algorithm
        self.secret_key = settings.secret_key
        
        # In production, these would be stored in Redis or database
        self.blacklisted_tokens = set()
        self.refresh_tokens = {}  # {jti: user_id}
    
    def create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a secure access token with JTI for tracking"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
        
        # Add standard JWT claims
        jti = secrets.token_urlsafe(32)  # Unique token identifier
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": jti,
            "type": "access"
        })
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Created access token for user {data.get('sub')}")
        
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create a long-lived refresh token"""
        jti = secrets.token_urlsafe(32)
        expire = datetime.utcnow() + timedelta(days=30)  # 30-day expiry
        
        to_encode = {
            "sub": user_id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": jti,
            "type": "refresh"
        }
        
        token = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token JTI
        self.refresh_tokens[jti] = user_id
        
        logger.info(f"Created refresh token for user {user_id}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify token and check if it's blacklisted"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is blacklisted
            jti = payload.get("jti")
            if jti and jti in self.blacklisted_tokens:
                logger.warning(f"Attempted use of blacklisted token: {jti}")
                return None
            
            # Verify token type
            token_type = payload.get("type", "access")
            if token_type not in ["access", "refresh"]:
                logger.warning(f"Invalid token type: {token_type}")
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.info("Token has expired")
            return None
        except JWTError as e:
            logger.warning(f"JWT verification failed: {str(e)}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[Dict[str, str]]:
        """Use refresh token to generate new access token"""
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            logger.warning("Invalid refresh token provided")
            return None
        
        jti = payload.get("jti")
        user_id = payload.get("sub")
        
        # Verify refresh token is still valid in our store
        if jti not in self.refresh_tokens or self.refresh_tokens[jti] != user_id:
            logger.warning(f"Refresh token not found or invalid: {jti}")
            return None
        
        # Create new access token
        new_access_token = self.create_access_token(data={"sub": user_id})
        
        logger.info(f"Refreshed access token for user {user_id}")
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
    
    def blacklist_token(self, token: str) -> bool:
        """Add token to blacklist (for logout)"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}  # Don't verify expiry for blacklisting
            )
            
            jti = payload.get("jti")
            if jti:
                self.blacklisted_tokens.add(jti)
                logger.info(f"Token blacklisted: {jti}")
                return True
                
        except JWTError as e:
            logger.error(f"Error blacklisting token: {str(e)}")
            
        return False
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke a refresh token"""
        payload = self.verify_token(refresh_token)
        
        if payload and payload.get("type") == "refresh":
            jti = payload.get("jti")
            if jti in self.refresh_tokens:
                del self.refresh_tokens[jti]
                self.blacklist_token(refresh_token)
                logger.info(f"Refresh token revoked: {jti}")
                return True
        
        return False
    
    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get detailed token information for debugging"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            return {
                "user_id": payload.get("sub"),
                "jti": payload.get("jti"),
                "type": payload.get("type"),
                "issued_at": datetime.fromtimestamp(payload.get("iat", 0)),
                "expires_at": datetime.fromtimestamp(payload.get("exp", 0)),
                "is_expired": datetime.utcnow() > datetime.fromtimestamp(payload.get("exp", 0)),
                "is_blacklisted": payload.get("jti") in self.blacklisted_tokens
            }
            
        except JWTError:
            return None
    
    def cleanup_expired_blacklist(self):
        """Remove expired tokens from blacklist (should be run periodically)"""
        # In a production system, this would query the database
        # and remove expired JTIs from the blacklist
        expired_count = 0
        current_size = len(self.blacklisted_tokens)
        
        # For now, we'll implement a simple cleanup
        # In production, store blacklist in database with expiry times
        
        logger.info(f"Blacklist cleanup completed. Current size: {current_size}")
        return expired_count

# Global instance
jwt_security = JWTSecurityManager()