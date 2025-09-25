import time
from collections import defaultdict
from typing import Dict, List
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from utils.logger import logger

class RateLimiter:
    """Rate limiter for authentication endpoints"""
    
    def __init__(self, max_requests: int = 5, window_seconds: int = 300):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in the time window
            window_seconds: Time window in seconds (default: 5 minutes)
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed for the client IP"""
        now = time.time()
        client_requests = self.requests[client_ip]
        
        # Remove old requests outside the window
        self.requests[client_ip] = [
            req_time for req_time in client_requests 
            if now - req_time < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return False
        
        # Add current request
        self.requests[client_ip].append(now)
        return True
    
    def get_reset_time(self, client_ip: str) -> int:
        """Get time until rate limit resets"""
        if client_ip not in self.requests or not self.requests[client_ip]:
            return 0
        
        oldest_request = min(self.requests[client_ip])
        reset_time = int(oldest_request + self.window_seconds - time.time())
        return max(0, reset_time)

# Global rate limiter instances
login_rate_limiter = RateLimiter(max_requests=5, window_seconds=300)  # 5 attempts per 5 minutes
register_rate_limiter = RateLimiter(max_requests=3, window_seconds=600)  # 3 attempts per 10 minutes

async def rate_limit_auth(request: Request, limiter: RateLimiter):
    """Middleware function to apply rate limiting"""
    client_ip = request.client.host if request.client else "unknown"
    
    if not limiter.is_allowed(client_ip):
        reset_time = limiter.get_reset_time(client_ip)
        logger.warning(f"Rate limit exceeded for {client_ip}. Reset in {reset_time} seconds.")
        
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Too many requests",
                "message": f"Rate limit exceeded. Try again in {reset_time} seconds.",
                "retry_after": reset_time
            }
        )