import logging
import sys
from datetime import datetime
from pathlib import Path
from config import settings

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup structured logging for the application"""
    
    logger = logging.getLogger(name)
    
    # Don't add handlers multiple times
    if logger.handlers:
        return logger
    
    # Set log level based on environment
    if settings.environment == "development" or settings.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    logger.addHandler(console_handler)
    
    # File handler for production
    if settings.environment == "production":
        # Ensure logs directory exists
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"healthcare_chatbot_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
    
    return logger

# Create default logger
logger = setup_logger("healthcare_chatbot")