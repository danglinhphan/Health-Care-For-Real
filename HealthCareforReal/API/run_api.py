#!/usr/bin/env python3
"""
Launch script for Qwen LoRA API
"""

import argparse
import sys
import os
import yaml
import uvicorn
from pathlib import Path

def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = "config/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file {config_path} not found, using defaults")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def main():
    parser = argparse.ArgumentParser(description="Qwen LoRA API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    api_config = config.get("api", {})
    host = args.host or api_config.get("host", "0.0.0.0")
    port = args.port or api_config.get("port", 8000)
    workers = args.workers or api_config.get("workers", 1)
    reload = args.reload or api_config.get("reload", False)
    log_level = args.log_level or api_config.get("log_level", "info")
    
    print(f"Starting Qwen LoRA API server on {host}:{port}")
    print(f"Workers: {workers}, Reload: {reload}, Log level: {log_level}")
    
    # Start server
    uvicorn.run(
        "src.api_server:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
        log_level=log_level,
        access_log=True,
        app_dir=str(Path(__file__).parent)
    )

if __name__ == "__main__":
    main()