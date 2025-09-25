#!/usr/bin/env python3
"""
Example client for Qwen LoRA API
"""

import requests
import json
import time
from typing import Iterator

class QwenAPIClient:
    """Client for Qwen LoRA API with Gemini-compatible interface"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
    def list_models(self) -> dict:
        """List available models"""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_name: str = "qwen-lora") -> dict:
        """Get model information"""
        response = self.session.get(f"{self.base_url}/v1/models/{model_name}")
        response.raise_for_status()
        return response.json()
    
    def generate_content(self,
                        prompt: str,
                        temperature: float = 0.7,
                        top_p: float = 0.9,
                        top_k: int = 40,
                        max_tokens: int = 1024,
                        stop_sequences: list = None) -> dict:
        """Generate content (non-streaming)"""
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generation_config": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
                "stop_sequences": stop_sequences
            },
            "stream": False
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/models/qwen-lora:generateContent",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def stream_generate_content(self,
                               prompt: str,
                               temperature: float = 0.7,
                               top_p: float = 0.9,
                               top_k: int = 40,
                               max_tokens: int = 1024,
                               stop_sequences: list = None) -> Iterator[dict]:
        """Generate content (streaming)"""
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generation_config": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_tokens,
                "stop_sequences": stop_sequences
            },
            "stream": True
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/models/qwen-lora:generateContent",
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                data = line[6:].decode()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    yield chunk
                except json.JSONDecodeError:
                    continue
    
    def chat(self, messages: list, **kwargs) -> dict:
        """Multi-turn chat"""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({
                "role": role,
                "parts": [{"text": msg["content"]}]
            })
        
        payload = {
            "contents": contents,
            "generation_config": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40),
                "max_output_tokens": kwargs.get("max_tokens", 1024),
                "stop_sequences": kwargs.get("stop_sequences")
            },
            "stream": kwargs.get("stream", False)
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/models/qwen-lora:generateContent",
            json=payload,
            stream=kwargs.get("stream", False)
        )
        response.raise_for_status()
        
        if kwargs.get("stream", False):
            return self._parse_stream_response(response)
        else:
            return response.json()
    
    def _parse_stream_response(self, response):
        """Parse streaming response"""
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                data = line[6:].decode()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    yield chunk
                except json.JSONDecodeError:
                    continue

def main():
    """Example usage"""
    
    # Initialize client
    client = QwenAPIClient("http://localhost:8000")
    
    print("=== Qwen LoRA API Client Example ===\n")
    
    # Test health check
    try:
        models = client.list_models()
        print("✓ API is healthy")
        print(f"Available models: {len(models['models'])}")
    except Exception as e:
        print(f"✗ API health check failed: {e}")
        return
    
    # Example 1: Simple generation
    print("\n1. Simple Generation:")
    try:
        response = client.generate_content(
            prompt="Explain artificial intelligence in simple terms.",
            temperature=0.7,
            max_tokens=200
        )
        
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        usage = response.get("usage_metadata", {})
        
        print(f"Response: {text}")
        print(f"Tokens used: {usage.get('total_tokens', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Generation failed: {e}")
    
    # Example 2: Streaming generation
    print("\n2. Streaming Generation:")
    try:
        print("Response: ", end="", flush=True)
        
        for chunk in client.stream_generate_content(
            prompt="Write a haiku about programming.",
            temperature=0.8,
            max_tokens=100
        ):
            if "candidates" in chunk:
                text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                print(text, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"✗ Streaming failed: {e}")
    
    # Example 3: Multi-turn chat
    print("\n3. Multi-turn Chat:")
    try:
        messages = [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"role": "user", "content": "Can you give me an example?"}
        ]
        
        response = client.chat(messages, temperature=0.6, max_tokens=150)
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        
        print(f"Assistant: {text}")
        
    except Exception as e:
        print(f"✗ Chat failed: {e}")
    
    # Example 4: Performance test
    print("\n4. Performance Test:")
    try:
        start_time = time.time()
        
        response = client.generate_content(
            prompt="Count from 1 to 10.",
            temperature=0.1,
            max_tokens=50
        )
        
        end_time = time.time()
        latency = end_time - start_time
        
        usage = response.get("usage_metadata", {})
        total_tokens = usage.get("total_tokens", 0)
        
        print(f"Latency: {latency:.2f}s")
        print(f"Tokens/sec: {total_tokens/latency:.1f}" if latency > 0 else "N/A")
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")

if __name__ == "__main__":
    main()