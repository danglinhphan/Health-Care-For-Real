#!/bin/bash
# Qwen LoRA API cURL Examples

API_BASE="http://localhost:8000"

echo "=== Qwen LoRA API cURL Examples ==="

# Health check
echo -e "\n1. Health Check:"
curl -s "${API_BASE}/" | python -m json.tool

# List models
echo -e "\n2. List Models:"
curl -s "${API_BASE}/v1/models" | python -m json.tool

# Get model info
echo -e "\n3. Get Model Info:"
curl -s "${API_BASE}/v1/models/qwen-lora" | python -m json.tool

# Simple generation
echo -e "\n4. Simple Generation:"
curl -s -X POST "${API_BASE}/v1/models/qwen-lora:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "What is artificial intelligence?"}]
      }
    ],
    "generation_config": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_output_tokens": 200
    }
  }' | python -m json.tool

# Streaming generation
echo -e "\n5. Streaming Generation:"
curl -s -X POST "${API_BASE}/v1/models/qwen-lora:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Write a short poem about technology."}]
      }
    ],
    "generation_config": {
      "temperature": 0.8,
      "max_output_tokens": 100
    },
    "stream": true
  }'

# Multi-turn conversation
echo -e "\n6. Multi-turn Conversation:"
curl -s -X POST "${API_BASE}/v1/models/qwen-lora:generateContent" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      {
        "role": "user",
        "parts": [{"text": "Hello, how are you?"}]
      },
      {
        "role": "model",
        "parts": [{"text": "Hello! I am doing well, thank you for asking. How can I help you today?"}]
      },
      {
        "role": "user",
        "parts": [{"text": "Can you explain quantum computing?"}]
      }
    ],
    "generation_config": {
      "temperature": 0.6,
      "max_output_tokens": 300
    }
  }' | python -m json.tool

echo -e "\n=== Examples Complete ==="