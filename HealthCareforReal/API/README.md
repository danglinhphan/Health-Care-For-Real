# Qwen LoRA API

A Gemini API-compatible interface for Qwen LoRA fine-tuned models. This API provides a RESTful interface that matches the Google Gemini API format, making it easy to integrate with existing Gemini-compatible applications.

## Features

- 🚀 **Gemini API Compatible**: Drop-in replacement for Gemini API endpoints
- ⚡ **Fast Inference**: Optimized for RTX 4090 with BF16 precision
- 🔄 **Streaming Support**: Real-time response streaming
- 🐳 **Docker Ready**: Containerized deployment with GPU support
- 📊 **Token Counting**: Accurate token usage tracking
- 🔧 **Configurable**: Flexible configuration via YAML

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (RTX 4090 recommended)
- PyTorch 2.0+
- Trained Qwen LoRA adapters

### Installation

1. **Clone or extract the API export**:
   ```bash
   cd qwen-api-export
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package**:
   ```bash
   pip install -e .
   ```

### Running the API

#### Method 1: Using the run script (Recommended)

```bash
python run_api.py --host 0.0.0.0 --port 8000
```

#### Method 2: Using uvicorn directly

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

#### Method 3: Using Docker

```bash
# Build the image
docker build -t qwen-lora-api .

# Run with GPU support
docker run --gpus all -p 3001:8000 qwen-lora-api
```

#### Method 4: Using Docker Compose

```bash
docker-compose up -d
```

## API Usage

The API follows the Gemini API format exactly. Here are some examples:

### Basic Text Generation

```python
import requests

url = "http://localhost:3001/v1/models/qwen-lora:generateContent"
payload = {
    "contents": [
        {
            "role": "user",
            "parts": [{"text": "Explain quantum computing in simple terms"}]
        }
    ],
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_output_tokens": 1024
    }
}

response = requests.post(url, json=payload)
result = response.json()
print(result["candidates"][0]["content"]["parts"][0]["text"])
```

### Streaming Generation

```python
import requests

url = "http://localhost:3001/v1/models/qwen-lora:generateContent"
payload = {
    "contents": [
        {
            "role": "user", 
            "parts": [{"text": "Write a short story about AI"}]
        }
    ],
    "stream": True
}

response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line.startswith(b"data: "):
        data = line[6:].decode()
        if data != "[DONE]":
            import json
            chunk = json.loads(data)
            text = chunk["candidates"][0]["content"]["parts"][0]["text"]
            print(text, end="", flush=True)
```

### List Models

```python
import requests

response = requests.get("http://localhost:3001/v1/models")
models = response.json()
print(models)
```

## Configuration

Edit `config/config.yaml` to customize the API:

```yaml
model:
  name: "Qwen/Qwen3-1.7B"
  adapter_path: "./models/qwen_lora_adapters"
  device: "auto"
  torch_dtype: "bfloat16"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1

generation:
  default_temperature: 0.7
  default_top_p: 0.9
  default_top_k: 40
  default_max_tokens: 1024
```

## Endpoints

### Core Endpoints

- `GET /` - Health check
- `GET /v1/models` - List available models
- `GET /v1/models/{model_name}` - Get model information
- `POST /v1/models/qwen-lora:generateContent` - Generate content
- `POST /v1/models/qwen-lora:streamGenerateContent` - Stream generate content

### Request Format

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [{"text": "Your prompt here"}]
    }
  ],
  "generation_config": {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 1024,
    "stop_sequences": ["STOP"]
  },
  "stream": false
}
```

### Response Format

```json
{
  "candidates": [
    {
      "content": {
        "role": "model",
        "parts": [{"text": "Generated response"}]
      },
      "finish_reason": "STOP",
      "index": 0
    }
  ],
  "usage_metadata": {
    "prompt_tokens": 15,
    "candidates_tokens": 127,
    "total_tokens": 142
  },
  "model_version": "qwen-lora-v1"
}
```

## Performance Optimization

### GPU Memory Management

The API automatically optimizes for RTX 4090:

- BF16 precision for Ada Lovelace architecture
- Flash Attention 2 for faster inference
- Memory-efficient loading and caching
- Automatic garbage collection

### Concurrent Requests

- Uses thread-safe inference engine
- Single-threaded execution to prevent OOM
- Request queuing for stability

## Deployment

### Production Deployment

1. **Use a reverse proxy** (nginx/traefik):
   ```nginx
   location /api/ {
       proxy_pass http://localhost:3001/;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```

2. **Set up monitoring**:
   - Health checks: `GET /`
   - Metrics collection via logs
   - GPU monitoring

3. **Scale horizontally**:
   - Run multiple instances on different GPUs
   - Use load balancer for distribution

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Specify GPU devices
- `PYTORCH_CUDA_ALLOC_CONF`: CUDA memory allocation config
- `HF_HOME`: Hugging Face cache directory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce `max_output_tokens` in generation config
   - Lower batch size or use gradient checkpointing
   - Restart the API to clear GPU cache

2. **Model Loading Errors**:
   - Verify adapter path in config
   - Check GPU memory availability
   - Ensure proper CUDA installation

3. **Slow Inference**:
   - Enable Flash Attention 2
   - Use BF16 precision
   - Optimize CUDA settings

### Logs

Check logs for detailed error information:
```bash
# View logs in real-time
tail -f logs/api.log

# Check Docker logs
docker logs qwen-api
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs for error details
- Ensure GPU compatibility and drivers