# Chat UI Loading Issue - Troubleshooting Guide

## Problem
The UI shows loading but doesn't respond when chatting.

## Root Cause
Your custom API is very slow (50+ seconds response time), causing frontend timeouts.

## Fixes Applied

### 1. **Backend Timeout Extended** (`Backend/tasksapi/controllers.py`)
- Increased HTTP client timeout from 30s to 120s (2 minutes)
- This prevents backend from timing out while waiting for your API

### 2. **Frontend Timeout Extended** (`Frontend/lib/api.ts`)
- Added 3-minute timeout with AbortController
- Better error messages for timeouts
- Improved error handling for slow responses

### 3. **API Performance Optimized** (`API/src/api_server.py`)
- Reduced `max_tokens` from 1024 to 256 (faster generation)
- Reduced `top_k` from 40 to 20 (faster sampling)
- These changes make responses faster but slightly shorter

## Testing the Fix

### Quick Test
```bash
# Test the custom API directly
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Full Test
```bash
# 1. Start all services
docker-compose up

# 2. Or start individually:
# Terminal 1: Custom API
cd API && python run_api.py

# Terminal 2: Backend
cd Backend && uvicorn app:app --port 3002

# Terminal 3: Frontend
cd Frontend && npm run dev
```

## Current Status
- ✅ Backend running (port 3002)
- ✅ Custom API running (port 3001) 
- ✅ API responding (but slowly ~50s)
- ✅ Timeout fixes applied
- ⏳ Performance optimizations applied

## Expected Behavior Now
1. **UI Loading**: You'll see loading indicator
2. **Wait Time**: 30-60 seconds for response (instead of infinite loading)
3. **Response**: Should get shorter but faster responses
4. **Error Handling**: Better error messages if something fails

## Further Optimizations (if still slow)

### Option 1: Use Smaller Model
If still too slow, consider using a smaller/faster model in your inference engine.

### Option 2: GPU Acceleration
Ensure your API is using GPU if available:
```python
# In inference_engine.py, check device settings
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Option 3: Reduce Context Length
Limit conversation history to last few messages to speed up processing.

## Debug Commands

### Check Service Status
```bash
# Check if services are running
curl http://localhost:3002/health  # Backend
curl http://localhost:3001/        # Custom API
```

### Monitor Logs
```bash
# Watch backend logs
docker-compose logs -f backend

# Watch API logs  
docker-compose logs -f custom-api
```

## Contact
If issues persist, the problem is likely in the inference engine performance. Consider:
1. Using a faster model
2. GPU acceleration
3. Model quantization
4. Reducing model parameters