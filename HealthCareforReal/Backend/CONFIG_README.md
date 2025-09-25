# Backend Configuration System

## 📁 Files Created

### 1. `config.py` - Main Configuration Class
- **Purpose**: Centralized configuration management using Pydantic Settings
- **Features**:
  - Environment variable support
  - Default values with production overrides
  - Type validation
  - Cached settings instance

### 2. `.env.example` - Environment Template
- **Purpose**: Template for environment variables
- **Usage**: Copy to `.env` and customize for your environment

### 3. Updated Files
- **`controllers.py`**: Uses `settings.gemini_api_key` and `settings.gemini_model`
- **`db.py`**: Uses `settings.database_url` and connection settings
- **`app.py`**: Uses `settings.allowed_origins` and app configuration
- **`utils.py`**: Uses `settings.secret_key` and JWT settings

## 🔧 Configuration Categories

### Database Settings
```python
db_host: str = "localhost"
db_port: int = 3306
db_user: str = "user"
db_password: str = "pass"
db_name: str = "tasksdb"
```

### Security Settings
```python
secret_key: str = "your-secret-key-here"
algorithm: str = "HS256"
access_token_expire_minutes: int = 30
```

### AI/Gemini Settings
```python
gemini_api_key: str = "your-api-key"
gemini_model: str = "gemini-2.0-flash-exp"
```

### CORS & Server Settings
```python
allowed_origins: list = ["http://localhost:3000"]
host: str = "0.0.0.0"
port: int = 8000
```

## 🚀 Usage Examples

### In Code
```python
from config import settings

# Database URL
DATABASE_URL = settings.database_url

# API Key
client = genai.Client(api_key=settings.gemini_api_key)

# JWT Settings
SECRET_KEY = settings.secret_key
```

### Environment Variables
```bash
# Override in .env file
GEMINI_API_KEY=your-real-api-key
SECRET_KEY=super-secure-production-key
DB_PASSWORD=secure-password
```

### Docker Override
```yaml
environment:
  - ENVIRONMENT=production
  - DB_HOST=mariadb
  - GEMINI_API_KEY=${GEMINI_API_KEY}
```

## ✅ Benefits

1. **Centralized**: All config in one place
2. **Type Safe**: Pydantic validation
3. **Environment Aware**: Development vs Production
4. **Secure**: Secrets in environment variables
5. **Docker Ready**: Easy container configuration
6. **Cached**: Performance optimized with @lru_cache

## 🔐 Security Best Practices

1. **Never commit .env** - Add to .gitignore
2. **Use strong secrets** in production
3. **Rotate API keys** regularly
4. **Use environment-specific configs**
5. **Validate settings** on startup
