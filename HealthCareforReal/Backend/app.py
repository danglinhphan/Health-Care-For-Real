from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tasksapi.routes import router as api_router
from config import settings

app = FastAPI(
    title=settings.app_name,
    description="API for managing tasks and conversations",
    version="0.0.1",
    debug=settings.debug,
)

# Enable CORS - This must be added BEFORE including routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "environment": settings.environment}

# Prefix is used to group routes under a common path
app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)