"""
Combined AgroSensor Server - Serves both API and Dashboard
"""
import os
import shutil
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import the main app
from main import app as api_app

# Get the directory where this script is located
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

# Create FastAPI app
app = FastAPI(title="AgroSensor Combined", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the API app (remove /api prefix since router already has it)
app.mount("/", api_app)

# Serve dashboard static files
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="static")
    
    @app.get("/")
    async def serve_dashboard():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))
    
    @app.get("/sensor/dashboard")
    async def serve_dashboard_path():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))
    
    # Serve CSS and JS files directly
    @app.get("/style.css")
    async def serve_css():
        return FileResponse(str(DASHBOARD_DIR / "style.css"))
    
    @app.get("/app.js")
    async def serve_js():
        return FileResponse(str(DASHBOARD_DIR / "app.js"))
    
    @app.get("/sensor/static/style.css")
    async def serve_sensor_css():
        return FileResponse(str(DASHBOARD_DIR / "style.css"))
    
    @app.get("/sensor/static/app.js")
    async def serve_sensor_js():
        return FileResponse(str(DASHBOARD_DIR / "app.js"))
else:
    raise FileNotFoundError(f"Dashboard directory not found: {DASHBOARD_DIR}")

if __name__ == "__main__":
    print("Combined AgroSensor server starting on port 8000")
    print("Access dashboard at: http://127.0.0.1:8000/sensor/dashboard")
    print("Access API at: http://127.0.0.1:8000/api/")
    
    uvicorn.run(
        "combined_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
