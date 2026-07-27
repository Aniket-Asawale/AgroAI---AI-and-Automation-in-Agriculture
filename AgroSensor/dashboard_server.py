"""
AgroSensor Dashboard Server
Runs the dashboard on port 8502 while the API runs on port 8000
"""

import os
import shutil
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Get the directory where this script is located
DASHBOARD_DIR = Path(__file__).parent / "dashboard"

# Create FastAPI app for dashboard only
app = FastAPI(title="AgroSensor Dashboard", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static dashboard files
if DASHBOARD_DIR.exists():
    # Mount static files at the correct paths - this should handle /sensor/static/*
    app.mount("/sensor/static", StaticFiles(directory=str(DASHBOARD_DIR)), name="sensor_static")
    
    @app.get("/")
    async def serve_dashboard():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))
    
    @app.get("/sensor/dashboard")
    async def serve_dashboard_path():
        return FileResponse(str(DASHBOARD_DIR / "index.html"))
    
    # Explicit routes for /sensor/static/ files (required for tunnel)
    @app.get("/sensor/static/style.css")
    async def serve_sensor_static_css():
        return FileResponse(str(DASHBOARD_DIR / "style.css"))
    
    @app.get("/sensor/static/app.js")
    async def serve_sensor_static_js():
        return FileResponse(str(DASHBOARD_DIR / "app.js"))
    
    # Fallback routes for direct access (local testing)
    @app.get("/style.css")
    async def serve_css():
        return FileResponse(str(DASHBOARD_DIR / "style.css"))
    
    @app.get("/app.js")
    async def serve_js():
        return FileResponse(str(DASHBOARD_DIR / "app.js"))
else:
    raise FileNotFoundError(f"Dashboard directory not found: {DASHBOARD_DIR}")

if __name__ == "__main__":
    print("Dashboard server starting on port 8502")
    print("Access dashboard at: http://127.0.0.1:8502")
    print("API should be running on: http://127.0.0.1:8000")
    
    uvicorn.run(
        "dashboard_server:app",
        host="127.0.0.1",
        port=8502,
        reload=False,
    )
