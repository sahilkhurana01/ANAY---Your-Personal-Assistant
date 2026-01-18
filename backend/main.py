"""
ANAY - Local Personal AI Assistant
Main Application Entry Point (FastAPI)

Architecture:
- Frontend (React/Voice) connects via WebSocket
- Agent Loop (WebSocketManager) handles "Listen -> Plan -> Execute -> Respond"
- Planner (TaskPlanner) uses Groq LLM + Persistent Context
- Executor (Tools) performs deterministic actions
"""
import uvicorn
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Import our Core Agent Manager
from websocket_manager import WebSocketManager

# Load Environment Variables
load_dotenv()

# Set UTF-8 encoding for Windows console to prevent Unicode errors
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Configure Logging with UTF-8 encoding
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(stream=sys.stdout),  # Use stdout with UTF-8
        logging.FileHandler("anay_backend.log", encoding='utf-8')
    ]
)

# Force UTF-8 encoding on stdout/stderr for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ANAY Personal Assistant",
    description="Local Execution-First AI Agent Service",
    version="2.0.0"
)

# Enable CORS (Allow all for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent Manager (Singleton)
manager = WebSocketManager()

@app.on_event("startup")
async def startup_event():
    """Check system readiness on startup."""
    logger.info("ANAY Backend Starting...")
    
    # 1. Check Execution Context
    if os.path.exists("execution_context.json"):
        logger.info("[OK] Persistent Memory Found")
    else:
        logger.warning("[WARNING] No Context Found - Creating new memory...")
        with open("execution_context.json", "w") as f:
            f.write("{}")

    # 2. Check Groq Key (CRITICAL)
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        logger.info("[OK] Groq API Key Detected - High Speed Mode Active")
    else:
        logger.error("[ERROR] GROQ_API_KEY MISSING! ANAY requires Groq for fast planning.")

    # 3. Start System Metrics Broadcast
    import asyncio
    asyncio.create_task(manager.broadcast_metrics())
    logger.info("[OK] System Metrics Broadcast Active")

    logger.info("[READY] ANAY Agent Ready to Serve.")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "running", "agent": "ANAY", "mode": "execution-first"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main Agent Loop Interface.
    Handlers real-time voice/text interaction.
    """
    await manager.connect(websocket)
    try:
        # Enter the Agent Loop: Listen -> Think -> Act -> Respond
        await manager.handle_voice_session(websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"Agent Loop Error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    # Auto-run if executed directly
    logger.info("Starting Server on Port 8000...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)