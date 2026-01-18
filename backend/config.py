"""Configuration and environment variables for ANAY backend."""
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Read API keys from api.txt file
def read_api_keys():
    """Read API keys from api.txt file in the parent directory."""
    api_file = Path(__file__).parent.parent / "api.txt"
    api_keys = {}
    
    if api_file.exists():
        try:
            with open(api_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                    if key == "Deepgram":
                        api_keys['DEEPGRAM_API_KEY'] = value
                    elif key == "Eleven Labs":
                        api_keys['ELEVENLABS_API_KEY'] = value
                    elif key == "OpenAI":
                        api_keys['OPENAI_API_KEY'] = value
        except Exception as e:
            print(f"Warning: Could not read api.txt: {e}")
    
    return api_keys

# Load API keys from file
file_api_keys = read_api_keys()

# Google Gemini API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or file_api_keys.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Deepgram STT Configuration
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY") or file_api_keys.get("DEEPGRAM_API_KEY", "")

# ElevenLabs TTS Configuration
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or file_api_keys.get("ELEVENLABS_API_KEY", "")

# Server Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8080,http://localhost:5173").split(",")

# AI Configuration
MAX_CONTEXT_MESSAGES = 20
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TOKENS = 1024

# System Control Configuration
ALLOWED_FILE_EXTENSIONS = ['.txt', '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.jpg', '.jpeg', '.png', '.gif', '.mp4', '.mp3', '.wav']
MAX_PATH_LENGTH = 500
