"""
Configuration for RAG AI System
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
EXCEL_DIR = BASE_DIR / "excel_files"
STATIC_DIR = BASE_DIR / "static"
CHROMA_DIR = BASE_DIR / "chroma_db"

# RAG Settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))

# Models
EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash-exp"

# Server
HOST = "0.0.0.0"
PORT = 8000
RELOAD = True
