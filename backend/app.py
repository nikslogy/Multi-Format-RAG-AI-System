"""
RAG AI System - Production API
Clean, professional FastAPI server
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.core.rag_engine_v2 import RAGEngineV2
from src.processors.document_processor import DocumentProcessor
from src.utils.logger_config import rag_logger

# Load environment
load_dotenv()

# Configuration
API_KEY = os.getenv("GOOGLE_API_KEY")
UPLOAD_DIR = Path("./uploads")
EXCEL_DIR = Path("./excel_files")
STATIC_DIR = Path("./static")

# Create directories
UPLOAD_DIR.mkdir(exist_ok=True)
EXCEL_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# Initialize components
rag_engine = RAGEngineV2(api_key=API_KEY)
doc_processor = DocumentProcessor()

# FastAPI app
app = FastAPI(
    title="RAG AI System",
    description="Intelligent document analysis with multi-file support",
    version="2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== MODELS =====

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list
    has_context: bool
    query_type: Optional[str] = None
    strategy: Optional[str] = None
    processing_time_ms: Optional[float] = None
    structured_data: Optional[Any] = None  # Excel table data
    code_executed: Optional[bool] = None

class StatsResponse(BaseModel):
    num_documents: int
    total_documents: int
    total_chunks: int
    total_words: int
    collection_name: str
    documents: list

# ===== API ENDPOINTS =====

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve main page"""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return index_file.read_text()
    return "<h1>RAG AI System V2</h1><p>Upload documents and ask questions!</p>"

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        rag_logger.info(f"📤 Upload request: {file.filename}")

        # Validate file type
        ext = Path(file.filename).suffix.lower()
        supported = ['.pdf', '.docx', '.txt', '.xlsx', '.xls', '.csv', '.json']

        if ext not in supported:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported: {', '.join(supported)}"
            )

        # Save temporarily
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        rag_logger.info(f"💾 Saved to: {temp_path}")

        # Process based on file type
        if ext in ['.xlsx', '.xls', '.csv']:
            # Excel/CSV - create metadata only
            metadata_chunk = doc_processor.create_excel_metadata(str(temp_path), file.filename)

            if not metadata_chunk:
                raise HTTPException(status_code=500, detail="Failed to process Excel file")

            # Copy to permanent storage
            perm_path = EXCEL_DIR / file.filename
            os.replace(temp_path, perm_path)

            # Update file path in metadata (use absolute path)
            metadata_chunk[1][0]['file_path'] = str(perm_path.absolute())

            # Add to RAG
            num_chunks = rag_engine.add_documents(metadata_chunk[0], metadata_chunk[1])

            rag_logger.info(f"✅ Excel file processed: {num_chunks} metadata chunks")

            return {
                "message": "Excel file uploaded successfully",
                "filename": file.filename,
                "chunks_created": num_chunks,
                "file_type": "excel",
                "row_count": metadata_chunk[1][0].get('row_count'),
                "col_count": metadata_chunk[1][0].get('col_count')
            }

        else:
            # Regular document - process and chunk
            chunks, metadatas = doc_processor.process_document(str(temp_path), file.filename)

            if not chunks:
                raise HTTPException(status_code=500, detail="Failed to process document")

            # Add to RAG
            num_chunks = rag_engine.add_documents(chunks, metadatas)

            # Clean up temp file
            temp_path.unlink()

            rag_logger.info(f"✅ Document processed: {num_chunks} chunks")

            return {
                "message": "Document uploaded successfully",
                "filename": file.filename,
                "chunks_created": num_chunks,
                "file_type": "document"
            }

    except HTTPException:
        raise
    except Exception as e:
        rag_logger.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents with intelligent routing"""
    try:
        rag_logger.info(f"❓ Query: {request.question}")

        result = rag_engine.query(request.question)

        return QueryResponse(**result)

    except Exception as e:
        rag_logger.error(f"❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        docs = rag_engine.list_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document"""
    try:
        num_deleted = rag_engine.delete_document(filename)

        if num_deleted == 0:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "message": "Document deleted",
            "filename": filename,
            "chunks_deleted": num_deleted
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        stats = rag_engine.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/reset")
async def reset_system():
    """Reset entire system (delete all documents)"""
    try:
        rag_engine.reset_collection()
        return {"message": "System reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Run
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting RAG AI System V2...")
    print(f"📁 Upload directory: {UPLOAD_DIR.absolute()}")
    print(f"📊 Excel directory: {EXCEL_DIR.absolute()}")
    print(f"🌐 Server: http://localhost:8000\n")

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
