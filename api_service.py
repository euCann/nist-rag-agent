"""
NIST RAG Agent - FastAPI REST Service
Production-ready API for querying NIST standards.
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from agent import NistRagAgent

# Load environment
load_dotenv()

# Global agent instance
agent: Optional[NistRagAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup."""
    global agent
    print("Initializing NIST RAG Agent...")
    agent = NistRagAgent()
    print("Agent ready!")
    yield
    # Cleanup on shutdown
    agent = None


# FastAPI app
app = FastAPI(
    title="NIST RAG Agent API",
    description="Conversational AI for NIST cybersecurity standards using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., description="Question about NIST standards")
    session_id: str = Field(default="default", description="Session ID for chat history")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID used")
    sources: Optional[list] = Field(default=None, description="Source documents used")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    model: str


# Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "NIST RAG Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=agent.model
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the NIST RAG agent.
    
    - **question**: Your question about NIST standards
    - **session_id**: Session identifier for maintaining chat history
    
    Returns the agent's answer with citations.
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        result = agent.query(
            question=request.question,
            session_id=request.session_id
        )
        
        return QueryResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            sources=None  # Could add source tracking here
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/history/{session_id}")
async def clear_history(session_id: str):
    """Clear chat history for a session."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    agent.clear_history(session_id)
    return {"message": f"History cleared for session {session_id}"}


@app.get("/sessions")
async def list_sessions():
    """List active sessions."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "sessions": list(agent.session_histories.keys()),
        "count": len(agent.session_histories)
    }


# Run with: uvicorn api_service:app --reload
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting NIST RAG Agent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
