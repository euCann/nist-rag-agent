"""
NIST RAG Agent V2 - FastAPI Service
REST API endpoints for the enhanced agent with HuggingFace dataset
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
from agent_v2 import NistRagAgentV2
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="NIST RAG Agent V2 API",
    description="Enhanced conversational AI for NIST cybersecurity standards using 530K+ training examples",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance
agent: Optional[NistRagAgentV2] = None


# Request/Response Models
class QueryRequest(BaseModel):
    """Query request schema."""
    question: str = Field(..., description="User's question about NIST standards", min_length=1)
    session_id: str = Field(default="default", description="Session ID for conversation context")
    top_k: Optional[int] = Field(default=None, description="Override number of documents to retrieve")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What's new in NIST Cybersecurity Framework 2.0?",
                "session_id": "user123",
                "top_k": 5
            }
        }


class QueryResponse(BaseModel):
    """Query response schema."""
    answer: str = Field(..., description="Agent's answer")
    session_id: str = Field(..., description="Session ID used")
    timestamp: str = Field(..., description="Response timestamp")
    model: str = Field(..., description="LLM model used")
    sources_used: int = Field(..., description="Number of sources consulted")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "NIST Cybersecurity Framework 2.0 introduces...",
                "session_id": "user123",
                "timestamp": "2026-01-07T10:30:00Z",
                "model": "gpt-4o",
                "sources_used": 3
            }
        }


class ClearHistoryRequest(BaseModel):
    """Clear history request schema."""
    session_id: str = Field(..., description="Session ID to clear")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    agent_initialized: bool
    dataset_stats: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    """Statistics response schema."""
    total_examples: int
    dataset_split: str
    active_sessions: int
    model: str
    top_k: int
    cache_directory: str


# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent
    print("="*70)
    print("NIST RAG Agent V2 API - Starting Up")
    print("="*70)
    
    try:
        print("\n⏳ Initializing agent...")
        print("   First run will download dataset (~7GB)")
        print("   Subsequent runs use cached data\n")
        
        agent = NistRagAgentV2(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            top_k=int(os.getenv("TOP_K", "5")),
            dataset_split=os.getenv("DATASET_SPLIT", "train")
        )
        
        print("✓ Agent initialized successfully!")
        stats = agent.get_dataset_stats()
        print(f"\n📊 Dataset: {stats['total_examples']:,} examples")
        print(f"📂 Split: {stats['split']}")
        print(f"🤖 Model: {agent.model}")
        print(f"🔍 Top-K: {agent.top_k}\n")
        
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        # Don't fail startup - agent will be None and endpoints will return 503


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("\n⏸️  Shutting down NIST RAG Agent V2 API")


# API Endpoints
@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "NIST RAG Agent V2 API",
        "version": "2.0.0",
        "description": "Enhanced conversational AI for NIST cybersecurity standards",
        "dataset": "ethanolivertroy/nist-cybersecurity-training",
        "examples": "530,912 training examples from 596 NIST publications",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "clear_history": "POST /clear-history",
            "stats": "GET /stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    global agent
    
    if agent is None:
        return HealthResponse(
            status="initializing",
            agent_initialized=False,
            dataset_stats=None
        )
    
    try:
        stats = agent.get_dataset_stats()
        return HealthResponse(
            status="healthy",
            agent_initialized=True,
            dataset_stats=stats
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            agent_initialized=True,
            dataset_stats={"error": str(e)}
        )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get agent statistics."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    stats = agent.get_dataset_stats()
    
    return StatsResponse(
        total_examples=stats['total_examples'],
        dataset_split=stats['split'],
        active_sessions=stats['num_sessions'],
        model=agent.model,
        top_k=agent.top_k,
        cache_directory=stats['cache_dir']
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_agent(request: QueryRequest):
    """
    Query the NIST RAG agent with a question.
    
    The agent uses multiple tools to answer:
    1. nist_lookup - Search across all 596 NIST publications
    2. control_lookup - Specific security control queries (SP 800-53)
    3. search_by_document - Document-scoped searches
    4. web_search - External fallback via DuckDuckGo
    
    Maintains conversation context per session_id.
    """
    global agent
    
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Please try again in a few moments."
        )
    
    try:
        # Override top_k if specified
        original_top_k = agent.top_k
        if request.top_k is not None:
            agent.retriever = agent.vectorstore.as_retriever(
                search_kwargs={"k": request.top_k}
            )
        
        # Query the agent
        result = agent.query(
            question=request.question,
            session_id=request.session_id
        )
        
        # Restore original top_k
        if request.top_k is not None:
            agent.retriever = agent.vectorstore.as_retriever(
                search_kwargs={"k": original_top_k}
            )
        
        # Count sources from intermediate steps
        sources_used = len(result.get("intermediate_steps", []))
        
        return QueryResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            timestamp=datetime.utcnow().isoformat() + "Z",
            model=agent.model,
            sources_used=sources_used
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/clear-history", tags=["Session"])
async def clear_history(request: ClearHistoryRequest):
    """Clear conversation history for a session."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.clear_history(session_id=request.session_id)
        return {
            "status": "success",
            "message": f"History cleared for session: {request.session_id}"
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing history: {str(e)}"
        )


@app.get("/sessions", tags=["Session"])
async def list_sessions():
    """List all active sessions."""
    global agent
    
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    sessions = []
    for session_id, history in agent.session_histories.items():
        sessions.append({
            "session_id": session_id,
            "message_count": len(history.messages)
        })
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.get("/dataset-info", tags=["Dataset"])
async def get_dataset_info():
    """Get detailed information about the dataset."""
    return {
        "name": "nist-cybersecurity-training",
        "version": "1.1",
        "source": "https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training",
        "statistics": {
            "total_examples": "530,912",
            "training_examples": "424,729",
            "validation_examples": "106,183",
            "nist_publications": 596,
            "document_sections": "263,252",
            "semantic_chunks": "136,320",
            "security_controls": "88,126",
            "definitions": "43,214",
            "working_links": "61,480"
        },
        "coverage": {
            "FIPS": "Federal Information Processing Standards",
            "SP_800": "Security Controls, Risk Management, Cryptography, etc.",
            "SP_1800": "NIST Practice Guides",
            "CSWP": "Cybersecurity White Papers (CSF 2.0, Zero Trust, PQC, IoT)",
            "IR": "Interagency/Internal Reports"
        },
        "new_in_v1_1": [
            "Added CSWP series (23 documents)",
            "Fixed 6,150 broken DOI links",
            "Validated 124,946 total links",
            "Increased from 523,706 to 530,912 examples"
        ]
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api_service_v2:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info")
    )
