"""
NIST RAG Agent - FastAPI REST Service
Production-ready API for querying NIST standards.
Now with HuggingFace dataset support (530K+ examples from 596 publications)
Secured with authentication, rate limiting, and input validation.
"""

import os
import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, generate_latest
import ipaddress

from agent import NistRagAgent
from security import (
    api_key_auth,
    ValidatedQueryRequest,
    safe_error_response,
    validate_env_vars,
    security_logger,
    PromptSanitizer
)

# Load environment
load_dotenv()

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Validate environment variables on startup
try:
    validate_env_vars()
except ValueError as e:
    security_logger.error(f"Startup failed: {e}")
    raise

# Global agent instance
agent: Optional[NistRagAgent] = None

# Rate limiter (tightened for production defaults)
limiter = Limiter(key_func=get_remote_address, default_limits=["30/hour"])

# Prometheus metrics
query_counter = Counter(
    'nist_rag_queries_total',
    'Total number of queries',
    ['status', 'endpoint']
)
query_duration = Histogram(
    'nist_rag_query_duration_seconds',
    'Query processing duration',
    ['endpoint']
)
failed_auth_counter = Counter(
    'nist_rag_failed_auth_total',
    'Failed authentication attempts'
)



class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size."""
    
    def __init__(self, app, max_size: int = 1024 * 100):  # 100KB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request body too large"}
                )
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup."""
    global agent
    print("="*70)
    print("NIST RAG Agent API - Starting Up (Secured)")
    print("="*70)
    
    # Validate environment
    try:
        validate_env_vars()
    except ValueError as e:
        print(f"\n❌ Environment validation failed: {e}")
        raise
    
    # Check if HuggingFace dataset should be used
    use_huggingface = os.getenv("USE_HUGGINGFACE", "true").lower() == "true"
    
    if use_huggingface:
        print("\n⏳ Using HuggingFace dataset (530K+ examples)")
        print("   First run will download dataset (~7GB)")
    else:
        print("\n⏳ Using local embeddings")
    
    agent = NistRagAgent(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        top_k=int(os.getenv("TOP_K", "5")),
        use_huggingface=use_huggingface,
        dataset_split=os.getenv("DATASET_SPLIT", "train")
    )
    
    stats = agent.get_stats()
    print("\n✓ Agent ready!")
    print("\n📊 Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Security notice
    if not os.getenv("API_KEYS"):
        print("\n⚠️  WARNING: No API keys configured!")
        print("   Set API_KEYS environment variable for production")
    else:
        print("\n✓ API authentication enabled")
    
    print()
    
    yield
    # Cleanup on shutdown
    agent = None


# FastAPI app
app = FastAPI(
    title="NIST RAG Agent API",
    description="Conversational AI for NIST cybersecurity standards using RAG with 596 NIST publications",
    version="2.0.0",
    lifespan=lifespan
)

# Force HTTPS in production
if ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add request size limit middleware
app.add_middleware(RequestSizeLimitMiddleware, max_size=1024 * 100)  # 100KB

# CORS middleware - restricted
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
if ENVIRONMENT == "production" and not allowed_origins_env:
    raise ValueError("ALLOWED_ORIGINS must be set in production")

if not allowed_origins_env:
    # Safe defaults for development only
    ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost:8000"]
else:
    ALLOWED_ORIGINS = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Specific domains only
    allow_credentials=False,  # Changed to False for security
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600
)

# Optional IP allowlist
allowed_ip_cidrs = [c.strip() for c in os.getenv("ALLOWED_IP_CIDRS", "").split(",") if c.strip()]
allowed_networks = []
for cidr in allowed_ip_cidrs:
    try:
        allowed_networks.append(ipaddress.ip_network(cidr, strict=False))
    except ValueError:
        raise ValueError(f"Invalid CIDR in ALLOWED_IP_CIDRS: {cidr}")

if allowed_networks:
    @app.middleware("http")
    async def ip_allowlist(request: Request, call_next):
        client_ip = ipaddress.ip_address(request.client.host)
        if not any(client_ip in net for net in allowed_networks):
            return JSONResponse(status_code=403, content={"detail": "Access denied"})
        return await call_next(request)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Add HSTS in production
    if os.getenv("ENVIRONMENT") == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response


# Request/Response Models
class QueryResponse(BaseModel):
    """Query response model."""
    answer: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session ID used")
    sources: Optional[list] = Field(default=None, description="Source documents used")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model: str = Field(..., description="LLM model in use")


# Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "NIST RAG Agent API",
        "version": "2.0.0",
        "dataset": "ethanolivertroy/nist-cybersecurity-training (596 publications, 530K+ examples)",
        "docs": "/docs",
        "health": "/health",
        "stats": "/stats",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint (no authentication required)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model=agent.model
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/stats", response_model=dict)
@limiter.limit("10/minute")
async def get_stats(
    request: Request,
    api_key: str = Depends(api_key_auth.verify)
):
    """Get agent statistics (requires authentication)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return agent.get_stats()


@app.post("/query", response_model=QueryResponse)
@limiter.limit("5/minute")
async def query(
    request: Request,
    query_request: ValidatedQueryRequest,
    api_key: str = Depends(api_key_auth.verify)
):
    """
    Query the NIST RAG agent (requires authentication).
    
    - **question**: Your question about NIST standards (1-2000 chars)
    - **session_id**: Session identifier for chat history (alphanumeric, default: "default")
    
    Returns the agent's answer with citations.
    
    Rate limit: 5 requests per minute per IP.
    """
    if agent is None:
        query_counter.labels(status='unavailable', endpoint='query').inc()
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    start_time = time.time()
    
    try:
        # Log request
        security_logger.info(
            f"Query from {request.client.host}: "
            f"session={query_request.session_id}, "
            f"length={len(query_request.question)}"
        )
        
        # Execute query
        with query_duration.labels(endpoint='query').time():
            result = agent.query(
                question=query_request.question,
                session_id=query_request.session_id
            )
        
        query_counter.labels(status='success', endpoint='query').inc()
        
        return QueryResponse(
            answer=result["answer"],
            session_id=result["session_id"],
            sources=None
        )
    
    except ValueError as e:
        # User input errors
        query_counter.labels(status='invalid_input', endpoint='query').inc()
        security_logger.warning(f"Validation error from {request.client.host}: {e}")
        raise HTTPException(status_code=400, detail="Invalid input provided")
    
    except Exception as e:
        # System errors
        query_counter.labels(status='error', endpoint='query').inc()
        error_info = safe_error_response(e, request)
        raise HTTPException(
            status_code=error_info["status_code"],
            detail=error_info["detail"]
        )
    finally:
        duration = time.time() - start_time
        security_logger.info(f"Query completed in {duration:.2f}s")


@app.delete("/history/{session_id}")
@limiter.limit("10/minute")
async def clear_history(
    request: Request,
    session_id: str,
    api_key: str = Depends(api_key_auth.verify)
):
    """Clear chat history for a session (requires authentication)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Validate session_id
        sanitized_id = PromptSanitizer.sanitize_session_id(session_id)
        agent.clear_history(sanitized_id)
        
        security_logger.info(f"History cleared for session {sanitized_id}")
        return {"message": f"History cleared for session {sanitized_id}"}
    
    except ValueError as e:
        security_logger.warning(f"Invalid session_id: {session_id}")
        raise HTTPException(status_code=400, detail="Invalid session ID")


@app.get("/sessions")
@limiter.limit("10/minute")
async def list_sessions(
    request: Request,
    api_key: str = Depends(api_key_auth.verify)
):
    """List active sessions (requires authentication)."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "sessions": list(agent.session_histories.keys()),
        "count": len(agent.session_histories)
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    if exc.status_code == 401:
        failed_auth_counter.inc()
    
    security_logger.warning(
        f"HTTP {exc.status_code} from {request.client.host}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    error_info = safe_error_response(exc, request)
    return JSONResponse(
        status_code=error_info["status_code"],
        content={"detail": error_info["detail"]}
    )


# Run with: uvicorn api_service:app --reload
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"Starting NIST RAG Agent API on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
