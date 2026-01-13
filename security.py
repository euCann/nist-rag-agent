"""
Security utilities for NIST RAG Agent
Implements authentication, rate limiting, input validation, and sanitization
"""

import os
import re
import secrets
import logging
from typing import Optional, Set
from datetime import datetime, timedelta
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(level=logging.INFO)
security_logger = logging.getLogger("security")


class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from logs."""
    
    PATTERNS = [
        (re.compile(r'sk-[A-Za-z0-9]{48}'), '***REDACTED_API_KEY***'),
        (re.compile(r'"password"\s*:\s*"[^"]*"'), '"password": "***REDACTED***"'),
        (re.compile(r'Bearer\s+[A-Za-z0-9_-]+'), 'Bearer ***REDACTED***'),
    ]
    
    def filter(self, record):
        if hasattr(record, 'msg'):
            msg = str(record.msg)
            for pattern, replacement in self.PATTERNS:
                msg = pattern.sub(replacement, msg)
            record.msg = msg
        return True


# Add filter to root logger
for handler in logging.root.handlers:
    handler.addFilter(SensitiveDataFilter())


class PromptSanitizer:
    """Sanitize user input to prevent prompt injection attacks."""
    
    BLACKLIST_PATTERNS = [
        r'ignore\s+(all\s+)?previous\s+instructions',
        r'you\s+are\s+now\s+',
        r'system\s+prompt',
        r'jailbreak',
        r'DAN\s+mode',
        r'developer\s+mode',
        r'godmode',
        r'<\|im_start\|>',
        r'<\|im_end\|>',
        r'<\|endoftext\|>',
    ]
    
    SUSPICIOUS_PATTERNS = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onclick=',
        r'\.\.\/',
        r'\$\{',
        r'\{\{',
        r'exec\(',
        r'eval\(',
    ]
    
    @classmethod
    def sanitize(cls, text: str, max_length: int = 2000) -> str:
        """
        Sanitize user input.
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
            
        Raises:
            ValueError: If input contains malicious patterns
        """
        if not text:
            raise ValueError("Input cannot be empty")
        
        # Truncate excessive length
        if len(text) > max_length:
            security_logger.warning(f"Input truncated from {len(text)} to {max_length} chars")
            text = text[:max_length]
        
        # Check for prompt injection patterns
        for pattern in cls.BLACKLIST_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                security_logger.warning(f"Potential prompt injection detected: {pattern}")
                raise ValueError("Potentially malicious input detected")
        
        # Check for other suspicious patterns
        for pattern in cls.SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                security_logger.warning(f"Suspicious pattern detected: {pattern}")
                raise ValueError("Invalid characters in input")
        
        # Remove control characters except newline, tab, carriage return
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        return text.strip()
    
    @classmethod
    def sanitize_session_id(cls, session_id: str) -> str:
        """Sanitize session ID to prevent path traversal."""
        if not session_id:
            raise ValueError("Session ID cannot be empty")
        
        # Only allow alphanumeric, underscore, hyphen
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            raise ValueError("Invalid session ID format")
        
        if len(session_id) > 64:
            raise ValueError("Session ID too long")
        
        # Check for path traversal
        if '..' in session_id or '/' in session_id or '\\' in session_id:
            security_logger.warning(f"Path traversal attempt in session ID: {session_id}")
            raise ValueError("Invalid session ID")
        
        return session_id


class APIKeyAuth:
    """API Key authentication handler."""
    
    def __init__(self):
        self.security = HTTPBearer()
        self.api_keys: Set[str] = self._load_api_keys()
        
        if not self.api_keys:
            # Generate a default key for development
            default_key = os.getenv("DEFAULT_API_KEY")
            if default_key:
                self.api_keys.add(default_key)
            else:
                security_logger.warning(
                    "No API keys configured. Set API_KEYS or DEFAULT_API_KEY environment variable."
                )
    
    def _load_api_keys(self) -> Set[str]:
        """Load API keys from environment."""
        keys_str = os.getenv("API_KEYS", "")
        if not keys_str:
            return set()
        
        keys = set()
        for key in keys_str.split(","):
            key = key.strip()
            if key:
                keys.add(key)
        
        return keys
    
    async def verify(
        self, 
        credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
    ) -> str:
        """
        Verify API key from Authorization header.
        
        Args:
            credentials: HTTP credentials from request
            
        Returns:
            Verified API key
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.api_keys:
            # Allow access if no keys configured (dev mode)
            security_logger.warning("No API keys configured - allowing unauthenticated access")
            return "dev-mode"
        
        token = credentials.credentials
        
        if token not in self.api_keys:
            security_logger.warning(f"Invalid API key attempted")
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return token
    
    @staticmethod
    def generate_key() -> str:
        """Generate a secure API key."""
        return secrets.token_urlsafe(32)


class SessionManager:
    """Manage user sessions with expiration and limits."""
    
    def __init__(self, max_sessions: int = 1000, ttl_minutes: int = 60):
        self.sessions = {}
        self.session_timestamps = {}
        self.max_sessions = max_sessions
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def create_session(self) -> str:
        """Create a new session with cryptographically secure ID."""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {"created": datetime.now(), "data": {}}
        self.session_timestamps[session_id] = datetime.now()
        self._cleanup_old_sessions()
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """Check if session exists and is not expired."""
        if session_id not in self.sessions:
            return False
        
        if datetime.now() - self.session_timestamps[session_id] > self.ttl:
            self.delete_session(session_id)
            return False
        
        # Update timestamp on access
        self.session_timestamps[session_id] = datetime.now()
        return True
    
    def get_session(self, session_id: str):
        """Get session data."""
        if not self.validate_session(session_id):
            return None
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str):
        """Delete a session."""
        self.sessions.pop(session_id, None)
        self.session_timestamps.pop(session_id, None)
    
    def _cleanup_old_sessions(self):
        """Remove expired and excess sessions."""
        now = datetime.now()
        
        # Remove expired sessions
        expired = [
            sid for sid, ts in self.session_timestamps.items()
            if now - ts > self.ttl
        ]
        for sid in expired:
            self.delete_session(sid)
        
        # Enforce max sessions limit
        if len(self.sessions) > self.max_sessions:
            # Remove oldest sessions
            sorted_sessions = sorted(
                self.session_timestamps.items(),
                key=lambda x: x[1]
            )
            for session_id, _ in sorted_sessions[:100]:
                self.delete_session(session_id)
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        return {
            "active_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "ttl_minutes": self.ttl.total_seconds() / 60
        }


def validate_env_vars():
    """
    Validate critical environment variables on startup.
    
    Raises:
        ValueError: If any validation fails
    """
    errors = []
    
    # Validate OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        errors.append("OPENAI_API_KEY is required")
    elif not re.match(r'^sk-[A-Za-z0-9_-]{32,}', api_key):
        errors.append("OPENAI_API_KEY format appears invalid")
    
    # Validate model name (whitelist)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    allowed_models = [
        "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-4-turbo", 
        "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
    ]
    if model not in allowed_models:
        errors.append(f"OPENAI_MODEL must be one of: {', '.join(allowed_models)}")
    
    # Validate numeric values
    top_k = os.getenv("TOP_K", "5")
    if not top_k.isdigit() or not (1 <= int(top_k) <= 20):
        errors.append("TOP_K must be a number between 1-20")
    
    # Validate dataset split
    split = os.getenv("DATASET_SPLIT", "train")
    if split not in ["train", "valid"]:
        errors.append("DATASET_SPLIT must be 'train' or 'valid'")
    
    # Validate boolean flags
    use_hf = os.getenv("USE_HUGGINGFACE", "true").lower()
    if use_hf not in ["true", "false", "1", "0"]:
        errors.append("USE_HUGGINGFACE must be 'true' or 'false'")
    
    if errors:
        error_msg = "Environment validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        security_logger.error(error_msg)
        raise ValueError(error_msg)
    
    security_logger.info("Environment variables validated successfully")
    return True


class ValidatedQueryRequest(BaseModel):
    """Validated query request with security checks."""
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Your question about NIST standards"
    )
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=64,
        regex=r"^[a-zA-Z0-9_-]+$",
        description="Session identifier"
    )
    
    @validator('question')
    def sanitize_question(cls, v):
        """Sanitize question input."""
        return PromptSanitizer.sanitize(v)
    
    @validator('session_id')
    def sanitize_session(cls, v):
        """Sanitize session ID."""
        return PromptSanitizer.sanitize_session_id(v)


def safe_error_response(error: Exception, request: Request = None) -> dict:
    """
    Create safe error response without exposing internals.
    
    Args:
        error: Exception that occurred
        request: Optional request object for logging
        
    Returns:
        Safe error response dict
    """
    # Log full error details
    if request:
        security_logger.error(
            f"Error processing request from {request.client.host}: "
            f"{type(error).__name__}: {str(error)}"
        )
    else:
        security_logger.error(f"{type(error).__name__}: {str(error)}")
    
    # Return generic error to client
    if isinstance(error, ValueError):
        return {
            "status_code": 400,
            "detail": "Invalid input provided"
        }
    elif isinstance(error, HTTPException):
        return {
            "status_code": error.status_code,
            "detail": error.detail
        }
    else:
        return {
            "status_code": 500,
            "detail": "An error occurred processing your request"
        }


# Singleton instances
api_key_auth = APIKeyAuth()
session_manager = SessionManager()
