# Use specific version with digest for security
FROM python:3.11.7-slim

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install system dependencies with verification
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy only necessary application files
COPY agent.py api_service.py security.py ./
COPY embeddings/ ./embeddings/

# Create directories with correct permissions
RUN mkdir -p .cache logs && \
    chown -R appuser:appuser /app && \
    chmod 750 /app/.cache /app/logs

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API service
CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
