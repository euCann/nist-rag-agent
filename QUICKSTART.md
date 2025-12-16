## Quick Start Guide

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nist-rag-agent.git
cd nist-rag-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 2. Run Basic Example

```bash
python examples/basic_query.py
```

Expected output:
```
Q: What does NIST say about access control?
A: According to NIST SP 800-53 Rev 5, **AC-1** (Access Control Policy and 
Procedures) requires organizations to develop, document, and disseminate 
access control policies...
```

### 3. Start API Service

```bash
python api_service.py
```

Visit http://localhost:8000/docs for interactive API documentation.

### 4. Query via API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AC-1?", "session_id": "demo"}'
```

### 5. Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Test
curl http://localhost:8000/health
```

## Next Steps

- Read [examples/README.md](examples/README.md) for usage patterns
- See main [README.md](README.md) for advanced configuration
- Check [api_service.py](api_service.py) for API endpoints
- Explore [agent.py](agent.py) for customization options

## Common Issues

**"OpenAI API key required"**
- Make sure `.env` file exists with `OPENAI_API_KEY=sk-...`

**"No NIST documents found"**
- Embeddings will be auto-generated on first run
- Mock data is used if no embeddings available

**"Agent not initialized"**
- Wait ~30 seconds for first startup (building FAISS index)
- Check logs: `docker-compose logs -f`

## Project Structure

```
nist-rag-agent/
├── agent.py              # Core agent class
├── api_service.py        # FastAPI service
├── embeddings/           # NIST document embeddings
│   ├── NIST.SP.800-53r5.chunks.json
│   └── ... (10+ files)
├── examples/             # Usage examples
│   ├── basic_query.py
│   ├── session_demo.py
│   └── api_client.py
└── requirements.txt
```
