# NIST RAG Agent - Examples

This directory contains usage examples for the NIST RAG Agent with HuggingFace dataset support.

## Dataset Information

The agent now uses the [ethanolivertroy/nist-cybersecurity-training](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training) dataset with:
- **596 NIST publications**
- **530K+ training examples**
- Coverage: FIPS, SP 800/1800, IR, CSWP (CSF 2.0, Zero Trust, PQC)

**Note**: First run downloads ~7GB dataset and creates FAISS index (~30 min total, one-time).

## Examples

### 1. Basic Query (`basic_query.py`)
Simple question-answering without chat history:

```bash
python basic_query.py
```

Demonstrates:
- Agent initialization with HuggingFace dataset
- Single-turn queries
- Citation format
- New topics: CSF 2.0, Zero Trust, PQC

### 2. Session Demo (`session_demo.py`)
Multi-turn conversations with context:

```bash
python session_demo.py
```

Demonstrates:
- Multiple users with separate sessions
- Context preservation across queries
- History clearing
- Access to 596 publications

### 3. API Client (`api_client.py`)
REST API usage via requests:

```bash
# Start the API first
python api_service.py

# Then run the client
python api_client.py
```

Demonstrates:
- Health checks
- Query endpoint
- Statistics endpoint (new!)
- Session management
- History clearing via API

## Quick Start

```python
from agent import NistRagAgent

# Initialize
agent = NistRagAgent()

# Query
response = agent.query(
    "What does NIST say about access control?",
    session_id="my_session"
)

print(response["answer"])
```

## API Usage

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is AC-1?",
    "session_id": "user123"
  }'
```

## Advanced

See the main README.md for:
- Custom embeddings
- Tool customization
- Docker deployment
- Configuration options
