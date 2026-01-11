## Quick Start Guide

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/nist-rag-agent.git
cd nist-rag-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (includes HuggingFace datasets)
pip install -r requirements.txt

# Configure environment
export OPENAI_API_KEY="your-api-key-here"

# Optional: Configure dataset usage
export USE_HUGGINGFACE="true"    # Use HuggingFace dataset (default, 530K+ examples)
export DATASET_SPLIT="train"      # Use training split (424K examples)
```

**Note**: First run will download ~7GB HuggingFace dataset (one-time, 5-15 minutes) and create FAISS index (10-20 minutes). Subsequent runs are instant (<30 seconds).

### 2. Run Basic Example

```bash
python examples/basic_query.py
```

Expected output:
```
⏳ Using HuggingFace dataset (530K+ examples)
✓ Loaded 424,729 examples from NIST dataset
Creating FAISS index...

Q: What does NIST say about access control?
A: According to NIST SP 800-53 Rev 5, **AC-1** (Access Control Policy and 
Procedures) requires organizations to develop, document, and disseminate 
access control policies...
```

### 3. Try New Capabilities

```python
from agent import NistRagAgent

agent = NistRagAgent()

# Ask about CSF 2.0 (new!)
agent.query("What's new in NIST Cybersecurity Framework 2.0?")

# Zero Trust Architecture (new!)
agent.query("What are the core principles of Zero Trust?")

# Post-Quantum Cryptography (new!)
agent.query("What is NIST's guidance on post-quantum cryptography?")
```

### 4. Start API Service

```bash
python api_service.py
```

Visit http://localhost:8000/docs for interactive API documentation.

### 5. Query via API

```bash
# Basic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Zero Trust Architecture?", "session_id": "demo"}'

# Get agent statistics
curl http://localhost:8000/stats
```

### 6. Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Test
curl http://localhost:8000/health
```

### 7. Legacy Mode (Local Embeddings)

If you prefer to use local embeddings instead of the HuggingFace dataset:

```bash
export USE_HUGGINGFACE="false"
python api_service.py
```

Or in code:
```python
agent = NistRagAgent(use_huggingface=False)
```

## Next Steps

- Read [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for dataset details
- See [examples/README.md](examples/README.md) for usage patterns
- Check main [README.md](README.md) for advanced configuration
- Explore [agent.py](agent.py) for customization options

## Dataset Information

**HuggingFace Dataset**: [ethanolivertroy/nist-cybersecurity-training](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training)

- **596 NIST publications**
- **530K+ training examples**
- Includes: FIPS, SP 800/1800, IR, CSWP (CSF 2.0, Zero Trust, PQC)
- First run: ~30 min setup (one-time)
- Subsequent runs: <30 sec startup

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
