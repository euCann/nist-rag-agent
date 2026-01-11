# NIST RAG Agent 🤖

> Conversational AI assistant for NIST cybersecurity standards and OSCAL compliance, powered by Retrieval-Augmented Generation (RAG)
> **Now with 530K+ training examples from 596 NIST publications** via [HuggingFace dataset](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is This?

A production-ready AI agent that answers questions about NIST cybersecurity frameworks using:
- **RAG (Retrieval-Augmented Generation)** - Searches actual NIST documents, not hallucinations
- **HuggingFace Dataset** - 596 NIST publications with 530K+ training examples
- **LangChain** - Multi-tool agent with chat history
- **FAISS** - Vector similarity search with cached indices
- **OpenAI** - GPT-4 for intelligent responses

Perfect for security assessors, compliance professionals, and anyone working with NIST standards.

## ✨ Features

- 📚 **Comprehensive NIST Coverage**: 596 publications with 530K+ examples
  - **FIPS** - Federal Information Processing Standards
  - **SP 800 Series** - Security Controls (800-53), Risk Management (800-37), CUI (800-171), and more
  - **SP 1800 Series** - NIST Practice Guides
  - **CSWP Series** - Cybersecurity White Papers:
    - ✨ NIST Cybersecurity Framework (CSF) 2.0
    - ✨ Zero Trust Architecture (SP 800-207)
    - ✨ Post-Quantum Cryptography guidance
    - ✨ IoT Cybersecurity Labeling
    - ✨ Privacy Framework v1.0
  - **IR Series** - Interagency/Internal Reports
- 🔄 **Automatic Dataset Updates** - HuggingFace integration keeps content current
- 🔍 **Intelligent Tool Selection**: RAG → Control lookup → Document search → Web fallback
- 💬 **Session-based Chat History**: Contextual conversations per user
- 🎯 **Quality Citations**: Includes source documents and metadata
- 🚀 **FastAPI Service**: Production-ready REST API
- 🐳 **Docker Ready**: Containerized deployment
- ⚡ **Cached Indices**: Fast startup after first run

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.10
openai >= 1.0
langchain >= 0.1
datasets >= 2.14.0  # For HuggingFace dataset
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nist-rag-agent.git
cd nist-rag-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-api-key-here"

# Optional: Configure dataset usage
export USE_HUGGINGFACE="true"      # Use HuggingFace dataset (default)
export DATASET_SPLIT="train"        # Use training split (424K examples)
```

### Basic Usage

```python
from agent import NistRagAgent

# Initialize the agent (first run downloads ~7GB dataset)
agent = NistRagAgent()

# Ask a question about latest NIST standards
response = agent.query(
    question="What's new in NIST Cybersecurity Framework 2.0?",
    session_id="user123"
)

print(response["answer"])
```

### Example Queries

```python
# CSF 2.0 (new!)
agent.query("What is the Govern function in CSF 2.0?")

# Zero Trust Architecture (new!)
agent.query("What are the core principles of Zero Trust?")

# Security Controls
agent.query("Explain control AC-2 in detail")

# Post-Quantum Cryptography (new!)
agent.query("What is NIST's guidance on post-quantum cryptography?")

# Follow-up questions (uses context)
agent.query("What are the implementation requirements?")
```

### Run as API Service

```bash
# Start the FastAPI server
python api_service.py

# Query via REST
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Zero Trust Architecture?", "session_id": "user123"}'

# Get agent statistics
curl http://localhost:8000/stats

# Interactive API docs
open http://localhost:8000/docs
```

### Legacy Mode (Local Embeddings)

If you prefer to use local embeddings instead of the HuggingFace dataset:

```python
from agent import NistRagAgent

# Use local embeddings
agent = NistRagAgent(use_huggingface=False)
```

Or via environment variable:
```bash
export USE_HUGGINGFACE="false"
python api_service.py
```

## � Dataset Information

### HuggingFace Dataset
The agent now uses the [ethanolivertroy/nist-cybersecurity-training](https://huggingface.co/datasets/ethanolivertroy/nist-cybersecurity-training) dataset with:

| Metric | Value |
|--------|-------|
| **Total Examples** | 530,912 |
| **Training Split** | 424,729 examples |
| **Validation Split** | 106,183 examples |
| **NIST Publications** | 596 documents |
| **Document Sections** | 263,252 |
| **Security Controls** | 88,126 |
| **Definitions** | 43,214 |
| **Validated Links** | 61,480 |

### Dataset Coverage
- **FIPS** - Federal Information Processing Standards
- **SP 800 Series** - Full collection (53 documents)
- **SP 1800 Series** - NIST Practice Guides
- **CSWP Series** - Cybersecurity White Papers (NEW!)
  - NIST CSF 2.0
  - Zero Trust Architecture
  - Post-Quantum Cryptography
  - IoT Security
  - Privacy Framework
- **IR Series** - Interagency/Internal Reports

### First Run Setup
- **Download**: ~7GB dataset (one-time)
- **Indexing**: 10-20 minutes to create FAISS index
- **Subsequent Runs**: <30 seconds (uses cache)

## 📁 Project Structure

```
nist-rag-agent/
├── agent.py                 # Core RAG agent (now with HuggingFace support)
├── api_service.py          # FastAPI REST service
├── .cache/                 # HuggingFace dataset cache (auto-created)
│   └── huggingface/
│       ├── datasets/       # Downloaded dataset
│       └── faiss_index_*/  # Cached FAISS indices
├── embeddings/             # Legacy: Local NIST embeddings (optional)
│   ├── README.md
│   └── ... (local files)
├── examples/              # Usage examples
│   ├── basic_query.py
│   ├── session_demo.py
│   └── api_client.py
├── requirements.txt       # Now includes 'datasets' package
├── Dockerfile
└── README.md
```

## 🛠️ Advanced Usage

### Custom Embeddings

Add your own NIST documents:

```python
from tools.embedding_builder import build_embeddings

# Build embeddings from PDF
build_embeddings(
    pdf_path="NIST.SP.800-XX.pdf",
    output_dir="embeddings/"
)
```

### Multi-Session Chat

```python
# User Alice asks about access control
agent.query("What is AC-1?", session_id="alice")

# Later, Alice asks a follow-up
agent.query("What are the requirements?", session_id="alice")
# Agent remembers we're talking about AC-1

# User Bob has a separate conversation
agent.query("What is IR-4?", session_id="bob")
```

### Tool Customization

```python
# Add your own tools
from langchain_core.tools import tool

@tool("custom_tool")
def my_custom_tool(query: str) -> str:
    """Your custom NIST-related functionality"""
    return "Custom response"

agent = NistRagAgent(extra_tools=[my_custom_tool])
```

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t nist-rag-agent .

# Run the container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  nist-rag-agent

# Or use docker-compose
docker-compose up -d
```

## 📊 Supported NIST Publications

| Document | Description | Chunks |
|----------|-------------|--------|
| SP 800-53 Rev 5 | Security and Privacy Controls | ~2,500 |
| SP 800-37 Rev 2 | Risk Management Framework | ~800 |
| SP 800-171 Rev 3 | CUI Protection | ~600 |
| SP 800-60 Vol 2 Rev 1 | Information Types | ~1,200 |
| SP 800-63-3 | Digital Identity | ~900 |
| SP 800-30 Rev 1 | Risk Assessment | ~700 |
| SP 800-137 | Continuous Monitoring | ~400 |
| SP 800-18 Rev 1 | Security Plans | ~300 |
| CSWP 29 | AI Risk Management | ~500 |

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Test specific functionality
pytest tests/test_agent.py::test_access_control_query
```

## 🔧 Configuration

Edit `.env` to customize:

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o  # or gpt-4, gpt-3.5-turbo

# Azure OpenAI (alternative)
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_DEPLOYMENT=...

# LangChain (optional)
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true

# RAG Configuration
TOP_K_RESULTS=3
CHUNK_SIZE=1000
EMBEDDING_MODEL=text-embedding-ada-002
```

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional NIST publications (800-137A, 800-161, etc.)
- Enhanced citation formatting
- OSCAL integration (SSP generation, profile validation)
- Performance optimizations
- UI/UX (Streamlit, Gradio)

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **NIST** for publishing open cybersecurity standards
- **LangChain** for the agent framework
- **OpenAI** for GPT models and embeddings

## 📮 Contact

Built by a federal cybersecurity professional working with AI-assisted development.

- GitHub Issues: [Report bugs or request features](https://github.com/yourusername/nist-rag-agent/issues)
- Discussions: [Ask questions or share ideas](https://github.com/yourusername/nist-rag-agent/discussions)

---

**Note**: This tool provides information retrieval only. Always verify compliance requirements with official NIST publications and your organization's policies.
