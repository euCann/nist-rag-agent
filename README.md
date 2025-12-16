# NIST RAG Agent 🤖

> Conversational AI assistant for NIST cybersecurity standards and OSCAL compliance, powered by Retrieval-Augmented Generation (RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 What is This?

A production-ready AI agent that answers questions about NIST cybersecurity frameworks (SP 800-53, 800-37, 800-171, etc.) using:
- **RAG (Retrieval-Augmented Generation)** - Searches actual NIST documents, not hallucinations
- **LangChain** - Multi-tool agent with chat history
- **FAISS** - Vector similarity search over 10+ NIST publications
- **OpenAI/Azure OpenAI** - GPT-4 for intelligent responses

Perfect for security assessors, compliance professionals, and anyone working with NIST standards.

## ✨ Features

- 📚 **Pre-indexed NIST Documents**: 10+ publications ready to query
  - NIST SP 800-53 Rev 5 (Security Controls)
  - NIST SP 800-37 Rev 2 (Risk Management Framework)
  - NIST SP 800-171 Rev 3 (CUI Protection)
  - NIST SP 800-60, 800-63, 800-30, and more
- 🔍 **Intelligent Tool Selection**: RAG → Control lookup → Web search fallback
- 💬 **Session-based Chat History**: Contextual conversations per user
- 🎯 **Citation**: Always includes Control ID, Title, URL, Section
- 🚀 **FastAPI Service**: REST API ready for integration
- 🐳 **Docker Ready**: Containerized deployment

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.10
openai >= 1.0
langchain >= 0.1
```

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/nist-rag-agent.git
cd nist-rag-agent

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Basic Usage

```python
from agent import NistRagAgent

# Initialize the agent
agent = NistRagAgent()

# Ask a question
response = agent.query(
    question="What does NIST say about access control?",
    session_id="user123"
)

print(response["answer"])
# Includes citations: Control ID, Title, URL
```

### Run as API Service

```bash
# Start the FastAPI server
python api_service.py

# Query via REST
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain AC-1", "session_id": "user123"}'
```

## 📁 Project Structure

```
nist-rag-agent/
├── agent.py                 # Core RAG agent implementation
├── api_service.py          # FastAPI REST service
├── embeddings/             # Pre-built NIST document embeddings
│   ├── NIST.SP.800-53r5.chunks.json
│   ├── NIST.SP.800-53r5.chunks.npy
│   └── ... (10+ documents)
├── tools/                  # Custom LangChain tools
│   ├── nist_lookup.py
│   ├── control_detail.py
│   └── web_search.py
├── examples/              # Usage examples
│   ├── basic_query.py
│   ├── batch_analysis.py
│   └── session_demo.py
├── tests/                 # Test suite
├── requirements.txt
├── .env.example
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
