# Model Provider Configuration Guide

The NIST RAG Agent now supports **multiple LLM and embedding providers** through a provider-agnostic architecture.

## 🌐 Supported Providers

### LLM Providers
| Provider | Models | API Key Required | Notes |
|----------|--------|------------------|-------|
| **🔥 LiteLLM** | 100+ providers | Varies | **Universal interface** |
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-3.5 | ✅ Yes | Default provider |
| **Anthropic** | Claude 3.5 Sonnet, Haiku | ✅ Yes | Excellent reasoning |
| **Google** | Gemini 1.5 Pro, Flash | ✅ Yes | Fast & capable |
| **Azure OpenAI** | GPT-4, GPT-3.5 | ✅ Yes | Enterprise option |
| **Ollama** | Llama 3.1, Mistral, etc. | ❌ No | **100% local & free** |
| **HuggingFace** | Various open models | ✅ Optional | API or local |
| **AWS Bedrock** | Claude, Titan, etc. | ✅ Yes | AWS account required |
| **Cohere** | Command models | ✅ Yes | Good for RAG |

### Embedding Providers
| Provider | Models | Free Option |
|----------|--------|-------------|
| **OpenAI** | text-embedding-3-small/large | ✅ With API key |
| **HuggingFace** | sentence-transformers/* | ✅ **Free & local** |
| **Ollama** | nomic-embed-text | ✅ **Free & local** |
| **Google** | embedding-001 | ✅ With API key |
| **Cohere** | embed-english-v3.0 | ✅ With API key |

## 🚀 Quick Start

### Option 1: Using Presets (Easiest)

```python
from agent import NistRagAgentV2

# OpenAI GPT-4 (default)
agent = NistRagAgentV2(
    model_config="openai-gpt4",
    embedding_config="openai"
)

# 🔥 LiteLLM - Universal interface for any provider!
agent = NistRagAgentV2(
    model_config="litellm-gpt4",      # or "litellm-claude", "litellm-gemini"
    embedding_config="litellm"
)

# Anthropic Claude
agent = NistRagAgentV2(
    model_config="anthropic-claude",
    embedding_config="openai"
)

# Local Ollama (100% free!)
agent = NistRagAgentV2(
    model_config="ollama-llama3",
    embedding_config="ollama"
)

# Google Gemini
agent = NistRagAgentV2(
    model_config="google-gemini",
    embedding_config="google"
)
```

### Option 2: Custom Configuration

```python
from agent import NistRagAgentV2
from model_config import ModelConfig, EmbeddingConfig

# Custom LLM
llm_config = ModelConfig(
    provider="openai",
    model_name="gpt-4-turbo-preview",
    temperature=0.2,
    max_tokens=2000
)

# Local embeddings (free!)
embed_config = EmbeddingConfig(
    provider="huggingface",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

agent = NistRagAgentV2(
    model_config=llm_config,
    embedding_config=embed_config
)
```

## 📋 Available Presets

### LLM Presets
```python
"openai-gpt4"          # GPT-4o (recommended)
"openai-gpt4-turbo"    # GPT-4 Turbo
"openai-gpt35"         # GPT-3.5 Turbo (faster, cheaper)
"litellm-gpt4"         # GPT-4 via LiteLLM (with retries, fallbacks)
"litellm-claude"       # Claude via LiteLLM
"litellm-gemini"       # Gemini via LiteLLM
"anthropic-claude"     # Claude 3.5 Sonnet
"anthropic-haiku"      # Claude Haiku (fast)
"google-gemini"        # Gemini 1.5 Pro
"google-gemini-flash"  # Gemini Flash (fast)
"ollama-llama3"        # Llama 3.1 8B (local)
"ollama-mistral"       # Mistral 7B (local)
```

### Embedding Presets
```python
"openai"          # text-embedding-3-small
"openai-large"    # text-embedding-3-large (better quality)
"huggingface"     # all-MiniLM-L6-v2 (local, free)
"ollama"          # nomic-embed-text (local, free)
"google"          # embedding-001
```

## 🔑 API Key Setup

### Environment Variables

Create a `.env` file:

```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Google
GOOGLE_API_KEY=AI...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# HuggingFace (optional for remote inference)
HUGGINGFACE_API_KEY=hf_...

# Cohere
COHERE_API_KEY=...

# AWS (for Bedrock)
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

### Setting Keys Programmatically

```python
from model_config import ModelConfig

config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    api_key="sk-your-key-here"  # Override env var
)
```

## 💡 Usage Examples

### Cost-Effective: Local Embeddings + Cloud LLM

```python
# Best of both worlds: powerful cloud LLM + free local embeddings
agent = NistRagAgentV2(
    model_config="openai-gpt4",      # Quality reasoning
    embedding_config="huggingface"    # Free local embeddings
)
```

### 🔥 LiteLLM: Universal Interface for 100+ Providers

**Why LiteLLM?**
- Single interface for OpenAI, Anthropic, Google, AWS, Azure, and 100+ more
- Automatic retries and fallbacks
- Built-in cost tracking
- Request caching
- Load balancing
- Easy provider switching

```python
from model_config import ModelConfig

# Use any provider through LiteLLM
# Format: "provider/model-name"

# OpenAI via LiteLLM
config = ModelConfig(
    provider="litellm",
    model_name="openai/gpt-4o"
)

# Anthropic via LiteLLM
config = ModelConfig(
    provider="litellm",
    model_name="anthropic/claude-3-5-sonnet-20241022"
)

# Google via LiteLLM
config = ModelConfig(
    provider="litellm",
    model_name="gemini/gemini-1.5-pro"
)

# AWS Bedrock via LiteLLM
config = ModelConfig(
    provider="litellm",
    model_name="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
)

# Azure OpenAI via LiteLLM
config = ModelConfig(
    provider="litellm",
    model_name="azure/your-deployment-name",
    additional_kwargs={
        "api_version": "2024-02-15-preview",
        "api_base": "https://your-resource.openai.azure.com/"
    }
)

agent = NistRagAgentV2(model_config=config)
```

**LiteLLM with Fallbacks:**
```python
config = ModelConfig(
    provider="litellm",
    model_name="openai/gpt-4o",
    additional_kwargs={
        "fallbacks": ["anthropic/claude-3-5-sonnet-20241022"],
        "num_retries": 3
    }
)
```

### 100% Free & Local: Ollama

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

```python
agent = NistRagAgentV2(
    model_config="ollama-llama3",
    embedding_config="ollama"
)
# No API costs, all runs locally!
```

### Azure OpenAI (Enterprise)

```python
from model_config import ModelConfig, EmbeddingConfig

llm = ModelConfig(
    provider="azure_openai",
    model_name="gpt-4",  # Your Azure deployment name
    base_url="https://your-resource.openai.azure.com/",
    additional_kwargs={"api_version": "2024-02-15-preview"}
)

embed = EmbeddingConfig(
    provider="azure_openai",
    model_name="text-embedding-ada-002",
    base_url="https://your-resource.openai.azure.com/"
)

agent = NistRagAgentV2(model_config=llm, embedding_config=embed)
```

### Mix & Match Providers

```python
# Claude for reasoning + Google embeddings
agent = NistRagAgentV2(
    model_config="anthropic-claude",
    embedding_config="google"
)

# Gemini + local embeddings
agent = NistRagAgentV2(
    model_config="google-gemini",
    embedding_config="huggingface"
)
```

## 📦 Installation

### Base Installation (OpenAI only)
```bash
pip install -r requirements.txt
```

### Add Anthropic Support
```bash
pip install langchain-anthropic
```

### Add Google Gemini Support
```bash
pip install langchain-google-genai
```

### Add Local Ollama Support
```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1:8b
pip install langchain-community  # Already in requirements
```

### Add AWS Bedrock Support
```bash
pip install langchain-aws boto3
```

### Add Cohere Support
```bash
pip install langchain-cohere
```

## 🔧 Advanced Configuration

### Custom Temperature & Tokens

```python
config = ModelConfig(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.5,  # More creative
    max_tokens=4000,  # Longer responses
    additional_kwargs={
        "top_p": 0.9,
        "top_k": 40
    }
)
```

### Custom Base URL (Self-Hosted)

```python
config = ModelConfig(
    provider="ollama",
    model_name="llama3.1:8b",
    base_url="http://192.168.1.100:11434"  # Remote Ollama server
)
```

### Retry & Timeout Settings

```python
config = ModelConfig(
    provider="openai",
    model_name="gpt-4o",
    additional_kwargs={
        "request_timeout": 60,
        "max_retries": 3
    }
)
```

## 🆚 Provider Comparison

| Provider | Speed | Quality | Cost | Privacy |
|----------|-------|---------|------|---------|
| **GPT-4** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$$ | ☁️ Cloud |
| **Claude** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | $$ | ☁️ Cloud |
| **Gemini** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $ | ☁️ Cloud |
| **GPT-3.5** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | $ | ☁️ Cloud |
| **Ollama** | ⭐⭐⭐ | ⭐⭐⭐ | **FREE** | 🔒 Local |

## 🧪 Testing Different Providers

```bash
# Run multi-provider demo
python examples/multi_provider_demo.py
```

## 🐛 Troubleshooting

### "Module not found" errors
Install the specific provider package:
```bash
pip install langchain-anthropic  # For Anthropic
pip install langchain-google-genai  # For Google
```

### Ollama connection refused
```bash
# Start Ollama service
ollama serve

# Or check if it's running
curl http://localhost:11434/api/tags
```

### API key not found
```bash
# Check your .env file
cat .env | grep API_KEY

# Or set directly
export OPENAI_API_KEY="sk-..."
```

## 📚 Additional Resources

- [model_config.py](model_config.py) - Source configuration module
- [examples/multi_provider_demo.py](examples/multi_provider_demo.py) - Examples for each provider
- [Ollama Documentation](https://ollama.ai)
- [LangChain Providers](https://python.langchain.com/docs/integrations/providers)

## 🔄 Backward Compatibility

Old initialization still works but shows deprecation warning:

```python
# Old way (deprecated but still works)
agent = NistRagAgentV2(
    openai_api_key="sk-...",
    model="gpt-4o"
)

# New way (recommended)
agent = NistRagAgentV2(
    model_config="openai-gpt4"
)
```
