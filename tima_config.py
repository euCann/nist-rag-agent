"""
Quick TIMA Configuration for NIST RAG Agent
Pre-configured for your TIMA infrastructure
"""

from model_config import ModelConfig, EmbeddingConfig

# Your TIMA LiteLLM configuration
TIMA_API_KEY = "sk-NvLMYtdkU9f4g3uNk-b6Sw"
TIMA_BASE_URL = "http://meshnet:4000"  # Or use http://192.168.1.198:4000

# Available TIMA Models
TIMA_MODELS = {
    "gpt-oss-20b": ModelConfig(
        provider="litellm",
        model_name="oss20b",
        api_key=TIMA_API_KEY,
        base_url=TIMA_BASE_URL,
        temperature=0.7,
        max_tokens=2000
    ),
    "llama-70b": ModelConfig(
        provider="litellm",
        model_name="3.170b",
        api_key=TIMA_API_KEY,
        base_url=TIMA_BASE_URL,
        temperature=0.7,
        max_tokens=2000
    ),
    "qwen-coder-7b": ModelConfig(
        provider="litellm",
        model_name="7binstruct",
        api_key=TIMA_API_KEY,
        base_url=TIMA_BASE_URL,
        temperature=0.3,  # Lower temp for code
        max_tokens=2000
    ),
    "qwen-1.5b": ModelConfig(
        provider="litellm",
        model_name="qwen1.5b",
        api_key=TIMA_API_KEY,
        base_url=TIMA_BASE_URL,
        temperature=0.7,
        max_tokens=1000
    ),
}

# Embedding options
TIMA_EMBEDDINGS = {
    "local-huggingface": EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ),
    "openai": EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-small"
    ),
}


def get_tima_agent(model="gpt-oss-20b", embeddings="local-huggingface"):
    """
    Quick helper to get a pre-configured TIMA agent
    
    Args:
        model: One of "gpt-oss-20b", "llama-70b", "qwen-coder-7b", "qwen-1.5b"
        embeddings: One of "local-huggingface", "openai"
    
    Returns:
        Configured NistRagAgentV2 instance
    """
    from agent import NistRagAgentV2
    
    return NistRagAgentV2(
        model_config=TIMA_MODELS[model],
        embedding_config=TIMA_EMBEDDINGS[embeddings]
    )


# Example usage:
if __name__ == "__main__":
    # Quick test
    agent = get_tima_agent("gpt-oss-20b", "local-huggingface")
    response = agent.query("What is NIST SP 800-53?")
    print(response["answer"][:200], "...")
