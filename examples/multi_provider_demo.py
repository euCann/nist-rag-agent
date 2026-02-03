"""
Examples of using NIST RAG Agent with different model providers
Demonstrates provider-agnostic configuration
"""

from agent import NistRagAgentV2
from model_config import ModelConfig, EmbeddingConfig


def example_openai():
    """Example: OpenAI GPT-4"""
    print("\n" + "="*80)
    print("Example 1: OpenAI GPT-4")
    print("="*80)
    
    # Using preset
    agent = NistRagAgentV2(
        model_config="openai-gpt4",
        embedding_config="openai"
    )
    
    response = agent.query("What is the NIST Cybersecurity Framework?")
    print(f"Answer: {response['answer'][:200]}...")


def example_anthropic():
    """Example: Anthropic Claude"""
    print("\n" + "="*80)
    print("Example 2: Anthropic Claude 3.5 Sonnet")
    print("="*80)
    
    agent = NistRagAgentV2(
        model_config="anthropic-claude",
        embedding_config="openai"  # Use OpenAI embeddings with Claude
    )
    
    response = agent.query("Explain Zero Trust Architecture")
    print(f"Answer: {response['answer'][:200]}...")


def example_google_gemini():
    """Example: Google Gemini"""
    print("\n" + "="*80)
    print("Example 3: Google Gemini Pro")
    print("="*80)
    
    agent = NistRagAgentV2(
        model_config="google-gemini",
        embedding_config="google"
    )
    
    response = agent.query("What are the key controls in SP 800-53?")
    print(f"Answer: {response['answer'][:200]}...")


def example_ollama_local():
    """Example: Ollama (Local LLMs)"""
    print("\n" + "="*80)
    print("Example 4: Ollama Llama 3.1 (Local)")
    print("="*80)
    print("Note: Requires Ollama running locally (ollama.ai)")
    
    agent = NistRagAgentV2(
        model_config="ollama-llama3",
        embedding_config="ollama"  # Local embeddings too!
    )
    
    response = agent.query("What is FIPS 140-3?")
    print(f"Answer: {response['answer'][:200]}...")


def example_custom_config():
    """Example: Custom configuration"""
    print("\n" + "="*80)
    print("Example 5: Custom Configuration")
    print("="*80)
    
    # Custom LLM config
    llm_config = ModelConfig(
        provider="openai",
        model_name="gpt-4-turbo-preview",
        temperature=0.3,  # Slightly more creative
        max_tokens=2000
    )
    
    # Use HuggingFace embeddings (free, local)
    embed_config = EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    agent = NistRagAgentV2(
        model_config=llm_config,
        embedding_config=embed_config
    )
    
    response = agent.query("What is post-quantum cryptography?")
    print(f"Answer: {response['answer'][:200]}...")


def example_azure_openai():
    """Example: Azure OpenAI"""
    print("\n" + "="*80)
    print("Example 6: Azure OpenAI")
    print("="*80)
    
    azure_config = ModelConfig(
        provider="azure_openai",
        model_name="gpt-4",  # Your deployment name
        base_url="https://your-resource.openai.azure.com/",
        additional_kwargs={
            "api_version": "2024-02-15-preview"
        }
    )
    
    azure_embed = EmbeddingConfig(
        provider="azure_openai",
        model_name="text-embedding-ada-002",  # Your deployment
        base_url="https://your-resource.openai.azure.com/"
    )
    
    agent = NistRagAgentV2(
        model_config=azure_config,
        embedding_config=azure_embed
    )
    
    response = agent.query("What is IR-4 incident handling?")
    print(f"Answer: {response['answer'][:200]}...")


def example_mixed_providers():
    """Example: Mix different providers"""
    print("\n" + "="*80)
    print("Example 7: Mixed Providers")
    print("="*80)
    print("Using Claude for reasoning + local HuggingFace embeddings")
    
    agent = NistRagAgentV2(
        model_config="anthropic-claude",      # Cloud LLM
        embedding_config="huggingface"         # Local embeddings
    )
    
    response = agent.query("Compare AC-1 and AC-2 controls")
    print(f"Answer: {response['answer'][:200]}...")


def compare_providers():
    """Compare the same query across different providers"""
    print("\n" + "="*80)
    print("Provider Comparison")
    print("="*80)
    
    query = "What is the purpose of the NIST CSF Govern function?"
    
    providers = [
        ("OpenAI GPT-4", "openai-gpt4", "openai"),
        ("Anthropic Claude", "anthropic-claude", "openai"),
        ("Google Gemini", "google-gemini", "google"),
    ]
    
    for name, model_preset, embed_preset in providers:
        try:
            print(f"\n--- {name} ---")
            agent = NistRagAgentV2(
                model_config=model_preset,
                embedding_config=embed_preset
            )
            response = agent.query(query, session_id=f"compare_{name}")
            print(f"Answer: {response['answer'][:150]}...")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║         NIST RAG Agent - Multi-Provider Examples                            ║
║                                                                              ║
║  Demonstrates using different LLM and embedding providers:                  ║
║  • OpenAI (GPT-4, GPT-3.5)                                                  ║
║  • Anthropic (Claude)                                                       ║
║  • Google (Gemini)                                                          ║
║  • Ollama (Local LLMs)                                                      ║
║  • Azure OpenAI                                                             ║
║  • HuggingFace (Local embeddings)                                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run examples (comment out providers you don't have configured)
    
    # Cloud providers
    try:
        example_openai()
    except Exception as e:
        print(f"OpenAI example failed: {e}")
    
    try:
        example_anthropic()
    except Exception as e:
        print(f"Anthropic example failed: {e}")
    
    try:
        example_google_gemini()
    except Exception as e:
        print(f"Google example failed: {e}")
    
    # Local provider (requires Ollama installed)
    try:
        example_ollama_local()
    except Exception as e:
        print(f"Ollama example failed (is Ollama running?): {e}")
    
    # Custom configs
    try:
        example_custom_config()
    except Exception as e:
        print(f"Custom config example failed: {e}")
    
    # Mixed providers
    try:
        example_mixed_providers()
    except Exception as e:
        print(f"Mixed provider example failed: {e}")
    
    # Comparison (requires multiple API keys)
    # compare_providers()
    
    print("\n✅ Examples completed!")
