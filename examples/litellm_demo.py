"""
LiteLLM Examples - Universal Interface for 100+ LLM Providers
Demonstrates using LiteLLM for provider-agnostic AI agent development
"""

from agent import NistRagAgentV2
from model_config import ModelConfig, EmbeddingConfig


def example_litellm_simple():
    """Example: LiteLLM with simple preset"""
    print("\n" + "="*80)
    print("Example 1: LiteLLM Simple Preset")
    print("="*80)
    
    # Using LiteLLM preset (easiest)
    agent = NistRagAgentV2(
        model_config="litellm-gpt4",
        embedding_config="openai"
    )
    
    response = agent.query("What is NIST SP 800-53?")
    print(f"Answer: {response['answer'][:200]}...")


def example_litellm_custom():
    """Example: Custom LiteLLM configuration"""
    print("\n" + "="*80)
    print("Example 2: Custom LiteLLM Configuration")
    print("="*80)
    
    config = ModelConfig(
        provider="litellm",
        model_name="openai/gpt-4o",  # Format: provider/model
        temperature=0.3,
        max_tokens=1500
    )
    
    agent = NistRagAgentV2(
        model_config=config,
        embedding_config="huggingface"  # Free local embeddings
    )
    
    response = agent.query("Explain Zero Trust Architecture")
    print(f"Answer: {response['answer'][:200]}...")


def example_litellm_with_fallbacks():
    """Example: LiteLLM with automatic fallbacks"""
    print("\n" + "="*80)
    print("Example 3: LiteLLM with Fallbacks")
    print("="*80)
    print("If GPT-4 fails, automatically falls back to Claude")
    
    config = ModelConfig(
        provider="litellm",
        model_name="openai/gpt-4o",
        additional_kwargs={
            "fallbacks": [
                "anthropic/claude-3-5-sonnet-20241022",
                "gemini/gemini-1.5-pro"
            ],
            "num_retries": 3
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    response = agent.query("What are the key NIST CSF 2.0 functions?")
    print(f"Answer: {response['answer'][:200]}...")


def example_litellm_multiple_providers():
    """Example: Easily switch between providers"""
    print("\n" + "="*80)
    print("Example 4: Switch Between Providers with LiteLLM")
    print("="*80)
    
    providers = [
        ("OpenAI GPT-4", "openai/gpt-4o"),
        ("Anthropic Claude", "anthropic/claude-3-5-sonnet-20241022"),
        ("Google Gemini", "gemini/gemini-1.5-pro"),
    ]
    
    query = "What is AC-2 in NIST SP 800-53?"
    
    for name, model in providers:
        try:
            print(f"\n--- Testing {name} ---")
            config = ModelConfig(
                provider="litellm",
                model_name=model
            )
            agent = NistRagAgentV2(
                model_config=config,
                embedding_config="huggingface"
            )
            response = agent.query(query, session_id=f"litellm_{name}")
            print(f"Response: {response['answer'][:150]}...")
        except Exception as e:
            print(f"Error with {name}: {e}")


def example_litellm_aws_bedrock():
    """Example: AWS Bedrock through LiteLLM"""
    print("\n" + "="*80)
    print("Example 5: AWS Bedrock via LiteLLM")
    print("="*80)
    print("Requires AWS credentials configured")
    
    config = ModelConfig(
        provider="litellm",
        model_name="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        additional_kwargs={
            "aws_region_name": "us-east-1"
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    response = agent.query("What is FIPS 140-3?")
    print(f"Answer: {response['answer'][:200]}...")


def example_litellm_azure():
    """Example: Azure OpenAI through LiteLLM"""
    print("\n" + "="*80)
    print("Example 6: Azure OpenAI via LiteLLM")
    print("="*80)
    
    config = ModelConfig(
        provider="litellm",
        model_name="azure/your-deployment-name",
        additional_kwargs={
            "api_base": "https://your-resource.openai.azure.com/",
            "api_version": "2024-02-15-preview"
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    response = agent.query("What is incident response?")
    print(f"Answer: {response['answer'][:200]}...")


def example_litellm_cost_tracking():
    """Example: LiteLLM with built-in cost tracking"""
    print("\n" + "="*80)
    print("Example 7: Cost Tracking with LiteLLM")
    print("="*80)
    print("LiteLLM automatically tracks API costs")
    
    config = ModelConfig(
        provider="litellm",
        model_name="openai/gpt-3.5-turbo",  # Cheaper model
        additional_kwargs={
            "mock_response": None,  # Set to test without actual API calls
            "success_callback": ["langfuse", "prometheus"]  # Optional logging
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    response = agent.query("What is the NIST Cybersecurity Framework?")
    print(f"Answer: {response['answer'][:200]}...")
    print("\nNote: Check LiteLLM logs for cost information")


def example_litellm_caching():
    """Example: LiteLLM with response caching"""
    print("\n" + "="*80)
    print("Example 8: Response Caching with LiteLLM")
    print("="*80)
    print("Cache responses to save costs and improve speed")
    
    config = ModelConfig(
        provider="litellm",
        model_name="openai/gpt-4o",
        additional_kwargs={
            "caching": True,  # Enable caching
            "cache_params": {
                "ttl": 3600  # Cache for 1 hour
            }
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    
    query = "What are the pillars of NIST CSF?"
    
    # First call (hits API)
    print("\nFirst query (API call)...")
    response = agent.query(query)
    print(f"Answer: {response['answer'][:150]}...")
    
    # Second call (cached)
    print("\nSecond query (cached)...")
    response = agent.query(query)
    print(f"Answer: {response['answer'][:150]}...")


def example_litellm_load_balancing():
    """Example: Load balancing across multiple deployments"""
    print("\n" + "="*80)
    print("Example 9: Load Balancing Multiple Deployments")
    print("="*80)
    
    config = ModelConfig(
        provider="litellm",
        model_name="openai/gpt-4o",
        additional_kwargs={
            "router": {
                "deployments": [
                    {"model_name": "openai/gpt-4o", "api_key": "key1"},
                    {"model_name": "anthropic/claude-3-5-sonnet", "api_key": "key2"},
                ],
                "routing_strategy": "least-busy"  # or "latency-based"
            }
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    response = agent.query("What is SP 800-171?")
    print(f"Answer: {response['answer'][:200]}...")


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    LiteLLM Examples for NIST RAG Agent                       ║
║                                                                              ║
║  LiteLLM provides a unified interface for 100+ LLM providers:               ║
║  • OpenAI, Anthropic, Google, Azure, AWS, Cohere, and more                 ║
║  • Automatic retries and fallbacks                                          ║
║  • Built-in cost tracking and caching                                       ║
║  • Load balancing and rate limiting                                         ║
║  • Single API for all providers                                             ║
║                                                                              ║
║  Installation: pip install litellm                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Run examples (comment out what you don't need)
    
    try:
        example_litellm_simple()
    except Exception as e:
        print(f"Simple example failed: {e}")
    
    try:
        example_litellm_custom()
    except Exception as e:
        print(f"Custom config example failed: {e}")
    
    try:
        example_litellm_with_fallbacks()
    except Exception as e:
        print(f"Fallbacks example failed: {e}")
    
    try:
        example_litellm_multiple_providers()
    except Exception as e:
        print(f"Multiple providers example failed: {e}")
    
    # Advanced examples (uncomment if you have the setup)
    # example_litellm_aws_bedrock()
    # example_litellm_azure()
    # example_litellm_cost_tracking()
    # example_litellm_caching()
    # example_litellm_load_balancing()
    
    print("\n✅ LiteLLM examples completed!")
    print("\n💡 Tips:")
    print("   - LiteLLM format: 'provider/model-name'")
    print("   - Set API keys in .env or pass directly")
    print("   - Use fallbacks for reliability")
    print("   - Enable caching to save costs")
    print("   - Check docs: https://docs.litellm.ai")
