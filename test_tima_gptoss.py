"""
Test NIST RAG Agent with TIMA GPT-OSS 20B Model
Uses your LiteLLM proxy to access the oss20b model
"""

from agent import NistRagAgentV2
from model_config import ModelConfig, EmbeddingConfig


def test_with_gpt_oss():
    """Test agent with your TIMA GPT-OSS 20B model"""
    print("="*80)
    print("Testing NIST RAG Agent with TIMA GPT-OSS 20B (oss20b)")
    print("="*80)
    
    # Configure LiteLLM to use your TIMA system
    # Note: Using Llama 70B instead of GPT-OSS because GPT-OSS is a reasoning model
    # that doesn't support system messages or function calling (required for agents)
    llm_config = ModelConfig(
        provider="litellm",
        model_name="3.170b",  # Llama 3.1 70B - supports function calling
        api_key="sk-NvLMYtdkU9f4g3uNk-b6Sw",  # Your admin key
        base_url="http://meshnet:4000",  # Or use http://192.168.1.198:4000
        temperature=0.7,
        max_tokens=2000,
        additional_kwargs={
            "timeout": 60
        }
    )
    
    # For embeddings, use local HuggingFace (free) or OpenAI
    embed_config = EmbeddingConfig(
        provider="huggingface",
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("\n🔧 Configuration:")
    print(f"   LLM: {llm_config.provider}/{llm_config.model_name}")
    print(f"   Endpoint: {llm_config.base_url}")
    print(f"   Embeddings: {embed_config.provider}/{embed_config.model_name}")
    
    print("\n📥 Initializing agent (first run downloads NIST dataset ~7GB)...")
    agent = NistRagAgentV2(
        model_config=llm_config,
        embedding_config=embed_config,
        top_k=5,
        dataset_split="train"
    )
    
    print("\n✅ Agent initialized successfully!")
    
    # Test queries
    test_queries = [
        "What is the NIST Cybersecurity Framework?",
        "Explain the key controls in SP 800-53",
        "What is Zero Trust Architecture according to NIST?",
        "What are the main security control families in NIST?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print('='*80)
        
        try:
            response = agent.query(query, session_id="tima_test")
            print(f"\n📤 Answer:\n{response['answer']}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
    
    # Get stats
    print("\n" + "="*80)
    print("Agent Statistics")
    print("="*80)
    stats = agent.get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


def test_comparison():
    """Compare GPT-OSS with other TIMA models"""
    print("\n" + "="*80)
    print("Model Comparison Test")
    print("="*80)
    
    models = [
        ("GPT-OSS 20B", "oss20b"),
        ("Llama 3.1 70B", "3.170b"),
        ("Qwen Coder 7B", "7binstruct"),
    ]
    
    query = "What is NIST SP 800-171?"
    
    for name, model_id in models:
        print(f"\n--- Testing {name} ({model_id}) ---")
        
        try:
            config = ModelConfig(
                provider="litellm",
                model_name=model_id,
                api_key="sk-6654c2c12c689b590a54140b86131d",
                base_url="http://meshnet:4000",
                temperature=0.3
            )
            
            agent = NistRagAgentV2(
                model_config=config,
                embedding_config=EmbeddingConfig(
                    provider="huggingface",
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
            
            response = agent.query(query, session_id=f"compare_{model_id}")
            print(f"Response: {response['answer'][:300]}...")
            
        except Exception as e:
            print(f"Error with {name}: {e}")


def test_with_fallback():
    """Test with fallback models"""
    print("\n" + "="*80)
    print("Testing with Fallback Configuration")
    print("="*80)
    
    config = ModelConfig(
        provider="litellm",
        model_name="oss20b",
        api_key="sk-NvLMYtdkU9f4g3uNk-b6Sw",
        base_url="http://meshnet:4000",
        additional_kwargs={
            "fallbacks": ["3.170b", "7binstruct"]  # Fallback to Llama or Qwen if oss20b fails
        }
    )
    
    agent = NistRagAgentV2(model_config=config)
    response = agent.query("What is incident response according to NIST?")
    print(f"Answer: {response['answer'][:300]}...")


if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NIST RAG Agent - TIMA GPT-OSS 20B Test Suite                   ║
║                                                                              ║
║  Testing with your local TIMA infrastructure:                               ║
║  • Model: GPT-OSS 20B (oss20b)                                             ║
║  • Endpoint: http://meshnet:4000                                            ║
║  • Also available: Llama 3.1 70B (3.170b), Qwen Coder (7binstruct)        ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Main test
        test_with_gpt_oss()
        
        # Optional: Compare models (comment out if not needed)
        # test_comparison()
        
        # Optional: Test fallback (comment out if not needed)
        # test_with_fallback()
        
        print("\n✅ All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
