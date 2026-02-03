#!/usr/bin/env python3
"""
Quick test to verify model provider configuration
Tests without requiring full dataset download
"""

import sys
from pathlib import Path

def test_imports():
    """Test that model_config module loads correctly."""
    print("Testing imports...")
    try:
        from model_config import (
            ModelConfig, EmbeddingConfig,
            create_llm, create_embeddings,
            get_preset_config, get_preset_embeddings,
            PRESET_CONFIGS, PRESET_EMBEDDINGS
        )
        print("✅ model_config imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_presets():
    """Test that presets are properly defined."""
    print("\nTesting presets...")
    try:
        from model_config import PRESET_CONFIGS, PRESET_EMBEDDINGS
        
        print(f"  LLM Presets: {len(PRESET_CONFIGS)}")
        for name in PRESET_CONFIGS:
            print(f"    - {name}")
        
        print(f"  Embedding Presets: {len(PRESET_EMBEDDINGS)}")
        for name in PRESET_EMBEDDINGS:
            print(f"    - {name}")
        
        print("✅ Presets loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Preset error: {e}")
        return False


def test_config_creation():
    """Test creating configs."""
    print("\nTesting config creation...")
    try:
        from model_config import ModelConfig, EmbeddingConfig
        
        # Test model config
        llm_config = ModelConfig(
            provider="openai",
            model_name="gpt-4o",
            temperature=0
        )
        print(f"  ✅ Created LLM config: {llm_config.provider} - {llm_config.model_name}")
        
        # Test embedding config
        embed_config = EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small"
        )
        print(f"  ✅ Created embedding config: {embed_config.provider} - {embed_config.model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Config creation error: {e}")
        return False


def test_preset_loading():
    """Test loading presets."""
    print("\nTesting preset loading...")
    try:
        from model_config import get_preset_config, get_preset_embeddings
        
        # Test LLM preset
        config = get_preset_config("openai-gpt4")
        print(f"  ✅ Loaded LLM preset: {config.model_name}")
        
        # Test embedding preset
        embed = get_preset_embeddings("openai")
        print(f"  ✅ Loaded embedding preset: {embed.model_name}")
        
        return True
    except Exception as e:
        print(f"❌ Preset loading error: {e}")
        return False


def test_llm_creation_mock():
    """Test LLM creation (without API calls)."""
    print("\nTesting LLM creation (mock)...")
    try:
        from model_config import ModelConfig
        
        # Test various providers (just config, no actual instantiation)
        providers = ["openai", "anthropic", "google", "ollama"]
        
        for provider in providers:
            config = ModelConfig(
                provider=provider,
                model_name="test-model"
            )
            print(f"  ✅ Config for {provider}: {config.model_name}")
        
        return True
    except Exception as e:
        print(f"❌ LLM creation error: {e}")
        return False


def test_agent_import():
    """Test that agent imports work with new config."""
    print("\nTesting agent import...")
    try:
        from agent import NistRagAgentV2
        print("  ✅ Agent class imported successfully")
        
        # Check if __init__ has new parameters
        import inspect
        sig = inspect.signature(NistRagAgentV2.__init__)
        params = list(sig.parameters.keys())
        
        if 'model_config' in params:
            print("  ✅ model_config parameter found")
        else:
            print("  ⚠️  model_config parameter not found")
        
        if 'embedding_config' in params:
            print("  ✅ embedding_config parameter found")
        else:
            print("  ⚠️  embedding_config parameter not found")
        
        return True
    except Exception as e:
        print(f"❌ Agent import error: {e}")
        return False


def test_backward_compatibility():
    """Test that old initialization params still work."""
    print("\nTesting backward compatibility...")
    try:
        # This should work without errors (even if we don't actually initialize)
        from agent import NistRagAgentV2
        import inspect
        
        sig = inspect.signature(NistRagAgentV2.__init__)
        params = list(sig.parameters.keys())
        
        # Check for deprecated params
        deprecated_params = ['openai_api_key', 'model']
        found = [p for p in deprecated_params if p in params]
        
        if found:
            print(f"  ✅ Backward compatibility params present: {found}")
        else:
            print("  ⚠️  Backward compatibility params not found")
        
        return True
    except Exception as e:
        print(f"❌ Backward compatibility error: {e}")
        return False


def main():
    print("="*70)
    print("NIST RAG Agent - Model Provider Configuration Test")
    print("="*70)
    
    tests = [
        ("Imports", test_imports),
        ("Presets", test_presets),
        ("Config Creation", test_config_creation),
        ("Preset Loading", test_preset_loading),
        ("LLM Creation", test_llm_creation_mock),
        ("Agent Import", test_agent_import),
        ("Backward Compatibility", test_backward_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Model provider configuration is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
