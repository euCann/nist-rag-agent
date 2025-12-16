"""
Quick test to verify NIST RAG Agent setup
Run this to check if everything is configured correctly
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if required packages are installed."""
    print("Checking Python packages...")
    required = [
        'langchain',
        'langchain_openai',
        'faiss',
        'fastapi',
        'uvicorn',
        'dotenv'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages: pip install {' '.join(missing)}")
        return False
    return True


def check_env():
    """Check if .env file exists and has API key."""
    print("\nChecking environment configuration...")
    
    if not Path('.env').exists():
        print("  ❌ .env file not found")
        print("  💡 Run: cp .env.example .env")
        print("  💡 Then add your OPENAI_API_KEY")
        return False
    
    print("  ✅ .env file exists")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'sk-your-key-here':
        print("  ❌ OPENAI_API_KEY not configured")
        print("  💡 Edit .env and add your actual API key")
        return False
    
    print("  ✅ OPENAI_API_KEY configured")
    return True


def check_embeddings():
    """Check if embeddings directory has data."""
    print("\nChecking embeddings...")
    
    embeddings_dir = Path('embeddings')
    if not embeddings_dir.exists():
        print("  ❌ embeddings/ directory not found")
        return False
    
    json_files = list(embeddings_dir.glob('*.chunks.json'))
    
    if not json_files:
        print("  ⚠️  No embedding files found")
        print("  💡 Agent will use mock data for demo")
        return True
    
    print(f"  ✅ Found {len(json_files)} NIST document embeddings")
    return True


def test_agent():
    """Try to initialize the agent."""
    print("\nTesting agent initialization...")
    
    try:
        from agent import NistRagAgent
        agent = NistRagAgent()
        print("  ✅ Agent initialized successfully")
        
        # Test a simple query
        print("\nTesting query...")
        response = agent.query("What is access control?", session_id="test")
        
        if response and response.get('answer'):
            print("  ✅ Query successful")
            print(f"\n  Sample response (first 200 chars):")
            print(f"  {response['answer'][:200]}...")
            return True
        else:
            print("  ❌ Query failed - no response")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("NIST RAG Agent - Setup Verification")
    print("="*60)
    
    checks = [
        ("Requirements", check_requirements),
        ("Environment", check_env),
        ("Embeddings", check_embeddings),
    ]
    
    passed = 0
    for name, check_func in checks:
        if check_func():
            passed += 1
    
    print("\n" + "="*60)
    
    if passed == len(checks):
        print("✅ All checks passed!")
        print("\nTesting agent...")
        if test_agent():
            print("\n" + "="*60)
            print("🎉 Setup complete and working!")
            print("="*60)
            print("\nNext steps:")
            print("  1. Try: python examples/basic_query.py")
            print("  2. Or:  python api_service.py")
            return 0
        else:
            print("\n⚠️  Agent test failed - check error messages above")
            return 1
    else:
        print(f"❌ {len(checks) - passed} check(s) failed")
        print("\nPlease fix the issues above and run this test again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
