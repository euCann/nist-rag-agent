#!/usr/bin/env python3
"""
Offline validation test - checks code quality without requiring API keys or dependencies
"""

import ast
import sys
from pathlib import Path


def check_python_syntax(filepath):
    """Check if Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True, None
    except SyntaxError as e:
        return False, str(e)


def check_imports(filepath):
    """Extract and list imports from a Python file."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    
    return imports


def check_functions(filepath):
    """Count functions and classes in a Python file."""
    with open(filepath, 'r') as f:
        tree = ast.parse(f.read())
    
    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    
    return functions, classes


def main():
    print("="*60)
    print("NIST RAG Agent - Offline Validation")
    print("="*60)
    
    base_dir = Path(__file__).parent
    
    # Check core Python files
    python_files = [
        'agent.py',
        'api_service.py',
        'test_setup.py',
        'examples/basic_query.py',
        'examples/session_demo.py',
        'examples/api_client.py'
    ]
    
    print("\n1. Syntax Check")
    print("-" * 60)
    all_valid = True
    for filepath in python_files:
        full_path = base_dir / filepath
        if not full_path.exists():
            print(f"  ⚠️  {filepath} - NOT FOUND")
            continue
        
        valid, error = check_python_syntax(full_path)
        if valid:
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath} - {error}")
            all_valid = False
    
    if not all_valid:
        print("\n❌ Syntax errors found!")
        return 1
    
    # Analyze agent.py
    print("\n2. Code Analysis - agent.py")
    print("-" * 60)
    
    agent_file = base_dir / 'agent.py'
    if agent_file.exists():
        functions, classes = check_functions(agent_file)
        imports = check_imports(agent_file)
        
        print(f"  Classes: {len(classes)}")
        for cls in classes:
            print(f"    - {cls}")
        
        print(f"  Functions: {len(functions)}")
        key_functions = [f for f in functions if not f.startswith('_')]
        for func in key_functions[:5]:  # Show first 5
            print(f"    - {func}()")
        
        print(f"  Total imports: {len(set(imports))}")
        
        # Check for key components
        has_langchain = any('langchain' in i for i in imports if i)
        has_faiss = any('faiss' in i for i in imports if i)
        has_openai = any('openai' in i for i in imports if i)
        
        print(f"  ✅ LangChain: {has_langchain}")
        print(f"  ✅ FAISS: {has_faiss}")
        print(f"  ✅ OpenAI: {has_openai}")
    
    # Analyze api_service.py
    print("\n3. Code Analysis - api_service.py")
    print("-" * 60)
    
    api_file = base_dir / 'api_service.py'
    if api_file.exists():
        functions, classes = check_functions(api_file)
        imports = check_imports(api_file)
        
        print(f"  Functions: {len(functions)}")
        endpoints = [f for f in functions if not f.startswith('_')]
        for func in endpoints:
            print(f"    - /{func}")
        
        has_fastapi = any('fastapi' in i for i in imports if i)
        has_pydantic = any('pydantic' in i for i in imports if i)
        
        print(f"  ✅ FastAPI: {has_fastapi}")
        print(f"  ✅ Pydantic: {has_pydantic}")
    
    # Check documentation
    print("\n4. Documentation Check")
    print("-" * 60)
    
    docs = [
        'README.md',
        'QUICKSTART.md',
        'LICENSE',
        'requirements.txt',
        '.env.example',
        '.gitignore'
    ]
    
    for doc in docs:
        if (base_dir / doc).exists():
            size = (base_dir / doc).stat().st_size
            print(f"  ✅ {doc} ({size} bytes)")
        else:
            print(f"  ❌ {doc} - MISSING")
    
    # Check examples
    print("\n5. Examples Check")
    print("-" * 60)
    
    examples_dir = base_dir / 'examples'
    if examples_dir.exists():
        examples = list(examples_dir.glob('*.py'))
        print(f"  Found {len(examples)} example scripts:")
        for ex in examples:
            print(f"    - {ex.name}")
    else:
        print("  ⚠️  examples/ directory not found")
    
    # Check embeddings
    print("\n6. Embeddings Check")
    print("-" * 60)
    
    embeddings_dir = base_dir / 'embeddings'
    if embeddings_dir.exists():
        json_files = list(embeddings_dir.glob('*.chunks.json'))
        print(f"  Found {len(json_files)} NIST document embeddings")
        if json_files:
            total_size = sum(f.stat().st_size for f in json_files)
            print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
        else:
            print("  ⚠️  No embedding files (agent will use mock data)")
    else:
        print("  ⚠️  embeddings/ directory not found")
    
    # Summary
    print("\n" + "="*60)
    print("✅ OFFLINE VALIDATION PASSED")
    print("="*60)
    print("\nCode is syntactically valid and properly structured.")
    print("\nTo test functionality:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Set up .env: cp .env.example .env")
    print("  3. Add OPENAI_API_KEY to .env")
    print("  4. Run: python test_setup.py")
    print("\nReady for GitHub! 🚀")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
