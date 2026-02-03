#!/bin/bash
# Quick publish script for NIST RAG Agent

echo "=========================================="
echo "NIST RAG Agent - GitHub Publish Guide"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Git not initialized. Run: git init"
    exit 1
fi

echo "✅ Local repository ready"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1️⃣  Create GitHub Repository"
echo "   - Go to: https://github.com/new"
echo "   - Repository name: nist-rag-agent"
echo "   - Description: Conversational AI for NIST cybersecurity standards using RAG"
echo "   - Public or Private: Your choice"
echo "   - ⚠️  DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2️⃣  Connect and Push (replace YOUR_USERNAME)"
echo ""
echo "git remote add origin https://github.com/YOUR_USERNAME/nist-rag-agent.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""
echo "3️⃣  After Push - Configure Repository"
echo "   - Add topics: nist, cybersecurity, rag, langchain, ai, compliance"
echo "   - Add description from above"
echo "   - Enable Issues and Discussions"
echo ""
echo "4️⃣  Optional - Create First Release"
echo "git tag -a v1.0.0 -m 'First release: NIST RAG Agent'"
echo "git push origin v1.0.0"
echo ""
echo "=========================================="
echo "Repository: $(pwd)"
echo "Files committed: $(git ls-files | wc -l | xargs)"
echo "=========================================="
