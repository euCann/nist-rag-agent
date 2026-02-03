#!/bin/bash
# Setup script for NIST RAG Agent

echo "Setting up NIST RAG Agent..."

# Copy embeddings from source location
EMBEDDINGS_SOURCE="../../../Python/NIST/Grounding Documents/rag_embeddings"
EMBEDDINGS_DEST="./embeddings"

if [ -d "$EMBEDDINGS_SOURCE" ]; then
    echo "Copying NIST embeddings..."
    cp "$EMBEDDINGS_SOURCE"/*.chunks.json "$EMBEDDINGS_DEST/" 2>/dev/null
    
    COUNT=$(ls -1 "$EMBEDDINGS_DEST"/*.chunks.json 2>/dev/null | wc -l)
    echo "Copied $COUNT embedding files"
else
    echo "Note: Embeddings not found at $EMBEDDINGS_SOURCE"
    echo "Agent will use mock data for demo purposes"
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your OPENAI_API_KEY"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Run: python examples/basic_query.py"
echo "3. Or start API: python api_service.py"
