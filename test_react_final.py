"""
Test the ReAct agent with TIMA Llama 70B model (non-interactive version).
"""

from model_config import ModelConfig, create_llm
from react_agent import ReActAgent
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


def create_faiss_index():
    """Create or load FAISS index with NIST documents."""
    cache_dir = ".cache/huggingface"
    faiss_cache = os.path.join(cache_dir, "faiss_index_train")
    
    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Try to load cached index
    if os.path.exists(faiss_cache):
        print(f"Loading cached FAISS index from {faiss_cache}...")
        vectorstore = FAISS.load_local(
            faiss_cache, 
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"✅ Loaded FAISS index with {vectorstore.index.ntotal} documents")
        return vectorstore
    
    # Load dataset
    print("Loading NIST dataset...")
    dataset = load_dataset(
        "ethanolivertroy/nist-cybersecurity-training",
        split="train",
        cache_dir=cache_dir
    )
    print(f"✅ Loaded {len(dataset)} examples")
    
    # Create documents
    print("Creating documents...")
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    for item in dataset:
        if 'text' in item and item['text']:
            chunks = text_splitter.split_text(item['text'])
            documents.extend(chunks)
    
    print(f"✅ Created {len(documents)} document chunks")
    
    # Create FAISS index
    print("Creating FAISS index...")
    vectorstore = FAISS.from_texts(documents, embeddings)
    
    # Cache it
    os.makedirs(cache_dir, exist_ok=True)
    vectorstore.save_local(faiss_cache)
    print(f"✅ Saved FAISS index to {faiss_cache}")
    
    return vectorstore


def main():
    print("="*80)
    print("NIST RAG Agent - ReAct Implementation with TIMA Llama 70B")
    print("="*80)
    
    # Create FAISS index
    vectorstore = create_faiss_index()
    
    # Create LLM
    print("\nInitializing Llama 70B model...")
    model_config = ModelConfig(
        provider="litellm",
        model_name="3.170b",
        api_key="sk-NvLMYtdkU9f4g3uNk-b6Sw",
        base_url="http://meshnet:4000",
        temperature=0.7,
        max_tokens=2000
    )
    llm = create_llm(model_config)
    print("✅ Model initialized")
    
    # Define tools
    def nist_lookup(query: str) -> str:
        """Search NIST documents for information related to the query."""
        results = vectorstore.similarity_search(query, k=3)
        if not results:
            return "No relevant NIST documents found."
        
        combined = "\n\n".join([
            f"Document {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(results)
        ])
        return combined
    
    def control_lookup(control_id: str) -> str:
        """Look up a specific NIST control by its ID (e.g., AC-1, SC-7)."""
        query = f"NIST control {control_id}"
        results = vectorstore.similarity_search(query, k=2)
        if not results:
            return f"Control {control_id} not found."
        
        return "\n\n".join([doc.page_content for doc in results])
    
    def search_by_document(document_name: str) -> str:
        """Search for content from a specific NIST document. Parameter: document_name (e.g., 'SP 800-53', 'Cybersecurity Framework')."""
        results = vectorstore.similarity_search(document_name, k=3)
        if not results:
            return f"No content found for document '{document_name}'."
        
        return "\n\n".join([doc.page_content for doc in results])
    
    tools = {
        "nist_lookup": nist_lookup,
        "control_lookup": control_lookup,
        "search_by_document": search_by_document
    }
    
    # Create ReAct agent
    agent = ReActAgent(llm, tools, max_iterations=5)
    
    # Test queries
    test_queries = [
        "What is the NIST Cybersecurity Framework?",
        "Explain the key controls in SP 800-53",
        "What is Zero Trust Architecture according to NIST?",
        "What are the main security control families in NIST?"
    ]
    
    all_results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        
        result = agent.run(query, verbose=True)
        all_results.append({
            'query': query,
            'answer': result['answer'],
            'iterations': result['iterations']
        })
        
        print(f"\n{'='*80}")
        print(f"FINAL ANSWER:")
        print('='*80)
        print(result['answer'])
        print(f"\nIterations used: {result['iterations']}")
    
    # Summary
    print(f"\n\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    
    for i, result in enumerate(all_results, 1):
        print(f"\n{i}. {result['query']}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Answer: {result['answer'][:150]}...")


if __name__ == "__main__":
    main()
