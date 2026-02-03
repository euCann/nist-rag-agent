"""
NIST RAG Agent V2 - Enhanced Implementation with HuggingFace Dataset
A conversational AI agent using the ethanolivertroy/nist-cybersecurity-training dataset
with 530K+ training examples from 596 NIST publications.
"""

import os
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.memory import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from datasets import load_dataset
import numpy as np

# Import model configuration utilities
from model_config import (
    ModelConfig, EmbeddingConfig,
    create_llm, create_embeddings,
    get_preset_config, get_preset_embeddings
)


class NistRagAgentV2:
    """
    Enhanced NIST RAG Agent V2 using HuggingFace Dataset.
    
    Features:
    - 530K+ training examples from 596 NIST publications
    - FIPS, SP (800/1800), IR, and CSWP series coverage
    - Pre-computed embeddings with FAISS indexing
    - Multi-tool agent (RAG, control lookup, web search)
    - Session-based chat history
    - Citation with source documents and metadata
    
    New in V2:
    - HuggingFace dataset integration
    - Support for CSWP (Cybersecurity White Papers)
    - Enhanced metadata extraction from dataset
    - Improved chunking with semantic coherence
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_config: Optional[Union[ModelConfig, str]] = None,
        embedding_config: Optional[Union[EmbeddingConfig, str]] = None,
        top_k: int = 5,
        use_precomputed_embeddings: bool = False,
        dataset_split: str = "train",
        # Deprecated params for backward compatibility
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the NIST RAG Agent V2 (Provider Agnostic).
        
        Args:
            cache_dir: Directory to cache the HuggingFace dataset
            model_config: ModelConfig instance or preset name (e.g., 'openai-gpt4', 'anthropic-claude')
            embedding_config: EmbeddingConfig instance or preset name (e.g., 'openai', 'huggingface')
            top_k: Number of similar chunks to retrieve
            use_precomputed_embeddings: Use precomputed embeddings from dataset
            dataset_split: Dataset split to use ('train' or 'valid')
            openai_api_key: [DEPRECATED] Use model_config instead
            model: [DEPRECATED] Use model_config instead
            
        Examples:
            # Using presets
            agent = NistRagAgentV2(model_config="openai-gpt4", embedding_config="openai")
            agent = NistRagAgentV2(model_config="anthropic-claude", embedding_config="huggingface")
            agent = NistRagAgentV2(model_config="ollama-llama3", embedding_config="ollama")
            
            # Using custom configs
            agent = NistRagAgentV2(
                model_config=ModelConfig(provider="openai", model_name="gpt-4"),
                embedding_config=EmbeddingConfig(provider="openai", model_name="text-embedding-3-small")
            )
        """
        # Load environment variables
        load_dotenv()
        
        # Handle backward compatibility
        if model_config is None:
            if model or openai_api_key:
                print("⚠️  Warning: 'model' and 'openai_api_key' parameters are deprecated.")
                print("   Use model_config='openai-gpt4' or ModelConfig(...) instead.")
                from model_config import ModelConfig
                model_config = ModelConfig(
                    provider="openai",
                    model_name=model or "gpt-4o",
                    api_key=openai_api_key
                )
            else:
                # Default to OpenAI GPT-4o
                model_config = "openai-gpt4"
        
        # Handle string presets
        if isinstance(model_config, str):
            model_config = get_preset_config(model_config)
        
        if embedding_config is None:
            embedding_config = "openai"  # Default
        
        if isinstance(embedding_config, str):
            embedding_config = get_preset_embeddings(embedding_config)
        
        # Store configs
        self.model_config = model_config
        self.embedding_config = embedding_config
        self.top_k = top_k
        self.dataset_split = dataset_split
        self.use_precomputed_embeddings = use_precomputed_embeddings
        
        # Set up cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".cache" / "huggingface"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize provider-agnostic components
        print(f"🤖 Initializing LLM: {model_config.provider} - {model_config.model_name}")
        print(f"📊 Initializing embeddings: {embedding_config.provider} - {embedding_config.model_name}")
        
        self.embeddings = create_embeddings(embedding_config)
        self.llm = create_llm(model_config)
        self.session_histories: Dict[str, ChatMessageHistory] = {}
        
        # Load dataset and create vector store
        print(f"Loading NIST cybersecurity dataset (split: {dataset_split})...")
        self.dataset = self._load_dataset()
        self.vectorstore = self._create_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        # Set up tools and agent
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
        
    def _load_dataset(self):
        """Load the NIST cybersecurity training dataset from HuggingFace."""
        try:
            dataset = load_dataset(
                "ethanolivertroy/nist-cybersecurity-training",
                split=self.dataset_split,
                cache_dir=str(self.cache_dir)
            )
            print(f"✓ Loaded {len(dataset)} examples from NIST dataset")
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def _extract_metadata_from_example(self, example: Dict) -> Dict[str, Any]:
        """Extract metadata from dataset example."""
        metadata = {}
        
        # Try to parse metadata field if it exists
        if "metadata" in example and example["metadata"]:
            import json
            try:
                if isinstance(example["metadata"], str):
                    meta = json.loads(example["metadata"])
                else:
                    meta = example["metadata"]
                
                metadata.update({
                    "source": meta.get("source", "Unknown"),
                    "type": meta.get("type", "section"),
                    "chunk_id": meta.get("chunk_id", 0)
                })
            except:
                pass
        
        # Extract from messages if available
        if "messages" in example:
            messages = example["messages"]
            if isinstance(messages, str):
                import json
                try:
                    messages = json.loads(messages)
                except:
                    messages = []
            
            # Look for source information in assistant message
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Try to extract source from "According to..." pattern
                    if "According to" in content:
                        import re
                        match = re.search(r'According to ([^,]+)', content)
                        if match and "source" not in metadata:
                            metadata["source"] = match.group(1).strip()
        
        return metadata
    
    def _create_documents_from_dataset(self) -> List[Document]:
        """Convert dataset examples to LangChain documents."""
        documents = []
        
        print("Processing dataset examples...")
        for idx, example in enumerate(self.dataset):
            if idx % 10000 == 0:
                print(f"  Processed {idx}/{len(self.dataset)} examples...")
            
            # Extract content - dataset has 'text' field directly
            content = ""
            if "text" in example:
                content = example["text"]
            elif "messages" in example:
                # Fallback: try messages format
                messages = example["messages"]
                if isinstance(messages, str):
                    import json
                    try:
                        messages = json.loads(messages)
                    except:
                        continue
                
                # Concatenate user and assistant messages as content
                for msg in messages:
                    if msg.get("role") in ["user", "assistant"]:
                        msg_content = msg.get("content", "")
                        if msg_content:
                            content += msg_content + "\n\n"
            
            if not content or not content.strip():
                continue
            
            # Extract metadata
            metadata = self._extract_metadata_from_example(example)
            metadata["dataset_index"] = idx
            
            # Create document
            doc = Document(
                page_content=content.strip(),
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"✓ Created {len(documents)} documents from dataset")
        return documents
    
    def _create_vectorstore(self) -> FAISS:
        """Create FAISS vector store from dataset."""
        # Check for cached FAISS index
        faiss_path = self.cache_dir / f"faiss_index_{self.dataset_split}"
        
        if faiss_path.exists():
            print(f"Loading cached FAISS index from {faiss_path}")
            try:
                return FAISS.load_local(
                    str(faiss_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load cached index: {e}")
                print("Creating new index...")
        
        # Create documents from dataset
        documents = self._create_documents_from_dataset()
        
        if not documents:
            raise ValueError("No documents created from dataset")
        
        # Create FAISS index with batching for large datasets
        print("Creating FAISS index (this may take a while)...")
        batch_size = 1000
        vectorstore = None
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"  Indexing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
            
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                batch_store = FAISS.from_documents(batch, self.embeddings)
                vectorstore.merge_from(batch_store)
        
        # Save for future use
        print(f"Saving FAISS index to {faiss_path}")
        vectorstore.save_local(str(faiss_path))
        
        return vectorstore
    
    def _create_tools(self) -> List:
        """Create LangChain tools for the agent."""
        
        @tool("nist_lookup")
        def nist_lookup(query: str) -> str:
            """
            Search NIST cybersecurity standards and guidance.
            Use for any NIST, OSCAL, or cybersecurity compliance question.
            Covers 596 NIST publications including:
            - FIPS (Federal Information Processing Standards)
            - SP 800 series (Security Controls, Risk Management, etc.)
            - SP 1800 series (Practice Guides)
            - IR (Interagency/Internal Reports)
            - CSWP (Cybersecurity White Papers, including CSF 2.0)
            
            Returns relevant content with source citations.
            """
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant NIST content found for this query."
            
            results = []
            seen_sources = set()
            
            for doc in docs[:self.top_k]:
                meta = doc.metadata
                source = meta.get('source', 'Unknown')
                
                # Avoid duplicate sources
                if source in seen_sources:
                    continue
                seen_sources.add(source)
                
                chunk_type = meta.get('type', 'section')
                
                result = f"""
**Source**: {source}
**Type**: {chunk_type}
**Content**:
{doc.page_content[:800]}...

---
"""
                results.append(result)
            
            return "\n".join(results)
        
        @tool("control_lookup")
        def control_lookup(control_id: str) -> str:
            """
            Fetch detailed information for a specific NIST control by Control ID.
            Use when the user asks about a specific control (e.g., AC-1, AU-2, IR-4).
            Searches SP 800-53 security controls catalog.
            """
            # Search for documents with this control ID
            query = f"control {control_id} security control"
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return f"No details found for control {control_id}. Try using nist_lookup for broader search."
            
            # Filter for most relevant
            relevant_docs = []
            for doc in docs:
                if control_id.upper() in doc.page_content.upper():
                    relevant_docs.append(doc)
            
            if not relevant_docs:
                relevant_docs = docs[:1]
            
            doc = relevant_docs[0]
            meta = doc.metadata
            
            return f"""
**Control ID**: {control_id}
**Source**: {meta.get('source', 'NIST SP 800-53')}
**Type**: {meta.get('type', 'control')}

**Details**:
{doc.page_content[:1000]}
"""
        
        @tool("search_by_document")
        def search_by_document(document_name: str) -> str:
            """
            Search for content within a specific NIST document.
            Examples: "SP 800-53", "NIST CSF", "SP 800-171", "FIPS 140-3"
            Returns a summary of key content from that document.
            """
            query = f"document {document_name}"
            docs = self.retriever.get_relevant_documents(query)
            
            # Filter for matching document
            matching_docs = [d for d in docs if document_name.upper() in d.metadata.get('source', '').upper()]
            
            if not matching_docs:
                matching_docs = docs[:2]
            
            results = []
            for doc in matching_docs[:3]:
                meta = doc.metadata
                results.append(f"""
**Source**: {meta.get('source', 'Unknown')}
**Excerpt**:
{doc.page_content[:600]}...
---
""")
            
            return "\n".join(results) if results else f"No content found for {document_name}"
        
        # Web search fallback
        web_search = DuckDuckGoSearchRun()
        
        return [nist_lookup, control_lookup, search_by_document, web_search]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools."""
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a highly knowledgeable NIST cybersecurity expert assistant. "
                "You have access to 596 NIST publications with 530,000+ training examples including:\n"
                "- FIPS (Federal Information Processing Standards)\n"
                "- SP 800 series (Security Controls, Risk Management, Cryptography, etc.)\n"
                "- SP 1800 series (Practice Guides)\n"
                "- NIST Cybersecurity Framework (CSF) 2.0\n"
                "- CSWP (Cybersecurity White Papers on Zero Trust, PQC, IoT, etc.)\n"
                "- IR (Interagency/Internal Reports)\n\n"
                "Prioritize tools: 1) nist_lookup 2) control_lookup 3) search_by_document 4) web search\n"
                "Always cite sources clearly with document names.\n"
                "Format control IDs in bold (e.g., **AC-1**).\n"
                "Be precise, thorough, and cite specific NIST guidance when available."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    def _get_chat_history(self, session_id: str) -> ChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self.session_histories:
            self.session_histories[session_id] = ChatMessageHistory()
        return self.session_histories[session_id]
    
    def query(
        self,
        question: str,
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Query the NIST RAG agent.
        
        Args:
            question: User's question about NIST standards
            session_id: Session identifier for chat history
            
        Returns:
            Dict with 'answer', 'session_id', and metadata
        """
        chat_history = self._get_chat_history(session_id)
        
        result = self.agent_executor.invoke({
            "input": question,
            "chat_history": chat_history.messages
        })
        
        # Update chat history
        chat_history.add_user_message(question)
        chat_history.add_ai_message(result["output"])
        
        return {
            "answer": result["output"],
            "session_id": session_id,
            "intermediate_steps": result.get("intermediate_steps", [])
        }
    
    def clear_history(self, session_id: str = "default"):
        """Clear chat history for a session."""
        if session_id in self.session_histories:
            self.session_histories[session_id].clear()
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        return {
            "total_examples": len(self.dataset),
            "split": self.dataset_split,
            "cache_dir": str(self.cache_dir),
            "num_sessions": len(self.session_histories)
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("NIST RAG Agent V2 - Enhanced with HuggingFace Dataset")
    print("530K+ examples from 596 NIST publications")
    print("="*70)
    
    print("\nInitializing agent (first run will download dataset)...")
    agent = NistRagAgentV2(
        top_k=3,
        dataset_split="train"
    )
    
    print("\nDataset Statistics:")
    stats = agent.get_dataset_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Demo Queries")
    print("="*70)
    
    # Example 1: CSF 2.0 question
    print("\n[Q1] What's new in NIST Cybersecurity Framework 2.0?")
    response = agent.query(
        "What's new in NIST Cybersecurity Framework 2.0?",
        session_id="demo"
    )
    print(f"\n[A1] {response['answer']}\n")
    
    # Example 2: Zero Trust
    print("\n[Q2] What does NIST say about Zero Trust Architecture?")
    response = agent.query(
        "What does NIST say about Zero Trust Architecture?",
        session_id="demo"
    )
    print(f"\n[A2] {response['answer']}\n")
    
    # Example 3: Specific control
    print("\n[Q3] Explain control AC-2 in detail")
    response = agent.query(
        "Explain control AC-2 in detail",
        session_id="demo"
    )
    print(f"\n[A3] {response['answer']}\n")
    
    # Example 4: Follow-up using context
    print("\n[Q4] What are the implementation requirements?")
    response = agent.query(
        "What are the implementation requirements?",
        session_id="demo"
    )
    print(f"\n[A4] {response['answer']}\n")
