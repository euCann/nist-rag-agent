"""
NIST RAG Agent - Core Implementation
A conversational AI agent for NIST cybersecurity standards using RAG.
Now powered by the ethanolivertroy/nist-cybersecurity-training dataset with 530K+ examples.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langchain.memory import ChatMessageHistory
from langchain.agents import create_tool_calling_agent, AgentExecutor

try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("Warning: 'datasets' package not installed. Install with: pip install datasets")


class NistRagAgent:
    """
    Conversational AI agent for NIST/OSCAL standards using RAG.
    
    Features:
    - Vector search over 596 NIST publications (530K+ examples via HuggingFace dataset)
    - Multi-tool agent (RAG, control lookup, document search, web search)
    - Session-based chat history
    - Intelligent citation with Control ID, Title, URL
    - Comprehensive coverage: FIPS, SP 800/1800, IR, CSWP (including CSF 2.0)
    """
    
    def __init__(
        self,
        embeddings_dir: Optional[Path] = None,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        top_k: int = 5,
        use_huggingface: bool = True,
        dataset_split: str = "train",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the NIST RAG Agent.
        
        Args:
            embeddings_dir: Directory containing local NIST document embeddings (legacy)
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o)
            top_k: Number of similar chunks to retrieve
            use_huggingface: Use HuggingFace dataset (530K+ examples from 596 publications)
            dataset_split: Dataset split to use ('train' or 'valid')
            cache_dir: Cache directory for HuggingFace dataset
        """
        # Load environment variables
        load_dotenv()
        
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key")
        
        # Configure logging to prevent API key exposure
        import logging
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        
        self.model = model
        self.top_k = top_k
        self.use_huggingface = use_huggingface and HUGGINGFACE_AVAILABLE
        self.dataset_split = dataset_split
        
        # Set up cache directory for HuggingFace with security validation
        if cache_dir is None:
            cache_dir = Path(__file__).parent / ".cache" / "huggingface"
        else:
            cache_dir = Path(cache_dir).resolve()
            # Ensure within allowed paths (prevent directory traversal)
            allowed_base = Path(__file__).parent.resolve()
            try:
                cache_dir.relative_to(allowed_base)
            except ValueError:
                raise ValueError(f"Cache directory must be within {allowed_base}")
        
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, mode=0o750, exist_ok=True)  # Restrictive permissions
        
        # Set up embeddings directory (legacy support)
        if embeddings_dir is None:
            embeddings_dir = Path(__file__).parent / "embeddings"
        self.embeddings_dir = Path(embeddings_dir)
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=self.model, temperature=0)
        self.session_histories: Dict[str, ChatMessageHistory] = {}
        self.dataset = None
        
        # Load or create vector store
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        # Set up tools and agent
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
        
    def _load_vectorstore(self) -> FAISS:
        """Load FAISS vector store from HuggingFace dataset or local embeddings."""
        if self.use_huggingface:
            return self._load_vectorstore_from_huggingface()
        else:
            return self._load_vectorstore_from_local()
    
    def _load_vectorstore_from_huggingface(self) -> FAISS:
        """Load FAISS vector store from HuggingFace dataset."""
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
                print(f"Failed to load cached index: {e}. Creating new index...")
        
        # Load dataset from HuggingFace
        print(f"Loading NIST dataset from HuggingFace (split: {self.dataset_split})...")
        print("⏳ First run will download ~7GB. Subsequent runs use cache.")
        
        try:
            self.dataset = load_dataset(
                "ethanolivertroy/nist-cybersecurity-training",
                split=self.dataset_split,
                cache_dir=str(self.cache_dir)
            )
            print(f"✓ Loaded {len(self.dataset):,} examples from NIST dataset")
        except Exception as e:
            print(f"❌ Failed to load HuggingFace dataset: {e}")
            print("Falling back to local embeddings...")
            self.use_huggingface = False
            return self._load_vectorstore_from_local()
        
        # Create documents from dataset
        documents = self._create_documents_from_dataset()
        
        if not documents:
            print("No documents created. Falling back to local embeddings...")
            self.use_huggingface = False
            return self._load_vectorstore_from_local()
        
        # Create FAISS index with batching for large datasets
        print("Creating FAISS index (this may take 10-20 minutes on first run)...")
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
    
    def _load_vectorstore_from_local(self) -> FAISS:
        """Load FAISS vector store from local embeddings directory (legacy)."""
        faiss_index_path = self.embeddings_dir / "faiss_index"
        
        if faiss_index_path.exists():
            print(f"Loading existing FAISS index from {faiss_index_path}")
            return FAISS.load_local(
                str(faiss_index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        # Load from JSON chunks and create new index
        print("Creating new FAISS index from local embeddings...")
        documents = self._load_nist_documents()
        
        if not documents:
            print("Warning: No NIST documents found. Using mock data for demo.")
            documents = self._create_mock_documents()
        
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the index for future use
        faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(faiss_index_path))
        
        return vectorstore
    
    def _create_documen cybersecurity standards and guidance.
            Use for any NIST, OSCAL, or cybersecurity compliance question.
            Covers 596 NIST publications including:
            - FIPS (Federal Information Processing Standards)
            - SP 800 series (Security Controls, Risk Management, etc.)
            - SP 1800 series (Practice Guides)
            - IR (Interagency/Internal Reports)
            - CSWP (Cybersecurity White Papers, including CSF 2.0, Zero Trust, PQC)
            
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
                
                control_id = meta.get('control_id', '')
                title = meta.get('title', '')
                chunk_type = meta.get('type', 'section')
                section = meta.get('section', '')
                
                result = f"""
**Source**: {source}
**Type**: {chunk_type}
"""
                if control_id:
                    result += f"**Control ID**: {control_id}\n"
                if title:
                    result += f"**Title**: {title}\n"
                if section:
                    result += f"**Section**: {section}\n"
                
                result += f"**Content**:\n{doc.page_content[:800]}...\n\n---\n"
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
        system_message = "You are a highly knowledgeable NIST cybersecurity expert assistant. "
        
        if self.use_huggingface:
            system_message += (
                "You have access to 596 NIST publications with 530,000+ training examples including:\n"
                "- FIPS (Federal Information Processing Standards)\n"
                "- SP 800 series (Security Controls, Risk Management, Cryptography, etc.)\n"
                "- SP 1800 series (Practice Guides)\n"
                "- NIST Cybersecurity Framework (CSF) 2.0\n"
                "- CSWP (Cybersecurity White Papers on Zero Trust, PQC, IoT, etc.)\n"
                "- IR (Interagency/Internal Reports)\n\n"
            )
        else:
            system_message += (
                "You have access to key NIST publications including SP 800-53, 800-37, 800-171, and more.\n\n"
            )
        
        system_message += (
            "Prioritize tools: 1) nist_lookup 2) control_lookup 3) search_by_document 4) web search\n"
            "Always cite sources clearly with document names.\n"
            "Format control IDs in bold (e.g., **AC-1**).\n"
            "Be precise, thorough, and cite specific NIST guidance when available."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Disable verbose in production to prevent log exposure
        is_production = os.getenv("ENVIRONMENT", "development") == "production"
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=not is_production,  # Only verbose in development
            handle_parsing_errors=True,
            max_iterations=5
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
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the agent and dataset."""
        stats = {
            "model": self.model,
            "top_k": self.top_k,
            "active_sessions": len(self.session_histories),
            "using_huggingface": self.use_huggingface
        }
        
        if self.use_huggingface and self.dataset:
            stats.update({
                "total_examples": len(self.dataset),
                "dataset_split": self.dataset_split,
                "cache_dir": str(self.cache_dir)
            })
        else:
            stats.update({
                "embeddings_dir": str(self.embeddings_dir)
            })
        
        return stats
            
            return "\n".join(results) if results else f"No content found for {document_name}"
        
        # Web search fallback
        web_search = DuckDuckGoSearchRun()
        
        return [nist_lookup, control_lookup, search_by_document, "section"),
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
            
            # Look for source in assistant message
            for msg in messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if "According to" in content:
                        import re
                        match = re.search(r'According to ([^,]+)', content)
                        if match and "source" not in metadata:
                            metadata["source"] = match.group(1).strip()
        
        return metadata
    
    def _load_nist_documents(self) -> List[Document]:
        """Load NIST documents from JSON chunk files."""
        import json
        
        documents = []
        
        if not self.embeddings_dir.exists():
            return documents
        
        # Look for all .chunks.json files
        for json_file in self.embeddings_dir.glob("*.chunks.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                doc_name = json_file.stem.replace('.chunks', '')
                
                for chunk in chunks:
                    doc = Document(
                        page_content=chunk.get('text', ''),
                        metadata={
                            'source': doc_name,
                            'chunk_id': chunk.get('chunk_id', ''),
                            'section': chunk.get('section', ''),
                            'control_id': chunk.get('control_id', ''),
                            'title': chunk.get('title', '')
                        }
                    )
                    documents.append(doc)
                
                print(f"Loaded {len(chunks)} chunks from {doc_name}")
            
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return documents
    
    def _create_mock_documents(self) -> List[Document]:
        """Create mock NIST documents for demo purposes."""
        return [
            Document(
                page_content="Access Control: Limit system access to authorized users. Organizations must develop, document, and disseminate access control policies.",
                metadata={"control_id": "AC-1", "title": "Access Control Policy and Procedures", "source": "NIST.SP.800-53r5", "section": "3.1"}
            ),
            Document(
                page_content="Audit and Accountability: Organizations must determine which events require auditing and maintain detailed audit logs for security monitoring.",
                metadata={"control_id": "AU-2", "title": "Audit Events", "source": "NIST.SP.800-53r5", "section": "3.2"}
            ),
            Document(
                page_content="Identification and Authentication: Each user and device must be uniquely identified before being granted system access.",
                metadata={"control_id": "IA-2", "title": "Identification and Authentication", "source": "NIST.SP.800-53r5", "section": "3.3"}
            ),
            Document(
                page_content="System and Communications Protection: Monitor and control communications at system boundaries to prevent unauthorized information transfers.",
                metadata={"control_id": "SC-7", "title": "Boundary Protection", "source": "NIST.SP.800-53r5", "section": "3.4"}
            ),
            Document(
                page_content="Incident Response: Organizations must establish an incident response capability including preparation, detection, analysis, containment, eradication, and recovery.",
                metadata={"control_id": "IR-4", "title": "Incident Handling", "source": "NIST.SP.800-53r5", "section": "3.5"}
            ),
        ]
    
    def _create_tools(self) -> List:
        """Create LangChain tools for the agent."""
        
        @tool("nist_lookup")
        def nist_lookup(query: str) -> str:
            """
            Search NIST/OSCAL controls and guidance.
            Use for any NIST, OSCAL, or cybersecurity compliance question.
            Returns relevant control text with metadata (Control ID, Title, Source, Section).
            """
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return "No relevant NIST/OSCAL controls found for this query."
            
            results = []
            for doc in docs[:self.top_k]:
                meta = doc.metadata
                control_id = meta.get('control_id', 'N/A')
                title = meta.get('title', 'N/A')
                source = meta.get('source', 'N/A')
                section = meta.get('section', 'N/A')
                
                result = f"""
**Control ID**: {control_id}
**Title**: {title}
**Source**: {source}
**Section**: {section}
**Content**: {doc.page_content}
---
"""
                results.append(result)
            
            return "\n".join(results)
        
        @tool("control_detail")
        def control_detail(control_id: str) -> str:
            """
            Fetch detailed information for a specific NIST control by Control ID.
            Use when the user asks about a specific control (e.g., AC-1, AU-2, IR-4).
            """
           ="*70)
    print("NIST RAG Agent - Enhanced with HuggingFace Dataset")
    print("="*70)
    
    print("\nInitializing agent...")
    if HUGGINGFACE_AVAILABLE:
        print("✓ HuggingFace dataset support available")
        print("  First run will download ~7GB dataset (one-time)")
    else:
        print("⚠️  Using local embeddings (install 'datasets' for full coverage)")
    
    agent = NistRagAgent()
    
    # Show stats
    stats = agent.get_stats()
    print("\n📊 Agent Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("Demo Queries")
    print("="*70)
    
    # Example 1: General question
    print("\n[Q1] What does NIST say about access control?")
    response = agent.query(
        "What does NIST say about access control?",
        session_id="demo"
    )
    print(f"\n[A1] {response['answer']}\n")
    
    # Example 2: Specific control
    print("\n[Q2] Tell me about AC-2")
    response = agent.query(
        "Tell me about AC-2",
        session_id="demo"
    )
    print(f"\n[A2] {response['answer']}\n")
    
    # Example 3: CSF 2.0 (if using HuggingFace)
    if agent.use_huggingface:
        print("\n[Q3] What's new in NIST Cybersecurity Framework 2.0?")
        response = agent.query(
            "What's new in NIST Cybersecurity Framework 2.0?",
            session_id="demo"
        )
        print(f"\n[A3] {response['answer']}\n")
    
    # Example 4: Follow-up question (uses chat history)
    print("\n[Q4] What are the implementation requirements?")
    response = agent.query(
        "What are the implementation requirements?",
        session_id="demo"
    )
    print(f"\n[A4] = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a knowledgeable NIST/OSCAL assistant helping with cybersecurity compliance. "
                "Prioritize tools in this order: 1) nist_lookup, 2) control_detail, 3) duckduckgo_search. "
                "Always cite sources (Control ID, Title, Source) when using nist_lookup. "
                "Make Control IDs easy to spot (use bold **AC-1** formatting). "
                "Be concise and accurate. If you don't know something, say so clearly."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_tool_calling_agent(self.llm, self.tools, prompt)
        
        # Disable verbose in production to prevent log exposure
        is_production = os.getenv("ENVIRONMENT", "development") == "production"
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=not is_production,  # Only verbose in development
            handle_parsing_errors=True
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
            question: User's question about NIST standards (already sanitized by API layer)
            session_id: Session identifier for chat history (already sanitized by API layer)
            
        Returns:
            Dict with 'answer' and metadata
        """
        # Input should already be sanitized by API layer, but validate basics
        if not question or len(question) > 2000:
            raise ValueError("Invalid question length")
        
        if not session_id or len(session_id) > 64:
            raise ValueError("Invalid session ID")
        
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
        # Validate session_id (defense in depth)
        if not session_id or len(session_id) > 64:
            raise ValueError("Invalid session ID")
        
        if session_id in self.session_histories:
            self.session_histories[session_id].clear()


# Example usage
if __name__ == "__main__":
    print("Initializing NIST RAG Agent...")
    agent = NistRagAgent()
    
    print("\n" + "="*60)
    print("NIST RAG Agent - Demo")
    print("="*60)
    
    # Example 1: General question
    print("\nQ: What does NIST say about access control?")
    response = agent.query(
        "What does NIST say about access control?",
        session_id="demo"
    )
    print(f"A: {response['answer']}\n")
    
    # Example 2: Specific control
    print("\nQ: Tell me about AC-1")
    response = agent.query(
        "Tell me about AC-1",
        session_id="demo"
    )
    print(f"A: {response['answer']}\n")
    
    # Example 3: Follow-up question (uses chat history)
    print("\nQ: What are its requirements?")
    response = agent.query(
        "What are its requirements?",
        session_id="demo"
    )
    print(f"A: {response['answer']}\n")
