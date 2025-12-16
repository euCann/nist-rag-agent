"""
NIST RAG Agent - Core Implementation
A conversational AI agent for NIST cybersecurity standards using RAG.
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


class NistRagAgent:
    """
    Conversational AI agent for NIST/OSCAL standards using RAG.
    
    Features:
    - Vector search over 10+ NIST publications
    - Multi-tool agent (RAG, control lookup, web search)
    - Session-based chat history
    - Intelligent citation with Control ID, Title, URL
    """
    
    def __init__(
        self,
        embeddings_dir: Optional[Path] = None,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-4o",
        top_k: int = 3
    ):
        """
        Initialize the NIST RAG Agent.
        
        Args:
            embeddings_dir: Directory containing NIST document embeddings
            openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI model to use (default: gpt-4o)
            top_k: Number of similar chunks to retrieve
        """
        # Load environment variables
        load_dotenv()
        
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY or pass openai_api_key")
        
        self.model = model
        self.top_k = top_k
        
        # Set up embeddings directory
        if embeddings_dir is None:
            embeddings_dir = Path(__file__).parent / "embeddings"
        self.embeddings_dir = Path(embeddings_dir)
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=self.model, temperature=0)
        self.session_histories: Dict[str, ChatMessageHistory] = {}
        
        # Load or create vector store
        self.vectorstore = self._load_vectorstore()
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.top_k})
        
        # Set up tools and agent
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()
        
    def _load_vectorstore(self) -> FAISS:
        """Load FAISS vector store from embeddings directory."""
        # Check if FAISS index exists
        faiss_index_path = self.embeddings_dir / "faiss_index"
        
        if faiss_index_path.exists():
            print(f"Loading existing FAISS index from {faiss_index_path}")
            return FAISS.load_local(
                str(faiss_index_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        
        # Otherwise, load from JSON chunks and create new index
        print("Creating new FAISS index from embeddings...")
        documents = self._load_nist_documents()
        
        if not documents:
            print("Warning: No NIST documents found. Using mock data for demo.")
            documents = self._create_mock_documents()
        
        vectorstore = FAISS.from_documents(documents, self.embeddings)
        
        # Save the index for future use
        faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(faiss_index_path))
        
        return vectorstore
    
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
            # Search for documents matching this control ID
            query = f"control_id:{control_id}"
            docs = self.retriever.get_relevant_documents(query)
            
            if not docs:
                return f"No details found for control {control_id}. Try using nist_lookup instead."
            
            # Return the most relevant document
            doc = docs[0]
            meta = doc.metadata
            
            return f"""
**Control ID**: {meta.get('control_id', control_id)}
**Title**: {meta.get('title', 'N/A')}
**Source**: {meta.get('source', 'N/A')}
**Section**: {meta.get('section', 'N/A')}

**Details**:
{doc.page_content}
"""
        
        # Web search fallback
        web_search = DuckDuckGoSearchRun()
        
        return [nist_lookup, control_detail, web_search]
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools."""
        
        prompt = ChatPromptTemplate.from_messages([
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
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
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
            question: User's question about NIST standards
            session_id: Session identifier for chat history
            
        Returns:
            Dict with 'answer' and metadata
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
