"""
NIST RAG Agent V2 - API Client Example
Python client for interacting with the V2 API service
"""

import requests
from typing import Optional, Dict, Any, List
import json


class NistRagClientV2:
    """Client for NIST RAG Agent V2 API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API service
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        
    def query(
        self,
        question: str,
        session_id: str = "default",
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Send a query to the agent.
        
        Args:
            question: Question about NIST standards
            session_id: Session ID for conversation context
            top_k: Override number of documents to retrieve
            
        Returns:
            Dict with answer and metadata
        """
        payload = {
            "question": question,
            "session_id": session_id
        }
        
        if top_k is not None:
            payload["top_k"] = top_k
        
        response = self.session.post(
            f"{self.base_url}/query",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def clear_history(self, session_id: str) -> Dict[str, str]:
        """
        Clear conversation history for a session.
        
        Args:
            session_id: Session ID to clear
            
        Returns:
            Status message
        """
        response = self.session.post(
            f"{self.base_url}/clear-history",
            json={"session_id": session_id}
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        response = self.session.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions."""
        response = self.session.get(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get detailed dataset information."""
        response = self.session.get(f"{self.base_url}/dataset-info")
        response.raise_for_status()
        return response.json()


def demo_basic_usage():
    """Demo basic client usage."""
    print("="*80)
    print("NIST RAG Agent V2 - API Client Demo")
    print("="*80 + "\n")
    
    # Initialize client
    client = NistRagClientV2("http://localhost:8000")
    
    # Health check
    print("1. Health Check")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Agent Initialized: {health['agent_initialized']}\n")
    
    # Get stats
    print("2. Dataset Statistics")
    stats = client.get_stats()
    print(f"   Total Examples: {stats['total_examples']:,}")
    print(f"   Dataset Split: {stats['dataset_split']}")
    print(f"   Model: {stats['model']}")
    print(f"   Active Sessions: {stats['active_sessions']}\n")
    
    # Query examples
    print("3. Example Queries")
    print("-" * 80)
    
    queries = [
        "What's new in NIST Cybersecurity Framework 2.0?",
        "What does NIST say about Zero Trust Architecture?",
        "Explain control AC-2 in detail"
    ]
    
    session_id = "demo"
    
    for i, question in enumerate(queries, 1):
        print(f"\n[Q{i}] {question}")
        
        try:
            response = client.query(question, session_id=session_id)
            print(f"\n[A{i}] {response['answer']}")
            print(f"\n     Timestamp: {response['timestamp']}")
            print(f"     Sources Used: {response['sources_used']}")
            print(f"     Session: {response['session_id']}")
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        print("\n" + "-" * 80)
    
    # List sessions
    print("\n4. Active Sessions")
    sessions = client.list_sessions()
    print(f"   Total Sessions: {sessions['total_sessions']}")
    for session in sessions['sessions']:
        print(f"   - {session['session_id']}: {session['message_count']} messages")
    
    # Clear history
    print(f"\n5. Clear History for session '{session_id}'")
    result = client.clear_history(session_id)
    print(f"   {result['message']}")
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)


def demo_conversational():
    """Demo conversational interaction."""
    print("\n" + "="*80)
    print("Conversational Demo")
    print("="*80 + "\n")
    
    client = NistRagClientV2("http://localhost:8000")
    session_id = "conversation_demo"
    
    conversation = [
        "What is the Risk Management Framework?",
        "What are the main steps in this framework?",
        "Can you explain step 3 in more detail?",
        "What documentation is required for that step?"
    ]
    
    for i, question in enumerate(conversation, 1):
        print(f"[Turn {i}] User: {question}")
        
        response = client.query(question, session_id=session_id)
        print(f"[Turn {i}] Agent: {response['answer'][:200]}...")
        print()
    
    # Clear when done
    client.clear_history(session_id)


def demo_document_search():
    """Demo document-specific searches."""
    print("\n" + "="*80)
    print("Document-Specific Search Demo")
    print("="*80 + "\n")
    
    client = NistRagClientV2("http://localhost:8000")
    
    searches = [
        ("SP 800-53", "What are the access control requirements?"),
        ("SP 800-171", "What does this say about protecting CUI?"),
        ("NIST CSF", "What are the core functions?"),
        ("SP 800-207", "What are the principles of Zero Trust?")
    ]
    
    for doc, question in searches:
        full_question = f"According to {doc}, {question}"
        print(f"📄 {doc}: {question}")
        
        response = client.query(full_question)
        print(f"   {response['answer'][:150]}...\n")


def demo_comparison():
    """Demo comparison queries."""
    print("\n" + "="*80)
    print("Comparison Queries Demo")
    print("="*80 + "\n")
    
    client = NistRagClientV2("http://localhost:8000")
    
    comparisons = [
        "What's the difference between SP 800-53 and SP 800-171?",
        "Compare the NIST RMF and CSF frameworks",
        "How does Zero Trust differ from traditional perimeter security?"
    ]
    
    for question in comparisons:
        print(f"🔄 {question}")
        response = client.query(question, top_k=5)  # Get more sources
        print(f"   {response['answer'][:200]}...\n")


def demo_csf_2_queries():
    """Demo CSF 2.0 specific queries."""
    print("\n" + "="*80)
    print("NIST CSF 2.0 Queries Demo (New in V2!)")
    print("="*80 + "\n")
    
    client = NistRagClientV2("http://localhost:8000")
    
    csf_queries = [
        "What's new in NIST Cybersecurity Framework 2.0?",
        "What is the Govern function in CSF 2.0?",
        "How does CSF 2.0 address supply chain risk management?",
        "What are the differences between CSF 1.1 and 2.0?"
    ]
    
    for question in csf_queries:
        print(f"🆕 {question}")
        response = client.query(question)
        print(f"   {response['answer'][:200]}...\n")


def demo_error_handling():
    """Demo error handling."""
    print("\n" + "="*80)
    print("Error Handling Demo")
    print("="*80 + "\n")
    
    client = NistRagClientV2("http://localhost:8000")
    
    # Test invalid endpoint
    try:
        response = requests.get(f"{client.base_url}/invalid")
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught expected error for invalid endpoint: {e.response.status_code}")
    
    # Test with empty question
    try:
        response = client.query("")
    except requests.exceptions.HTTPError as e:
        print(f"✓ Caught expected error for empty question: {e.response.status_code}")
    
    print("\n✓ Error handling working correctly")


if __name__ == "__main__":
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                   NIST RAG Agent V2 - API Client Examples                    ║
║                                                                              ║
║  Make sure the API service is running:                                      ║
║  $ python api_service.py                                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # Check if API is available
        client = NistRagClientV2()
        health = client.health_check()
        
        if health['status'] != 'healthy':
            print("⚠️  API is starting up. Please wait...")
            sys.exit(1)
        
        # Run demos
        demo_basic_usage()
        demo_conversational()
        demo_document_search()
        demo_comparison()
        demo_csf_2_queries()
        demo_error_handling()
        
        print("\n✅ All demos completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API service")
        print("   Make sure the API is running: python api_service.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
