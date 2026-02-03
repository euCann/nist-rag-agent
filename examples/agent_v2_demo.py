"""
NIST RAG Agent V2 - Demo Script
Demonstrates the enhanced capabilities using the HuggingFace dataset
"""

from agent import NistRagAgentV2
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def demo_basic_queries(agent: NistRagAgentV2):
    """Demo basic RAG queries."""
    print_section("Demo 1: Basic NIST Queries")
    
    queries = [
        "What is the NIST Cybersecurity Framework?",
        "Explain the concept of Zero Trust Architecture",
        "What are the main categories in SP 800-53?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"[Q{i}] {query}")
        response = agent.query(query, session_id="demo1")
        print(f"\n[A{i}] {response['answer']}\n")
        print("-" * 80)
        time.sleep(1)


def demo_control_lookup(agent: NistRagAgentV2):
    """Demo specific control lookups."""
    print_section("Demo 2: Security Control Lookups")
    
    controls = [
        "AC-2 Account Management",
        "AU-6 Audit Review, Analysis, and Reporting",
        "SC-7 Boundary Protection",
    ]
    
    for i, control in enumerate(controls, 1):
        print(f"[Q{i}] Tell me about {control}")
        response = agent.query(f"Tell me about {control}", session_id="demo2")
        print(f"\n[A{i}] {response['answer']}\n")
        print("-" * 80)
        time.sleep(1)


def demo_document_search(agent: NistRagAgentV2):
    """Demo searching within specific documents."""
    print_section("Demo 3: Document-Specific Searches")
    
    queries = [
        "What does SP 800-171 say about protecting CUI?",
        "Find information about post-quantum cryptography in NIST documents",
        "What guidance does NIST provide for IoT device security?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"[Q{i}] {query}")
        response = agent.query(query, session_id="demo3")
        print(f"\n[A{i}] {response['answer']}\n")
        print("-" * 80)
        time.sleep(1)


def demo_conversational(agent: NistRagAgentV2):
    """Demo conversational follow-up questions."""
    print_section("Demo 4: Conversational Context")
    
    conversation = [
        "What is the Risk Management Framework?",
        "What are the main steps in this framework?",
        "Can you explain step 3 in more detail?",
        "What documentation is required for that step?",
    ]
    
    for i, query in enumerate(conversation, 1):
        print(f"[Q{i}] {query}")
        response = agent.query(query, session_id="demo4")
        print(f"\n[A{i}] {response['answer']}\n")
        print("-" * 80)
        time.sleep(1)


def demo_csf_2_queries(agent: NistRagAgentV2):
    """Demo CSF 2.0 specific queries."""
    print_section("Demo 5: NIST CSF 2.0 (New in Dataset V2)")
    
    queries = [
        "What's new in NIST Cybersecurity Framework 2.0?",
        "What is the Govern function in CSF 2.0?",
        "How does CSF 2.0 address supply chain risk management?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"[Q{i}] {query}")
        response = agent.query(query, session_id="demo5")
        print(f"\n[A{i}] {response['answer']}\n")
        print("-" * 80)
        time.sleep(1)


def demo_comparison_queries(agent: NistRagAgentV2):
    """Demo comparison and analysis queries."""
    print_section("Demo 6: Comparative Analysis")
    
    queries = [
        "What's the difference between SP 800-53 and SP 800-171?",
        "Compare NIST CSF and ISO 27001 frameworks",
        "How does NIST address cloud security across different publications?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"[Q{i}] {query}")
        response = agent.query(query, session_id="demo6")
        print(f"\n[A{i}] {response['answer']}\n")
        print("-" * 80)
        time.sleep(1)


def show_statistics(agent: NistRagAgentV2):
    """Show agent statistics."""
    print_section("Agent Statistics")
    
    stats = agent.get_dataset_stats()
    print(f"Total Training Examples: {stats['total_examples']:,}")
    print(f"Dataset Split: {stats['split']}")
    print(f"Cache Directory: {stats['cache_dir']}")
    print(f"Active Sessions: {stats['num_sessions']}")
    print(f"Top-K Retrieval: {agent.top_k}")
    print(f"LLM Model: {agent.model}")


def main():
    """Run all demos."""
    print("="*80)
    print("NIST RAG Agent V2 - Comprehensive Demo")
    print("Using ethanolivertroy/nist-cybersecurity-training dataset")
    print("530K+ examples from 596 NIST publications")
    print("="*80)
    
    print("\n⏳ Initializing agent (first run will download ~7GB dataset)...")
    print("   Subsequent runs will use cached data.\n")
    
    agent = NistRagAgentV2(
        top_k=3,
        dataset_split="train"
    )
    
    print("✓ Agent initialized successfully!\n")
    
    # Show statistics
    show_statistics(agent)
    
    # Run demos
    try:
        demo_basic_queries(agent)
        demo_control_lookup(agent)
        demo_document_search(agent)
        demo_conversational(agent)
        demo_csf_2_queries(agent)
        demo_comparison_queries(agent)
        
        print_section("Demo Complete!")
        print("All demos finished successfully.")
        print(f"Total sessions created: {len(agent.session_histories)}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
