"""
Basic usage example for NIST RAG Agent
"""

from agent import NistRagAgent


def main():
    print("="*60)
    print("NIST RAG Agent - Basic Query Example")
    print("="*60)
    
    # Initialize the agent
    print("\nInitializing agent...")
    agent = NistRagAgent()
    
    # Example queries
    questions = [
        "What does NIST say about access control?",
        "Explain AC-1",
        "What are the requirements for incident response?",
        "Tell me about boundary protection",
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"{'='*60}")
        
        response = agent.query(question, session_id="example")
        print(f"\nA: {response['answer']}\n")


if __name__ == "__main__":
    main()
