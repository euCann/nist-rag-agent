"""
Session-based chat demonstration
Shows how the agent maintains context across multiple queries
"""

from agent import NistRagAgent


def main():
    print("="*60)
    print("NIST RAG Agent - Session Demo")
    print("="*60)
    
    agent = NistRagAgent()
    
    # Simulate two users having separate conversations
    print("\n" + "="*60)
    print("USER ALICE - Session")
    print("="*60)
    
    # Alice asks about access control
    print("\nAlice: What is AC-1?")
    response = agent.query("What is AC-1?", session_id="alice")
    print(f"Agent: {response['answer']}\n")
    
    # Alice follows up (agent remembers context)
    print("\nAlice: What are its main requirements?")
    response = agent.query("What are its main requirements?", session_id="alice")
    print(f"Agent: {response['answer']}\n")
    
    # Alice asks another follow-up
    print("\nAlice: How often should it be reviewed?")
    response = agent.query("How often should it be reviewed?", session_id="alice")
    print(f"Agent: {response['answer']}\n")
    
    # Now Bob starts a separate conversation
    print("\n" + "="*60)
    print("USER BOB - Separate Session")
    print("="*60)
    
    print("\nBob: What is IR-4?")
    response = agent.query("What is IR-4?", session_id="bob")
    print(f"Agent: {response['answer']}\n")
    
    print("\nBob: What steps does it require?")
    response = agent.query("What steps does it require?", session_id="bob")
    print(f"Agent: {response['answer']}\n")
    
    # Back to Alice (her context is preserved)
    print("\n" + "="*60)
    print("ALICE - Continuing Her Session")
    print("="*60)
    
    print("\nAlice: Who is responsible for that?")
    response = agent.query("Who is responsible for that?", session_id="alice")
    print(f"Agent: {response['answer']}\n")
    print("(Notice: Agent still remembers Alice was asking about AC-1)")
    
    # Clear Alice's history
    print("\n" + "="*60)
    print("Clearing Alice's history...")
    agent.clear_history("alice")
    print("History cleared!")
    print("="*60)


if __name__ == "__main__":
    main()
