"""
API client example using requests
"""

import requests
import json


API_URL = "http://localhost:8000"


def main():
    print("="*60)
    print("NIST RAG Agent - API Client Example")
    print("="*60)
    
    # Check health
    print("\n1. Health Check")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Query the agent
    print("\n2. Query: What is AC-1?")
    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": "What is AC-1?",
            "session_id": "api_demo"
        }
    )
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"\nAnswer:\n{data['answer']}")
    
    # Follow-up question
    print("\n3. Follow-up: What are its requirements?")
    response = requests.post(
        f"{API_URL}/query",
        json={
            "question": "What are its requirements?",
            "session_id": "api_demo"
        }
    )
    data = response.json()
    print(f"\nAnswer:\n{data['answer']}")
    
    # List sessions
    print("\n4. List Active Sessions")
    response = requests.get(f"{API_URL}/sessions")
    print(f"Sessions: {json.dumps(response.json(), indent=2)}")
    
    # Clear history
    print("\n5. Clear Session History")
    response = requests.delete(f"{API_URL}/history/api_demo")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to API.")
        print("Make sure the API is running:")
        print("  python api_service.py")
