import requests
import json

# Test configuration
API_URL = "http://127.0.0.1:8000/api/query"
TEST_QUERY = "What is a list in Python?"

def test_query():
    """Test the RAG query endpoint"""
    
    print("🔍 Testing Django RAG API...")
    print(f"Query: {TEST_QUERY}\n")
    
    try:
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"query": TEST_QUERY},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ SUCCESS!\n")
            print("=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(data["answer"])
            print("\n" + "=" * 80)
            print("SOURCES:")
            print("=" * 80)
            for i, source in enumerate(data.get("sources", []), 1):
                print(f"{i}. {source['source']} ({source['unit']})")
            
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed!")
        print("Make sure Django server is running: python manage.py runserver")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://127.0.0.1:8000/api/health")
        if response.status_code == 200:
            print("✅ Health check passed:", response.json())
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DJANGO RAG API TEST SUITE")
    print("=" * 80 + "\n")
    
    # First check if server is up
    print("1️⃣ Testing health endpoint...")
    test_health()
    
    print("\n2️⃣ Testing query endpoint...")
    test_query()
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)