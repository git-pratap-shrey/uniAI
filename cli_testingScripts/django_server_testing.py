import requests
import json
import time

# Test configuration
BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/api/query"
HEALTH_URL = f"{BASE_URL}/api/health"

# A diverse set of test cases across different subjects
TEST_CASES = [
    {
        "name": "UHV - Conceptual",
        "query": "Explain the difference between sukh and suvidha.",
        "subject": "UHV"
    },
    {
        "name": "DE - Syllabus",
        "query": "What is in the digital electronics syllabus?",
        "subject": "DIGITAL_ELECTRONICS"
    },
    {
        "name": "Cyber Security - Specific",
        "query": "What are the key provisions of the DPDP act?",
        "subject": "CYBER_SECURITY"
    },
    {
        "name": "Generic Fallback",
        "query": "How do I make a chocolate cake?",
        "subject": None
    }
]

def print_separator(char="=", length=80):
    print(char * length)

def test_health():
    """Test the health endpoint"""
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(HEALTH_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed!")
            print(f"   Model: {data.get('model')}")
            print(f"   Status: {data.get('status')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_single_query(name, query, subject=None, history=None):
    """Test a single RAG query"""
    print(f"\n🔍 Testing: {name}")
    print(f"   Query: {query}")
    if subject: print(f"   Subject: {subject}")
    
    payload = {"query": query}
    if subject: payload["subject"] = subject
    if history: payload["history"] = history

    try:
        start_time = time.time()
        response = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCCESS ({duration:.2f}s) | MODE: {data.get('mode')} | ROUTED AS: {data.get('expanded_query')[:50]}...")
            
            print_separator("-")
            print("ANSWER:")
            print(data.get("answer", "No answer returned."))
            
            sources = data.get("sources", [])
            if sources:
                print("\nSOURCES:")
                for i, src in enumerate(sources, 1):
                    source_name = src.get('source', 'unknown').split('/')[-1]
                    print(f"  {i}. {source_name} [Unit {src.get('unit', '?')}, Page {src.get('page_start', '?')}]")
            else:
                print("\n(No sources found)")
            print_separator("-")
            return data
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed! Is the server running?")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_conversational_flow():
    """Test multi-turn interaction"""
    print("\n3️⃣ Testing Multi-Turn Conversational Flow...")
    
    # First turn
    q1 = "Who is the Prime Minister of India?" # Out of syllabus but test history
    res1 = test_single_query("Turn 1 (General)", q1)
    
    if res1:
        # Second turn
        history = [
            {"role": "user", "content": q1},
            {"role": "assistant", "content": res1["answer"]}
        ]
        q2 = "Can you summarize that in 2 points?"
        test_single_query("Turn 2 (Follow-up)", q2, history=history)

if __name__ == "__main__":
    print_separator()
    print("DJANGO RAG SERVER - LATEST INTEGRATION TEST")
    print_separator()
    
    if test_health():
        print("\n2️⃣ Testing Subject-Specific Queries...")
        for case in TEST_CASES:
            test_single_query(case["name"], case["query"], case["subject"])
        
        test_conversational_flow()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)