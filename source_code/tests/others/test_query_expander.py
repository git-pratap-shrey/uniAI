import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from rag.query_expander import expand_query

def test_query_expander():
    cases = [
        ("explain tabular method in de", ["digital electronics", "quine mccluskey", "boolean minimization"]),
        ("what is phishing", ["social engineering attack", "phishing"]),
        ("what are CS security policies", ["cyber security"]),
        ("underneath the desk", []) # "de" shouldn't match "underneath"
    ]
    
    passed = 0
    for query, expected_in in cases:
        expanded = expand_query(query)
        print(f"Original: '{query}'")
        print(f"Expanded: '{expanded}'")
        
        all_found = True
        for expected in expected_in:
            if expected not in expanded:
                print(f"  ❌ Missing expected term: '{expected}'")
                all_found = False
                
        if query == "underneath the desk" and "digital electronics" in expanded:
            print("  ❌ Incorrectly expanded 'de' from 'underneath'")
            all_found = False
            
        if all_found:
            print("  ✅ Pass")
            passed += 1
            
        print("-" * 40)
        
    print(f"\nPassed {passed}/{len(cases)} cases.")
    if passed != len(cases):
        sys.exit(1)

if __name__ == "__main__":
    test_query_expander()
