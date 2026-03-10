import pytest
from source_code.rag.router import detect_subject

def test_router_basic():
    """Test that the router returns a valid subject."""
    query = "explain flip flop digital electronics"
    # detect_subject returns (subject, best_unit) or (subject, best_unit, used_llm)
    res = detect_subject(query)
    
    assert res is not None
    subject = res[0]
    assert subject is not None
    assert subject == "DIGITAL_ELECTRONICS"
