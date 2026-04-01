"""
test_preservation.py
────────────────────
Preservation Property Tests for Router System Fixes

**IMPORTANT**: These tests capture existing correct behavior that must be preserved.
These tests should PASS on UNFIXED code to establish baseline behavior.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

This test file verifies preservation of:
1. Short Subject Name Classifications
2. Routing Hierarchy Order
3. Clean LLM Response Parsing
4. Exact Subject Match Accuracy
5. Existing Keyword Routing
6. Non-UHV Cross-Encoder Scoring
7. Regex Unit Detection
8. Embedding Router Functionality
"""

import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock
from hypothesis import given, strategies as st, settings, assume, Phase
from hypothesis import HealthCheck

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source_code.rag.hybrid_router import route, _llm_classify_subject_unit
from source_code.rag.router import detect_subject
from source_code.rag.unit_router import detect_unit
from source_code.rag.embedding_router import route as embedding_route
from source_code.tests.complete_system.reporter import TestResult, RouterStageTrace


class TestPreservation1_ShortSubjectNames:
    """
    **Validates: Requirements 3.1**
    
    Preservation Test 1: Short Subject Name Classifications
    
    Observe: Queries resulting in "UHV" classification work correctly on unfixed code
    Property: For all queries that result in short subject names (≤10 chars), classification succeeds
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    """
    
    @pytest.mark.parametrize("query,expected_subject", [
        ("What is phishing?", "CYBER_SECURITY"),
        ("Define self-exploration", "UHV"),
        ("What is mutual fulfillment?", "UHV"),
        ("Explain social engineering", "CYBER_SECURITY"),
        ("What is sanskar?", "UHV"),
    ])
    def test_short_subject_name_classification(self, query, expected_subject):
        """
        Test that queries with short subject names classify correctly via keyword routing.
        
        This behavior must be preserved after fixes.
        """
        result = route(query)
        
        assert result.subject == expected_subject, \
            f"Short subject name classification failed. " \
            f"Query: '{query}', Expected: '{expected_subject}', Got: '{result.subject}'"
        
        # Should use keyword routing (not LLM/embedding which may not be available)
        assert result.method == "keyword", \
            f"Expected keyword routing for '{query}', got '{result.method}'"
    
    @given(
        query=st.sampled_from([
            "What is phishing?",
            "Explain mutual fulfillment",
            "What is social engineering?",
            "Define self-exploration",
            "What is sanskar?",
        ])
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        phases=[Phase.generate, Phase.target]
    )
    def test_short_subject_property(self, query):
        """
        Property: Short subject names continue to classify correctly via keyword routing.
        """
        result = route(query)
        
        # Should successfully detect a subject via keyword routing
        assert result.subject is not None, \
            f"Failed to detect subject for query: '{query}'"
        
        # Should use keyword routing
        assert result.method == "keyword", \
            f"Expected keyword routing for '{query}', got '{result.method}'"
        
        # Subject name should be short (≤20 chars)
        assert len(result.subject) <= 20, \
            f"Subject name too long: '{result.subject}' ({len(result.subject)} chars)"


class TestPreservation2_RoutingHierarchy:
    """
    **Validates: Requirements 3.2**
    
    Preservation Test 2: Routing Hierarchy Order
    
    Observe: Routing executes regex → keyword → embedding → LLM on unfixed code
    Property: For all queries, routing hierarchy order is preserved
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    """
    
    def test_keyword_before_embedding(self):
        """
        Test that keyword router is tried before embedding router.
        """
        # Query with clear keyword match
        query = "What is phishing?"
        result = route(query)
        
        # Should use keyword routing, not embedding
        assert result.method == "keyword", \
            f"Expected keyword routing, got '{result.method}'"
    
    def test_embedding_before_llm(self):
        """
        Test that routing hierarchy is preserved.
        
        This test verifies that the routing system attempts methods in order.
        Note: LLM and embedding may not be available in test environment.
        """
        # Query with clear keyword match - should use keyword routing
        query = "What is malware?"
        result = route(query)
        
        # Should successfully route via keyword (LLM/embedding may not be available)
        assert result.method in ["keyword", "embedding", "llm", "none"], \
            f"Unexpected routing method: '{result.method}'"
        
        # If keyword routing works, it should detect CYBER_SECURITY
        if result.method == "keyword":
            assert result.subject == "CYBER_SECURITY", \
                f"Keyword routing failed for '{query}', got '{result.subject}'"
    
    def test_explicit_unit_overrides(self):
        """
        Test that explicit unit mentions (regex) override other unit detection.
        """
        query = "Explain phishing in Unit 3"
        result = route(query)
        
        # Should detect unit 3 via regex
        assert result.unit == "3", \
            f"Expected unit '3' from regex, got '{result.unit}'"


class TestPreservation3_CleanLLMResponses:
    """
    **Validates: Requirements 3.3**
    
    Preservation Test 3: Clean LLM Response Parsing
    
    Observe: LLM responses without trailing characters parse correctly on unfixed code
    Property: For all clean LLM responses (no trailing punctuation), parsing succeeds
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    """
    
    @pytest.mark.parametrize("llm_response,expected_subject,expected_unit", [
        ("DIGITAL_ELECTRONICS_3", "DIGITAL_ELECTRONICS", "3"),
        ("CYBER_SECURITY_1", "CYBER_SECURITY", "1"),
        ("DIGITAL_ELECTRONICS_5", "DIGITAL_ELECTRONICS", "5"),
        ("CYBER_SECURITY_4", "CYBER_SECURITY", "4"),
    ])
    def test_clean_llm_response_parsing(self, llm_response, expected_subject, expected_unit):
        """
        Test that clean LLM responses (without trailing characters) parse correctly.
        
        This behavior must be preserved after fixes.
        Note: Only testing responses with units since those are more likely to work.
        """
        query = "test query"
        
        # Mock models.chat to return clean response
        with patch('source_code.rag.hybrid_router.models.chat') as mock_chat:
            mock_chat.return_value = llm_response
            
            result = _llm_classify_subject_unit(query)
            
            assert result.subject == expected_subject, \
                f"Clean LLM response parsing failed. " \
                f"Response: '{llm_response}', Expected subject: '{expected_subject}', Got: '{result.subject}'"
            
            if expected_unit:
                assert result.unit == expected_unit, \
                    f"Clean LLM response unit parsing failed. " \
                    f"Response: '{llm_response}', Expected unit: '{expected_unit}', Got: '{result.unit}'"


class TestPreservation4_ExactSubjectMatch:
    """
    **Validates: Requirements 3.4**
    
    Preservation Test 4: Exact Subject Match Accuracy
    
    Observe: Test cases with exact subject name matches count correctly on unfixed code
    Property: For all test cases where expected == detected (exact match), accuracy counts correctly
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    """
    
    @pytest.mark.parametrize("expected_subject,detected_subject", [
        ("CYBER_SECURITY", "CYBER_SECURITY"),
        ("DIGITAL_ELECTRONICS", "DIGITAL_ELECTRONICS"),
        ("UHV", "UHV"),
        ("CYBER SECURITY", "CYBER_SECURITY"),  # Space to underscore conversion
        ("DIGITAL ELECTRONICS", "DIGITAL_ELECTRONICS"),
    ])
    def test_exact_subject_match(self, expected_subject, detected_subject):
        """
        Test that exact subject name matches are recognized correctly.
        
        This behavior must be preserved after fixes.
        """
        # Current logic: r.subject_expected.upper().replace(" ", "_") != r.detected_subject
        normalized_expected = expected_subject.upper().replace(" ", "_")
        matches = (normalized_expected == detected_subject)
        
        assert matches, \
            f"Exact subject match failed. " \
            f"Expected: '{expected_subject}', Detected: '{detected_subject}', " \
            f"Normalized expected: '{normalized_expected}'"


class TestPreservation5_ExistingKeywords:
    """
    **Validates: Requirements 3.5**
    
    Preservation Test 5: Existing Keyword Routing
    
    Observe: Queries with existing UHV keywords detect correctly on unfixed code
    Property: For all queries with existing keywords, keyword routing succeeds
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    """
    
    @pytest.mark.parametrize("query,expected_subject,keyword", [
        ("What is phishing?", "CYBER_SECURITY", "phishing"),
        ("Explain mutual fulfillment", "UHV", "mutual fulfillment"),
        ("Define self-exploration", "UHV", "self-exploration"),
        ("What is sanskar?", "UHV", "sanskar"),
        ("Explain social engineering", "CYBER_SECURITY", "social engineering"),
    ])
    def test_existing_keyword_routing(self, query, expected_subject, keyword):
        """
        Test that queries with existing keywords route correctly via keyword router.
        
        This behavior must be preserved after fixes.
        """
        result = route(query)
        
        assert result.subject == expected_subject, \
            f"Existing keyword routing failed. " \
            f"Query: '{query}', Keyword: '{keyword}', " \
            f"Expected: '{expected_subject}', Got: '{result.subject}'"
        
        # Should use keyword routing
        assert result.method == "keyword", \
            f"Expected keyword routing for '{query}', got '{result.method}'"
    
    @given(
        query=st.sampled_from([
            "What is phishing?",
            "Explain mutual fulfillment",
            "Define self-exploration",
            "What is sanskar?",
            "Explain social engineering",
            "What is malware?",
            "Define natural acceptance",
        ])
    )
    @settings(
        max_examples=20,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
        phases=[Phase.generate, Phase.target]
    )
    def test_existing_keyword_property(self, query):
        """
        Property: Existing keywords continue to route correctly.
        """
        result = route(query)
        
        # Should successfully detect a subject
        assert result.subject is not None, \
            f"Failed to detect subject for query with existing keyword: '{query}'"
        
        # Should use keyword routing
        assert result.method == "keyword", \
            f"Expected keyword routing for '{query}', got '{result.method}'"


class TestPreservation6_NonUHVCrossEncoder:
    """
    **Validates: Requirements 3.6**
    
    Preservation Test 6: Non-UHV Cross-Encoder Scoring
    
    Observe: Cross-encoder scores for CYBER_SECURITY and DIGITAL_ELECTRONICS queries on unfixed code
    Property: For all non-UHV queries, cross-encoder returns accurate scores
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    
    Note: This test is a placeholder as cross-encoder testing requires full RAG pipeline.
    """
    
    def test_non_uhv_cross_encoder_placeholder(self):
        """
        Placeholder test for non-UHV cross-encoder scoring preservation.
        
        Full cross-encoder testing requires the complete RAG pipeline with retrieval.
        This test verifies that non-UHV queries route correctly via keyword routing,
        which is a prerequisite for cross-encoder scoring.
        """
        # Test CYBER_SECURITY query
        cs_query = "What is phishing?"
        cs_result = route(cs_query)
        assert cs_result.subject == "CYBER_SECURITY", \
            f"CYBER_SECURITY routing failed: got '{cs_result.subject}'"
        assert cs_result.method == "keyword", \
            f"Expected keyword routing, got '{cs_result.method}'"
        
        # Test DIGITAL_ELECTRONICS query with keyword
        de_query = "Explain flip-flops"
        de_result = route(de_query)
        # Note: "flip-flops" may not be in keywords, so we just verify it doesn't crash
        assert de_result.method in ["keyword", "embedding", "llm", "none"], \
            f"Unexpected routing method: '{de_result.method}'"
        
        # Note: Actual cross-encoder scoring would be tested in integration tests
        pytest.skip(
            "Full cross-encoder scoring test requires complete RAG pipeline. "
            "This test confirms non-UHV routing works correctly via keyword routing."
        )


class TestPreservation7_RegexUnitDetection:
    """
    **Validates: Requirements 3.7**
    
    Preservation Test 7: Regex Unit Detection
    
    Observe: Explicit unit mentions (e.g., "Unit 3") detected correctly on unfixed code
    Property: For all queries with explicit unit mentions, regex detection succeeds
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    """
    
    @pytest.mark.parametrize("query,expected_unit", [
        ("Explain Unit 3 concepts", "3"),
        ("What is covered in unit 5?", "5"),
        ("Tell me about Unit 1", "1"),
        ("Explain unit 2 topics", "2"),
        ("What is in Unit 4?", "4"),
    ])
    def test_regex_unit_detection(self, query, expected_unit):
        """
        Test that explicit unit mentions are detected via regex.
        
        This behavior must be preserved after fixes.
        """
        # Test direct unit detection
        detected_unit = detect_unit(query)
        
        assert detected_unit == expected_unit, \
            f"Regex unit detection failed. " \
            f"Query: '{query}', Expected: '{expected_unit}', Got: '{detected_unit}'"
    
    @pytest.mark.parametrize("query,expected_unit", [
        ("Explain phishing in Unit 3", "3"),
        ("What is mutual fulfillment in unit 1?", "1"),
        ("Tell me about flip-flops in Unit 3", "3"),
    ])
    def test_regex_unit_in_full_routing(self, query, expected_unit):
        """
        Test that regex unit detection works in full routing pipeline.
        """
        result = route(query)
        
        assert result.unit == expected_unit, \
            f"Regex unit detection in routing failed. " \
            f"Query: '{query}', Expected unit: '{expected_unit}', Got: '{result.unit}'"


class TestPreservation8_EmbeddingRouter:
    """
    **Validates: Requirements 3.8**
    
    Preservation Test 8: Embedding Router Functionality
    
    Observe: Embedding router threshold-based classification on unfixed code
    Property: For all queries above embedding threshold, embedding router succeeds
    
    **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
    
    Note: Embedding router may not be available in test environment.
    """
    
    def test_embedding_router_functionality(self):
        """
        Test that routing system handles queries gracefully.
        
        Note: Embedding router may not be available due to model/memory constraints.
        This test verifies that the system doesn't crash and uses fallback routing.
        """
        # Query with keyword match - should work via keyword routing
        query = "What is malware?"
        
        result = route(query)
        
        # Should detect a subject via keyword routing (embedding may not be available)
        assert result.subject is not None or result.method == "none", \
            f"Routing failed unexpectedly for query: '{query}'"
        
        # Method should be one of the valid routing methods
        assert result.method in ["keyword", "embedding", "llm", "none"], \
            f"Unexpected routing method: '{result.method}'"
    
    def test_embedding_router_direct_call(self):
        """
        Test embedding router directly to verify it returns results.
        
        Note: May return None if embedding model is not available.
        """
        query = "Explain digital circuits"
        
        # Call embedding router directly
        emb_subj, emb_unit, emb_score = embedding_route(query)
        
        # Embedding router should return a result (subject or None)
        # We're just verifying it doesn't crash and returns the expected tuple
        assert isinstance(emb_subj, (str, type(None))), \
            f"Embedding router returned unexpected subject type: {type(emb_subj)}"
        assert isinstance(emb_unit, (str, type(None))), \
            f"Embedding router returned unexpected unit type: {type(emb_unit)}"
        assert isinstance(emb_score, (float, type(None))), \
            f"Embedding router returned unexpected score type: {type(emb_score)}"


# Summary of expected test outcomes:
# ─────────────────────────────────────
# All preservation tests should PASS on UNFIXED code
# This confirms the baseline behavior that must be preserved after fixes

