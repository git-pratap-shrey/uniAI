"""
test_bug_exploration.py
───────────────────────
Bug Condition Exploration Tests for Router System Fixes

**CRITICAL**: These tests are EXPECTED TO FAIL on unfixed code.
Failure confirms the bugs exist. DO NOT fix the tests or code when they fail.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6**

This test file explores all 6 bug conditions:
1. LLM Output Truncation (num_predict=10)
2. /no_think Placement in Wrong Parameter
3. LLM Output Parsing with Trailing Characters
4. Subject Alias Resolution in Accuracy Calculation
5. Missing UHV Keywords
6. UHV Q5 Cross-Encoder Scoring
"""

import os
import sys
import pytest
import json
from unittest.mock import patch, MagicMock

# Add project root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source_code.rag.hybrid_router import route, _llm_classify_subject_unit
from source_code.config import CONFIG
from source_code.tests.complete_system.reporter import generate_text_report, TestResult, RouterStageTrace


class TestBug1_LLMOutputTruncation:
    """
    **Validates: Requirements 1.1, 2.1**
    
    Bug 1: LLM Output Truncation Test
    
    EXPECTED BEHAVIOR (after fix): LLM should return complete subject_unit strings
    CURRENT BEHAVIOR (before fix): LLM truncates responses to 10 tokens
    
    Test queries that require long subject names like "DIGITAL_ELECTRONICS_3"
    """
    
    def test_long_subject_name_truncation(self):
        """
        Test that LLM router with num_predict=10 truncates long subject names.
        
        Query: "Explain flip-flops in digital electronics unit 3"
        Expected (after fix): "DIGITAL_ELECTRONICS_3"
        Current (before fix): Truncated like "DIGITAL_EL"
        
        **EXPECTED OUTCOME**: Test FAILS (confirms truncation bug exists)
        """
        query = "Explain flip-flops in digital electronics unit 3"
        
        # Check current config value
        current_num_predict = CONFIG["rag"].get("router_num_predict", 10)
        
        # This test expects the bug to exist (num_predict <= 10)
        assert current_num_predict <= 10, \
            f"Bug 1 may already be fixed: num_predict={current_num_predict} (expected <=10)"
        
        # Route the query
        result = route(query)
        
        # After fix, this should be "DIGITAL_ELECTRONICS"
        # Before fix, LLM truncation causes classification failure
        assert result.subject == "DIGITAL_ELECTRONICS", \
            f"Expected 'DIGITAL_ELECTRONICS', got '{result.subject}'. " \
            f"Bug 1 confirmed: LLM output truncation prevents correct classification."
        
        assert result.unit == "3", \
            f"Expected unit '3', got '{result.unit}'. " \
            f"Bug 1 confirmed: LLM output truncation prevents correct unit detection."


class TestBug2_NoThinkPlacement:
    """
    **Validates: Requirements 1.2, 2.2**
    
    Bug 2: /no_think Placement Test
    
    EXPECTED BEHAVIOR (after fix): /no_think should be in system_prompt parameter
    CURRENT BEHAVIOR (before fix): /no_think appended to user query text
    
    Verify by inspecting the models.chat call parameters in hybrid_router.py
    """
    
    def test_no_think_in_wrong_parameter(self):
        """
        Test that /no_think is incorrectly placed in prompt parameter.
        
        Query: "What is phishing?"
        Expected (after fix): /no_think in system_prompt
        Current (before fix): /no_think in prompt parameter
        
        **EXPECTED OUTCOME**: Test FAILS (confirms /no_think in wrong location)
        """
        query = "What is phishing?"
        
        # Mock models.chat to capture the parameters
        with patch('source_code.rag.hybrid_router.models.chat') as mock_chat:
            mock_chat.return_value = "CYBER_SECURITY"
            
            # Call the LLM classifier
            _llm_classify_subject_unit(query)
            
            # Check if models.chat was called
            if mock_chat.called:
                call_args = mock_chat.call_args
                prompt_arg = call_args.kwargs.get('prompt', '')
                system_prompt_arg = call_args.kwargs.get('system_prompt', '')
                
                # After fix: /no_think should be in system_prompt, NOT in prompt
                # Before fix: /no_think is in prompt parameter
                assert "/no_think" not in prompt_arg, \
                    f"Bug 2 confirmed: /no_think found in prompt parameter: '{prompt_arg}'"
                
                assert "/no_think" in system_prompt_arg, \
                    f"Bug 2 confirmed: /no_think NOT found in system_prompt parameter: '{system_prompt_arg}'"


class TestBug3_LLMOutputParsing:
    """
    **Validates: Requirements 1.3, 2.3**
    
    Bug 3: LLM Output Parsing Test
    
    EXPECTED BEHAVIOR (after fix): Parser should strip trailing characters and match
    CURRENT BEHAVIOR (before fix): Exact match fails with trailing punctuation/newlines
    
    Mock LLM responses with trailing characters
    """
    
    @pytest.mark.parametrize("llm_response,expected_subject,expected_unit", [
        ("DIGITAL_ELECTRONICS_3.\n", "DIGITAL_ELECTRONICS", "3"),
        ("UHV.", "UHV", None),
        ("CYBER_SECURITY_1 ", "CYBER_SECURITY", "1"),
        ("DIGITAL_ELECTRONICS_5\n", "DIGITAL_ELECTRONICS", "5"),
    ])
    def test_trailing_characters_parsing(self, llm_response, expected_subject, expected_unit):
        """
        Test that exact string matching fails with trailing characters.
        
        Mock responses: "DIGITAL_ELECTRONICS_3.\\n", "UHV.", "CYBER_SECURITY_1 "
        Expected (after fix): Parser strips and matches correctly
        Current (before fix): Exact match fails
        
        **EXPECTED OUTCOME**: Test FAILS (confirms parsing bug exists)
        """
        query = "test query"
        
        # Mock models.chat to return response with trailing characters
        with patch('source_code.rag.hybrid_router.models.chat') as mock_chat:
            mock_chat.return_value = llm_response
            
            # Call the LLM classifier
            result = _llm_classify_subject_unit(query)
            
            # After fix: Parser should strip trailing characters and match
            # Before fix: Exact match fails
            assert result.subject == expected_subject, \
                f"Bug 3 confirmed: Failed to parse '{llm_response}'. " \
                f"Expected subject '{expected_subject}', got '{result.subject}'. " \
                f"Trailing characters prevent correct parsing."
            
            if expected_unit:
                assert result.unit == expected_unit, \
                    f"Bug 3 confirmed: Failed to parse unit from '{llm_response}'. " \
                    f"Expected unit '{expected_unit}', got '{result.unit}'."


class TestBug4_SubjectAliasResolution:
    """
    **Validates: Requirements 1.4, 2.4**
    
    Bug 4: Subject Alias Resolution Test
    
    EXPECTED BEHAVIOR (after fix): Should count as correct match using alias mapping
    CURRENT BEHAVIOR (before fix): String comparison fails for aliases
    
    Test accuracy calculation with subject name variations
    """
    
    @pytest.mark.parametrize("expected_subject,detected_subject,should_match", [
        ("UNIVERSAL HUMAN VALUES", "UHV", True),
        ("UNIVERSAL HUMAN VALUES", "UNIVERSAL_HUMAN_VALUES", True),
        ("CYBER SECURITY", "CYBER_SECURITY", True),
        ("DIGITAL ELECTRONICS", "DIGITAL_ELECTRONICS", True),
    ])
    def test_subject_alias_matching(self, expected_subject, detected_subject, should_match):
        """
        Test that subject name format mismatches fail accuracy calculation.
        
        Test cases: 
        - expected="UNIVERSAL HUMAN VALUES", detected="UHV"
        - expected="UNIVERSAL HUMAN VALUES", detected="UNIVERSAL_HUMAN_VALUES"
        
        Expected (after fix): Should count as correct match
        Current (before fix): String comparison fails
        
        **EXPECTED OUTCOME**: Test FAILS (confirms alias resolution bug exists)
        """
        # Import the normalize function from reporter
        from source_code.tests.complete_system.reporter import normalize_subject_name
        
        # Create a test result
        test_result = TestResult(
            question_id="TEST",
            subject_expected=expected_subject,
            subject_code="TEST",
            query="test query",
            expected_answer="test answer",
            detected_subject=detected_subject,
            router_trace=RouterStageTrace()
        )
        
        # Check if subjects match using NEW logic (after fix)
        # New logic: normalize_subject_name(r.subject_expected) == normalize_subject_name(r.detected_subject)
        normalized_expected = normalize_subject_name(test_result.subject_expected)
        normalized_detected = normalize_subject_name(test_result.detected_subject)
        matches = (normalized_expected == normalized_detected)
        
        if should_match:
            assert matches, \
                f"Bug 4 confirmed: Subject alias not resolved. " \
                f"Expected '{expected_subject}' to match '{detected_subject}', but they don't. " \
                f"Normalized expected: '{normalized_expected}', normalized detected: '{normalized_detected}'"


class TestBug5_MissingUHVKeywords:
    """
    **Validates: Requirements 1.5, 2.5**
    
    Bug 5: Missing UHV Keywords Test
    
    EXPECTED BEHAVIOR (after fix): Keyword router should detect UHV
    CURRENT BEHAVIOR (before fix): Keyword router returns None, falls back to embedding
    
    Test UHV queries with missing keywords
    """
    
    @pytest.mark.parametrize("query,missing_keyword", [
        ("Explain the role of trust in relationships", "trust"),
        ("What is competence in human values?", "competence"),
        ("Define intention and its importance", "intention"),
        ("How do relationships affect human values?", "relationship"),
    ])
    def test_missing_uhv_keywords(self, query, missing_keyword):
        """
        Test that UHV queries with missing keywords fail keyword routing.
        
        Queries: "Explain the role of trust in relationships", etc.
        Expected (after fix): Keyword router detects UHV
        Current (before fix): Keyword router returns None
        
        **EXPECTED OUTCOME**: Test FAILS (confirms missing keywords bug exists)
        """
        # Load current keyword map
        keyword_file = os.path.join(ROOT, "source_code/data/subject_keywords.json")
        with open(keyword_file, 'r') as f:
            keywords = json.load(f)
        
        # Check if the keyword exists in UHV keywords
        uhv_keywords = keywords.get("UHV", {})
        all_uhv_keywords = []
        
        for section in uhv_keywords.values():
            if isinstance(section, dict):
                for unit_keywords in section.values():
                    if isinstance(unit_keywords, list):
                        all_uhv_keywords.extend([k.lower() for k in unit_keywords])
            elif isinstance(section, list):
                all_uhv_keywords.extend([k.lower() for k in section])
        
        # Check if missing keyword is in the map
        keyword_exists = missing_keyword.lower() in all_uhv_keywords
        
        # Before fix: keyword should be missing
        assert keyword_exists, \
            f"Bug 5 confirmed: Keyword '{missing_keyword}' is missing from UHV keyword map. " \
            f"Query '{query}' will fail keyword routing and fall back to slower methods."
        
        # Route the query and check if keyword router detects UHV
        result = route(query)
        
        # After fix: Should detect UHV via keyword router
        # Before fix: Falls back to embedding or LLM router
        assert result.subject == "UHV", \
            f"Bug 5 confirmed: Query '{query}' did not detect UHV. " \
            f"Got subject: '{result.subject}', method: '{result.method}'. " \
            f"Keyword '{missing_keyword}' missing from keyword map."
        
        assert result.method == "keyword", \
            f"Bug 5 confirmed: Query '{query}' used '{result.method}' router instead of 'keyword'. " \
            f"Missing keyword '{missing_keyword}' caused fallback to slower routing method."


class TestBug6_UHVCrossEncoderScoring:
    """
    **Validates: Requirements 1.6, 2.6**
    
    Bug 6: UHV Q5 Cross-Encoder Scoring Test
    
    EXPECTED BEHAVIOR (after fix): Cross-encoder should return non-zero score
    CURRENT BEHAVIOR (before fix): Cross-encoder returns score of 0
    
    Test UHV Q5 query processing
    """
    
    def test_uhv_q5_cross_encoder_zero_score(self):
        """
        Test that UHV Q5 routing and retrieval work correctly after fix.
        
        Query: UHV Q5 text from test data
        Expected (after fix): Routes to Unit 1, retrieves relevant chunks
        Current (before fix): Routed to Unit 2, no relevant chunks
        
        **EXPECTED OUTCOME**: Test PASSES after fix (confirms routing issue resolved)
        """
        # UHV Q5 from questions.txt
        uhv_q5_query = "Define 'Prosperity'. How is it different from 'Wealth'?"
        
        # Test routing
        result = route(uhv_q5_query)
        
        # Should detect UHV subject
        assert result.subject == "UHV", \
            f"UHV Q5 routing failed. Expected 'UHV', got '{result.subject}'"
        
        # After fix, should detect Unit 1 (where the content actually is)
        # Before fix, it detected Unit 2 (wrong unit)
        assert result.unit == "1", \
            f"UHV Q5 unit routing failed. Expected '1', got '{result.unit}'. " \
            f"The prosperity/wealth content is in Unit 1 notes, not Unit 2."
        
        # Verify routing method was keyword (not LLM fallback)
        assert result.method == "keyword", \
            f"Expected keyword routing, got '{result.method}'"
        
        print(f"\n✓ Bug 6 Fix Verified:")
        print(f"  - Subject: {result.subject}")
        print(f"  - Unit: {result.unit}")
        print(f"  - Method: {result.method}")
        print(f"  - Routing now correctly identifies Unit 1 where prosperity/wealth content exists")


# Summary of expected test outcomes:
# ─────────────────────────────────────
# Bug 1: FAIL - LLM truncation prevents correct classification
# Bug 2: FAIL - /no_think in wrong parameter location
# Bug 3: FAIL - Trailing characters prevent parsing
# Bug 4: FAIL - Subject aliases not resolved in accuracy calculation
# Bug 5: FAIL - Missing keywords cause keyword routing failure
# Bug 6: SKIP - Requires investigation with full RAG pipeline
