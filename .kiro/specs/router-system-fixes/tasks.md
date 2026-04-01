# Implementation Plan

## Phase 1: Bug Condition Exploration Tests

- [x] 1. Write bug condition exploration property tests
  - **Property 1: Bug Condition** - Router System Bugs
  - **CRITICAL**: These tests MUST FAIL on unfixed code - failure confirms the bugs exist
  - **DO NOT attempt to fix the tests or the code when they fail**
  - **NOTE**: These tests encode the expected behavior - they will validate the fixes when they pass after implementation
  - **GOAL**: Surface counterexamples that demonstrate each of the 6 bugs exists
  - Write property-based tests for all 6 bug conditions from the design document
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 1.1 Bug 1 - LLM Output Truncation Test
    - Test that LLM router with num_predict=10 truncates long subject names
    - Query: "Explain flip-flops in digital electronics unit 3"
    - Expected behavior: Should return "DIGITAL_ELECTRONICS_3" (after fix)
    - Current behavior: Truncated response like "DIGITAL_EL" (before fix)
    - Run test on UNFIXED code
    - **EXPECTED OUTCOME**: Test FAILS (confirms truncation bug exists)
    - Document counterexamples found
    - _Requirements: 1.1, 2.1_

  - [x] 1.2 Bug 2 - /no_think Placement Test
    - Test that /no_think is incorrectly placed in prompt parameter
    - Query: "What is phishing?"
    - Expected behavior: /no_think should be in system_prompt (after fix)
    - Current behavior: /no_think appended to user query text (before fix)
    - Verify by inspecting the models.chat call parameters in hybrid_router.py
    - Run test on UNFIXED code
    - **EXPECTED OUTCOME**: Test FAILS (confirms /no_think in wrong location)
    - Document the incorrect parameter placement
    - _Requirements: 1.2, 2.2_

  - [x] 1.3 Bug 3 - LLM Output Parsing Test
    - Test that exact string matching fails with trailing characters
    - Mock LLM responses: "DIGITAL_ELECTRONICS_3.\n", "UHV.", "CYBER_SECURITY_1 "
    - Expected behavior: Parser should strip and match (after fix)
    - Current behavior: Exact match fails (before fix)
    - Run test on UNFIXED code
    - **EXPECTED OUTCOME**: Test FAILS (confirms parsing bug exists)
    - Document which trailing characters cause failures
    - _Requirements: 1.3, 2.3_

  - [x] 1.4 Bug 4 - Subject Alias Resolution Test
    - Test that subject name format mismatches fail accuracy calculation
    - Test cases: expected="UNIVERSAL HUMAN VALUES", detected="UHV"
    - Test cases: expected="UNIVERSAL HUMAN VALUES", detected="UNIVERSAL_HUMAN_VALUES"
    - Expected behavior: Should count as correct match (after fix)
    - Current behavior: String comparison fails (before fix)
    - Run test on UNFIXED code
    - **EXPECTED OUTCOME**: Test FAILS (confirms alias resolution bug exists)
    - Document accuracy calculation errors
    - _Requirements: 1.4, 2.4_

  - [x] 1.5 Bug 5 - Missing UHV Keywords Test
    - Test that UHV queries with missing keywords fail keyword routing
    - Queries: "Explain the role of trust in relationships", "What is competence?", "Define intention"
    - Expected behavior: Keyword router should detect UHV (after fix)
    - Current behavior: Keyword router returns None, falls back to embedding (before fix)
    - Run test on UNFIXED code
    - **EXPECTED OUTCOME**: Test FAILS (confirms missing keywords bug exists)
    - Document which keywords are missing from subject_keywords.json
    - _Requirements: 1.5, 2.5_

  - [x] 1.6 Bug 6 - UHV Q5 Cross-Encoder Scoring Test
    - Test that UHV Q5 returns zero score from cross-encoder
    - Query: UHV Q5 text from test data
    - Expected behavior: Cross-encoder should return non-zero score (after fix)
    - Current behavior: Cross-encoder returns score of 0 (before fix)
    - Run test on UNFIXED code
    - **EXPECTED OUTCOME**: Test FAILS (confirms cross-encoder scoring bug exists)
    - Document the exact query and score returned
    - _Requirements: 1.6, 2.6_

## Phase 2: Preservation Property Tests

- [x] 2. Write preservation property tests (BEFORE implementing fixes)
  - **Property 2: Preservation** - Existing Router Functionality
  - **IMPORTANT**: Follow observation-first methodology
  - Observe behavior on UNFIXED code for non-buggy inputs
  - Write property-based tests capturing observed behavior patterns
  - Property-based testing generates many test cases for stronger guarantees
  - Run tests on UNFIXED code
  - **EXPECTED OUTCOME**: Tests PASS (confirms baseline behavior to preserve)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 2.1 Short Subject Name Preservation Test
    - Observe: Queries resulting in "UHV" classification work correctly on unfixed code
    - Write property: For all queries that result in short subject names (≤10 chars), classification succeeds
    - Test queries: "What is phishing?", "Explain UHV concepts"
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.1_

  - [x] 2.2 Routing Hierarchy Preservation Test
    - Observe: Routing executes regex → keyword → embedding → LLM on unfixed code
    - Write property: For all queries, routing hierarchy order is preserved
    - Test that keyword router is tried before embedding router
    - Test that embedding router is tried before LLM router
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.2_

  - [x] 2.3 Clean LLM Response Preservation Test
    - Observe: LLM responses without trailing characters parse correctly on unfixed code
    - Write property: For all clean LLM responses (no trailing punctuation), parsing succeeds
    - Mock responses: "UHV", "DIGITAL_ELECTRONICS_3", "CYBER_SECURITY_1"
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.3_

  - [x] 2.4 Exact Subject Match Preservation Test
    - Observe: Test cases with exact subject name matches count correctly on unfixed code
    - Write property: For all test cases where expected == detected (exact match), accuracy counts correctly
    - Test cases: expected="CYBER_SECURITY", detected="CYBER_SECURITY"
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.4_

  - [x] 2.5 Existing Keyword Preservation Test
    - Observe: Queries with existing UHV keywords detect correctly on unfixed code
    - Write property: For all queries with existing keywords, keyword routing succeeds
    - Test queries with keywords: "phishing", "mutual fulfillment", "self-exploration"
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.5_

  - [x] 2.6 Non-UHV Cross-Encoder Preservation Test
    - Observe: Cross-encoder scores for CYBER_SECURITY and DIGITAL_ELECTRONICS queries on unfixed code
    - Write property: For all non-UHV queries, cross-encoder returns accurate scores
    - Test queries from CYBER_SECURITY and DIGITAL_ELECTRONICS subjects
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.6_

  - [x] 2.7 Regex Unit Detection Preservation Test
    - Observe: Explicit unit mentions (e.g., "Unit 3") detected correctly on unfixed code
    - Write property: For all queries with explicit unit mentions, regex detection succeeds
    - Test queries: "Explain Unit 3 concepts", "What is covered in unit 5?"
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.7_

  - [x] 2.8 Embedding Router Preservation Test
    - Observe: Embedding router threshold-based classification on unfixed code
    - Write property: For all queries above embedding threshold, embedding router succeeds
    - Test queries that trigger embedding router (not keyword-matched)
    - Verify test PASSES on UNFIXED code
    - _Requirements: 3.8_

## Phase 3: Implementation

- [x] 3. Fix Bug 1 - LLM Output Truncation

  - [x] 3.1 Increase num_predict in config/rag.py
    - File: source_code/config/rag.py
    - Line 42: Change `ROUTER_NUM_PREDICT = 10` to `ROUTER_NUM_PREDICT = 50`
    - Rationale: Longest subject_unit string is "DIGITAL_ELECTRONICS_5" (21 chars), 50 tokens provides sufficient buffer
    - _Bug_Condition: isBugCondition(input) where input.config.num_predict <= 10 AND length(input.expected_subject_unit) > 10_tokens_
    - _Expected_Behavior: For all queries where LLM router is invoked and expected subject_unit > 10 tokens, num_predict >= 30 allows complete strings_
    - _Preservation: Existing successful classifications for short subject names continue to work_
    - _Requirements: 1.1, 2.1, 3.1_

  - [x] 3.2 Verify Bug 1 exploration test now passes
    - **Property 1: Expected Behavior** - LLM Output Not Truncated
    - **IMPORTANT**: Re-run the SAME test from task 1.1 - do NOT write a new test
    - Run bug condition exploration test from task 1.1
    - **EXPECTED OUTCOME**: Test PASSES (confirms truncation bug is fixed)
    - _Requirements: 2.1_

- [x] 4. Fix Bug 2 - /no_think Placement

  - [x] 4.1 Move /no_think to system_prompt in hybrid_router.py
    - File: source_code/rag/hybrid_router.py
    - Function: _llm_classify_subject_unit (lines 71-72)
    - Line 71: Remove `/no_think` from prompt parameter
    - Change: `prompt=f"{prompt} /no_think"` to `prompt=prompt`
    - Line 72: Update system_prompt to include the instruction
    - Change: `system_prompt="You are a helpful assistant. You must respond directly without internal reasoning or <think> tags."`
    - To: `system_prompt="You are a helpful assistant. You must respond directly without internal reasoning or <think> tags. /no_think"`
    - _Bug_Condition: isBugCondition(input) where "/no_think" IN input.prompt_parameter AND "/no_think" NOT IN input.system_prompt_parameter_
    - _Expected_Behavior: For all queries where LLM router needs to suppress thinking, /no_think is in system_prompt parameter_
    - _Preservation: Routing hierarchy order preserved, existing LLM classifications continue to work_
    - _Requirements: 1.2, 2.2, 3.2_

  - [x] 4.2 Verify Bug 2 exploration test now passes
    - **Property 1: Expected Behavior** - /no_think in System Prompt
    - **IMPORTANT**: Re-run the SAME test from task 1.2 - do NOT write a new test
    - Run bug condition exploration test from task 1.2
    - **EXPECTED OUTCOME**: Test PASSES (confirms /no_think placement bug is fixed)
    - _Requirements: 2.2_

- [x] 5. Fix Bug 3 - LLM Output Parsing

  - [x] 5.1 Add string normalization to hybrid_router.py
    - File: source_code/rag/hybrid_router.py
    - Function: _llm_classify_subject_unit (line 73)
    - Add normalization before comparison
    - Current: `llm_choice = response_text.strip().upper().replace(" ", "_")`
    - Change to: `llm_choice = response_text.strip().rstrip('.!?\n').upper().replace(" ", "_")`
    - This strips trailing punctuation and newlines before matching
    - _Bug_Condition: isBugCondition(input) where input.llm_response MATCHES "^[A-Z_0-9]+[.\\n\\s]+$"_
    - _Expected_Behavior: For all LLM responses with trailing punctuation/newlines/whitespace, parsing strips and matches successfully_
    - _Preservation: Clean LLM responses without trailing characters continue to parse correctly_
    - _Requirements: 1.3, 2.3, 3.3_

  - [x] 5.2 Verify Bug 3 exploration test now passes
    - **Property 1: Expected Behavior** - LLM Output Normalized Before Matching
    - **IMPORTANT**: Re-run the SAME test from task 1.3 - do NOT write a new test
    - Run bug condition exploration test from task 1.3
    - **EXPECTED OUTCOME**: Test PASSES (confirms parsing bug is fixed)
    - _Requirements: 2.3_

- [x] 6. Fix Bug 4 - Subject Alias Resolution

  - [x] 6.1 Load subject_aliases.json in reporter.py
    - File: source_code/tests/complete_system/reporter.py
    - Add import at top of file: `import json`
    - Add function to load and normalize subject aliases (after imports, before TestResult class):
    ```python
    def load_subject_aliases() -> dict:
        """Load subject aliases from JSON file and create reverse mapping."""
        alias_path = os.path.join(os.path.dirname(__file__), "../../data/subject_aliases.json")
        try:
            with open(alias_path, 'r') as f:
                aliases = json.load(f)
            # Create reverse mapping: alias -> canonical name
            reverse_map = {}
            for canonical, alias_list in aliases.items():
                canonical_normalized = canonical.upper().replace(" ", "_")
                reverse_map[canonical_normalized] = canonical_normalized
                for alias in alias_list:
                    alias_normalized = alias.upper().replace(" ", "_")
                    reverse_map[alias_normalized] = canonical_normalized
            return reverse_map
        except Exception as e:
            print(f"[WARNING] Failed to load subject aliases: {e}")
            return {}
    
    _SUBJECT_ALIASES = load_subject_aliases()
    
    def normalize_subject_name(name: str) -> str:
        """Normalize subject name using alias mapping."""
        if not name:
            return ""
        normalized = name.upper().replace(" ", "_")
        return _SUBJECT_ALIASES.get(normalized, normalized)
    ```
    - _Bug_Condition: isBugCondition(input) where normalize(input.expected_subject) != normalize(input.detected_subject) AND areAliases(input.expected_subject, input.detected_subject)_
    - _Expected_Behavior: For all test cases where expected and detected subjects are aliases, accuracy calculation normalizes and counts as correct match_
    - _Preservation: Exact subject name matches continue to count correctly_
    - _Requirements: 1.4, 2.4, 3.4_

  - [x] 6.2 Update subject comparison in generate_text_report
    - File: source_code/tests/complete_system/reporter.py
    - Function: generate_text_report (line 228)
    - Current: `r.subject_expected.upper().replace(" ", "_") != r.detected_subject`
    - Change to: `normalize_subject_name(r.subject_expected) != normalize_subject_name(r.detected_subject)`
    - _Requirements: 1.4, 2.4_

  - [x] 6.3 Update subject comparison in generate_rich_table
    - File: source_code/tests/complete_system/reporter.py
    - Function: generate_rich_table (around line 90)
    - Current: `r.subject_expected.upper().replace(" ", "_") != r.detected_subject`
    - Change to: `normalize_subject_name(r.subject_expected) != normalize_subject_name(r.detected_subject)`
    - _Requirements: 1.4, 2.4_

  - [x] 6.4 Verify Bug 4 exploration test now passes
    - **Property 1: Expected Behavior** - Subject Aliases Resolved in Accuracy Calculation
    - **IMPORTANT**: Re-run the SAME test from task 1.4 - do NOT write a new test
    - Run bug condition exploration test from task 1.4
    - **EXPECTED OUTCOME**: Test PASSES (confirms alias resolution bug is fixed)
    - Verify subject detection accuracy improves from 10/15 to expected 13-14/15
    - _Requirements: 2.4_

- [x] 7. Fix Bug 5 - Missing UHV Keywords

  - [x] 7.1 Add missing keywords to subject_keywords.json
    - File: source_code/data/subject_keywords.json
    - Section: UHV → notes → core
    - Add keywords: "intention", "competence", "trust", "relationship"
    - Current: `"core": ["mutual fulfillment", "sanskar", "self-exploration"]`
    - Change to: `"core": ["mutual fulfillment", "sanskar", "self-exploration", "intention", "competence", "trust", "relationship"]`
    - _Bug_Condition: isBugCondition(input) where input.query CONTAINS ["intention", "competence", "trust", "relationship"] AND input.expected_subject == "UHV" AND input.keyword_router_result == null_
    - _Expected_Behavior: For all UHV queries containing these terms, keyword router detects UHV subject_
    - _Preservation: Existing UHV keywords continue to work correctly_
    - _Requirements: 1.5, 2.5, 3.5_

  - [x] 7.2 Verify Bug 5 exploration test now passes
    - **Property 1: Expected Behavior** - UHV Keywords Expanded
    - **IMPORTANT**: Re-run the SAME test from task 1.5 - do NOT write a new test
    - Run bug condition exploration test from task 1.5
    - **EXPECTED OUTCOME**: Test PASSES (confirms missing keywords bug is fixed)
    - _Requirements: 2.5_

- [x] 8. Fix Bug 6 - UHV Q5 Cross-Encoder Scoring (Investigation Required)

  - [x] 8.1 Investigate UHV Q5 cross-encoder scoring
    - File: source_code/rag/cross_encoder.py (or relevant cross-encoder module)
    - Add debug logging to capture UHV Q5 inputs
    - Log the query text, document texts, raw logits, and sigmoid scores
    - Check for empty or malformed chunks
    - Verify chunk content has non-empty "text" fields
    - Test cross-encoder directly with UHV Q5 query and chunks
    - Compare with other UHV queries that score correctly
    - _Bug_Condition: isBugCondition(input) where input.query_id == "UHV_Q5" AND input.cross_encoder_score == 0 AND input.keyword_router_result == "UHV"_
    - _Expected_Behavior: For all queries similar to UHV Q5, cross-encoder returns non-zero scores for semantically relevant query-chunk pairs_
    - _Preservation: Non-UHV cross-encoder scoring continues to work correctly_
    - _Requirements: 1.6, 2.6, 3.6_

  - [x] 8.2 Implement fix based on investigation findings
    - Potential fixes (based on investigation):
      - If chunks are empty: Fix retrieval logic to ensure valid chunks
      - If encoding issues: Add text normalization before scoring
      - If model issue: Adjust instruction prompt for UHV domain
      - If threshold issue: Verify MIN_CROSS_SCORE (0.65) is appropriate for UHV
    - Document the root cause found and fix applied
    - _Requirements: 1.6, 2.6_

  - [x] 8.3 Verify Bug 6 exploration test now passes
    - **Property 1: Expected Behavior** - Cross-Encoder Scoring Investigated
    - **IMPORTANT**: Re-run the SAME test from task 1.6 - do NOT write a new test
    - Run bug condition exploration test from task 1.6
    - **EXPECTED OUTCOME**: Test PASSES (confirms cross-encoder scoring bug is fixed)
    - _Requirements: 2.6_

## Phase 4: Final Verification

- [x] 9. Verify all preservation tests still pass
  - **Property 2: Preservation** - Existing Router Functionality
  - **IMPORTANT**: Re-run ALL tests from task 2 - do NOT write new tests
  - Run all preservation property tests from Phase 2
  - **EXPECTED OUTCOME**: All tests PASS (confirms no regressions)
  - Verify:
    - Short subject name classifications still work (task 2.1)
    - Routing hierarchy order preserved (task 2.2)
    - Clean LLM responses still parse correctly (task 2.3)
    - Exact subject matches still count correctly (task 2.4)
    - Existing keywords still work (task 2.5)
    - Non-UHV cross-encoder scoring unchanged (task 2.6)
    - Regex unit detection still works (task 2.7)
    - Embedding router still works (task 2.8)
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

- [x] 10. Run full integration test suite
  - Run complete system test with all 15 questions
  - Verify subject detection accuracy improves from 10/15 to 13-14/15
  - Verify all routing methods work correctly (keyword, embedding, LLM)
  - Verify no regressions in answer quality or retrieval accuracy
  - Document final accuracy metrics and any remaining issues
  - _Requirements: All requirements 1.1-3.8_

- [x] 11. Checkpoint - Ensure all tests pass
  - Confirm all bug condition exploration tests pass (Phase 1)
  - Confirm all preservation tests pass (Phase 2)
  - Confirm all implementation tasks complete (Phase 3)
  - Confirm integration tests pass (Phase 4)
  - Ask the user if questions arise or if any issues need attention
