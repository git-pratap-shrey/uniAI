# Router System Fixes Bugfix Design

## Overview

The hybrid router system contains 6 critical bugs that prevent accurate subject/unit detection and cause LLM output truncation. These bugs affect configuration parameters, string parsing logic, subject name normalization, and keyword mappings. The fix approach involves:

1. Increasing `num_predict` to prevent LLM output truncation
2. Moving `/no_think` from user message to system prompt
3. Adding robust string normalization to LLM output parsing
4. Implementing subject alias mapping for accuracy calculation
5. Expanding UHV keyword map with missing terms
6. Investigating cross-encoder scoring for UHV Q5

The fixes are minimal, targeted changes that preserve existing functionality while addressing each specific bug condition.

## Glossary

- **Bug_Condition (C)**: The conditions that trigger each of the 6 bugs in the router system
- **Property (P)**: The desired behavior when the bug conditions are met - correct classification, parsing, and scoring
- **Preservation**: Existing router behavior that must remain unchanged (regex → keyword → embedding → LLM hierarchy, successful classifications)
- **num_predict**: LLM parameter controlling maximum token generation length
- **hybrid_router**: The master routing module coordinating regex, keyword, embedding, and LLM strategies
- **subject_unit_router**: The LLM prompt template for subject/unit classification
- **_llm_classify_subject_unit**: Function in hybrid_router.py that uses LLM for classification
- **reporter.py**: Test reporting module that calculates subject detection accuracy
- **subject_keywords.json**: Keyword map file containing subject-specific terms for keyword routing
- **subject_aliases.json**: Mapping file for subject name variations (e.g., "UHV" = "UNIVERSAL_HUMAN_VALUES")
- **cross-encoder**: Neural reranking model (Qwen3-Reranker) that scores query-chunk relevance

## Bug Details

### Bug Condition

The bugs manifest across 6 distinct conditions in the router system:

**Bug 1 - LLM Output Truncation:**
The bug occurs when the LLM router attempts to return subject names longer than 10 tokens. The `num_predict=10` configuration truncates responses mid-string, causing classification failures.

**Bug 2 - /no_think Appended to User Message:**
The bug occurs when hybrid_router.py appends "/no_think" to the prompt parameter instead of the system_prompt. The qwen3.5:2b model interprets this as part of the query text rather than a system instruction.

**Bug 3 - Exact String Matching in LLM Output:**
The bug occurs when the LLM returns valid subject names with trailing punctuation, newlines, or whitespace (e.g., "DIGITAL_ELECTRONICS_3." or "UHV\n"). The exact string matching fails to recognize these as valid responses.

**Bug 4 - Subject Name Mismatch in Accuracy Calculation:**
The bug occurs when reporter.py compares subject names with different formats: "UNIVERSAL HUMAN VALUES" (test data with spaces) vs "UNIVERSAL_HUMAN_VALUES" (detected with underscores) vs "UHV" (subject code). The string comparison fails even though all refer to the same subject.

**Bug 5 - Missing UHV Keywords:**
The bug occurs when UHV queries contain common terms like "intention", "competence", "trust", or "relationship" that are not present in the keyword map. The keyword router fails to detect UHV, forcing fallback to slower routing methods.

**Bug 6 - UHV Q5 Cross-Encoder Scoring:**
The bug occurs when UHV Q5 is processed by the cross-encoder, returning a score of 0 despite keyword routing working correctly. This suggests a potential issue with the cross-encoder's handling of certain UHV query-chunk pairs.

**Formal Specification:**
```
FUNCTION isBugCondition(input)
  INPUT: input of type RouterInput (query, config, test_data)
  OUTPUT: boolean
  
  RETURN (
    // Bug 1: LLM truncation
    (input.router_method == "llm" 
     AND input.config.num_predict <= 10
     AND length(input.expected_subject_unit) > 10_tokens)
    
    OR
    
    // Bug 2: /no_think in wrong parameter
    (input.router_method == "llm"
     AND "/no_think" IN input.prompt_parameter
     AND "/no_think" NOT IN input.system_prompt_parameter)
    
    OR
    
    // Bug 3: LLM output has trailing characters
    (input.router_method == "llm"
     AND input.llm_response MATCHES "^[A-Z_0-9]+[.\\n\\s]+$")
    
    OR
    
    // Bug 4: Subject name format mismatch
    (input.test_mode == true
     AND normalize(input.expected_subject) != normalize(input.detected_subject)
     AND areAliases(input.expected_subject, input.detected_subject))
    
    OR
    
    // Bug 5: Missing UHV keywords
    (input.query CONTAINS ["intention", "competence", "trust", "relationship"]
     AND input.expected_subject == "UHV"
     AND input.keyword_router_result == null)
    
    OR
    
    // Bug 6: Cross-encoder returns 0 for UHV Q5
    (input.query_id == "UHV_Q5"
     AND input.cross_encoder_score == 0
     AND input.keyword_router_result == "UHV")
  )
END FUNCTION
```

### Examples

- **Bug 1 Example**: Query "Explain flip-flops in digital electronics unit 3" → LLM attempts to return "DIGITAL_ELECTRONICS_3" (20 chars) → Truncated to "DIGITAL_EL" (10 tokens) → Classification fails
- **Bug 2 Example**: LLM receives prompt "Classify: What is phishing? /no_think" → Model confused by "/no_think" in query text → Returns incorrect classification
- **Bug 3 Example**: LLM returns "DIGITAL_ELECTRONICS_3.\n" → Exact match against "DIGITAL_ELECTRONICS_3" fails → Classification marked as failed
- **Bug 4 Example**: Test expects "UNIVERSAL HUMAN VALUES" → Router detects "UHV" → String comparison fails → Accuracy incorrectly decremented
- **Bug 5 Example**: Query "Explain the role of trust in relationships" → Keyword router checks map → "trust" and "relationship" not in UHV keywords → Falls back to embedding router (slower)
- **Bug 6 Example**: UHV Q5 query processed → Keyword router detects UHV correctly → Cross-encoder scores chunks → Returns score of 0 → Potential retrieval failure

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- Routing hierarchy (regex → keyword → embedding → LLM) must execute in the correct order
- Successful classifications for short subject names (e.g., "UHV") must continue to work
- Keyword routing for existing keywords must continue to detect subjects correctly
- Embedding router threshold-based classification must continue to function
- Cross-encoder scoring for non-UHV queries must continue to return accurate scores
- Regex unit detection for explicit mentions (e.g., "Unit 3") must continue to work
- Session subject override functionality must continue to work

**Scope:**
All inputs that do NOT trigger the 6 specific bug conditions should be completely unaffected by these fixes. This includes:
- Queries that result in successful LLM classifications without truncation
- Queries where the LLM returns clean responses without trailing characters
- Test cases where subject names already match exactly
- Queries containing existing UHV keywords that work correctly
- Non-UHV queries processed by the cross-encoder
- All regex-based unit detection
- All keyword routing for non-UHV subjects

## Hypothesized Root Cause

Based on the bug description and code analysis, the root causes are:

1. **LLM Output Truncation (Bug 1)**: The `ROUTER_NUM_PREDICT = 10` value in config/rag.py is too small. Subject_unit strings like "DIGITAL_ELECTRONICS_3" require approximately 20-30 tokens when the LLM generates them. The configuration was likely set conservatively to reduce latency but didn't account for the longest possible subject names.

2. **Incorrect /no_think Placement (Bug 2)**: In hybrid_router.py line 71, `/no_think` is appended to the `prompt` parameter: `prompt=f"{prompt} /no_think"`. The qwen3.5:2b model expects system-level instructions in the `system_prompt` parameter, not as part of the user query. This causes the model to interpret "/no_think" as part of the question text.

3. **Exact String Matching (Bug 3)**: The parsing logic in `_llm_classify_subject_unit` (lines 73-82) uses exact string comparison: `if su.upper() == llm_choice`. LLMs commonly add trailing punctuation, newlines, or whitespace to responses. The code doesn't normalize the LLM output before comparison.

4. **Subject Name Format Inconsistency (Bug 4)**: The reporter.py accuracy calculation (line 228) compares strings directly: `r.subject_expected.upper().replace(" ", "_") != r.detected_subject`. This handles space-to-underscore conversion but doesn't account for subject code aliases like "UHV" vs "UNIVERSAL_HUMAN_VALUES". The subject_aliases.json file exists but isn't used in accuracy calculation.

5. **Incomplete UHV Keyword Map (Bug 5)**: The subject_keywords.json file contains UHV keywords but is missing common terms from the UHV curriculum. Terms like "intention", "competence", "trust", and "relationship" appear frequently in UHV queries but are absent from the keyword map, causing the keyword router to miss valid UHV queries.

6. **Cross-Encoder Scoring Issue (Bug 6)**: The cross-encoder returns a score of 0 for UHV Q5 despite keyword routing working correctly. Potential causes:
   - The query-chunk pair may have semantic mismatch despite keyword match
   - The cross-encoder model may have difficulty with UHV domain-specific terminology
   - The retrieved chunks may not actually be relevant to the query
   - There may be an issue with the cross-encoder input formatting for this specific query

## Correctness Properties

Property 1: Bug Condition - LLM Output Not Truncated

_For any_ query where the LLM router is invoked and the expected subject_unit string length exceeds 10 tokens, the fixed configuration SHALL set num_predict to at least 30 tokens, allowing the LLM to return complete subject_unit strings without truncation.

**Validates: Requirements 2.1**

Property 2: Bug Condition - /no_think in System Prompt

_For any_ query where the LLM router needs to suppress thinking behavior, the fixed code SHALL pass "/no_think" as part of the system_prompt parameter (or use model configuration), NOT append it to the user query, ensuring the model interprets it as a system instruction.

**Validates: Requirements 2.2**

Property 3: Bug Condition - LLM Output Normalized Before Matching

_For any_ LLM response that contains trailing punctuation, newlines, or whitespace, the fixed parsing logic SHALL strip these characters before performing string matching, successfully recognizing valid subject_unit responses.

**Validates: Requirements 2.3**

Property 4: Bug Condition - Subject Aliases Resolved in Accuracy Calculation

_For any_ test case where the expected and detected subject names are aliases of the same subject (e.g., "UNIVERSAL HUMAN VALUES" and "UHV"), the fixed accuracy calculation SHALL normalize both names using the subject_aliases.json mapping and count them as correct matches.

**Validates: Requirements 2.4**

Property 5: Bug Condition - UHV Keywords Expanded

_For any_ UHV query containing the terms "intention", "competence", "trust", or "relationship", the fixed keyword map SHALL include these terms in the UHV keyword list, allowing the keyword router to detect the UHV subject.

**Validates: Requirements 2.5**

Property 6: Bug Condition - Cross-Encoder Scoring Investigated

_For any_ query similar to UHV Q5, the investigation SHALL identify the root cause of the 0 score and implement a fix that ensures the cross-encoder returns non-zero scores for semantically relevant query-chunk pairs.

**Validates: Requirements 2.6**

Property 7: Preservation - Existing Successful Classifications

_For any_ query that currently results in successful classification (short subject names, clean LLM responses, exact subject name matches, existing keywords), the fixed code SHALL produce exactly the same classification results, preserving all existing functionality.

**Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.7, 3.8**

Property 8: Preservation - Routing Hierarchy Order

_For any_ query processed by the hybrid router, the fixed code SHALL execute the routing hierarchy in the same order (regex → keyword → embedding → LLM), preserving the existing routing logic and fallback behavior.

**Validates: Requirements 3.2**

Property 9: Preservation - Cross-Encoder for Non-UHV Queries

_For any_ non-UHV query processed by the cross-encoder, the fixed code SHALL continue to return accurate similarity scores, preserving the existing cross-encoder functionality for all other subjects.

**Validates: Requirements 3.6**

## Fix Implementation

### Changes Required

Assuming our root cause analysis is correct:

**File 1**: `source_code/config/rag.py`

**Changes**:
1. **Increase ROUTER_NUM_PREDICT**: Change from 10 to 50
   - Line 42: `ROUTER_NUM_PREDICT = 10` → `ROUTER_NUM_PREDICT = 50`
   - Rationale: Longest subject_unit string is "DIGITAL_ELECTRONICS_5" (21 chars). With LLM tokenization overhead, 50 tokens provides sufficient buffer.

**File 2**: `source_code/rag/hybrid_router.py`

**Function**: `_llm_classify_subject_unit`

**Specific Changes**:
1. **Move /no_think to system_prompt**: Modify the models.chat call
   - Line 71: Remove `/no_think` from prompt parameter
   - Line 72: Update system_prompt to include the instruction
   - Before: `prompt=f"{prompt} /no_think"`
   - After: `prompt=prompt` and update system_prompt to include no-think instruction

2. **Add string normalization to LLM output parsing**: Add normalization before comparison
   - Line 73: Add normalization function
   - Line 76: Apply normalization to llm_choice before comparison
   - Add: `llm_choice = response_text.strip().rstrip('.!?\n').upper().replace(" ", "_")`

**File 3**: `source_code/tests/complete_system/reporter.py`

**Function**: Subject accuracy calculation (in generate_text_report and generate_rich_table)

**Specific Changes**:
1. **Load subject_aliases.json**: Add import and load aliases at module level
   - Add function to load and normalize subject aliases
   - Create reverse mapping: {"UHV": "UNIVERSAL_HUMAN_VALUES", "UNIVERSAL HUMAN VALUES": "UNIVERSAL_HUMAN_VALUES", etc.}

2. **Normalize subject names before comparison**: Update comparison logic
   - Line 228 (in generate_text_report): Replace direct string comparison with alias-aware comparison
   - Add function: `normalize_subject_name(name: str, aliases: dict) -> str`
   - Logic: Convert to uppercase, replace spaces with underscores, check if it's an alias, return canonical name

3. **Update accuracy calculation**: Apply normalization in all subject comparison locations
   - generate_rich_table (line ~90)
   - generate_text_report (line ~228)

**File 4**: `source_code/data/subject_keywords.json`

**Section**: UHV keywords

**Specific Changes**:
1. **Add missing keywords to UHV notes section**: Expand the keyword list
   - Add to "notes" → "core": ["intention", "competence", "trust", "relationship"]
   - Alternatively, add to appropriate unit sections if they're unit-specific

2. **Add missing keywords to UHV syllabus section**: Ensure consistency
   - Add to "syllabus" → appropriate units based on curriculum analysis

3. **Add missing keywords to UHV pyq section**: Cover past year questions
   - Add to "pyq": ["intention", "competence", "trust", "relationship"]

**File 5**: `source_code/rag/cross_encoder.py` (Investigation Required)

**Function**: `rerank_cross_encoder`

**Investigation Steps**:
1. **Log UHV Q5 inputs**: Add debug logging to capture the exact query and chunks being scored
   - Log the query text
   - Log the document texts being scored
   - Log the raw logits and sigmoid scores from the model

2. **Check for empty or malformed chunks**: Verify chunk content
   - Ensure chunks have non-empty "text" fields
   - Check for encoding issues or special characters

3. **Test cross-encoder directly**: Isolate the cross-encoder scoring
   - Call models.rerank() directly with UHV Q5 query and chunks
   - Compare with other UHV queries that score correctly

4. **Potential fixes** (based on investigation findings):
   - If chunks are empty: Fix retrieval logic to ensure valid chunks are retrieved
   - If encoding issues: Add text normalization before scoring
   - If model issue: Consider adjusting the instruction prompt for UHV domain
   - If threshold issue: Verify MIN_CROSS_SCORE (0.65) is appropriate for UHV queries

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate each bug on unfixed code, then verify the fixes work correctly and preserve existing behavior. Each bug requires specific test cases to validate the fix.

### Exploratory Bug Condition Checking

**Goal**: Surface counterexamples that demonstrate each bug BEFORE implementing the fixes. Confirm or refute the root cause analysis for all 6 bugs.

**Test Plan**: Write tests that trigger each bug condition on the UNFIXED code, observe failures, and document the exact failure modes.

**Test Cases**:

1. **Bug 1 - LLM Truncation Test**: 
   - Query: "Explain flip-flops in digital electronics unit 3"
   - Expected: LLM should return "DIGITAL_ELECTRONICS_3"
   - Actual on unfixed: Truncated response (e.g., "DIGITAL_EL")
   - Will fail on unfixed code

2. **Bug 2 - /no_think Placement Test**:
   - Query: "What is phishing?"
   - Expected: LLM receives "/no_think" as system instruction
   - Actual on unfixed: LLM receives "/no_think" in query text, may return confused response
   - Will fail on unfixed code

3. **Bug 3 - LLM Output Parsing Test**:
   - Mock LLM response: "DIGITAL_ELECTRONICS_3.\n"
   - Expected: Parser should recognize as "DIGITAL_ELECTRONICS_3"
   - Actual on unfixed: Exact match fails, classification marked as failed
   - Will fail on unfixed code

4. **Bug 4 - Subject Alias Test**:
   - Test data: expected="UNIVERSAL HUMAN VALUES", detected="UHV"
   - Expected: Accuracy calculation should count as correct match
   - Actual on unfixed: String comparison fails, accuracy decremented
   - Will fail on unfixed code

5. **Bug 5 - Missing UHV Keywords Test**:
   - Query: "Explain the role of trust in relationships"
   - Expected: Keyword router should detect UHV
   - Actual on unfixed: Keyword router returns None, falls back to embedding router
   - Will fail on unfixed code

6. **Bug 6 - UHV Q5 Cross-Encoder Test**:
   - Query: UHV Q5 text (from test data)
   - Expected: Cross-encoder should return non-zero score
   - Actual on unfixed: Cross-encoder returns score of 0
   - Will fail on unfixed code

**Expected Counterexamples**:
- Bug 1: LLM responses truncated mid-string, classification failures for long subject names
- Bug 2: LLM confusion or incorrect classifications when "/no_think" is in query text
- Bug 3: Valid LLM responses with trailing punctuation not recognized
- Bug 4: Subject detection accuracy incorrectly calculated as 10/15 instead of 13-14/15
- Bug 5: UHV queries with "trust", "relationship" not detected by keyword router
- Bug 6: Cross-encoder score of 0 for UHV Q5 despite keyword routing success

### Fix Checking

**Goal**: Verify that for all inputs where each bug condition holds, the fixed code produces the expected behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition(input) DO
  result := fixed_router_system(input)
  ASSERT expectedBehavior(result)
END FOR
```

**Test Cases by Bug**:

1. **Bug 1 Fix Verification**:
   - Test with queries requiring long subject_unit strings
   - Assert: LLM returns complete strings without truncation
   - Assert: Classification succeeds for "DIGITAL_ELECTRONICS_3", "CYBER_SECURITY_5", etc.

2. **Bug 2 Fix Verification**:
   - Test with various queries using LLM router
   - Assert: "/no_think" is in system_prompt parameter, not prompt parameter
   - Assert: LLM returns correct classifications without confusion

3. **Bug 3 Fix Verification**:
   - Test with mock LLM responses containing trailing characters
   - Assert: Parser successfully strips and matches "UHV.\n", "DIGITAL_ELECTRONICS_3.", etc.
   - Assert: Classification succeeds for all valid responses with trailing characters

4. **Bug 4 Fix Verification**:
   - Test with subject name variations: "UHV" vs "UNIVERSAL_HUMAN_VALUES" vs "UNIVERSAL HUMAN VALUES"
   - Assert: Accuracy calculation normalizes and matches all aliases correctly
   - Assert: Subject detection accuracy increases from 10/15 to expected 13-14/15

5. **Bug 5 Fix Verification**:
   - Test with UHV queries containing "intention", "competence", "trust", "relationship"
   - Assert: Keyword router detects UHV subject
   - Assert: No fallback to embedding router for these queries

6. **Bug 6 Fix Verification**:
   - Test with UHV Q5 query
   - Assert: Cross-encoder returns non-zero score
   - Assert: Score accurately reflects semantic similarity

### Preservation Checking

**Goal**: Verify that for all inputs where the bug conditions do NOT hold, the fixed code produces the same results as the original code.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT original_router(input) = fixed_router(input)
END FOR
```

**Testing Approach**: Property-based testing is recommended for preservation checking because:
- It generates many test cases automatically across the input domain
- It catches edge cases that manual unit tests might miss
- It provides strong guarantees that behavior is unchanged for all non-buggy inputs

**Test Plan**: Observe behavior on UNFIXED code first for successful classifications, then write property-based tests capturing that behavior.

**Test Cases**:

1. **Short Subject Name Preservation**: 
   - Observe: Queries resulting in "UHV" classification work correctly on unfixed code
   - Test: Verify "UHV" classifications continue to work after fix
   - Assert: Same classification results before and after fix

2. **Routing Hierarchy Preservation**:
   - Observe: Routing executes regex → keyword → embedding → LLM on unfixed code
   - Test: Verify routing order remains unchanged after fix
   - Assert: Same routing method used for each query before and after fix

3. **Clean LLM Response Preservation**:
   - Observe: LLM responses without trailing characters parse correctly on unfixed code
   - Test: Verify clean responses continue to parse correctly after fix
   - Assert: Same parsing results before and after fix

4. **Exact Subject Match Preservation**:
   - Observe: Test cases with exact subject name matches count correctly on unfixed code
   - Test: Verify exact matches continue to count correctly after fix
   - Assert: Same accuracy calculation for exact matches before and after fix

5. **Existing Keyword Preservation**:
   - Observe: Queries with existing UHV keywords detect correctly on unfixed code
   - Test: Verify existing keywords continue to work after adding new keywords
   - Assert: Same keyword routing results before and after fix

6. **Non-UHV Cross-Encoder Preservation**:
   - Observe: Cross-encoder scores for CYBER_SECURITY and DIGITAL_ELECTRONICS queries on unfixed code
   - Test: Verify non-UHV cross-encoder scoring continues to work after fix
   - Assert: Same cross-encoder scores before and after fix

7. **Regex Unit Detection Preservation**:
   - Observe: Explicit unit mentions (e.g., "Unit 3") detected correctly on unfixed code
   - Test: Verify regex unit detection continues to work after fix
   - Assert: Same unit detection results before and after fix

8. **Embedding Router Preservation**:
   - Observe: Embedding router threshold-based classification on unfixed code
   - Test: Verify embedding router continues to work after fix
   - Assert: Same embedding router results before and after fix

### Unit Tests

- Test config value changes (num_predict = 50)
- Test /no_think placement in system_prompt parameter
- Test string normalization function with various inputs
- Test subject alias loading and normalization
- Test keyword map updates with new UHV terms
- Test cross-encoder with UHV Q5 after investigation

### Property-Based Tests

- Generate random queries with long subject names, verify no truncation
- Generate random LLM responses with trailing characters, verify parsing succeeds
- Generate random subject name variations, verify alias resolution works
- Generate random UHV queries with new keywords, verify keyword routing succeeds
- Generate random non-UHV queries, verify cross-encoder scoring unchanged

### Integration Tests

- Run full test suite (15 questions) and verify subject detection accuracy improves from 10/15 to 13-14/15
- Test complete routing flow for each bug scenario
- Verify routing hierarchy order preserved across all test cases
- Test cross-encoder scoring across all subjects after Bug 6 fix
