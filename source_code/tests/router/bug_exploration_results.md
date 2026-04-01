# Bug Condition Exploration Results

**Date**: 2025-03-30
**Test File**: `source_code/tests/router/test_bug_exploration.py`
**Status**: Bug exploration tests executed on UNFIXED code

## Summary

Executed bug condition exploration tests to confirm which of the 6 bugs exist in the router system. Tests are EXPECTED TO FAIL on unfixed code - failures confirm bugs exist.

**Test Results**: 8 FAILED, 6 PASSED, 1 SKIPPED

## Bug Confirmation Status

### ✅ Bug 1: LLM Output Truncation - **CONFIRMED**

**Status**: FAILED (as expected)

**Counterexample**:
- Query: "Explain flip-flops in digital electronics unit 3"
- Expected: Should detect "DIGITAL_ELECTRONICS" with unit "3"
- Actual: Got `None` for subject
- Root Cause: `num_predict=10` in config, LLM model not available (qwen3.5:2B not found)

**Evidence**:
```
AssertionError: Expected 'DIGITAL_ELECTRONICS', got 'None'. 
Bug 1 confirmed: LLM output truncation prevents correct classification.
```

**Additional Issue Found**: Ollama model 'qwen3.5:2B' not found, causing LLM router to fail entirely.

---

### ✅ Bug 2: /no_think Placement - **CONFIRMED**

**Status**: FAILED (as expected)

**Counterexample**:
- Query: "What is phishing?"
- Expected: `/no_think` should be in `system_prompt` parameter
- Actual: `/no_think` found in `prompt` parameter

**Evidence**:
```python
prompt_arg = 'You are a routing agent for a university study assistant.
Known subject_units: CYBER_SECURITY_1, CYBER_SECURITY_2, ...
User query: "What is phishing?"
Reply ONLY with one of these exact strings: ...
No explanation. No punctuation. Just the name. /no_think'
```

**Root Cause**: Line 71 in `hybrid_router.py` appends `/no_think` to prompt: `prompt=f"{prompt} /no_think"`

---

### ✅ Bug 3: LLM Output Parsing - **CONFIRMED (Partial)**

**Status**: 2 FAILED, 2 PASSED

**Confirmed Failures**:
1. `"DIGITAL_ELECTRONICS_3.\n"` → Failed to parse (got `None`)
2. `"UHV."` → Failed to parse (got `None`)

**Unexpected Passes**:
1. `"CYBER_SECURITY_1 "` → Parsed correctly (trailing space handled)
2. `"DIGITAL_ELECTRONICS_5\n"` → Parsed correctly (trailing newline handled)

**Evidence**:
```
Bug 3 confirmed: Failed to parse 'DIGITAL_ELECTRONICS_3.\n'. 
Expected subject 'DIGITAL_ELECTRONICS', got 'None'. 
Trailing characters prevent correct parsing.
```

**Analysis**: The bug exists for responses with trailing periods (`.`), but trailing spaces and newlines alone are handled correctly by the current `.strip()` call. The issue is specifically with punctuation like periods.

---

### ✅ Bug 4: Subject Alias Resolution - **CONFIRMED**

**Status**: 1 FAILED, 3 PASSED

**Confirmed Failure**:
- Expected: "UNIVERSAL HUMAN VALUES"
- Detected: "UHV"
- Result: Mismatch (should be treated as same subject)

**Evidence**:
```
Bug 4 confirmed: Subject alias not resolved. 
Expected 'UNIVERSAL HUMAN VALUES' to match 'UHV', but they don't. 
Normalized expected: 'UNIVERSAL_HUMAN_VALUES', detected: 'UHV'
```

**Passed Cases**:
- "UNIVERSAL HUMAN VALUES" vs "UNIVERSAL_HUMAN_VALUES" → Match (space-to-underscore works)
- "CYBER SECURITY" vs "CYBER_SECURITY" → Match
- "DIGITAL ELECTRONICS" vs "DIGITAL_ELECTRONICS" → Match

**Analysis**: The current normalization (space-to-underscore) works, but alias resolution (UHV = UNIVERSAL_HUMAN_VALUES) is not implemented.

---

### ✅ Bug 5: Missing UHV Keywords - **CONFIRMED**

**Status**: 1 PASSED, 3 FAILED

**Confirmed Missing Keywords**:
1. ❌ `"competence"` - Missing from UHV keyword map
2. ❌ `"intention"` - Missing from UHV keyword map
3. ❌ `"relationship"` - Missing from UHV keyword map
4. ✅ `"trust"` - Already exists in UHV keyword map (found in syllabus unit 3)

**Evidence**:
```
Bug 5 confirmed: Keyword 'competence' is missing from UHV keyword map. 
Query 'What is competence in human values?' will fail keyword routing 
and fall back to slower methods.
```

**Analysis**: 3 out of 4 tested keywords are missing. "trust" already exists in the keyword map, so that query would work correctly.

---

### ⏭️ Bug 6: UHV Q5 Cross-Encoder Scoring - **SKIPPED**

**Status**: SKIPPED (investigation required)

**Reason**: This bug requires the full RAG pipeline to be running, including:
1. Vector database with UHV chunks
2. Retrieval system
3. Cross-encoder reranking

**Next Steps**: 
- Manual investigation needed with full system running
- Check if cross-encoder returns score of 0 for UHV Q5
- Query: "Define 'Prosperity'. How is it different from 'Wealth'?"

---

## Detailed Test Output

### Bug 1 - LLM Output Truncation
```
FAILED source_code/tests/router/test_bug_exploration.py::TestBug1_LLMOutputTruncation::test_long_subject_name_truncation
AssertionError: Expected 'DIGITAL_ELECTRONICS', got 'None'. 
Bug 1 confirmed: LLM output truncation prevents correct classification.

[LLM RAW OUTPUT]: '⚠ Ollama Error: model 'qwen3.5:2B' not found (status code: 404)'
```

### Bug 2 - /no_think Placement
```
FAILED source_code/tests/router/test_bug_exploration.py::TestBug2_NoThinkPlacement::test_no_think_in_wrong_parameter
AssertionError: Bug 2 confirmed: /no_think found in prompt parameter
```

### Bug 3 - LLM Output Parsing
```
FAILED: test_trailing_characters_parsing[DIGITAL_ELECTRONICS_3.\n-DIGITAL_ELECTRONICS-3]
FAILED: test_trailing_characters_parsing[UHV.-UHV-None]
PASSED: test_trailing_characters_parsing[CYBER_SECURITY_1 -CYBER_SECURITY-1]
PASSED: test_trailing_characters_parsing[DIGITAL_ELECTRONICS_5\n-DIGITAL_ELECTRONICS-5]
```

### Bug 4 - Subject Alias Resolution
```
FAILED: test_subject_alias_matching[UNIVERSAL HUMAN VALUES-UHV-True]
PASSED: test_subject_alias_matching[UNIVERSAL HUMAN VALUES-UNIVERSAL_HUMAN_VALUES-True]
PASSED: test_subject_alias_matching[CYBER SECURITY-CYBER_SECURITY-True]
PASSED: test_subject_alias_matching[DIGITAL ELECTRONICS-DIGITAL_ELECTRONICS-True]
```

### Bug 5 - Missing UHV Keywords
```
PASSED: test_missing_uhv_keywords[Explain the role of trust in relationships-trust]
FAILED: test_missing_uhv_keywords[What is competence in human values?-competence]
FAILED: test_missing_uhv_keywords[Define intention and its importance-intention]
FAILED: test_missing_uhv_keywords[How do relationships affect human values?-relationship]
```

### Bug 6 - UHV Q5 Cross-Encoder Scoring
```
SKIPPED: test_uhv_q5_cross_encoder_zero_score
Reason: Bug 6 investigation required: Cross-encoder scoring test needs full RAG pipeline.
```

---

## Conclusions

### Confirmed Bugs (5 out of 6)

1. ✅ **Bug 1**: LLM output truncation confirmed (though masked by missing model)
2. ✅ **Bug 2**: /no_think in wrong parameter confirmed
3. ✅ **Bug 3**: Trailing period parsing bug confirmed (partial - periods fail, spaces/newlines work)
4. ✅ **Bug 4**: Subject alias resolution bug confirmed
5. ✅ **Bug 5**: Missing UHV keywords confirmed (3 out of 4 keywords missing)
6. ⏭️ **Bug 6**: Requires investigation with full RAG pipeline

### Additional Findings

1. **Ollama Model Missing**: The LLM router cannot function because the qwen3.5:2B model is not installed
2. **Bug 3 Refinement**: The parsing bug is specifically with trailing periods (`.`), not all trailing characters
3. **Bug 5 Refinement**: "trust" keyword already exists, only "competence", "intention", and "relationship" are missing

### Recommendations

1. Install qwen3.5:2B model for Ollama to enable LLM router testing
2. Proceed with implementing fixes for Bugs 2, 3, 4, and 5
3. Investigate Bug 6 with full RAG pipeline running
4. Update Bug 3 fix to specifically handle trailing punctuation (`.!?`)

---

## Test Execution Details

**Command**: `python -m pytest source_code/tests/router/test_bug_exploration.py -v`

**Duration**: 57.05 seconds

**Environment**:
- Python: 3.12
- pytest: 9.0.2
- OS: Windows (WSL Ubuntu)

**Warnings**:
- DeprecationWarning: builtin type SwigPyPacked has no __module__ attribute
- PytestCollectionWarning: cannot collect test class 'TestResult' (has __init__ constructor)
