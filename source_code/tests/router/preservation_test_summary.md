# Preservation Property Tests Summary

## Overview

This document summarizes the preservation property tests created for the router-system-fixes bugfix spec. These tests capture existing correct behavior that must be preserved after implementing the 6 bug fixes.

## Test Execution Results

**Status**: ✅ All tests PASS on unfixed code (34 passed, 1 skipped)
**Date**: 2025-03-30
**Test File**: `source_code/tests/router/test_preservation.py`

## Test Coverage

### 1. Short Subject Name Preservation (Requirements 3.1)

**Tests**: 6 tests (5 parametrized + 1 property-based)

**Behavior Preserved**:
- Queries with short subject names (≤20 chars) classify correctly via keyword routing
- Tested subjects: CYBER_SECURITY, UHV
- Tested keywords: phishing, self-exploration, mutual fulfillment, sanskar, social engineering

**Key Assertions**:
- Subject detection succeeds for queries with existing keywords
- Keyword routing method is used (not LLM/embedding fallback)
- Subject names remain short (≤20 characters)

### 2. Routing Hierarchy Preservation (Requirements 3.2)

**Tests**: 3 tests

**Behavior Preserved**:
- Keyword router is tried before embedding router
- Routing hierarchy order: regex → keyword → embedding → LLM
- Explicit unit mentions (regex) override other unit detection methods

**Key Assertions**:
- Keyword routing takes precedence when keywords match
- Explicit "Unit 3" mentions are detected via regex
- Routing methods execute in correct order

### 3. Clean LLM Response Preservation (Requirements 3.3)

**Tests**: 4 parametrized tests

**Behavior Preserved**:
- LLM responses without trailing characters parse correctly
- Subject and unit extraction works for clean responses
- Tested responses: DIGITAL_ELECTRONICS_3, CYBER_SECURITY_1, etc.

**Key Assertions**:
- Clean responses like "DIGITAL_ELECTRONICS_3" parse to subject="DIGITAL_ELECTRONICS", unit="3"
- No trailing punctuation/whitespace in test cases
- Parsing logic handles subject_unit format correctly

### 4. Exact Subject Match Preservation (Requirements 3.4)

**Tests**: 5 parametrized tests

**Behavior Preserved**:
- Exact subject name matches are recognized correctly
- Space-to-underscore conversion works (e.g., "CYBER SECURITY" → "CYBER_SECURITY")
- Direct string comparison succeeds for exact matches

**Key Assertions**:
- "CYBER_SECURITY" == "CYBER_SECURITY" (exact match)
- "CYBER SECURITY" normalizes to "CYBER_SECURITY" (space conversion)
- All three subjects tested: CYBER_SECURITY, DIGITAL_ELECTRONICS, UHV

### 5. Existing Keyword Preservation (Requirements 3.5)

**Tests**: 6 tests (5 parametrized + 1 property-based)

**Behavior Preserved**:
- Queries with existing keywords route correctly via keyword router
- Tested keywords: phishing, mutual fulfillment, self-exploration, sanskar, social engineering
- Keyword routing method is used consistently

**Key Assertions**:
- Keyword router detects correct subject for each query
- Method is "keyword" (not embedding/LLM fallback)
- All existing keywords continue to work

### 6. Non-UHV Cross-Encoder Preservation (Requirements 3.6)

**Tests**: 1 test (skipped - requires full RAG pipeline)

**Behavior Preserved**:
- Non-UHV queries route correctly via keyword routing
- Prerequisite for cross-encoder scoring is maintained

**Key Assertions**:
- CYBER_SECURITY queries route correctly
- DIGITAL_ELECTRONICS queries route correctly
- Full cross-encoder testing deferred to integration tests

### 7. Regex Unit Detection Preservation (Requirements 3.7)

**Tests**: 8 tests (5 direct + 3 full routing)

**Behavior Preserved**:
- Explicit unit mentions (e.g., "Unit 3") detected via regex
- Regex detection works in both direct calls and full routing pipeline
- Unit numbers 1-5 all tested

**Key Assertions**:
- "Unit 3" → unit="3"
- "unit 5" → unit="5" (case insensitive)
- Regex unit detection works in full routing context

### 8. Embedding Router Preservation (Requirements 3.8)

**Tests**: 2 tests

**Behavior Preserved**:
- Embedding router returns expected tuple format (subject, unit, score)
- System handles embedding unavailability gracefully
- Fallback routing works when embedding fails

**Key Assertions**:
- Embedding router returns (str|None, str|None, float|None)
- System doesn't crash when embedding unavailable
- Keyword routing works as fallback

## Property-Based Testing

The test suite includes property-based tests using Hypothesis for:
- Short subject name classification (20 examples)
- Existing keyword routing (20 examples)

These tests generate multiple test cases automatically to provide stronger guarantees about preserved behavior.

## Test Environment Notes

- **LLM Model**: qwen3.5:2B (not available in test environment)
- **Embedding Model**: May not be available due to memory constraints
- **Fallback**: Tests focus on keyword routing which works reliably
- **Integration Tests**: Full LLM/embedding/cross-encoder testing deferred to integration tests

## Conclusion

All preservation tests pass on the unfixed code, confirming the baseline behavior that must be preserved after implementing the 6 bug fixes. The tests cover:

- ✅ Keyword routing functionality
- ✅ Regex unit detection
- ✅ Subject name normalization
- ✅ Clean LLM response parsing
- ✅ Routing hierarchy order
- ✅ Embedding router interface

These tests will be re-run after each bug fix to ensure no regressions are introduced.

