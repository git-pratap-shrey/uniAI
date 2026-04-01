# Bugfix Requirements Document

## Introduction

The hybrid router system contains multiple critical bugs that prevent accurate subject/unit detection and cause LLM output truncation. These bugs affect the router's ability to correctly classify queries, leading to reduced subject detection accuracy (currently 10/15 instead of expected 13-14/15) and LLM router failures. The bugs span configuration issues, string parsing problems, and incomplete keyword mappings.

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN the LLM router is invoked with `num_predict=10` in config/rag.py THEN the system truncates LLM output to 10 tokens, cutting off subject names like "DIGITAL_ELECTRONICS_3" (20 characters) mid-string

1.2 WHEN hybrid_router.py appends "/no_think" to the user message in the prompt parameter THEN the qwen3.5:2b model receives "/no_think" as part of the query text instead of as a system instruction, causing confusion and incorrect classifications

1.3 WHEN hybrid_router._llm_classify_subject_unit parses LLM output using exact string matching THEN the system fails to match responses with trailing periods, newlines, or whitespace variations (e.g., "DIGITAL_ELECTRONICS_3." or "UHV\n")

1.4 WHEN reporter.py calculates subject accuracy by comparing "UNIVERSAL HUMAN VALUES" (from test data) against "UNIVERSAL_HUMAN_VALUES" (detected subject with underscores) THEN the system fails to match even though both refer to the same subject, and additionally fails when the stored subject code is "UHV"

1.5 WHEN a UHV query contains words like "intention", "competence", "trust", or "relationship" THEN the keyword router fails to detect the UHV subject because these common UHV terms are missing from the keyword map in source_code/data/subject_keywords.json

1.6 WHEN UHV Q5 is processed by the cross-encoder THEN the system returns a score of 0 despite keyword routing working correctly, indicating a potential issue with the cross-encoder scoring mechanism for certain UHV queries

### Expected Behavior (Correct)

2.1 WHEN the LLM router is invoked THEN the system SHALL set `num_predict` to a value sufficient to accommodate the longest subject_unit string (at least 30 tokens) to prevent truncation of valid responses

2.2 WHEN hybrid_router.py needs to suppress thinking behavior THEN the system SHALL pass "/no_think" as part of the system_prompt parameter or use a separate model configuration, not append it to the user query

2.3 WHEN hybrid_router._llm_classify_subject_unit parses LLM output THEN the system SHALL strip whitespace, newlines, and trailing punctuation before performing string matching to handle common LLM output variations

2.4 WHEN reporter.py calculates subject accuracy THEN the system SHALL normalize both expected and detected subject names by converting spaces to underscores and handling subject code aliases (e.g., "UHV" = "UNIVERSAL_HUMAN_VALUES") before comparison

2.5 WHEN a UHV query contains words like "intention", "competence", "trust", or "relationship" THEN the keyword router SHALL detect the UHV subject by including these terms in the UHV keyword map

2.6 WHEN UHV Q5 is processed by the cross-encoder THEN the system SHALL return a non-zero score that accurately reflects the semantic similarity between the query and retrieved chunks

### Unchanged Behavior (Regression Prevention)

3.1 WHEN the LLM router successfully classifies queries with short subject names (e.g., "UHV") THEN the system SHALL CONTINUE TO correctly parse and return these classifications

3.2 WHEN hybrid_router.py processes queries through keyword and embedding routers THEN the system SHALL CONTINUE TO execute the routing hierarchy (regex → keyword → embedding → LLM) in the correct order

3.3 WHEN the LLM router receives well-formed responses without trailing punctuation THEN the system SHALL CONTINUE TO parse them correctly

3.4 WHEN reporter.py calculates subject accuracy for queries where expected and detected subjects already match exactly THEN the system SHALL CONTINUE TO count these as correct matches

3.5 WHEN keyword routing works correctly for existing UHV keywords THEN the system SHALL CONTINUE TO detect UHV subject for queries containing those keywords

3.6 WHEN the cross-encoder processes non-UHV queries THEN the system SHALL CONTINUE TO return accurate similarity scores

3.7 WHEN queries contain explicit unit mentions (e.g., "Unit 3") THEN the system SHALL CONTINUE TO detect and extract the unit number via regex

3.8 WHEN the embedding router successfully classifies queries above the threshold THEN the system SHALL CONTINUE TO return the detected subject and unit before falling back to the LLM router
