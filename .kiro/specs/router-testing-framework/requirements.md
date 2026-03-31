# Requirements Document

## Introduction

This document specifies requirements for a comprehensive router testing and debugging system. The system will provide structured debug tracing, standardized output formats, automated testing capabilities, and validation mechanisms for the multi-stage routing pipeline (regex, keyword, embedding, and LLM routing strategies).

## Glossary

- **Router_System**: The hybrid routing system that determines subject and unit classification for user queries
- **Debug_Trace**: A structured data object capturing execution details from all routing stages
- **Routing_Stage**: One of four classification methods: regex, keyword, embedding, or LLM
- **Test_Runner**: The automated batch testing component that executes test cases
- **Output_Parser**: The component that extracts subject and unit information from router outputs
- **Test_Dataset**: A collection of test cases with expected routing outcomes
- **Validation_System**: The component that verifies router outputs against expected results

## Requirements

### Requirement 1: Debug Trace Data Structure

**User Story:** As a developer, I want a standardized debug trace structure, so that I can analyze the complete routing pipeline execution for any query.

#### Acceptance Criteria

1. THE Debug_Trace SHALL capture the original query text
2. THE Debug_Trace SHALL record the final routing decision including subject and unit
3. THE Debug_Trace SHALL store execution details from each Routing_Stage attempted
4. FOR EACH Routing_Stage, THE Debug_Trace SHALL record the stage name, confidence score, and routing result
5. THE Debug_Trace SHALL include timing information for each Routing_Stage
6. THE Debug_Trace SHALL record which Routing_Stage produced the final decision
7. THE Debug_Trace SHALL be serializable to JSON format

### Requirement 2: Standardized Output Format

**User Story:** As a developer, I want consistent output formats across all routing stages, so that I can reliably parse and compare results.

#### Acceptance Criteria

1. THE Router_System SHALL return subject and unit as separate fields
2. WHEN a Routing_Stage produces a result, THE Router_System SHALL include a confidence score between 0.0 and 1.0
3. THE Router_System SHALL identify which Routing_Stage produced each result
4. WHEN no subject is detected, THE Router_System SHALL return null for the subject field
5. WHEN no unit is detected, THE Router_System SHALL return null for the unit field
6. THE Router_System SHALL format unit identifiers as numeric strings

### Requirement 3: Output Parsing

**User Story:** As a developer, I want to parse router outputs automatically, so that I can extract subject and unit information for validation.

#### Acceptance Criteria

1. THE Output_Parser SHALL extract subject names from router responses
2. THE Output_Parser SHALL extract unit numbers from router responses
3. WHEN parsing fails, THE Output_Parser SHALL return null values with an error indicator
4. THE Output_Parser SHALL handle both combined format (SUBJECT_UNIT) and separate format outputs
5. THE Output_Parser SHALL normalize subject names to match the keyword map format

### Requirement 4: LLM Router Enhancement

**User Story:** As a developer, I want enhanced LLM router logging, so that I can debug LLM-based routing decisions.

#### Acceptance Criteria

1. WHEN the LLM Routing_Stage executes, THE Router_System SHALL log the model name used
2. WHEN the LLM Routing_Stage executes, THE Router_System SHALL log the complete prompt sent
3. WHEN the LLM Routing_Stage executes, THE Router_System SHALL log the raw response received
4. WHEN the LLM Routing_Stage executes, THE Router_System SHALL store this information in the Debug_Trace
5. THE Router_System SHALL record LLM response parsing steps in the Debug_Trace

### Requirement 5: Full Pipeline Debug Integration

**User Story:** As a developer, I want debug tracing integrated into the complete routing pipeline, so that I can trace execution through all stages.

#### Acceptance Criteria

1. WHEN a query is routed, THE Router_System SHALL create a Debug_Trace object
2. THE Router_System SHALL populate the Debug_Trace with data from each Routing_Stage executed
3. WHEN a Routing_Stage succeeds, THE Router_System SHALL record the success in the Debug_Trace and skip remaining stages
4. WHEN a Routing_Stage fails, THE Router_System SHALL record the failure in the Debug_Trace and proceed to the next stage
5. THE Router_System SHALL return the Debug_Trace along with the routing result

### Requirement 6: Final Decision Logic

**User Story:** As a developer, I want clear final decision logic, so that I can understand how the router selects between multiple stage results.

#### Acceptance Criteria

1. THE Router_System SHALL attempt stages in the order: regex, keyword, embedding, LLM
2. WHEN a Routing_Stage produces a result with confidence above its threshold, THE Router_System SHALL use that result as the final decision
3. WHEN explicit unit detection via regex succeeds, THE Router_System SHALL override unit results from other stages
4. WHEN a session subject is provided, THE Router_System SHALL override subject results from all stages
5. WHEN no Routing_Stage produces a valid result, THE Router_System SHALL return null for both subject and unit

### Requirement 7: Test Dataset Structure

**User Story:** As a developer, I want a comprehensive test dataset, so that I can validate router behavior across diverse query types.

#### Acceptance Criteria

1. THE Test_Dataset SHALL include test cases for explicit unit mentions
2. THE Test_Dataset SHALL include test cases for subject-specific technical terms
3. THE Test_Dataset SHALL include test cases for ambiguous queries requiring LLM routing
4. THE Test_Dataset SHALL include test cases for generic non-syllabus queries
5. FOR EACH test case, THE Test_Dataset SHALL specify the expected subject
6. FOR EACH test case, THE Test_Dataset SHALL specify the expected unit when applicable
7. FOR EACH test case, THE Test_Dataset SHALL specify the expected routing method when applicable
8. THE Test_Dataset SHALL be stored in JSON format

### Requirement 8: Batch Test Runner

**User Story:** As a developer, I want automated batch testing, so that I can validate router performance across all test cases efficiently.

#### Acceptance Criteria

1. THE Test_Runner SHALL load test cases from the Test_Dataset
2. FOR EACH test case, THE Test_Runner SHALL execute the Router_System
3. FOR EACH test case, THE Test_Runner SHALL compare actual results against expected results
4. THE Test_Runner SHALL calculate overall routing accuracy as a percentage
5. THE Test_Runner SHALL calculate subject detection accuracy separately
6. THE Test_Runner SHALL calculate unit detection accuracy separately
7. THE Test_Runner SHALL report the distribution of routing methods used
8. WHEN a test case fails, THE Test_Runner SHALL output the query, expected result, and actual result
9. THE Test_Runner SHALL support running a subset of test cases via filtering

### Requirement 9: Debug Logging and Persistence

**User Story:** As a developer, I want debug traces saved to files, so that I can analyze routing behavior offline.

#### Acceptance Criteria

1. THE Validation_System SHALL save Debug_Trace objects to JSON files
2. THE Validation_System SHALL organize debug files by timestamp
3. THE Validation_System SHALL include the test case identifier in the debug filename
4. WHEN batch testing completes, THE Validation_System SHALL generate a summary report file
5. THE summary report SHALL include accuracy metrics for all routing stages
6. THE summary report SHALL list all failed test cases with details

### Requirement 10: System Validation

**User Story:** As a developer, I want validation that the system produces correct outputs, so that I can trust the router in production.

#### Acceptance Criteria

1. THE Validation_System SHALL verify that all Routing_Stage outputs conform to the standardized format
2. THE Validation_System SHALL verify that confidence scores are within the valid range
3. THE Validation_System SHALL verify that unit identifiers are numeric strings when present
4. THE Validation_System SHALL verify that subject names match entries in the keyword map
5. WHEN validation fails, THE Validation_System SHALL report the specific validation error
6. THE Validation_System SHALL support validation of individual routing results and batch results

### Requirement 11: Parser and Serializer Round-Trip Testing

**User Story:** As a developer, I want round-trip testing for debug trace serialization, so that I can ensure no data loss during persistence.

#### Acceptance Criteria

1. WHEN a Debug_Trace is serialized to JSON, THE Router_System SHALL produce valid JSON
2. THE Router_System SHALL provide a deserializer that reconstructs Debug_Trace objects from JSON
3. FOR ALL valid Debug_Trace objects, serializing then deserializing SHALL produce an equivalent object
4. WHEN deserialization fails, THE Router_System SHALL return a descriptive error message
5. THE Test_Runner SHALL include round-trip tests for Debug_Trace serialization
