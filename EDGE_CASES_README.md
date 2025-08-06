# Edge Case Handling for RAG System

## Overview

This document describes the edge case handling system implemented to address the problem of information being split across multiple chunks in a RAG (Retrieval-Augmented Generation) system.

## Problem Statement

### The Multi-Chunk Information Problem

When documents are chunked for vector storage, information needed to answer a question may be split across multiple chunks. This leads to:

1. **Incomplete Answers**: The system only retrieves partial information
2. **Incorrect Answers**: Missing context leads to wrong conclusions
3. **Low Confidence**: Similarity scores may be low due to fragmented information

### Example Scenario

**Original Document:**
```
A社の売上がB社より高い、その差が1000万円。B社の売り上げがC社より高い、その差が500万円。A社の売上が5000万円でした。
```

**After Chunking:**
- Chunk 1: "A社の売上がB社より高い"
- Chunk 2: "その差が1000万円"
- Chunk 3: "B社の売上がC社より高い"
- Chunk 4: "その差が500万円"
- Chunk 5: "A社の売上が5000万円でした"

**User Query:** "ABC 3社の売上が最も高いのはどれですか？"

**Problem:** No single chunk contains enough information to answer the question completely.

## Solution Strategies

### Strategy 1: Full Document Context

**Approach:** Retrieve the complete document context containing the search hits.

**Process:**
1. Get top-k similar chunks
2. Group chunks by source document
3. Retrieve all chunks from the most relevant document
4. Extract evidence from the full context using LLM

**Advantages:**
- Provides complete context
- Maintains document coherence
- High accuracy for document-level questions

**Disadvantages:**
- May include irrelevant information
- Higher computational cost
- Requires document-level metadata

### Strategy 2: Adaptive Chunking

**Approach:** Start with large chunks and refine if needed.

**Process:**
1. Try with current chunking strategy
2. Check if context can answer the question using LLM
3. If insufficient, try with larger context
4. Iteratively refine until sufficient information is found

**Advantages:**
- Adaptive to question complexity
- Balances precision and recall
- Handles varying information needs

**Disadvantages:**
- Multiple LLM calls required
- May not find optimal chunk size
- Higher latency

### Strategy 3: Multi-Chunk Aggregation

**Approach:** Aggregate information from multiple high-similarity chunks.

**Process:**
1. Get top-k chunks with similarity scores
2. Filter chunks above similarity threshold
3. Combine relevant chunks
4. Extract evidence from combined context

**Advantages:**
- Handles distributed information
- Uses similarity scores for filtering
- Efficient for multi-topic questions

**Disadvantages:**
- May miss low-similarity but relevant chunks
- Requires careful threshold tuning
- Potential information redundancy

## Implementation

### Core Classes

#### `ChunkInfo`
```python
@dataclass
class ChunkInfo:
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    start_char: int = 0
    end_char: int = 0
```

#### `EvidenceResult`
```python
@dataclass
class EvidenceResult:
    evidence_text: str
    source_chunks: List[ChunkInfo]
    confidence_score: float
    strategy_used: str
    is_complete: bool
```

#### `EdgeCaseHandler`
Main class that implements all three strategies and provides automatic strategy selection.

### Key Methods

#### `handle_edge_case(query: str, strategy: str = "auto")`
Main entry point that automatically selects and applies the best strategy.

#### `_check_if_context_can_answer(query: str, context: str)`
Uses LLM to determine if the provided context can answer the question.

#### `_extract_evidence_from_context(query: str, context: str)`
Uses LLM to extract relevant evidence from the context.

## Usage

### Basic Usage

```python
from edge_case_handler import EdgeCaseHandler

# Initialize handler
edge_handler = EdgeCaseHandler(openai_api_key, chroma_path)

# Handle edge case automatically
result = edge_handler.handle_edge_case("Your question here")

# Use specific strategy
result = edge_handler.handle_edge_case("Your question here", strategy="strategy_1")
```

### Integration with RAG System

The edge case handler is integrated into the main RAG system (`rag.py`) and is automatically triggered when:

1. No results are found (empty result set)
2. Similarity scores are below threshold (0.3)
3. User explicitly requests edge case handling

### Testing

Run the test suite to verify edge case handling:

```bash
python3 test_edge_cases.py
```

## Configuration

### Similarity Thresholds

- **High similarity threshold**: 0.7 (for Strategy 3)
- **Low similarity threshold**: 0.3 (for triggering edge case handling)

### Chunking Parameters

- **Default chunk size**: 300 characters
- **Chunk overlap**: 100 characters
- **Batch size**: 5000 chunks per batch

### LLM Parameters

- **Model**: GPT-3.5-turbo
- **Max tokens**: 500 for evidence extraction
- **Temperature**: Default (0.7)

## Performance Considerations

### Computational Cost

1. **Strategy 1**: O(n) where n is document chunks
2. **Strategy 2**: O(m) where m is number of LLM calls
3. **Strategy 3**: O(k) where k is number of high-similarity chunks

### Memory Usage

- **Chunk storage**: ~1KB per chunk
- **Evidence extraction**: Temporary storage for context
- **LLM calls**: Minimal memory overhead

### Latency

- **Strategy 1**: ~2-5 seconds
- **Strategy 2**: ~3-8 seconds (multiple LLM calls)
- **Strategy 3**: ~1-3 seconds

## Best Practices

### When to Use Each Strategy

1. **Strategy 1**: Document-level questions, complete context needed
2. **Strategy 2**: Complex questions, adaptive approach preferred
3. **Strategy 3**: Multi-topic questions, efficiency important

### Threshold Tuning

- Adjust similarity thresholds based on your dataset
- Monitor false positives/negatives
- Consider domain-specific requirements

### Error Handling

- Always implement fallback mechanisms
- Log strategy selection and performance
- Monitor LLM API usage and costs

## Future Enhancements

### Planned Improvements

1. **Hybrid Strategies**: Combine multiple strategies for better results
2. **Learning-based Selection**: Use ML to predict optimal strategy
3. **Caching**: Cache evidence extraction results
4. **Parallel Processing**: Process multiple strategies concurrently

### Research Directions

1. **Dynamic Chunking**: Adaptive chunk sizes based on content
2. **Semantic Chunking**: Chunk based on semantic boundaries
3. **Cross-document Evidence**: Evidence spanning multiple documents
4. **Confidence Calibration**: Better confidence scoring

## Troubleshooting

### Common Issues

1. **No evidence found**: Check similarity thresholds and chunking
2. **Incomplete answers**: Try Strategy 1 or 2
3. **High latency**: Use Strategy 3 or reduce LLM calls
4. **Low confidence**: Adjust thresholds or improve chunking

### Debug Mode

Enable debug output by setting environment variable:
```bash
export RAG_DEBUG=1
```

## Conclusion

The edge case handling system provides robust solutions for the multi-chunk information problem in RAG systems. By implementing three complementary strategies and automatic selection, the system can handle complex queries that require information from multiple chunks.

The system is designed to be:
- **Robust**: Handles various edge cases
- **Efficient**: Minimizes unnecessary processing
- **Accurate**: Provides complete and correct answers
- **Extensible**: Easy to add new strategies

For questions or issues, please refer to the test suite or contact the development team. 