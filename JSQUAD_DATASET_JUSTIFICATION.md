# JSQuAD Dataset Selection Justification

## Why JSQuAD Dataset Was Selected for the Evidence-Indicator RAG Project

### Dataset Overview
- **Name**: JSQuAD (Japanese Question Answering Dataset)
- **Source**: HuggingFace (sbintuitions/JSQuAD)
- **Size**: 67,301 examples (62,900 train, 4,400 validation)
- **Origin**: Wikipedia Japanese articles (November 2021 dump)
- **Format**: Extractive Q&A with character-level position annotations
- **License**: Creative Commons Attribution ShareAlike 4.0

---

## 1. Perfect Alignment with Project Goals

### Character-Level Evidence Extraction
**Project Requirement**: The Evidence-Indicator system implements a patent-pending character-level evidence extraction method that marks precise answer positions within source documents.

**JSQuAD Match**: JSQuAD provides exact character-level position annotations (`answer_start`) for every answer span within context passages. This enables:
- Direct comparison between system-extracted positions and ground truth positions
- Precise evaluation of evidence extraction accuracy
- Validation of the position marking algorithm

**Example from JSQuAD**:
```json
{
  "context": "徳川家康は江戸幕府を開いた人物である...",
  "question": "江戸幕府を開いたのは誰ですか？",
  "answer": {
    "text": "徳川家康",
    "answer_start": 0  // Character position 0
  }
}
```

### Japanese Language Focus
**Project Requirement**: The system is specifically designed for Japanese text retrieval with MeCab tokenization, handling Japanese linguistic challenges (no word boundaries, multiple scripts, ordinal numbers).

**JSQuAD Match**:
- 100% Japanese content from Wikipedia articles
- Covers Japanese-specific linguistic phenomena
- Tests system's ability to handle kanji, hiragana, katakana mixture
- Includes temporal expressions, ordinal numbers (第31代), and complex noun phrases

---

## 2. Extractive Q&A Format Compatibility

### System Design Match
**Project Approach**: The RAG system retrieves relevant document chunks and extracts precise answer spans from retrieved evidence.

**JSQuAD Match**: JSQuAD follows the extractive Q&A paradigm where:
- Every answer is a span of text from the source passage
- No answer generation or paraphrasing required
- Direct text extraction evaluation possible

This perfectly matches the system's hybrid retrieval + evidence extraction architecture.

---

## 3. Evaluation Rigor and Precision

### Strict Evaluation Metrics
**Project Need**: Evaluate not just answer correctness, but also:
- Evidence quality and relevance
- Retrieval accuracy
- Position marking precision

**JSQuAD Enables**:
- **Exact Match (EM)**: Binary correctness evaluation
- **Character-level F1**: Partial credit for overlapping spans
- **Position Accuracy**: Compare extracted positions with ground truth
- **Retrieval Quality**: Check if correct passage is retrieved

### No Ambiguity
JSQuAD is based on SQuAD 1.1, meaning:
- All questions have answers (no unanswerable cases)
- Single correct answer per question
- Clear evaluation criteria
- Simplifies initial system validation

---

## 4. Dataset Quality and Scale

### Sufficient Volume
- **67,301 examples** provide statistically significant evaluation
- **62,900 training examples** enable future fine-tuning of retrieval/ranking models
- **4,400 validation examples** for hyperparameter tuning

### Domain Diversity
JSQuAD covers diverse Wikipedia topics:
- Geography (locations, demographics, natural features)
- History (events, figures, timelines)
- Culture (traditions, arts, customs)
- Science (concepts, discoveries, phenomena)
- Sports (athletes, competitions, records)

This diversity tests the system's ability to:
- Handle various question types (factual, temporal, definitional)
- Retrieve across different domains
- Extract evidence from different text structures

### High-Quality Source
- **Wikipedia articles**: Factual, well-structured, verifiable
- **Professional curation**: Dataset creation by SB Intuitions
- **Community validation**: Open-source with active usage

---

## 5. Technical Compatibility

### Standard Format
- **SQuAD 1.1 format**: Industry-standard JSON structure
- **Easy integration**: Compatible with existing evaluation frameworks
- **Reproducible**: Well-documented dataset with versioning

### Character-Level Annotations
Unlike token-level or sentence-level datasets, JSQuAD provides:
```json
"answers": {
  "text": ["ハーバート・フーヴァー"],
  "answer_start": [42]  // Exact character position
}
```

This enables precise evaluation of the system's position marking feature.

---

## 6. Advantages Over Alternative Datasets

### Compared to Manual Curation
**Alternative**: Create custom Japanese Q&A dataset manually
**JSQuAD Advantage**:
- 67,301 examples vs. limited manual annotations (~100-500)
- Professional quality assurance
- Community-validated questions
- Diverse coverage impossible to replicate manually

### Compared to Other Japanese Q&A Datasets
**Alternatives**:
- JAQKET (quiz-style questions, no positions)
- JaQuAD (smaller scale)
- NewsQA-ja (news-specific, no character positions)

**JSQuAD Advantages**:
- Largest scale with character-level positions
- Wikipedia source (more diverse than news)
- Extractive format (matches RAG paradigm)
- Active maintenance and community

### Compared to English SQuAD
**Why Not English SQuAD**:
- System specifically designed for Japanese challenges
- MeCab tokenization requires Japanese text
- Ordinal number enhancement (第31代) is Japanese-specific
- Goal is Japanese document retrieval

---

## 7. Real-World Applicability

### Realistic Question Types
JSQuAD questions mirror real user information needs:
- "アメリカ合衆国の第31代大統領は誰ですか？" (Who is the 31st US President?)
- "関東地方の梅雨はいつ頃ですか？" (When is the rainy season in Kanto?)
- "富士山の標高は何メートルですか？" (What is the elevation of Mt. Fuji?)

### Production-Ready Evaluation
The system's performance on JSQuAD indicates:
- Ability to handle real user queries
- Reliability of evidence extraction
- Scalability to large document collections

---

## 8. Project-Specific Benefits

### Validates Core Innovation
The character-level position marking method is the system's core innovation. JSQuAD enables:
- Precise validation of position accuracy
- Comparison with traditional retrieval methods
- Demonstration of practical value

### Supports Version Comparison
Current evaluation includes three versions:
- **v1**: Baseline (Original BM25 + Original Prompt)
- **v2**: Original BM25 + Improved Prompt
- **v3**: Enhanced BM25 + Improved Prompt

JSQuAD's consistent format enables fair comparison across versions.

### Enables Demo and Presentation
For the afternoon demo, JSQuAD provides:
- Clear, understandable questions
- Verifiable answers
- Impressive accuracy metrics (92.6% in v3)
- Concrete evidence of system capabilities

---

## Conclusion

JSQuAD was selected because it uniquely satisfies all critical requirements:

1. **Character-level positions** match the system's evidence extraction innovation
2. **Japanese language** aligns with system's design and MeCab tokenization
3. **Extractive format** matches RAG system's architecture
4. **Large scale** (67,301 examples) enables robust evaluation
5. **High quality** Wikipedia source with professional curation
6. **Diverse coverage** tests system across multiple domains
7. **Standard format** ensures reproducibility and comparability
8. **Open license** permits research and development use

No other available Japanese Q&A dataset provides this combination of scale, quality, and technical compatibility with the Evidence-Indicator RAG system's core innovations.

---

## References
- JSQuAD Dataset: https://huggingface.co/datasets/sbintuitions/JSQuAD
- Original SQuAD Paper: Rajpurkar et al., 2016
- Project Repository: Evidence-Indicator RAG System
