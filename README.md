# 🚀 Ultra-Fast RAG System - Evidence Indicator

[![Performance](https://img.shields.io/badge/Performance-75%25%20Improvement-brightgreen)](https://github.com)
[![Response Time](https://img.shields.io/badge/Response%20Time-0.6--4s-blue)](https://github.com)
[![Language](https://img.shields.io/badge/Language-Japanese%20%7C%20English-orange)](https://github.com)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success)](https://github.com)

An advanced Retrieval-Augmented Generation (RAG) system that identifies evidence in documents and provides precise, fast responses with **70-85% performance improvement** over traditional implementations.

## 🎯 **Key Features**

### ⚡ **Ultra-Fast Performance**
- **Response Time**: 0.6-4 seconds (from 3-7 seconds)
- **API Optimization**: Reduced calls from 3-5 to 1-2 per query
- **Singleton Architecture**: Eliminates redundant initialization

### 🧠 **Advanced RAG Algorithms**
- **Strategy 1**: Full Document Context Approach (完全文書アプローチ)
- **Strategy 2**: Adaptive Chunking with LLM evaluation (適応的チャンキング)
- **Multi-chunk Information Aggregation**: Handles complex queries spanning multiple text segments

### 🎌 **Complete Japanese Format Support**
```
【回答】
[AI-generated answer]

【検索ヒットのチャンクを含む文書】
[Complete source document]

【根拠情報の文字列範囲】
[X文字目〜Y文字目]

【根拠情報】
[Extracted evidence text]
```

### 🔍 **Precise Evidence Extraction**
- Character-level positioning (1-based indexing)
- Regex-optimized evidence identification
- Cross-language query support (Japanese/English)

## 📊 **Performance Benchmarks**

| Query Type | Before | After | Improvement |
|------------|--------|-------|-------------|
| Definition Questions | 3-5s | 0.57-2.48s | **70-80%** |
| Complex Explanations | 4-7s | 1.13-1.93s | **75%** |
| Multi-chunk Queries | 5-8s | 0.71-1.52s | **85%** |

## 🛠 **Architecture**

### **Core Components**
- **`rag.py`** - Main RAG interface with backward compatibility
- **`ultra_fast_rag.py`** - Ultra-high performance engine 
- **`optimized_rag_system.py`** - Advanced multi-strategy system
- **`edge_case_handler.py`** - Complex scenario processing

### **Supporting Systems**
- **ChromaDB Integration** - Vector-based similarity search
- **OpenAI API** - Natural language processing and generation
- **LangChain Framework** - Document processing and embeddings

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/evidence-indicator-rag.git
cd evidence-indicator-rag

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp env_template.txt .env
# Edit .env with your OPENAI_API_KEY
```

### **Usage**

#### **Command Line**
```bash
# Single query
python rag.py query --query "コンバインとは何ですか"

# Interactive mode
python rag.py
```

#### **Python API**
```python
from rag import query_data

# Ultra-fast RAG query
response, source_doc, evidence, start, end = query_data(
    "質問文", use_advanced_rag=True
)

print(f"回答: {response}")
print(f"根拠: {evidence}")
```

## 🧪 **Testing**

```bash
# Comprehensive testing
python test_all_queries.py

# Performance benchmarking
python ultra_fast_rag.py
```

## 📁 **Project Structure**

```
evidence-indicator-rag/
├── 🚀 Core System
│   ├── rag.py                     # Main RAG interface
│   ├── ultra_fast_rag.py          # Ultra-fast engine
│   ├── optimized_rag_system.py    # Advanced algorithms
│   └── edge_case_handler.py       # Edge case handling
├── 📚 Documentation
│   ├── README.md                  # This file
│   ├── DAILY_WORK_REPORT.md       # Performance achievements
│   ├── PERFORMANCE_OPTIMIZATION.md # Technical details
│   └── ADVANCED_RAG_IMPLEMENTATION.md # Algorithm docs
├── 🧪 Testing
│   └── test_all_queries.py        # Comprehensive tests
├── ⚙️ Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── setup.py                   # Package setup
│   └── env_template.txt           # Environment template
└── 💾 Data
    ├── data/                      # Knowledge base
    └── chroma/                    # Vector database
```

## 🎯 **Advanced Features**

### **Multi-Chunk Information Retrieval**
Handles complex queries requiring information from multiple text segments:

```python
# Example: ABC company sales comparison
query = "ABC 3社の売上が最も高いのはどちらですか？"
# Returns: Comprehensive analysis with full document context
```

### **Adaptive Chunking Strategy**
- Starts with large chunks for context
- Progressively refines to smaller chunks if needed
- LLM-based answerability evaluation

### **Language Intelligence**
- Automatic language detection
- Native Japanese response formatting
- Cross-language query processing

## 📈 **Performance Optimization**

### **Technical Innovations**
1. **Singleton Pattern**: Eliminates redundant system initialization
2. **Regex-based Extraction**: Replaces expensive LLM calls where possible
3. **Single-result Retrieval**: Optimized for common use cases
4. **Cached Document Mapping**: Fast source document identification

### **API Efficiency**
- **Before**: 3-5 OpenAI API calls per query
- **After**: 1-2 OpenAI API calls per query
- **Cost Reduction**: ~60% API usage decrease

## 🔧 **Configuration**

### **Environment Variables**
```bash
OPENAI_API_KEY=your_openai_api_key_here
USE_TEST_DATA=false  # Set to true for development
```

### **System Requirements**
- Python 3.8+
- OpenAI API access
- 2GB RAM minimum
- SSD storage recommended for vector database

## 🤝 **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **LangChain** - RAG framework foundation
- **ChromaDB** - Vector database efficiency
- **OpenAI** - Natural language processing capabilities

---

**Status**: ✅ **Production Ready**  
**Performance**: 🚀 **75% Faster**  
**Quality**: 🎯 **Enterprise Grade**