# 🔍 Evidence Indicator RAG System - Streamlit Frontend

A beautiful and interactive web interface for the Evidence Indicator RAG System, built with Streamlit.

## 🚀 Features

- **Interactive Query Interface**: Easy-to-use text input for queries
- **Real-time Results**: Instant display of RAG system responses
- **Performance Monitoring**: Track query processing times and system performance
- **Query History**: Save and reuse previous queries
- **Beautiful UI**: Modern, responsive design with Japanese language support
- **Quick Query Buttons**: Pre-defined queries for quick testing
- **Performance Charts**: Visualize system performance over time
- **🎯 Enhanced Japanese Support**: New Ichikara dataset with rich metadata and reference validation
- **📚 Instruction-Response Pairs**: Structured Q&A format for better query understanding
- **🔗 Source Verification**: Built-in reference validation and attribution

## 📋 Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Evidence Indicator RAG System backend (optional)

## 🛠️ Installation

1. **Clone or navigate to the frontend directory:**
   ```bash
   cd rag-streamlit-frontend
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv streamlit-env
   source streamlit-env/bin/activate  # On Windows: streamlit-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Basic Usage

1. **Start the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

3. **Start querying:**
   - Type your question in the text area
   - Click "🔍 検索実行" to submit
   - View results in the beautiful interface

### Advanced Usage

- **Quick Queries**: Use the pre-defined query buttons for quick testing
- **Query History**: Access previous queries from the sidebar
- **Performance Monitoring**: View real-time performance metrics and charts
- **API Configuration**: Configure backend API settings in the sidebar
- **🎯 Japanese Queries**: Try Japanese language queries like "上高地について教えて"
- **🔍 Enhanced Search**: Leverage the new Ichikara dataset for better results

## 📊 Features Overview

### Main Interface
- **Query Input**: Large text area for entering questions
- **Quick Query Buttons**: Pre-defined queries for testing
- **Results Display**: Beautifully formatted results with:
  - 【回答】: LLM-generated answer
  - 【検索ヒットのチャンクを含む文書】: Complete source document
  - 【根拠情報】: Extracted evidence with character positions
  - 【参考情報】: Source references and timestamps (new!)
  - 【更新日時】: Content update timestamps (new!)

### Sidebar Features
- **API Configuration**: Set backend API URL
- **Query History**: Access and reuse previous queries
- **Quick Actions**: Clear history and reset metrics
- **Statistics**: View query counts and performance metrics

### Performance Monitoring
- **Real-time Metrics**: Processing time tracking
- **Performance Charts**: Visual performance trends
- **Statistics Panel**: Query counts and average processing times

## 🆕 New: Ichikara Dataset Integration

### What's New
The system now includes the **Ichikara RAG Dataset** (`ichikara-rag-sampleToMF.json`), providing:

- **🏔️ Tourism Content**: Detailed guides about Japanese destinations
- **🏥 Medical Information**: Traditional medicine and health advice  
- **💕 Lifestyle Guidance**: Dating and relationship advice
- **📚 Educational Materials**: Structured instructional content

### Enhanced Capabilities
- **Instruction-Response Pairs**: Better understanding of user queries
- **Rich Metadata**: References, timestamps, and source validation
- **Japanese Language Expertise**: High-quality Japanese content
- **Reference Verification**: Built-in source attribution

### Quick Start with Ichikara Dataset

1. **Test the integration:**
   ```bash
   python script/ichikara_dataset_integration.py
   ```

2. **Try Japanese queries:**
   - "上高地について教えて" (Tell me about Kamikochi)
   - "観光地のアクセス方法は？" (How to access tourist spots?)
   - "伝統的な治療法について" (About traditional treatments)

3. **View enhanced results** with source references and timestamps

For detailed integration instructions, see: [📖 Ichikara Integration Guide](docs/ICHIKARA_INTEGRATION_GUIDE.md)

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# API Configuration
API_BASE_URL=http://localhost:8000
API_TIMEOUT=30

# Streamlit Configuration  
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost

# OpenAI Configuration (for Ichikara dataset)
OPENAI_API_KEY=your_openai_api_key_here
```

### Ichikara Dataset Configuration

The new dataset can be configured through `config/ichikara_config.py`:

```python
# Customize chunk sizes, search parameters, and quality settings
from config.ichikara_config import get_config, get_chunk_settings

chunk_settings = get_chunk_settings()
chunk_settings["chunk_size"] = 400  # Larger chunks
chunk_settings["chunk_overlap"] = 150  # More overlap
```

## 🔧 System Architecture

### Core Components
- **Streamlit Frontend**: Beautiful web interface
- **UltraFastRAG**: High-performance RAG engine
- **ChromaDB**: Vector database for embeddings
- **OpenAI Embeddings**: Text vectorization
- **🎯 Ichikara Integrator**: Enhanced dataset processor

### Data Flow
1. **User Query** → Streamlit Interface
2. **Query Processing** → Enhanced RAG System
3. **Vector Search** → ChromaDB + Ichikara Dataset
4. **Result Generation** → Rich metadata + source validation
5. **Response Display** → Beautiful UI with references

## 📈 Performance & Quality

### Enhanced Metrics
- **Query Processing Time**: Real-time performance tracking
- **Result Quality**: Metadata richness and reference validation
- **Content Coverage**: Japanese language expertise expansion
- **Source Reliability**: Built-in verification capabilities

### Quality Assurance
- **Content Validation**: JSON structure and content quality checks
- **Reference Verification**: Source URL validation and timestamp tracking
- **Metadata Enrichment**: Enhanced search and filtering capabilities
- **Performance Optimization**: Batch processing and caching

## 🚨 Troubleshooting

### Common Issues

1. **Dataset Integration Problems**
   ```bash
   # Validate configuration
   python config/ichikara_config.py
   
   # Test integration
   python script/ichikara_dataset_integration.py
   ```

2. **Performance Issues**
   - Check chunk size settings in configuration
   - Monitor memory usage during large dataset processing
   - Verify OpenAI API key and rate limits

3. **Japanese Content Issues**
   - Ensure proper UTF-8 encoding
   - Check Japanese language processing settings
   - Validate text chunking parameters

### Getting Help

- **📖 Documentation**: Check the integration guide
- **🔧 Configuration**: Review `config/ichikara_config.py`
- **🐛 Issues**: Check GitHub repository for known problems
- **💬 Support**: Review troubleshooting section in integration guide

## 🔮 Future Roadmap

### Upcoming Features
- **Multi-language Expansion**: Beyond Japanese content
- **Advanced Analytics**: Deep insights into dataset usage
- **Automated Updates**: Sync with source content changes
- **Confidence Scoring**: Reliability metrics for responses

### Integration Enhancements
- **Unified Query Interface**: Seamless system integration
- **Advanced Metadata Search**: Enhanced filtering capabilities
- **Performance Optimization**: Further speed improvements
- **Quality Metrics**: Automated content assessment

## 📚 Additional Resources

- **🎯 Ichikara Integration Guide**: [docs/ICHIKARA_INTEGRATION_GUIDE.md](docs/ICHIKARA_INTEGRATION_GUIDE.md)
- **🔧 Configuration Reference**: [config/ichikara_config.py](config/ichikara_config.py)
- **🚀 Integration Scripts**: [script/ichikara_dataset_integration.py](script/ichikara_dataset_integration.py)
- **📊 Dataset Analysis**: See the comprehensive dataset exploration above

## 🤝 Contributing

We welcome contributions to enhance the system:

1. **Dataset Improvements**: Add more content categories
2. **Performance Optimization**: Enhance processing speed
3. **Feature Development**: New capabilities and integrations
4. **Documentation**: Improve guides and examples

---

**🚀 Ready to explore the enhanced Japanese RAG system? Start with the Ichikara dataset integration! 🇯🇵✨**