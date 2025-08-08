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
- **Debug Mode**: Advanced debugging and error handling
- **API Health Monitoring**: Real-time backend API status checking
- **Fallback Simulation**: Works even when backend is unavailable

## 📋 Prerequisites

- Python 3.9+
- Virtual environment (recommended)
- Evidence Indicator RAG System backend (optional)

## 🛠️ Installation

1. **Navigate to the frontend directory:**
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

### Quick Start

1. **Start the Streamlit application:**
   ```bash
   streamlit run app_enhanced.py
   ```

2. **Open your browser and navigate to:**
   ```
   http://localhost:8501
   ```

3. **Start querying:**
   - Type your question in the text area
   - Click "🔍 検索実行" to submit
   - View results in the beautiful interface

### Alternative Startup Methods

**Using the startup script:**
```bash
./run_streamlit.sh
```

**Using the basic app:**
```bash
streamlit run app.py
```

**Using the test UI:**
```bash
streamlit run test_ui.py
```

## 📊 Features Overview

### Main Interface
- **Query Input**: Large text area for entering questions
- **Quick Query Buttons**: Pre-defined queries for testing
- **Results Display**: Beautifully formatted results with:
  - 【回答】: LLM-generated answer
  - 【検索ヒットのチャンクを含む文書】: Complete source document
  - 【根拠情報】: Extracted evidence with character positions

### Sidebar Features
- **Debug Mode**: Toggle for advanced debugging information
- **API Configuration**: Set backend API URL
- **API Health Monitoring**: Real-time status checking
- **Query History**: Access and reuse previous queries
- **Quick Actions**: Clear history and reset metrics
- **Statistics**: View query counts and performance metrics

### Performance Monitoring
- **Real-time Metrics**: Processing time tracking
- **Performance Charts**: Visual performance trends
- **Statistics Panel**: Query counts and average processing times
- **API Status Indicators**: Visual feedback on backend connectivity

### Debug Features
- **Debug Mode**: Shows detailed API responses and error information
- **Error Handling**: Graceful fallback to simulation mode
- **API Health Checks**: Automatic backend availability monitoring
- **Detailed Logging**: Comprehensive error tracking and reporting

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
```

### Configuration File

The `config.py` file contains additional settings:

- Default queries for quick access
- UI configuration parameters
- Performance thresholds
- Color schemes

## 🔧 Development

### Project Structure

```
rag-streamlit-frontend/
├── app.py              # Basic Streamlit application
├── app_enhanced.py     # Enhanced version with debugging
├── test_ui.py          # UI testing script
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── run_streamlit.sh    # Startup script
├── README.md          # This file
├── streamlit-env/     # Virtual environment
└── .env              # Environment variables (create this)
```

### File Descriptions

- **`app.py`**: Basic Streamlit application with core functionality
- **`app_enhanced.py`**: Enhanced version with debug mode, API health monitoring, and error handling
- **`test_ui.py`**: Simple test script to verify UI components work correctly
- **`config.py`**: Centralized configuration management
- **`run_streamlit.sh`**: Automated startup script with environment checks

### Adding New Features

1. **New UI Components**: Add to `app_enhanced.py`
2. **Configuration**: Update `config.py`
3. **Dependencies**: Add to `requirements.txt`

## 🌐 Integration

### Backend Integration

To connect to the actual RAG backend:

1. **Update API URL**: Set `API_BASE_URL` in `.env`
2. **Start Backend**: Ensure the RAG backend is running
3. **Test Connection**: Use the API health check in the sidebar

### Example API Integration

The enhanced app automatically handles:
- API health checking
- Connection timeouts
- Error fallbacks
- Simulation mode when backend is unavailable

## 📈 Performance

- **Fast Loading**: Optimized for quick startup
- **Responsive UI**: Smooth interactions and real-time updates
- **Memory Efficient**: Session state management for optimal performance
- **Scalable**: Designed to handle multiple concurrent users
- **Fallback Mode**: Works even without backend connectivity

## 🎨 UI/UX Features

- **Japanese Language Support**: Full Japanese interface
- **Responsive Design**: Works on desktop and mobile
- **Dark/Light Mode**: Automatic theme detection
- **Accessibility**: Keyboard navigation and screen reader support
- **Error Handling**: Graceful error messages and recovery
- **Visual Feedback**: Loading spinners, success/error indicators
- **Interactive Charts**: Real-time performance visualization

## 🔍 Debugging

### Debug Mode

Enable debug mode in the sidebar to see:
- Detailed API responses
- Error stack traces
- Performance metrics
- Session state information

### Common Issues

1. **Port Already in Use**:
   ```bash
   streamlit run app_enhanced.py --server.port 8502
   ```

2. **API Connection Issues**:
   - Check API_BASE_URL in `.env`
   - Verify backend is running
   - Use API health check in sidebar
   - App will automatically fall back to simulation mode

3. **Dependencies Issues**:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Debug Commands

Run with debug information:
```bash
streamlit run app_enhanced.py --logger.level debug
```

## 🧪 Testing

### UI Testing

Run the test UI to verify components:
```bash
streamlit run test_ui.py
```

### API Testing

Test API connectivity:
```python
import requests
response = requests.get("http://localhost:8000/health")
print(response.json())
```

## 📝 License

This project is part of the Evidence Indicator RAG System.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

**🔍 Evidence Indicator RAG System | Streamlit Frontend | Powered by Streamlit** 