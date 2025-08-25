#!/bin/bash

# Evidence Indicator RAG System - Streamlit Frontend Startup Script

echo "🔍 Starting Evidence Indicator RAG System - Streamlit Frontend"
echo "=============================================================="

# Check if virtual environment exists
if [ ! -d "streamlit-env" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv streamlit-env"
    echo "   source streamlit-env/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source streamlit-env/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating default configuration..."
    cat > .env << EOF
# Streamlit Frontend Configuration
API_BASE_URL=http://localhost:8000
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
EOF
    echo "✅ Created default .env file"
fi

# Start Streamlit
echo "🚀 Starting Streamlit application..."
echo "📱 Open your browser to: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run streamlit_app.py --server.port 8501 --server.address localhost 