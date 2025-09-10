#!/bin/bash

echo "🚀 Advanced RAG Evidence Indicator - Quick Start"
echo "================================================"

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Test the system
echo "🧪 Testing the system..."
python3 script/test_system.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 System is ready!"
    echo ""
    echo "Next steps:"
    echo "1. Set your OPENAI_API_KEY:"
    echo "   export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "2. Run the main system:"
    echo "   python3 script/rag_evidence_indicator_new.py"
    echo ""
    echo "3. Or test configuration:"
    echo "   python3 config.py"
    echo ""
    echo "4. For help, see README.md"
else
    echo ""
    echo "❌ System test failed. Please check the errors above."
    exit 1
fi