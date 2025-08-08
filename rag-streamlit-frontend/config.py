#!/usr/bin/env python3
"""
Configuration file for Evidence Indicator RAG System Streamlit Frontend
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))

# Streamlit Configuration
STREAMLIT_SERVER_PORT = int(os.getenv("STREAMLIT_SERVER_PORT", "8501"))
STREAMLIT_SERVER_ADDRESS = os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost")

# Default Queries for Quick Access
DEFAULT_QUERIES = [
    "コンバインとは何ですか",
    "音位転倒について説明してください", 
    "どのような農業機械がありますか",
    "What is a combine harvester?",
    "日本語の言語現象はどんなものがありますか",
    "A社とB社とC社の中で売上が最も高いのはどちらですか"
]

# UI Configuration
MAX_QUERY_HISTORY = 20
MAX_PERFORMANCE_METRICS = 100
DEFAULT_CHART_HEIGHT = 400

# Performance Thresholds
FAST_QUERY_THRESHOLD = 1.0  # seconds
SLOW_QUERY_THRESHOLD = 3.0  # seconds

# Colors for UI
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c', 
    'warning': '#ff7f0e',
    'error': '#d62728',
    'info': '#17a2b8'
} 