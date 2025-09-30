#!/usr/bin/env python3
"""
Modern RAG System - Streamlit Frontend
Integrated with all four architectural improvements
"""

import streamlit as st
import time
import json
import sys
import os
from typing import Dict, Optional

# Add improvements directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
improvements_dir = os.path.join(parent_dir, "improvements")
sys.path.insert(0, improvements_dir)

# Import modern RAG components
try:
    from config_driven_rag import ConfigDrivenRAGSystem
    from semantic_keyword_extractor import SemanticKeywordExtractor
    from llm_intent_classifier import LLMIntentClassifier
    from dynamic_context_generator import DynamicContextGenerator

    MODERN_RAG_AVAILABLE = True
    print("✅ Modern RAG components imported successfully")
except ImportError as e:
    print(f"⚠️ Modern RAG import failed: {e}")
    MODERN_RAG_AVAILABLE = False

# Initialize global variables
modern_rag_system = None

def initialize_modern_rag():
    """Initialize the modern RAG system"""
    global modern_rag_system

    if not MODERN_RAG_AVAILABLE:
        return False

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY not set in environment variables")
        return False

    try:
        # Configure the modern RAG system
        config = {
            "llm_model": "gpt-4o-mini",
            "llm_temperature": 0.1,
            "use_llm_intent": True,
            "use_dynamic_context": True,
            "keyword_max_count": 10,
            "intent_confidence_threshold": 0.8,
            "similarity_threshold": 0.3,
            "domains": {
                "agriculture": {"semantic_boosting": True, "boost_factor": 1.2},
                "technology": {"semantic_boosting": False, "boost_factor": 1.0},
                "general": {"semantic_boosting": False, "boost_factor": 1.0}
            }
        }

        modern_rag_system = ConfigDrivenRAGSystem(config_dict=config)
        return True

    except Exception as e:
        st.error(f"❌ Failed to initialize Modern RAG system: {e}")
        return False

def main():
    """Main Streamlit application"""

    # Page configuration
    st.set_page_config(
        page_title="Modern RAG System",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Application header
    st.title("🚀 Modern RAG System")
    st.markdown("### Zero Hardcoded Architecture • Multilingual • Configuration-Driven")
    st.markdown("---")

    # Initialize system
    if modern_rag_system is None:
        with st.spinner("Initializing Modern RAG System..."):
            system_ready = initialize_modern_rag()
    else:
        system_ready = True

    # Sidebar for system information and configuration
    with st.sidebar:
        st.header("🛠️ System Status")

        if system_ready and MODERN_RAG_AVAILABLE:
            st.success("✅ Modern RAG System Ready")

            st.markdown("### 🎯 Architecture Improvements")
            st.markdown("✅ **Semantic Keywords** (No hardcoded patterns)")
            st.markdown("✅ **LLM Intent Understanding** (No hardcoded rules)")
            st.markdown("✅ **Dynamic Context Generation** (No fixed templates)")
            st.markdown("✅ **Configuration-Driven** (No hardcoded values)")

            st.markdown("---")

            # Configuration options
            st.header("⚙️ Configuration")

            # Domain selection
            domain = st.selectbox(
                "Select Domain",
                ["agriculture", "technology", "general"],
                index=0,
                help="Choose the domain for semantic boosting"
            )

            # Language detection
            auto_language = st.checkbox("Auto-detect Language", value=True)

            # Advanced settings
            with st.expander("🔧 Advanced Settings"):
                max_keywords = st.slider("Max Keywords", 5, 15, 10)
                intent_threshold = st.slider("Intent Confidence Threshold", 0.5, 1.0, 0.8)

        else:
            st.error("❌ Modern RAG System Not Available")
            if not MODERN_RAG_AVAILABLE:
                st.markdown("**Missing Components:**")
                st.markdown("- Configuration-driven RAG system")
                st.markdown("- Semantic keyword extractor")
                st.markdown("- LLM intent classifier")
                st.markdown("- Dynamic context generator")

    # Main interface
    if system_ready and MODERN_RAG_AVAILABLE:

        # Query input
        col1, col2 = st.columns([3, 1])

        with col1:
            query = st.text_input(
                "Enter your question (any language):",
                placeholder="e.g., コンバインとは何ですか / What is a combine harvester? / 农业机械有哪些类型？",
                key="user_query"
            )

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            search_button = st.button("🔍 Search", type="primary")

        # Example queries
        st.markdown("### 💡 Example Queries")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🇯🇵 Japanese Definition"):
                st.session_state.user_query = "コンバインとは何ですか"

        with col2:
            if st.button("🇺🇸 English Classification"):
                st.session_state.user_query = "What types of agricultural machinery exist?"

        with col3:
            if st.button("🇯🇵 Japanese Comparison"):
                st.session_state.user_query = "普通型と自立型の違いは何ですか"

        # Process query
        if search_button and query:
            process_modern_rag_query(query, domain)
        elif query and query != st.session_state.get('last_processed_query', ''):
            # Auto-process on input change
            process_modern_rag_query(query, domain)
            st.session_state.last_processed_query = query

    else:
        st.warning("⚠️ Please check system configuration and API keys")

        st.markdown("### 🔧 Setup Instructions")
        st.markdown("""
        1. Set your `OPENAI_API_KEY` environment variable
        2. Ensure all modern RAG components are installed
        3. Check the improvements directory path
        """)

def process_modern_rag_query(query: str, domain: str):
    """Process query using modern RAG system"""

    if not query.strip():
        return

    # Create context chunks for demonstration
    context_chunks = [
        "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
        "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
        "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。",
        "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。",
        "Agricultural machinery includes various types of equipment used in farming operations for improved efficiency and productivity.",
        "Modern farming relies heavily on mechanization to handle large-scale crop production and harvesting processes."
    ]

    with st.spinner("🧠 Processing with Modern RAG System..."):
        try:
            start_time = time.time()

            # Call modern RAG system
            result = modern_rag_system.query(
                question=query,
                domain=domain,
                context_chunks=context_chunks
            )

            processing_time = time.time() - start_time

            if result.get("error"):
                st.error(f"❌ Error: {result['answer']}")
                return

            # Display results
            display_modern_rag_results(result, query, processing_time)

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")

def display_modern_rag_results(result: Dict, query: str, processing_time: float):
    """Display modern RAG results with detailed analysis"""

    # Main answer
    st.markdown("### 💬 Answer")
    answer_container = st.container()
    with answer_container:
        st.markdown(f"**{result['answer']}**")

    # Analysis details
    st.markdown("### 📊 Analysis Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Intent Analysis")
        analysis = result["query_analysis"]

        intent_container = st.container()
        with intent_container:
            st.markdown(f"**Intent:** {analysis['intent']}")
            st.markdown(f"**Confidence:** {analysis['intent_confidence']:.2%}")
            st.markdown(f"**Domain:** {analysis['domain']}")

            # Intent confidence indicator
            confidence_color = "green" if analysis['intent_confidence'] > 0.8 else "orange" if analysis['intent_confidence'] > 0.6 else "red"
            st.markdown(f"<div style='background-color: {confidence_color}; padding: 5px; border-radius: 5px; color: white; text-align: center;'>Confidence: {analysis['intent_confidence']:.1%}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 🔑 Semantic Keywords")
        keywords = analysis["keywords"]

        if keywords:
            # Display keywords as badges
            keyword_html = ""
            for keyword in keywords:
                keyword_html += f'<span style="background-color: #e1f5fe; padding: 2px 8px; margin: 2px; border-radius: 10px; font-size: 12px;">{keyword}</span> '
            st.markdown(keyword_html, unsafe_allow_html=True)
        else:
            st.markdown("*No keywords extracted*")

    # Context and processing information
    st.markdown("### 🔄 Processing Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Processing Time", f"{processing_time:.2f}s")

    with col2:
        st.metric("Keywords Found", len(analysis["keywords"]))

    with col3:
        st.metric("Intent Confidence", f"{analysis['intent_confidence']:.1%}")

    # Technical details in expandable section
    with st.expander("🔧 Technical Details"):
        st.json({
            "configuration": result["configuration"],
            "context_info": {
                "method": result["context_info"]["method"],
                "context_preview": result["context_info"]["enhanced_context"][:100] + "..."
            },
            "processing": result["processing"]
        })

    # Comparison with original system
    st.markdown("### 📈 Architecture Comparison")

    comparison_data = {
        "Feature": ["Keyword Extraction", "Intent Understanding", "Context Generation", "Configuration", "Multilingual Support"],
        "Original System": ["Hardcoded patterns", "Hardcoded rules", "Fixed templates", "Code-based", "Japanese only"],
        "Modern System": ["Semantic embeddings", "LLM-based", "Dynamic generation", "Config-driven", "Auto-detection"]
    }

    st.table(comparison_data)

if __name__ == "__main__":
    main()