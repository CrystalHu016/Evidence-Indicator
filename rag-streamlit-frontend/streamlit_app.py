#!/usr/bin/env python3
"""
Evidence Indicator RAG System - Streamlit Frontend
Comprehensive UI for interacting with the RAG backend
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from io import StringIO
import csv
import logging
from typing import Any, Dict, Optional, Tuple
import streamlit.components.v1 as components

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from parent directory where .env file is located
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_path = os.path.join(parent_dir, '.env')
    load_dotenv(env_path)
    print("✅ Environment variables loaded from .env file")
    
    # Set environment variables for proper file paths instead of changing working directory
    os.environ['CHROMA_PATH'] = os.path.join(parent_dir, 'chroma')
    os.environ['DATA_PATH'] = os.path.join(parent_dir, 'data', 'single_20240229.json')
    print(f"✅ Environment variables set - CHROMA_PATH: {os.environ['CHROMA_PATH']}")
    print(f"✅ Environment variables set - DATA_PATH: {os.environ['DATA_PATH']}")
    
except ImportError:
    print("⚠️ python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️ Error loading .env file: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)

# =============================================================================
# CONFIGURATION
# =============================================================================

class AppConfig:
    """Application configuration"""
    API_BASE_URL = "http://localhost:8000"  # Backend RAG API
    DEFAULT_TIMEOUT = 30
    BATCH_TIMEOUT = 120
    MAX_HISTORY_ITEMS = 50
    PAGE_TITLE = "根拠提示装置 | Evidence Indicator RAG System"
    PAGE_ICON = "🔍"

# Sample queries for different categories
SAMPLE_QUERIES = {
    "Agriculture (農業)": [
        "コンバインとは何ですか",
        "農業機械の種類について教えてください",
        "コンバインとは何かとその構造を説明してください"
    ],
    "Language (言語学)": [
        "音位転倒について説明してください"
    ],
    "Technology (技術)": [
        "AI技術の最新動向",
        "機械学習の応用例",
        "自然言語処理の手法について"
    ],
    "General (一般)": [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain deep learning concepts"
    ]
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def initialize_language():
    """Initialize UI language preference in session state."""
    if 'ui_language' not in st.session_state:
        st.session_state.ui_language = 'ja'  # default to Japanese; options: 'ja' | 'en'
    # Coerce any legacy 'bi' to 'ja'
    if st.session_state.ui_language == 'bi':
        st.session_state.ui_language = 'ja'

def t(japanese_text: str, english_text: str) -> str:
    """Translate helper. Returns text based on UI language setting."""
    mode = st.session_state.get('ui_language', 'bi')
    if mode == 'ja':
        return japanese_text
    if mode == 'en':
        return english_text
    # bilingual: show JP / EN
    return f"{japanese_text} / {english_text}"

def language_selector_in_sidebar():
    """Render language selector in sidebar."""
    with st.sidebar:
        options = ["日本語", "English"]
        current_label = {"ja": "日本語", "en": "English"}.get(st.session_state.get('ui_language', 'ja'), "日本語")
        choice = st.radio(
            "Language / 言語",
            options,
            index=options.index(current_label),
            horizontal=True,
            key="lang_radio",
        )
        mapped = {"日本語": "ja", "English": "en"}[choice]
        # Update only if changed then rerun once to apply everywhere
        if st.session_state.get('ui_language') != mapped:
            st.session_state.ui_language = mapped
            st.rerun()

def inject_global_styles():
    """Inject lightweight CSS to beautify the UI."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.2rem; }
        h1, h2, h3 { letter-spacing: 0.2px; }
        .evidence-box { background: #fdfbe6; border: 1px solid #f6e58d; padding: 12px; border-radius: 8px; }
        .source-box { background: #f7f9fc; border: 1px solid #e5eaf2; padding: 12px; border-radius: 8px; }
        div[data-testid="stMetricValue"] { color: #2b8a3e; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def initialize_session_state():
    """Initialize session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'settings' not in st.session_state:
        st.session_state.settings = {
            'single_timeout': AppConfig.DEFAULT_TIMEOUT,
            'batch_timeout': AppConfig.BATCH_TIMEOUT,
            'show_technical_details': True,
            'show_timestamps': True,
            'auto_scroll_results': True,
            'max_history': AppConfig.MAX_HISTORY_ITEMS
        }
    if 'cache_cleared' not in st.session_state:
        st.session_state.cache_cleared = False

def validate_query(query: str) -> Tuple[bool, str]:
    """Validate query input"""
    if not query or not query.strip():
        return False, t("クエリを入力してください", "Please enter a query")
    if len(query.strip()) < 2:
        return False, t("クエリが短すぎます", "Query is too short")
    if len(query.strip()) > 1000:
        return False, t("クエリが長すぎます（1000文字以内）", "Query is too long (max 1000 chars)")
    return True, ""

# =============================================================================
# API FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=False, ttl=15)
def call_health_check(api_url: str) -> bool:
    """Check API health"""
    try:
        # Try to import backend integration first
        from backend_integration import test_backend_connection
        if test_backend_connection():
            return True
    except ImportError:
        pass
    
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        return response.status_code == 200
    except:
        pass
    
    # Always return True for simulation mode
    return True

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_single_query_cached(api_url: str, query: str, timeout_seconds: int, cache_version: str = "v20_llm_relevance") -> Tuple[Optional[Dict], Optional[str]]:
    """Pure function for fetching a single query result; safe to cache."""
    # Try backend integration first (this is the primary method)
    try:
        from backend_integration import call_backend_query
        # Get system mode from session state, default to "ultra_fast_original" for speed
        system_mode = st.session_state.get('system_mode', 'ultra_fast_original')
        result, error = call_backend_query(query, system_mode)
        if result and not error:
            return result, None
        elif error:
            # If backend integration has an error, return it directly (don't fall back to simulation)
            return None, error
    except ImportError:
        # If backend_integration module not available
        pass
    except Exception as e:
        # If there's any other error with backend integration
        return None, f"Backend error: {str(e)}"

    # Try HTTP API
    try:
        if api_url:
            response = requests.post(
                f"{api_url}/query",
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=timeout_seconds
            )
            if response.status_code == 200:
                return response.json(), None
    except Exception:
        pass

    # No more hardcoded simulations - return None to indicate no data found
    # This will force the system to show a proper "no results" message
    return None, "No simulation data available. Please ensure the RAG backend is running and the vector database is built with your JSON dataset."

def call_single_query(api_url: str, query: str) -> Tuple[Optional[Dict], Optional[str]]:
    """Call the single query endpoint with caching and spinner."""
    try:
        with st.spinner("🔄 処理中..."):
            timeout_seconds = st.session_state.settings.get('single_timeout', 30)
            return _fetch_single_query_cached(api_url, query, timeout_seconds, "v20_llm_relevance")
    except Exception as e:
        return None, str(e)



# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

@st.cache_data(show_spinner=False, ttl=300)
def compute_effective_range(source_text: str, start_char: int, end_char: int, evidence_text: str) -> Tuple[int, int]:
    """Return an adjusted 1-based (start, end) range that best matches evidence_text if available."""
    if source_text and evidence_text:
        idx = source_text.find(evidence_text)
        if idx != -1:
            # Convert to 1-based inclusive range
            start = idx + 1
            end = idx + len(evidence_text)
            return start, end
    # Fallback to provided range
    start = max(1, start_char)
    end = min(len(source_text), end_char) if source_text else end_char
    return start, end

@st.cache_data(show_spinner=False, ttl=300)
def highlight_rag_evidence_in_source(source_text: str, evidence_text: str, char_ranges: list = None) -> str:
    """Highlight the RAG-identified evidence chunk at specific character positions.

    Args:
        source_text: The full source document
        evidence_text: The evidence text (for fallback if no char_ranges provided)
        char_ranges: List of tuples [(start1, end1), (start2, end2), ...] with 1-based positions
    """
    if not source_text:
        return f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px;
                    font-family: 'Hiragino Sans', sans-serif; line-height: 1.8; border: 1px solid #e0e0e0;">
            {source_text}
        </div>
        """

    # If no char_ranges provided, fall back to old logic
    if not char_ranges:
        highlighted_text = source_text
        if evidence_text:
            evidence_clean = evidence_text.strip()
            # Strategy: Split evidence by newlines first (LLM uses newlines to separate sentences)
            if '\n' in evidence_clean:
                sentences = [s.strip() for s in evidence_clean.split('\n') if s.strip()]
            else:
                sentences = evidence_clean.replace('！', '。').replace('？', '。').split('。')

            sentences_to_highlight = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) >= 10 and sentence in source_text:
                    sentences_to_highlight.append(sentence)

            # Highlight all matching sentences individually
            for sentence in sentences_to_highlight:
                highlight_span = f'<span style="background-color: #ffff00; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ffcc00;">{sentence}</span>'
                highlighted_text = highlighted_text.replace(sentence, f'__HIGHLIGHT_MARKER_{sentences_to_highlight.index(sentence)}__', 1)

            # Replace markers with actual highlights
            for idx, sentence in enumerate(sentences_to_highlight):
                highlight_span = f'<span style="background-color: #ffff00; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ffcc00;">{sentence}</span>'
                highlighted_text = highlighted_text.replace(f'__HIGHLIGHT_MARKER_{idx}__', highlight_span)
    else:
        # New logic: Highlight only at specific character ranges
        # Sort ranges by start position (descending) to avoid position shifts
        sorted_ranges = sorted(char_ranges, key=lambda x: x[0], reverse=True)

        highlighted_text = source_text
        for start_pos, end_pos in sorted_ranges:
            # Convert to 0-based indexing
            start_idx = start_pos - 1
            end_idx = end_pos  # end_pos is already inclusive in 1-based, so end_idx in 0-based

            # Ensure indices are valid
            if 0 <= start_idx < len(source_text) and start_idx < end_idx <= len(source_text):
                text_to_highlight = source_text[start_idx:end_idx]
                highlight_span = f'<span style="background-color: #ffff00; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ffcc00;">{text_to_highlight}</span>'

                # Replace only this specific range
                highlighted_text = (
                    highlighted_text[:start_idx] +
                    highlight_span +
                    highlighted_text[end_idx:]
                )

    html_content = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px;
                font-family: 'Hiragino Sans', sans-serif; line-height: 1.8; border: 1px solid #e0e0e0;">
        {highlighted_text}
    </div>
    """
    return html_content

def display_results():
    """Display the query results in Japanese format with highlighting"""
    if 'last_result' not in st.session_state:
        return
        
    result = st.session_state.last_result
    query = st.session_state.last_query
    
    st.markdown("---")
    st.header(t("📋 検索結果", "📋 Results"))
    
    # Query info
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(t(f"🔍 クエリ: {query}", f"🔍 Query: {query}"))
    with col2:
        processing_time = result.get('processing_time', 0)
        st.metric(t("⚡ 処理時間", "⚡ Time"), t(f"{processing_time:.2f}秒", f"{processing_time:.2f}s"))
    
    # Results in Japanese format
    st.markdown(t("### 【回答】", "### Answer"))
    answer = result.get('answer', '回答が見つかりませんでした。')
    st.write(answer)
    
    st.markdown(t("### 【検索ヒットのチャンクを含む文書】", "### Source document that contains the hit chunk"))
    source_doc = result.get('source_document', '文書が見つかりませんでした。')
    start_char = result.get('start_char', 0)
    end_char = result.get('end_char', 0)
    evidence_text = result.get('evidence_text', '')

    # Get extracted evidences (Strategy 3)
    evidences = result.get('evidences', [])
    valid_evidences = [e for e in evidences if not e.get('is_empty', True)]

    # Use extracted evidence for highlighting (more precise than whole chunk)
    if valid_evidences:
        # Combine all extracted evidences for highlighting
        extracted_texts = [e.get('extracted_evidence', '') for e in valid_evidences]
        combined_extracted = '\n'.join(extracted_texts)
        display_evidence = combined_extracted
    else:
        display_evidence = evidence_text

    # Compute adjusted range based on evidence text for consistency
    eff_start, eff_end = compute_effective_range(source_doc, start_char, end_char, display_evidence)

    # Detect document prefix (e.g., "文档1: ", "文档2: ") and calculate offset
    import re
    doc_prefix_match = re.match(r'^文档\d+:\s*', source_doc)
    prefix_offset = len(doc_prefix_match.group(0)) if doc_prefix_match else 0

    # Extract the actual document content (without prefix) for display
    display_source_doc = source_doc[prefix_offset:] if prefix_offset > 0 else source_doc

    # Calculate character ranges for each extracted sentence separately
    # display_evidence contains the fine-grained extraction (e.g. 2 sentences separated by newline)
    sentence_ranges = []
    char_position_ranges = []  # Store tuples for highlighting: [(start1, end1), (start2, end2), ...]

    if display_evidence and display_evidence != '根拠情報なし':
        # Split by newlines to get individual sentences
        sentences = [s.strip() for s in display_evidence.split('\n') if s.strip()]

        for sentence in sentences:
            # Search in the display document (without prefix)
            if sentence in display_source_doc:
                # Position in the display document (1-indexed)
                start_pos_display = display_source_doc.index(sentence) + 1
                end_pos_display = start_pos_display + len(sentence) - 1
                sentence_ranges.append(f"{start_pos_display}文字目～{end_pos_display}文字目")

                # For highlighting in the original source_doc (with prefix), add offset
                start_pos_orig = start_pos_display + prefix_offset
                end_pos_orig = end_pos_display + prefix_offset
                char_position_ranges.append((start_pos_orig, end_pos_orig))

    # Show highlighted version with RAG evidence (pass char_position_ranges)
    st.markdown(t("**💡 根拠部分のハイライト表示:**", "**💡 Highlighted evidence:**"))
    highlighted_html = highlight_rag_evidence_in_source(source_doc, display_evidence, char_position_ranges)
    st.markdown(highlighted_html, unsafe_allow_html=True)

    st.markdown(t("**📄 元の文書:**", "**📄 Original document:**"))

    st.text_area(t("文書内容", "Document"), source_doc, height=200, key="source_display")

    # Evidence information - use display_evidence (extracted evidence, not full chunk)
    evidence_text = result.get('evidence_text', '根拠情報なし')

    # Display all sentence ranges
    if sentence_ranges:
        ranges_text = "、".join(sentence_ranges)
        st.markdown(t(f"### 【根拠情報の文字列範囲】{ranges_text}",
                     f"### Evidence character ranges: {ranges_text}"))
    else:
        # Fallback to old logic
        st.markdown(t(f"### 【根拠情報の文字列範囲】{eff_start}文字目～{eff_end}文字目",
                     f"### Evidence character range: {eff_start} to {eff_end}"))

    st.markdown(t("### 【根拠情報】", "### Evidence"))

    # Strategy 3: Display multiple evidences (if available)
    evidences = result.get('evidences', [])

    if evidences and len(evidences) > 0:
        valid_evidences = [e for e in evidences if not e.get('is_empty', True)]

        if valid_evidences:
            # Simply display the extracted evidence without chunk/similarity metadata
            for evidence in valid_evidences:
                extracted = evidence.get('extracted_evidence', '')
                st.info(extracted)
        else:
            # Should not use evidence_text here, use display_evidence instead
            st.info(display_evidence if display_evidence else evidence_text)
    else:
        # Should not use evidence_text here, use display_evidence instead
        st.info(display_evidence if display_evidence else evidence_text)

    # Additional metadata
    if st.session_state.settings.get('show_technical_details', True):
        with st.expander(t("📊 技術詳細", "📊 Technical details")):
            col1, col2, col3 = st.columns(3)
            with col1:
                confidence = result.get('confidence', 0)
                st.metric(t("信頼度", "Confidence"), f"{confidence:.2f}")
            with col2:
                model = result.get('model', 'Unknown')
                st.metric(t("モデル", "Model"), model)
            with col3:
                timestamp = result.get('timestamp', time.time())
                timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                st.write(t(f"**タイムスタンプ:** {timestamp_str}", f"**Timestamp:** {timestamp_str}"))

# =============================================================================
# HISTORY MANAGEMENT
# =============================================================================

def add_to_history(query: str, result: dict):
    """Add query and result to history"""
    history_item = {
        'timestamp': datetime.now(),
        'query': query,
        'answer': result.get('answer', ''),
        'processing_time': result.get('processing_time', 0),
        'confidence': result.get('confidence', 0),
        'evidence_text': result.get('evidence_text', ''),
        'start_char': result.get('start_char', 0),
        'end_char': result.get('end_char', 0)
    }
    st.session_state.query_history.append(history_item)
    
    # Keep only last N queries
    max_history = st.session_state.settings.get('max_history', AppConfig.MAX_HISTORY_ITEMS)
    if len(st.session_state.query_history) > max_history:
        st.session_state.query_history = st.session_state.query_history[-max_history:]

def export_history():
    """Export query history to CSV"""
    if not st.session_state.query_history:
        st.error(t("エクスポートする履歴がありません", "No history to export"))
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.query_history)
    
    # Convert timestamp to string for CSV
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Create CSV
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False, encoding='utf-8')
    
    # Download button
    st.download_button(
        label=t("📥 履歴をCSVでダウンロード", "Download history as CSV"),
        data=csv_buffer.getvalue(),
        file_name=f"rag_query_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def query_history_interface():
    """Interface for viewing and managing query history"""
    st.markdown("---")
    st.header(t("📚 クエリ履歴", "Query history"))
    
    if not st.session_state.query_history:
        st.info(t("まだ履歴がありません。", "No history yet."))
        return
    
    # History controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("📊 履歴をエクスポート", "Export history")):
            export_history()
    with col2:
        if st.button(t("🗑️ 履歴をクリア", "Clear history")):
            st.session_state.query_history = []
            st.success(t("履歴をクリアしました！", "History cleared!"))
            st.rerun()
    with col3:
        show_count = st.selectbox(t("表示件数", "Items to show"), [5, 10, 20, 50], index=1)
    
    # Performance chart
    if len(st.session_state.query_history) > 1:
        df_history = pd.DataFrame(st.session_state.query_history[-20:])
        fig = px.line(
            df_history, x='timestamp', y='processing_time',
            title=t('処理時間の推移', 'Processing time over queries'),
            labels={'processing_time': t('処理時間(秒)', 'Time (s)'), 'timestamp': t('時刻', 'Time')}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display history
    history_to_show = st.session_state.query_history[-show_count:]
    
    for i, item in enumerate(reversed(history_to_show), 1):
        timestamp_str = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        
        with st.expander(f"{i}. {item['query'][:60]}... ({timestamp_str})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(t("**クエリ:**", "**Query:**"))
                st.write(item['query'])
                st.markdown(t("**回答:**", "**Answer:**"))
                st.write(item['answer'])
                st.markdown(t("**根拠:**", "**Evidence:**"))
                st.info(item['evidence_text'])
            
            with col2:
                st.metric(t("処理時間", "Time"), t(f"{item['processing_time']:.2f}秒", f"{item['processing_time']:.2f}s"))
                st.metric(t("信頼度", "Confidence"), f"{item['confidence']:.2f}")
                st.metric(t("根拠範囲", "Range"), f"{item['start_char']}-{item['end_char']}")
                
                if st.button(t("🔄 再実行", "Re-run"), key=f"rerun_{i}"):
                    # Re-run the query
                    result, error = call_single_query("", item['query'])
                    if result:
                        st.session_state.last_result = result
                        st.session_state.last_query = item['query']
                        st.rerun()



# =============================================================================
# SETTINGS INTERFACE
# =============================================================================

def settings_interface():
    """Settings and configuration interface"""
    with st.sidebar:
        st.header(t("⚙️ 設定", "Settings"))
        language_selector_in_sidebar()
        
        # History settings
        st.subheader(t("履歴設定", "History settings"))
        max_history = st.slider(t("最大履歴件数", "Max history items"), 10, 100, 
                               st.session_state.settings['max_history'])
        
        # Store settings in session state
        st.session_state.settings = {
            'single_timeout': 30,
            'batch_timeout': 120,
            'show_technical_details': True,
            'show_timestamps': True,
            'auto_scroll_results': True,
            'max_history': max_history
        }
        
        # Store system mode in session state (fixed to enhanced mode)
        st.session_state.system_mode = "enhanced"
        
        # Sample queries
        st.subheader(t("📝 サンプルクエリ", "Sample queries"))
        category = st.selectbox(t("カテゴリ", "Category"), list(SAMPLE_QUERIES.keys()), key="category_select")
        
        for i, sample_query in enumerate(SAMPLE_QUERIES[category]):
            if st.button(sample_query, key=f"sample_{category}_{i}"):
                st.session_state.selected_sample_query = sample_query
                st.session_state["query_input"] = sample_query
                st.rerun()

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application"""
    # Page configuration
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        page_icon=AppConfig.PAGE_ICON,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_language()
    initialize_session_state()
    inject_global_styles()
    
    # Main title and description
    st.title(t("🔍 根拠提示装置", "🔍 Evidence Indicator RAG System"))
    st.markdown(t(
        """
        **高速検索・根拠抽出システム** - 日本語対応RAGシステム
        
        このシステムは、質問に対して根拠となる情報を含む回答を日本語で提供します。
        """,
        """
        **High-speed retrieval and evidence extraction** - RAG system
        
        This system provides answers with supporting evidence for your questions.
        """
    ))
    
    # Real RAG system indicator
    st.success(t(
        "🚀 **実RAGシステム動作中** - システムはあなたのJSONデータセットを使用して動作しています。",
        "**Real RAG System Active** - The system is running with your JSON dataset."
    ))
    st.markdown("---")
    
    # Settings interface (sidebar)
    settings_interface()
    
    # Main query interface
    st.header(t("📝 クエリ入力", "Query input"))
    
    # Handle sample query selection
    if 'selected_sample_query' in st.session_state:
        default_query = st.session_state.selected_sample_query
        # Don't delete immediately, let it persist for the button click
    else:
        default_query = ""
    
    query_text = st.text_area(
        t("質問を入力してください:", "Enter your question:"),
        value=st.session_state.get("query_input", default_query),
        height=100,
        placeholder=t("例: コンバインとは何ですか", "e.g., What is a combine harvester?")
    )
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(t("🚀 検索実行", "🚀 Search"), type="primary"):
            # Use the query text or the selected sample query
            current_query = query_text.strip() or st.session_state.get('selected_sample_query', '').strip()
            
            if current_query:
                valid, error_msg = validate_query(current_query)
                if not valid:
                    st.error(f"❌ {error_msg}")
                else:
                    # Call query function directly (no API check needed)
                    result, error = call_single_query("", current_query)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Store result in session state for display
                        st.session_state.last_result = result
                        st.session_state.last_query = current_query
                        
                        # Add to history
                        add_to_history(current_query, result)
                        
                        # Clear the selected sample query after successful processing
                        if 'selected_sample_query' in st.session_state:
                            del st.session_state.selected_sample_query
                        st.session_state.pop("query_input", None)
                        
                        st.success(t("✅ クエリが正常に処理されました！", "Query processed successfully!"))
                        if st.session_state.settings.get('auto_scroll_results', True):
                            st.rerun()
            else:
                st.error(t("クエリを入力してください", "Please enter a query"))
    
    with col2:
        if st.button(t("🔄 クリア", "🔄 Clear")):
            st.session_state.pop('last_result', None)
            st.session_state.pop('last_query', None)
            st.session_state.pop('selected_sample_query', None)
            st.rerun()
    
    with col3:
        if st.button(t("📊 履歴表示", "📊 Show history")):
            st.session_state.show_history = True
    
    # Cache clear button
    if st.button(t("🔄 キャッシュクリア", "🔄 Clear Cache"), help=t("キャッシュをクリアして最新の結果を取得", "Clear cache to get latest results")):
        st.cache_data.clear()
        st.session_state.cache_cleared = True
        st.success(t("✅ キャッシュをクリアしました！", "✅ Cache cleared!"))
        st.rerun()
    
    # Display results
    display_results()
    
    # Query history interface
    if st.session_state.get('show_history', False):
        query_history_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            Evidence Indicator RAG System v1.0
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()