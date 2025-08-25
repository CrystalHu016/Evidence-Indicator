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
    MAX_HISTORY_ITEMS = 20
    PAGE_TITLE = "発明原稿 根拠提示装置 | Evidence Indicator RAG System"
    PAGE_ICON = "🔍"

# Sample queries for different categories
SAMPLE_QUERIES = {
    "Agriculture (農業)": [
        "コンバインとは何ですか",
        "稲作の手順を説明してください",
        "コンバインの種類と用途の違いを比較してください",
        "日本の農業機械の分類について詳しく説明してください"
    ],
    "Language & Literature (言語・文学)": [
        "音位転倒について説明してください",
        "アナグラムとは何ですか",
        "真っ赤な嘘の意味を教えてください"
    ],
    "Science & Nature (科学・自然現象)": [
        "環水平アークとは何ですか",
        "毛細管現象について説明してください",
        "蛙化現象とはどのような現象ですか"
    ],
    "Culture & Entertainment (文化・エンターテイメント)": [
        "つちのこについて説明してください",
        "不知火型とはどのような型ですか",
        "オレンジペコーとは何ですか"
    ],
    "Food & Beverage (飲食)": [
        "犬に食べさせてはいけないものは何ですか",
        "蒸留酒と非蒸留酒の分類について"
    ],
    "Psychology & Phenomena (心理・現象)": [
        "ゲシュタルト崩壊とは何ですか",
        "意味飽和について説明してください"
    ],
    "Sports & Games (スポーツ・ゲーム)": [
        "ペイントボールとは何ですか",
        "日本がオリンピックで金メダルを取れない種目"
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

# @st.cache_data(show_spinner=False, ttl=300)  # Disabled to test filtering changes
def _fetch_single_query_cached(api_url: str, query: str, timeout_seconds: int, cache_version: str = "v9_fixed_paths") -> Tuple[Optional[Dict], Optional[str]]:
    """Pure function for fetching a single query result; safe to cache."""
    # Try the new query handler with proper path management
    try:
        from query_handler import query_rag_system
        import streamlit as st
        use_multi_chunk = st.session_state.settings.get('use_multi_chunk', True) if 'st' in globals() else True
        result, error = query_rag_system(query, use_multi_chunk)
        if result and not error:
            return result, None
        elif error:
            return None, error
    except ImportError:
        # Fallback to original backend integration
        try:
            from backend_integration import call_backend_query
            import streamlit as st
            use_multi_chunk = st.session_state.settings.get('use_multi_chunk', True) if 'st' in globals() else True
            result, error = call_backend_query(query, use_multi_chunk)
            if result and not error:
                return result, None
            elif error:
                return None, error
        except ImportError:
            pass
        except Exception as e:
            return None, f"Backend error: {str(e)}"
    except Exception as e:
        # If there's any other error with query handler
        return None, f"Query handler error: {str(e)}"

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
            return _fetch_single_query_cached(api_url, query, timeout_seconds, "v8_force_reload_fix")
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
def highlight_evidence_in_source(source_text: str, start_char: int, end_char: int) -> str:
    """Create highlighted version of source text using a 1-based inclusive range."""
    if not source_text:
        return source_text
    # Normalize
    start_char = max(1, start_char)
    end_char = min(len(source_text), end_char)
    if end_char < start_char:
        return source_text
    # Adjust for 0-based slicing; end is exclusive in Python, but our end_char is inclusive
    start_idx = start_char - 1
    end_idx = end_char
    before = source_text[:start_idx]
    highlighted = source_text[start_idx:end_idx]
    after = source_text[end_idx:]
    html_content = f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; 
                font-family: 'Hiragino Sans', sans-serif; line-height: 1.8; border: 1px solid #e0e0e0;">
        {before}<span style="background-color: #ffff00; padding: 3px 6px; border-radius: 4px; 
                           font-weight: bold; border: 1px solid #ffcc00;">{highlighted}</span>{after}
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
    # Compute adjusted range based on evidence text for consistency
    eff_start, eff_end = compute_effective_range(source_doc, start_char, end_char, evidence_text)
    
    # Show highlighted version
    if eff_start > 0 and eff_end > eff_start:
        st.markdown(t("**💡 根拠部分のハイライト表示:**", "**💡 Highlighted evidence:**"))
        highlighted_html = highlight_evidence_in_source(source_doc, eff_start, eff_end)
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        st.markdown(t("**📄 元の文書:**", "**📄 Original document:**"))
    
    st.text_area(t("文書内容", "Document"), source_doc, height=200, key="source_display")
    
    # Evidence information
    evidence_text = result.get('evidence_text', '根拠情報なし')
    
    st.markdown(t(f"### 【根拠情報の文字列範囲】{eff_start}文字目～{eff_end}文字目",
                 f"### Evidence character range: {eff_start} to {eff_end}"))
    
    st.markdown(t("### 【根拠情報】", "### Evidence"))
    st.info(evidence_text)
    
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
        show_count = st.selectbox(t("表示件数", "Items to show"), [5, 10, 15, 20], index=1)
    
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
        st.header(t("⚙️ 設定", "⚙️ Settings"))
        language_selector_in_sidebar()
        
        # Query processing settings
        st.subheader(t("🔧 クエリ処理設定", "🔧 Query processing settings"))
        use_multi_chunk = st.checkbox(
            t("複雑クエリで多級ブロック分析を使用", "Use multi-chunk analysis for complex queries"),
            value=st.session_state.settings.get('use_multi_chunk', True),
            help=t("複雑な質問に対して段階的なブロック分析を実行します", "Performs progressive chunk analysis for complex queries")
        )
        
        # History settings
        st.subheader(t("📚 履歴設定", "📚 History settings"))
        max_history = st.slider(t("最大履歴件数", "Max history items"), 5, 20, 
                               min(st.session_state.settings['max_history'], 20))
        
        # Store settings in session state
        st.session_state.settings = {
            'single_timeout': 30,
            'batch_timeout': 120,
            'show_technical_details': True,
            'show_timestamps': True,
            'auto_scroll_results': True,
            'max_history': max_history,
            'use_multi_chunk': use_multi_chunk
        }
        
        # Sample queries
        st.subheader(t("📝 サンプルクエリ", "📝 Sample queries"))
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
    st.title(t("🔍 発明原稿 根拠提示装置", "🔍 Evidence Indicator RAG System"))
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
            st.session_state.pop('query_input', None)
            st.session_state.pop('show_history', None)
            st.rerun()
    
    with col3:
        if st.button(t("📊 履歴表示", "📊 Show history")):
            st.session_state.show_history = True
    
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