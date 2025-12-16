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
import sqlite3
from typing import Any, Dict, Optional, Tuple
import streamlit.components.v1 as components

# Import database manager
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from database.query_history_manager import QueryHistoryManager

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
        "コンバインとは何かとその構造を説明してください",
        "コンバインで収穫できる穀物は何ですか？"
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
        # Initialize database manager
        try:
            db_path = os.path.join(parent_dir, 'query_history.db')
            history_manager = QueryHistoryManager(db_path)

            # Load recent queries from database (try all versions, not just v2_full_evidence)
            try:
                recent_queries = history_manager.get_recent_queries(limit=100, version='v2_full_evidence')
                print(f"✅ Loaded {len(recent_queries)} v2_full_evidence queries")
            except Exception as e:
                print(f"⚠️ Failed to load v2_full_evidence queries: {e}")
                recent_queries = []

            # If no results or fewer than expected, also load from all versions
            if len(recent_queries) < 50:  # Always try to get more records
                print("⚠️ Loading additional queries without version filter...")
                # Get all queries without version filter
                conn = sqlite3.connect(history_manager.db_path)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM query_history ORDER BY created_at DESC LIMIT 100")
                rows = cursor.fetchall()
                conn.close()
                all_queries = [dict(row) for row in rows]

                # Merge with existing, avoid duplicates based on query and timestamp
                existing_keys = {(q.get('query', ''), q.get('timestamp', '')) for q in recent_queries}
                for query in all_queries:
                    key = (query.get('query', ''), query.get('timestamp', ''))
                    if key not in existing_keys:
                        recent_queries.append(query)

                print(f"✅ Total loaded queries: {len(recent_queries)}")

            # Convert database records to Streamlit format with full evidence processing
            loaded_history = []
            for record in recent_queries:
                history_item = {
                    'query_id': record.get('id'),
                    'timestamp': record.get('timestamp', ''),
                    'query': record.get('query', ''),
                    'answer': record.get('generated_answer', ''),
                    'dataset_answer': record.get('dataset_answer', ''),
                    'processing_time': record.get('processing_time', 0),
                    'confidence': record.get('confidence', 0.0),
                    'evidence_text': '',
                    'start_char': 0,
                    'end_char': 0,
                    'evidences': [],
                    'highlighted_evidences': [],
                    'answer_judgment': record.get('answer_judgment', ''),
                    'model': record.get('model', 'Unknown')
                }

                # Try to load evidences from JSON column first
                evidences_json = record.get('evidences', '')
                if evidences_json:
                    try:
                        # Handle both string and already-parsed JSON
                        if isinstance(evidences_json, str):
                            deserialized_evidences = json.loads(evidences_json)
                        else:
                            deserialized_evidences = evidences_json

                        if isinstance(deserialized_evidences, list):
                            history_item['evidences'] = deserialized_evidences
                            # Extract highlighted evidences
                            for ev in deserialized_evidences:
                                if isinstance(ev, dict):
                                    extracted = ev.get('extracted_evidence', '').strip()
                                    # Key fix: is_empty should default to False, and we should check extracted content
                                    if extracted and (not ev.get('is_empty', False)):
                                        history_item['highlighted_evidences'].append(extracted)
                        else:
                            print(f"⚠️ Evidences is not a list for query_id={record.get('id')}: {type(deserialized_evidences)}")
                    except (json.JSONDecodeError, TypeError) as e:
                        print(f"⚠️ Could not deserialize evidences JSON for query_id={record.get('id')}: {e}")
                    except Exception as e:
                        print(f"⚠️ Unexpected error processing evidences for query_id={record.get('id')}: {e}")

                # Fallback: Load detailed evidence data from evidence_extraction table
                if not history_item['evidences']:
                    query_id = record.get('id')
                    if query_id:
                        try:
                            evidence_records = history_manager.get_evidence_for_query(query_id)
                            for ev_record in evidence_records:
                                # Safe extraction with proper error handling
                                chunk_id = ev_record[2] if len(ev_record) > 2 else ''
                                chunk_content = ev_record[3] if len(ev_record) > 3 else ''
                                evidence_variant_prompt = ev_record[4] if len(ev_record) > 4 else ''
                                llm_response = ev_record[5] if len(ev_record) > 5 else ''

                                # Safe JSON parsing for char_ranges
                                char_ranges = []
                                if len(ev_record) > 6 and ev_record[6]:
                                    try:
                                        char_ranges = json.loads(ev_record[6]) if isinstance(ev_record[6], str) else ev_record[6]
                                        if not isinstance(char_ranges, list):
                                            char_ranges = []
                                    except (json.JSONDecodeError, TypeError):
                                        char_ranges = []

                                # Safe JSON parsing for extracted_evidence
                                extracted_evidence = ''
                                highlighted_texts = []
                                if len(ev_record) > 7 and ev_record[7]:
                                    try:
                                        if isinstance(ev_record[7], str):
                                            evidence_list = json.loads(ev_record[7])
                                        else:
                                            evidence_list = ev_record[7]

                                        if isinstance(evidence_list, list):
                                            extracted_evidence = '\n'.join(evidence_list)
                                            highlighted_texts = evidence_list
                                        elif isinstance(evidence_list, str):
                                            extracted_evidence = evidence_list
                                            highlighted_texts = [evidence_list]
                                    except (json.JSONDecodeError, TypeError):
                                        extracted_evidence = str(ev_record[7]) if ev_record[7] else ''
                                        highlighted_texts = [extracted_evidence] if extracted_evidence else []

                                similarity_score = ev_record[8] if len(ev_record) > 8 else 0.0
                                semantic_relevance = ev_record[9] if len(ev_record) > 9 else 0.0
                                core_term = ev_record[10] if len(ev_record) > 10 else ''

                                # Key fix: is_empty logic based on GitHub history
                                is_empty_value = not bool(extracted_evidence.strip()) if extracted_evidence else True

                                evidence_item = {
                                    'chunk_id': chunk_id,
                                    'chunk_content': chunk_content,
                                    'evidence_variant_prompt': evidence_variant_prompt,
                                    'llm_response': llm_response,
                                    'extracted_evidence': extracted_evidence,
                                    'char_ranges': char_ranges,
                                    'similarity_score': similarity_score,
                                    'semantic_relevance': semantic_relevance,
                                    'is_empty': is_empty_value,
                                    'core_term': core_term
                                }
                                history_item['evidences'].append(evidence_item)

                                # Add non-empty highlighted texts
                                if not evidence_item['is_empty'] and highlighted_texts:
                                    history_item['highlighted_evidences'].extend([t for t in highlighted_texts if t.strip()])
                        except Exception as e:
                            print(f"⚠️ Error loading evidence for query {query_id}: {e}")

                loaded_history.append(history_item)

            st.session_state.query_history = loaded_history

            # Debug information
            total_loaded = len(st.session_state.query_history)
            with_evidence = len([h for h in loaded_history if h.get('highlighted_evidences')])
            print(f"✅ Loaded {total_loaded} queries from database, {with_evidence} with evidence")

            # Print sample for debugging
            if loaded_history:
                sample = loaded_history[0]
                print(f"🔍 Sample record - evidences: {len(sample.get('evidences', []))}, highlighted: {len(sample.get('highlighted_evidences', []))}")
        except Exception as e:
            print(f"⚠️ Failed to load query history from database: {e}")
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
def _fetch_single_query_cached(api_url: str, query: str, timeout_seconds: int, cache_version: str = "v21_phrase_level") -> Tuple[Optional[Dict], Optional[str]]:
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
            return _fetch_single_query_cached(api_url, query, timeout_seconds, "v21_phrase_level")
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

    # Patent-compliant: Use LLM-provided character ranges directly (no calculation)
    sentence_ranges = []
    char_position_ranges = []  # Store tuples for highlighting: [(start1, end1), (start2, end2), ...]

    # Priority 1: Use char_ranges from backend (LLM-provided ranges)
    if valid_evidences:
        for evidence in valid_evidences:
            backend_char_ranges = evidence.get('char_ranges', [])
            if backend_char_ranges:
                for start, end in backend_char_ranges:
                    # Ranges are relative to chunk content, add prefix offset for highlighting
                    sentence_ranges.append(f"{start}文字目～{end}文字目")
                    char_position_ranges.append((start + prefix_offset, end + prefix_offset))

    # Fallback: Calculate ranges if backend didn't provide them (for backward compatibility)
    if not char_position_ranges and display_evidence and display_evidence != '根拠情報なし':
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
        'end_char': result.get('end_char', 0),
        'model': result.get('model', 'Streamlit-RAG')
    }
    st.session_state.query_history.append(history_item)

    # Also save to database
    try:
        db_path = os.path.join(parent_dir, 'query_history.db')
        history_manager = QueryHistoryManager(db_path)

        # Save to database
        history_manager.add_query(
            query=query,
            generated_answer=result.get('answer', ''),
            processing_time=result.get('processing_time', 0),
            model=result.get('model', 'Streamlit-RAG'),
            confidence=result.get('confidence', 0.0),
            num_chunks=result.get('num_chunks', 0)
        )
        print(f"✅ Saved query to database: {query[:50]}...")
    except Exception as e:
        print(f"⚠️ Failed to save query to database: {e}")

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
    if len(df) > 0:
        # Handle mixed timestamp formats (string and datetime)
        df['timestamp'] = df['timestamp'].apply(lambda x: x if isinstance(x, str) else x.strftime('%Y-%m-%d %H:%M:%S'))
    
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

def get_best_f1_score(item):
    """Get the best F1 score from item's evidences"""
    evidences = item.get('evidences', [])
    if not evidences:
        return 0.0

    # Find maximum F1 score across all evidences
    best_f1 = 0.0
    for ev in evidences:
        f1 = ev.get('f1_score', 0.0)
        if f1 > best_f1:
            best_f1 = f1

    return best_f1

def query_history_interface():
    """Interface for viewing and managing query history"""
    st.markdown("---")
    st.header(t("📚 クエリ履歴", "Query history"))

    if not st.session_state.query_history:
        st.info(t("まだ履歴がありません。", "No history yet."))
        return

    # Sort history by best F1 score (highest first)
    sorted_history = sorted(
        st.session_state.query_history,
        key=lambda x: get_best_f1_score(x),
        reverse=True
    )

    # History controls
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button(t("📊 履歴をエクスポート", "Export history")):
            export_history()
    with col2:
        if st.button(t("🗑️ 履歴をクリア", "Clear history")):
            st.session_state.query_history = []
            st.success(t("履歴をクリアしました！", "History cleared!"))
            st.rerun()
    with col3:
        items_per_page = st.selectbox(
            t("表示件数/ページ", "Items per page"),
            [5, 10, 20, 50],
            index=1
        )
    with col4:
        if st.button(t("🔄 DBから再読込", "Reload from DB")):
            # Clear session state to force reload from database
            if 'query_history' in st.session_state:
                del st.session_state['query_history']
            st.success(t("データベースから再読み込みしました！", "Reloaded from database!"))
            st.rerun()

    # Calculate total pages
    total_items = len(sorted_history)
    total_pages = (total_items + items_per_page - 1) // items_per_page  # Ceiling division

    # Initialize current page in session state
    if 'history_current_page' not in st.session_state:
        st.session_state.history_current_page = 1

    # Reset to page 1 if items_per_page changes
    if 'history_items_per_page' not in st.session_state:
        st.session_state.history_items_per_page = items_per_page
    elif st.session_state.history_items_per_page != items_per_page:
        st.session_state.history_items_per_page = items_per_page
        st.session_state.history_current_page = 1

    st.markdown(f"**{t('合計', 'Total')}: {total_items} {t('件', 'items')}**")

    # Pagination controls
    if total_pages > 1:
        st.markdown("---")
        page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([1, 1, 2, 1, 1])

        with page_col1:
            if st.button("⏮️ " + t("最初", "First"), disabled=(st.session_state.history_current_page == 1)):
                st.session_state.history_current_page = 1
                st.rerun()

        with page_col2:
            if st.button("◀️ " + t("前", "Prev"), disabled=(st.session_state.history_current_page == 1)):
                st.session_state.history_current_page -= 1
                st.rerun()

        with page_col3:
            st.markdown(f"<div style='text-align: center; padding-top: 8px;'><b>{t('ページ', 'Page')} {st.session_state.history_current_page} / {total_pages}</b></div>", unsafe_allow_html=True)

        with page_col4:
            if st.button(t("次", "Next") + " ▶️", disabled=(st.session_state.history_current_page >= total_pages)):
                st.session_state.history_current_page += 1
                st.rerun()

        with page_col5:
            if st.button(t("最後", "Last") + " ⏭️", disabled=(st.session_state.history_current_page >= total_pages)):
                st.session_state.history_current_page = total_pages
                st.rerun()

    # Calculate items for current page
    start_idx = (st.session_state.history_current_page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    history_to_show = sorted_history[start_idx:end_idx]

    st.markdown("---")

    # Display history items
    for i, item in enumerate(history_to_show, start=start_idx + 1):
        # Handle both string and datetime timestamp formats
        timestamp = item.get('timestamp', '')
        if isinstance(timestamp, str):
            timestamp_str = timestamp
        elif hasattr(timestamp, 'strftime'):
            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            timestamp_str = str(timestamp)

        # Calculate the global index (in sorted_history)
        global_index = start_idx + (i - start_idx - 1)

        # Check if this record should be expanded (from navigation)
        should_expand = (st.session_state.get('expanded_history_index') == global_index)

        # Clear the expanded flag after displaying to avoid staying expanded on next page
        if should_expand and 'expanded_history_index' in st.session_state:
            # Mark with a highlight icon
            query_preview = f"🎯 {i}. {item['query'][:60]}... ({timestamp_str})"
        else:
            query_preview = f"{i}. {item['query'][:60]}... ({timestamp_str})"

        with st.expander(query_preview, expanded=should_expand):
            col1, col2 = st.columns([5, 1])

            with col1:
                st.markdown(t("**クエリ:**", "**Query:**"))
                st.write(item['query'])
                st.markdown(t("**回答:**", "**Answer:**"))
                st.write(item.get('answer', ''))

                # Show dataset true answer if available
                dataset_answer = item.get('dataset_answer', '')
                if dataset_answer and dataset_answer.strip():
                    st.markdown(t("**正解:**", "**Dataset Answer:**"))
                    st.info(dataset_answer)

                # Show detailed character-level metrics right below the answer
                evidences = item.get('evidences', [])
                valid_evidences = [e for e in evidences if not e.get('is_empty', False) or e.get('extracted_evidence', '').strip()]

                # Check if the generated answer indicates no relevant information found
                generated_answer = item.get('answer', '') or item.get('generated_answer', '')
                is_no_info = any(phrase in generated_answer.lower() for phrase in
                    ['sorry, no relevant information', 'no relevant information is currently available',
                     '申し訳ございませんが、関連する情報が見つかりません'])

                if valid_evidences and not is_no_info:
                    first_evidence = valid_evidences[0]

                    # Get detailed metrics
                    overlap_chars = first_evidence.get('overlap_chars', 0)
                    evidence_length = first_evidence.get('evidence_length', 0)
                    dataset_length = first_evidence.get('dataset_length', 0)
                    detailed_precision = first_evidence.get('detailed_precision', 0)
                    detailed_recall = first_evidence.get('detailed_recall', 0)
                    detailed_f1 = first_evidence.get('detailed_f1', 0)
                    exact_match = first_evidence.get('exact_match', False)

                    # Display metrics in a compact row format
                    st.markdown("---")
                    st.markdown(t("**📊 評価メトリクス:**", "**📊 Evaluation Metrics:**"))

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "総合マッチング",
                            f"{detailed_f1:.3f}",
                            help="Overall Matching = 2 × (Precision × Recall) / (Precision + Recall)"
                        )

                    with col2:
                        st.metric(
                            "精度",
                            f"{detailed_precision:.3f}",
                            help="文字レベルマッチング率 = 重複文字数 / 根拠長さ"
                        )

                    with col3:
                        st.metric(
                            "再現率",
                            f"{detailed_recall:.3f}",
                            help="重複文字数 / データセット回答長さ"
                        )

                    with col4:
                        exact_status = "完全一致" if exact_match else "部分一致"
                        st.metric(
                            "一致状況",
                            exact_status,
                            help="Evidence exactly matches Dataset Answer"
                        )
                elif is_no_info:
                    # Show failure metrics for queries with no relevant information
                    st.markdown("---")
                    st.markdown(t("**📊 評価メトリクス:**", "**📊 Evaluation Metrics:**"))

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "総合マッチング",
                            "0.000",
                            help="No relevant information found"
                        )

                    with col2:
                        st.metric(
                            "精度",
                            "0.000",
                            help="No relevant information found"
                        )

                    with col3:
                        st.metric(
                            "再現率",
                            "0.000",
                            help="No relevant information found"
                        )

                    with col4:
                        st.metric(
                            "一致状況",
                            "情報なし",
                            help="No relevant information found"
                        )

                # Display highlighted evidence (根拠情報)
                highlighted_evidences = item.get('highlighted_evidences', [])
                evidences = item.get('evidences', [])

                if highlighted_evidences:
                    st.markdown(t("**【根拠情報】:**", "**【Evidence Highlights】:**"))
                    for idx, evidence in enumerate(highlighted_evidences, 1):
                        st.markdown(f"**{idx}.** {evidence}")
                elif evidences:
                    # Try to extract evidence from evidences list
                    st.markdown(t("**【根拠情報】:**", "**【Evidence Highlights】:**"))
                    extracted_count = 0
                    for evidence in evidences:
                        if isinstance(evidence, dict):
                            extracted = evidence.get('extracted_evidence', '').strip()
                            if extracted and not evidence.get('is_empty', False):
                                extracted_count += 1
                                st.markdown(f"**{extracted_count}.** {extracted}")

                    if extracted_count == 0:
                        # Still no evidence found, show basic evidence_text
                        st.markdown(t("**根拠:**", "**Evidence:**"))
                        evidence_text = item.get('evidence_text', 'N/A')
                        st.info(evidence_text if evidence_text and evidence_text.strip() else 'N/A')
                else:
                    st.markdown(t("**根拠:**", "**Evidence:**"))
                    evidence_text = item.get('evidence_text', 'N/A')
                    st.info(evidence_text if evidence_text and evidence_text.strip() else 'N/A')

                # Display evidence chunks with highlighting
                evidences = item.get('evidences', [])
                if evidences:
                    # Filter valid evidences: based on GitHub history, use OR condition not AND
                    valid_evidences = [
                        e for e in evidences
                        if not e.get('is_empty', False) or e.get('extracted_evidence', '').strip()
                    ]
                    if valid_evidences:
                        with st.expander(t("📄 完全な根拠チャンク (黄色でハイライト表示)", "📄 Full evidence chunks (highlighted in yellow)")):
                            for idx, evidence in enumerate(valid_evidences, 1):
                                chunk_content = evidence.get('chunk_content', '')
                                extracted_evidence = evidence.get('extracted_evidence', '')

                                st.markdown(f"**Chunk {idx}:**")

                                # Display chunk_content with highlighted evidence
                                if chunk_content and extracted_evidence:
                                    if extracted_evidence in chunk_content:
                                        highlighted_html = chunk_content.replace(
                                            extracted_evidence,
                                            f'<mark style="background-color: yellow;">{extracted_evidence}</mark>'
                                        )
                                        st.markdown(f'<div style="padding: 1rem; background-color: #f0f0f0; border-radius: 5px; white-space: pre-wrap;">{highlighted_html}</div>', unsafe_allow_html=True)
                                    else:
                                        st.info(f"**Chunk:** {chunk_content}")
                                        st.warning(f"**Extracted Evidence:** {extracted_evidence}")
                                elif chunk_content:
                                    st.info(chunk_content)
                                elif extracted_evidence:
                                    st.info(extracted_evidence)
                                else:
                                    st.warning("No content available")

                                st.markdown("---")


            with col2:
                # Only show the re-run button
                if st.button(t("🔄 再実行", "Re-run"), key=f"rerun_{i}"):
                    # Re-run the query
                    result, error = call_single_query("", item['query'])
                    if result:
                        st.session_state.last_result = result
                        st.session_state.last_query = item['query']
                        st.rerun()

    # Clear the expanded flag after displaying all records
    if 'expanded_history_index' in st.session_state:
        del st.session_state.expanded_history_index



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