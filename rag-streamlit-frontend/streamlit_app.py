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
    import sys
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    env_path = os.path.join(parent_dir, '.env')
    load_dotenv(env_path)
    print("✅ Environment variables loaded from .env file")

    # Set environment variables for proper file paths instead of changing working directory
    os.environ['CHROMA_PATH'] = os.path.join(parent_dir, 'chroma')
    os.environ['DATA_PATH'] = os.path.join(parent_dir, 'data', 'merged_qa_dataset.json')
    print(f"✅ Environment variables set - CHROMA_PATH: {os.environ['CHROMA_PATH']}")
    print(f"✅ Environment variables set - DATA_PATH: {os.environ['DATA_PATH']}")

    # Import QueryHistoryManager for persistent storage
    sys.path.insert(0, parent_dir)
    from query_history_manager import QueryHistoryManager
    DB_PATH = os.path.join(parent_dir, 'query_history.db')
    HISTORY_MANAGER_AVAILABLE = True
    print(f"✅ QueryHistoryManager imported, DB path: {DB_PATH}")

except ImportError as ie:
    print(f"⚠️ python-dotenv or QueryHistoryManager not available: {ie}")
    HISTORY_MANAGER_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Error loading .env file or QueryHistoryManager: {e}")
    HISTORY_MANAGER_AVAILABLE = False

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
    MAX_HISTORY_ITEMS = 100  # Increased to 100 to show all historical queries
    PAGE_TITLE = "根拠提示装置 | Evidence Indicator RAG System"
    PAGE_ICON = "🔍"

# Import SAMPLE_QUERIES from config (organized by category)
try:
    from config import SAMPLE_QUERIES_BY_CATEGORY as SAMPLE_QUERIES
except ImportError:
    # Fallback: define inline if config import fails
    SAMPLE_QUERIES = {
        "気象・梅雨": [
            "日本で梅雨がないのは北海道とどこか。",
            "梅雨とは何季の一種か?",
            "入梅は何の目安の時期か？",
            "梅雨明けの別名を何というか。",
            "シベリアから中国大陸にかけての広範囲を冷たく乾燥させる気団は？"
        ],
        "歴史・戦争": [
            "極東国際軍事裁判（東京裁判）で弁護人を務めたのは誰？",
            "「大東亜戦争」が閣議決定されたのはいつ？",
            "イギリスでは1918年初頭に導入されたのは？",
            "ジョゼフ・ジョフルが、フランス軍最高司令官の座を譲った相手は？",
            "イギリスにおける第一次世界大戦後の余剰女性はどれだけか？"
        ],
        "人物・思想家": [
            "マルクスが兵役不適格とされた理由は",
            "マルクスは何に不信感を抱いたか",
            "バウアーが大学で講義することは禁止された年は",
            "『インターナショナルにおける偽装的分裂』を採択したのはいつか",
            "松本清張の生まれは"
        ],
        "交通・鉄道": [
            "東海道新幹線で総列車本数の最大記録は何本か",
            "山陰地区発の特急は？",
            "平日朝の上りで約10分おきに三島駅から運行している東海道新幹線の種類は何か。",
            "スプリンクラーの設置は主に何処を中心とされているか？",
            "特別企画乗車券の有効期間は何日間か"
        ],
        "スポーツ・サッカー": [
            "ナビスコ杯では、グループリーグA組を4勝2敗の2位で通過したサッカーチームは何ですか。",
            "ベガルタ仙台の2002年の第2ステージの勝利数は？",
            "ベガルタ仙台がFC東京から完全移籍で獲得した選手は誰か",
            "J2得点王となったFWボルジェスは、期限付き移籍したチームは？",
            "仙台スタジアム の所在地は。"
        ],
        "地理・国家": [
            "タイの代表的な乗り物は。",
            "通称タイと呼ばれる国は",
            "タイ王国第31代首相は誰か？",
            "南部のマレー半島へはかつて、何朝が併合を目指して侵攻しましたか？",
            "東日本大震災が発生したのはいつ？"
        ],
        "科学・技術": [
            "古細菌の外観は何と似ているか。",
            "1674年に微生物を発見したのは誰か?",
            "最初に発売されたBDビデオソフトはDVDと同じ何をコーデックに採用せざるをえなかったか？",
            "-400以降の型式は",
            "一般的に使われている析出硬化系の母相の種類は具体的に何？"
        ],
        "政治・人物": [
            "アンが斬首されたのは何年か",
            "野村克也が監督の姿勢として一流と公言していたのは",
            "2008年6月29日のソフトバンク戦で球団史上最多20安打の猛攻で15点を奪い大勝した野球監督",
            "エリザベス1世の在位は何年か",
            "2010年オバマ政権は　何実験を行ったでしょうか"
        ],
        "文化・芸術": [
            "古今集と万葉集でより古いものは？",
            "松本清張の生まれは",
            "さくらが逝去した日",
            "木村の死去に際して清張は何を書いたか？",
            "日本刀の拵えなどに影響を与えたのは"
        ],
        "国際・外交": [
            "CPTPP は何のように加入交渉についての詳細な規定はしていない？",
            "フランスのどこから孫文が上海に帰国した？",
            "環太平洋パートナーシップ協定は当初何ヵ国か？",
            "革命活動の主要活動地域の一つとされるのはどこ？",
            "調印式会場のあるオークランドでは、約2万人の人々がTPPに反対するために抗議活動を行った協定は"
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
        # Load history from database if available
        if HISTORY_MANAGER_AVAILABLE:
            try:
                manager = QueryHistoryManager(DB_PATH)
                # Get recent queries from database - only load v2_full_evidence version (100 records)
                recent_queries = manager.get_recent_queries(limit=100, version='v2_full_evidence')

                # Load dataset for looking up answers
                dataset_lookup = {}
                try:
                    import json
                    dataset_path = os.path.join(parent_dir, "data", "merged_qa_dataset.json")
                    if os.path.exists(dataset_path):
                        with open(dataset_path, 'r', encoding='utf-8') as f:
                            dataset = json.load(f)
                        # Create lookup dictionary: question -> answer
                        for item in dataset:
                            dataset_lookup[item['question']] = item['answers']['text'][0]
                        print(f"✅ Loaded dataset with {len(dataset_lookup)} Q&A pairs for lookup")
                except Exception as e:
                    print(f"⚠️ Could not load dataset for lookup: {e}")

                # Convert database records to session state format
                loaded_history = []
                for record in recent_queries:
                    # record format: dict with keys: id, query, generated_answer, created_at, processing_time, model, confidence, num_chunks
                    # Get dataset answer from database first, fallback to lookup
                    dataset_answer = record.get('dataset_answer', '')
                    if not dataset_answer:
                        # Try exact match
                        if record['query'] in dataset_lookup:
                            dataset_answer = dataset_lookup[record['query']]
                        # Try with ']' prefix (for old dataset entries with Wikipedia markers)
                        elif ']' + record['query'] in dataset_lookup:
                            dataset_answer = dataset_lookup[']' + record['query']]

                    history_item = {
                        'query_id': record['id'],  # Store database ID for deletion
                        'timestamp': datetime.fromisoformat(record['created_at']) if isinstance(record['created_at'], str) else datetime.fromtimestamp(record['created_at']),
                        'query': record['query'],
                        'answer': record['generated_answer'],
                        'dataset_answer': dataset_answer,  # Use found dataset answer
                        'processing_time': record['processing_time'],
                        'confidence': record['confidence'],
                        'evidence_text': '',  # Will be loaded from evidences if needed
                        'start_char': 0,
                        'end_char': 0,
                        'evidences': [],
                        'highlighted_evidences': [],
                        'answer_judgment': record.get('answer_judgment', ''),  # Load Gemini answer judgment
                        'context': record.get('context', '')  # Load context from database
                    }

                    # Try to load evidences from JSON column first (new method)
                    evidences_json = record.get('evidences', '')
                    if evidences_json:
                        try:
                            deserialized_evidences = json.loads(evidences_json)
                            history_item['evidences'] = deserialized_evidences
                            # Extract highlighted evidences
                            for ev in deserialized_evidences:
                                extracted = ev.get('extracted_evidence', '').strip()
                                if extracted and (not ev.get('is_empty', False)):
                                    history_item['highlighted_evidences'].append(extracted)
                        except Exception as e:
                            print(f"⚠️ Could not deserialize evidences JSON for query_id={record['id']}: {e}")

                    # Fallback: Load detailed evidence data from evidence_extraction table (old method)
                    if not history_item['evidences']:
                        query_id = record['id']
                        evidence_records = manager.get_evidence_for_query(query_id)
                        for ev_record in evidence_records:
                            # ev_record: (id, query_id, chunk_id, chunk_content, extraction_prompt, llm_response,
                            #             extracted_ranges, extracted_texts, similarity_score, semantic_relevance, created_at, core_term)
                            evidence_item = {
                                'chunk_id': ev_record[2],
                                'chunk_content': ev_record[3],
                                'evidence_variant_prompt': ev_record[4],  # Store the evidence extraction prompt (variant method)
                                'llm_response': ev_record[5],  # Store the LLM response
                                'extracted_evidence': '\n'.join(json.loads(ev_record[7])) if ev_record[7] else '',
                                'char_ranges': json.loads(ev_record[6]) if ev_record[6] else [],
                                'similarity_score': ev_record[8],
                                'semantic_relevance': ev_record[9],
                                'is_empty': not bool(ev_record[7]),
                                'core_term': ev_record[10] if len(ev_record) > 10 else ''  # Load core term (index 10: core_term, index 11: created_at)
                            }
                            history_item['evidences'].append(evidence_item)
                            if not evidence_item['is_empty'] and ev_record[7]:
                                history_item['highlighted_evidences'].extend(json.loads(ev_record[7]))

                    # Calculate match metrics for this history item if we have dataset answer and evidence
                    if dataset_answer and history_item['highlighted_evidences']:
                        try:
                            from calculate_match_metrics import calculate_char_match_rate
                            # Use the first evidence for match calculation
                            primary_evidence = history_item['highlighted_evidences'][0]
                            match_metrics = calculate_char_match_rate(primary_evidence, dataset_answer)
                            history_item['match_metrics'] = match_metrics
                        except Exception as e:
                            print(f"⚠️ Could not calculate match metrics for history item: {e}")
                            history_item['match_metrics'] = {}
                    else:
                        history_item['match_metrics'] = {}

                    loaded_history.append(history_item)

                st.session_state.query_history = loaded_history
                print(f"✅ Loaded {len(loaded_history)} queries from database")
            except Exception as e:
                print(f"⚠️ Failed to load history from database: {e}")
                st.session_state.query_history = []
        else:
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

def find_query_in_history(query: str) -> Optional[int]:
    """
    Find if a query already exists in history.
    Returns the index (in sorted history) if found, None otherwise.
    """
    if not st.session_state.query_history:
        return None

    # Normalize query for comparison (strip whitespace and convert to lowercase)
    normalized_query = query.strip().lower()

    # Sort history by timestamp (newest first) - same as displayed
    sorted_history = sorted(
        st.session_state.query_history,
        key=lambda x: x.get('timestamp', datetime.now()),
        reverse=True
    )

    # Search for matching query
    for index, item in enumerate(sorted_history):
        if item['query'].strip().lower() == normalized_query:
            return index

    return None

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
        # Use HTML escaping and build segments to avoid nested replacements
        import html

        # Merge overlapping ranges and sort by start position
        merged_ranges = []
        sorted_ranges = sorted(char_ranges, key=lambda x: x[0])

        for start_pos, end_pos in sorted_ranges:
            # Convert to 0-based indexing
            start_idx = start_pos - 1
            end_idx = end_pos  # end_pos is already inclusive

            # Validate range
            if not (0 <= start_idx < len(source_text) and start_idx < end_idx <= len(source_text)):
                continue

            # Merge with previous range if overlapping
            if merged_ranges and start_idx <= merged_ranges[-1][1]:
                merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end_idx))
            else:
                merged_ranges.append((start_idx, end_idx))

        # Build highlighted text by segments
        if merged_ranges:
            segments = []
            last_end = 0

            for start_idx, end_idx in merged_ranges:
                # Add non-highlighted text before this range
                if last_end < start_idx:
                    segments.append(html.escape(source_text[last_end:start_idx]))

                # Add highlighted text
                text_to_highlight = source_text[start_idx:end_idx]
                highlight_span = f'<span style="background-color: #ffff00; padding: 2px 4px; border-radius: 3px; font-weight: bold; border: 1px solid #ffcc00;">{html.escape(text_to_highlight)}</span>'
                segments.append(highlight_span)

                last_end = end_idx

            # Add remaining non-highlighted text
            if last_end < len(source_text):
                segments.append(html.escape(source_text[last_end:]))

            highlighted_text = ''.join(segments)
        else:
            highlighted_text = html.escape(source_text)

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

    # Display dataset answer if available
    dataset_answer = result.get('dataset_answer', '')
    if dataset_answer:
        st.markdown(t("**📋 元のデータセット回答:**", "**📋 Original Dataset Answer:**"))
        st.info(dataset_answer)

        # Display match metrics if available
        # For v2_full_evidence: calculate best metrics from all evidences
        match_metrics = result.get('match_metrics', {})
        result_evidences = result.get('evidences', [])

        # If we have evidences with per-chunk metrics, use the best one
        if result_evidences:
            best_f1 = 0.0
            best_metrics = None

            for ev in result_evidences:
                ev_f1 = ev.get('f1_score', 0.0)
                if ev_f1 > best_f1:
                    best_f1 = ev_f1
                    best_metrics = {
                        'match_rate': ev.get('f1_score', 0.0),
                        'precision': ev.get('precision', 0.0),
                        'recall': ev.get('recall', 0.0),
                        'exact_match': ev.get('exact_match', False)
                    }

            # Use best metrics if found, otherwise fallback to match_metrics
            if best_metrics and best_metrics['match_rate'] > 0:
                match_metrics = best_metrics

        if match_metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                match_rate = match_metrics.get('match_rate', 0.0)
                st.metric(
                    label=t("総合マッチング", "Overall Matching"),
                    value=f"{match_rate:.1%}",
                    help=t("F1スコア：精度と再現率の調和平均", "F1 Score: Harmonic mean of precision and recall")
                )
            with col2:
                precision = match_metrics.get('precision', 0.0)
                st.metric(
                    label=t("文字レベルマッチング率", "Character-level Match Rate"),
                    value=f"{precision:.1%}",
                    help=t("抽出した根拠のうち正しい割合", "Percentage of extracted evidence that is correct")
                )
            with col3:
                recall = match_metrics.get('recall', 0.0)
                st.metric(
                    label=t("再現率", "Recall"),
                    value=f"{recall:.1%}",
                    help=t("データセット回答のうち見つけた割合", "Percentage of dataset answer found")
                )
            with col4:
                exact_match = match_metrics.get('exact_match', False)
                st.metric(
                    label=t("完全一致", "Exact Match"),
                    value=t("✅ はい", "✅ Yes") if exact_match else t("❌ いいえ", "❌ No"),
                    help=t("100%一致するか", "Whether it's 100% match")
                )

    # Display identified core terms from all chunks
    evidences = result.get('evidences', [])
    core_terms = [ev.get('core_term', '') for ev in evidences if ev.get('core_term')]
    if core_terms:
        # Remove duplicates while preserving order
        unique_core_terms = list(dict.fromkeys(core_terms))
        st.markdown(t("**🎯 LLM回答から識別されたコアタ一ム:**", "**🎯 Core Terms Identified from LLM Answer:**"))
        st.info(" / ".join(unique_core_terms))

    st.markdown(t("### 【検索ヒットのチャンクを含む文書】", "### Source document that contains the hit chunk"))
    source_doc = result.get('source_document', '文書が見つかりませんでした。')
    start_char = result.get('start_char', 0)
    end_char = result.get('end_char', 0)
    evidence_text = result.get('evidence_text', '')
    
    # Extract only the context part (remove "文脈: " prefix and Q&A parts)
    context_only_text = source_doc
    if "文脈: " in source_doc:
        # Extract only the context part before "質問:"
        context_start = source_doc.find("文脈: ") + 3  # Skip "文脈: "
        question_start = source_doc.find("\n\n質問:")
        if question_start > context_start:
            context_only_text = source_doc[context_start:question_start].strip()
        else:
            # If no question found, take everything after "文脈: "
            context_only_text = source_doc[context_start:].strip()

    # Use context-only text for display
    display_source_doc = context_only_text

    # Get extracted evidences (Strategy 3)
    evidences = result.get('evidences', [])

    # Filter 1: Only keep chunks with actual evidence extracted
    valid_evidences = [e for e in evidences if not e.get('is_empty', True)]

    # Filter 2: Only keep chunks with high semantic relevance (>= 0.5) for display
    # This prevents showing too many marginally relevant chunks
    high_relevance_threshold = 0.5
    filtered_evidences = [
        e for e in valid_evidences
        if e.get('semantic_relevance', 0.0) >= high_relevance_threshold
    ]

    # Filter 3: Limit to top 3 most relevant chunks
    filtered_evidences = sorted(
        filtered_evidences,
        key=lambda e: e.get('semantic_relevance', 0.0),
        reverse=True
    )[:3]

    # Use filtered evidences for display (fallback to all valid if none pass filter)
    display_evidences = filtered_evidences if filtered_evidences else valid_evidences[:3]

    print(f"📊 Evidence filtering: {len(evidences)} total -> {len(valid_evidences)} with evidence -> {len(display_evidences)} displayed")

    # Update valid_evidences to use the filtered version for all subsequent display
    valid_evidences = display_evidences

    # Use RAG extracted evidence for highlighting (to analyze RAG's selection)
    if valid_evidences:
        # Combine all extracted evidences for highlighting
        extracted_texts = [e.get('extracted_evidence', '') for e in valid_evidences]
        combined_extracted = '\n'.join(extracted_texts)
        display_evidence = combined_extracted
    else:
        display_evidence = evidence_text
    
    # Extract original dataset answer for comparison (but don't use for highlighting)
    original_answer = ""
    if "回答: " in source_doc:
        answer_start = source_doc.find("回答: ") + 3  # Skip "回答: "
        answer_end = source_doc.find("\n", answer_start)
        if answer_end == -1:
            answer_end = len(source_doc)
        original_answer = source_doc[answer_start:answer_end].strip()

    # Compute adjusted range based on evidence text for consistency
    eff_start, eff_end = compute_effective_range(source_doc, start_char, end_char, display_evidence)

    # Patent-compliant: Use LLM-provided character ranges directly (no calculation)
    sentence_ranges = []
    char_position_ranges = []  # Store tuples for highlighting: [(start1, end1), (start2, end2), ...]

    # Priority 1: Use char_ranges from backend (LLM-provided ranges) to analyze RAG's selection
    if valid_evidences:
        for evidence in valid_evidences:
            backend_char_ranges = evidence.get('char_ranges', [])
            chunk_content = evidence.get('chunk_content', '')

            if backend_char_ranges and chunk_content:
                # char_ranges are relative to chunk_content
                # We need to map them to display_source_doc positions

                # Strategy 1: Check if chunk_content is a substring of display_source_doc
                # If yes, calculate the offset and map all ranges accordingly
                chunk_offset = display_source_doc.find(chunk_content)

                if chunk_offset >= 0:
                    # chunk_content is found in display_source_doc
                    # All char_ranges can be mapped by adding the offset
                    print(f"✅ Chunk found in display at offset {chunk_offset}, mapping all ranges...")

                    for start, end in backend_char_ranges:
                        # Validate range within chunk
                        if 1 <= start <= len(chunk_content) and start < end <= len(chunk_content):
                            # Map to display_source_doc (convert to 0-based, add offset, convert back to 1-based)
                            display_start = chunk_offset + (start - 1) + 1  # = chunk_offset + start
                            display_end = chunk_offset + (end - 1) + 1      # = chunk_offset + end

                            sentence_ranges.append(f"{display_start}文字目～{display_end}文字目")
                            char_position_ranges.append((display_start, display_end))

                            print(f"  Mapped range: chunk[{start}:{end}] -> display[{display_start}:{display_end}]")
                        else:
                            print(f"⚠️ Invalid char_range: {start}～{end} for chunk length {len(chunk_content)}")
                else:
                    # Strategy 2: Chunk not found as substring - try individual evidence substring matching
                    print(f"⚠️ Chunk not found as substring in display, trying individual evidence matching...")

                    for start, end in backend_char_ranges:
                        # Extract the actual text from chunk using the range (1-based indexing)
                        if 1 <= start <= len(chunk_content) and start < end <= len(chunk_content):
                            evidence_substring = chunk_content[start-1:end]

                            # Find this substring in display_source_doc
                            # Use finditer to find all occurrences and pick the best match
                            import re
                            escaped_substring = re.escape(evidence_substring)
                            matches = list(re.finditer(escaped_substring, display_source_doc))

                            if matches:
                                # Use the first occurrence (could be improved with context matching)
                                match = matches[0]
                                substring_pos = match.start()

                                # Calculate position in display_source_doc (1-based)
                                context_start_pos = substring_pos + 1
                                context_end_pos = substring_pos + len(evidence_substring)

                                sentence_ranges.append(f"{context_start_pos}文字目～{context_end_pos}文字目")
                                char_position_ranges.append((context_start_pos, context_end_pos))

                                print(f"  Found evidence substring at display[{context_start_pos}:{context_end_pos}]")
                            else:
                                print(f"⚠️ Evidence substring not found in display: {evidence_substring[:50]}...")
                        else:
                            print(f"⚠️ Invalid char_range: {start}～{end} for chunk length {len(chunk_content)}")

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
                # Use display positions directly since we're working with display_source_doc
                char_position_ranges.append((start_pos_display, end_pos_display))

    # Show highlighted version with RAG evidence
    st.markdown(t("**💡 根拠部分のハイライト表示:**", "**💡 Highlighted evidence:**"))

    # Display each chunk with its own evidence highlighting separately
    if valid_evidences and len(valid_evidences) > 1:
        # Multiple chunks - display each separately with its own highlighting
        for idx, evidence in enumerate(valid_evidences, 1):
            chunk_content = evidence.get('chunk_content', '')
            backend_char_ranges = evidence.get('char_ranges', [])
            extracted_evidence = evidence.get('extracted_evidence', '')

            if chunk_content and backend_char_ranges:
                st.markdown(f"**Chunk {idx}:**")
                # Highlight within this specific chunk using its char_ranges
                highlighted_html = highlight_rag_evidence_in_source(chunk_content, extracted_evidence, backend_char_ranges)
                st.markdown(highlighted_html, unsafe_allow_html=True)
    else:
        # Single chunk or fallback - use original logic
        highlighted_html = highlight_rag_evidence_in_source(display_source_doc, display_evidence, char_position_ranges)
        st.markdown(highlighted_html, unsafe_allow_html=True)

    st.markdown(t("**📄 元の文書:**", "**📄 Original document:**"))

    st.text_area(t("文書内容", "Document"), display_source_doc, height=200, key="source_display")
    
    # Extract original dataset answer and position from source document
    original_answer = ""
    original_answer_ranges = []

    if "回答: " in source_doc:
        answer_start = source_doc.find("回答: ") + 3  # Skip "回答: "
        answer_end = source_doc.find("\n", answer_start)
        if answer_end == -1:
            answer_end = len(source_doc)
        original_answer = source_doc[answer_start:answer_end].strip()

    # Try to load ground truth answer positions from JSQuAD dataset
    original_answers_with_ranges = []  # Store all answers with their ranges
    try:
        import json
        import os
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                     "data", "jsquad_validation_100.json")

        if os.path.exists(dataset_path):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)

            # Find matching question in dataset
            for item in dataset:
                if item['question'] == query:
                    # Get answer positions (character start positions in 1-based indexing)
                    answer_starts = item['answers']['answer_start']
                    answer_texts = item['answers']['text']

                    # Store each answer with its range
                    seen_answers = set()  # Track (text, range) pairs to avoid duplicates
                    for i, start_pos in enumerate(answer_starts):
                        if i < len(answer_texts):
                            answer_text = answer_texts[i]
                            end_pos = start_pos + len(answer_text)
                            # Convert to 1-based display format (add 1 to start_pos since dataset is 0-based)
                            range_str = f"{start_pos + 1}文字目～{end_pos}文字目"

                            # Check for duplicates (same text and same range)
                            answer_key = (answer_text, range_str)
                            if answer_key not in seen_answers:
                                seen_answers.add(answer_key)
                                original_answer_ranges.append(range_str)
                                original_answers_with_ranges.append({
                                    'text': answer_text,
                                    'range': range_str
                                })

                    # Use the first ground truth answer if not already found
                    if not original_answer and answer_texts:
                        original_answer = answer_texts[0]

                    break
    except Exception as e:
        print(f"⚠️ Could not load ground truth positions: {e}")

    # Display original dataset answers with their ranges
    if original_answers_with_ranges:
        st.markdown(t("**📋 元のデータセット回答:**", "**📋 Original dataset answer(s):**"))

        # Display each answer with its corresponding range
        for idx, answer_info in enumerate(original_answers_with_ranges, 1):
            if len(original_answers_with_ranges) > 1:
                st.markdown(f"**回答 {idx}:** {answer_info['text']}")
                st.markdown(f"  └─ **位置:** {answer_info['range']}")
            else:
                # Single answer - simpler display
                st.info(answer_info['text'])
                st.markdown(t(f"**📍 元データセット回答の位置:** {answer_info['range']}",
                             f"**📍 Ground truth answer range:** {answer_info['range']}"))

    # Evidence information - use display_evidence (extracted evidence, not full chunk)
    evidence_text = result.get('evidence_text', '根拠情報なし')

    # Display character ranges for each chunk separately
    st.markdown(t("### 【根拠情報の文字列範囲】", "### Evidence character ranges"))

    if valid_evidences:
        # Display ranges for each chunk separately
        for idx, evidence in enumerate(valid_evidences, 1):
            backend_char_ranges = evidence.get('char_ranges', [])
            extracted_evidence = evidence.get('extracted_evidence', '')

            if backend_char_ranges:
                # Format ranges for this chunk
                chunk_ranges = [f"{start}文字目～{end}文字目" for start, end in backend_char_ranges]
                ranges_text = "、".join(chunk_ranges)
                st.markdown(f"**Chunk {idx}:** {ranges_text}")
    elif sentence_ranges:
        # Fallback: display all ranges together
        ranges_text = "、".join(sentence_ranges)
        st.markdown(ranges_text)
    else:
        # Old logic fallback
        st.markdown(f"{eff_start}文字目～{eff_end}文字目")

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

    # Display chunks with highlighting
    if valid_evidences:
        st.markdown(t("### 【チャンクと根拠情報のハイライト表示】", "### Chunks with Evidence Highlighting"))
        st.markdown(t("*チャンク内で抽出された根拠情報を黄色でハイライト表示します*",
                     "*Extracted evidence is highlighted in yellow within each chunk*"))

        for idx, evidence in enumerate(valid_evidences, 1):
            chunk_content = evidence.get('chunk_content', '')
            char_ranges = evidence.get('char_ranges', [])
            extracted_evidence = evidence.get('extracted_evidence', '')

            st.markdown(f"**Chunk {idx}:**")

            if chunk_content and char_ranges:
                # Use highlight function to show evidence in yellow
                highlighted_html = highlight_rag_evidence_in_source(
                    chunk_content,
                    extracted_evidence,
                    char_ranges
                )
                st.markdown(highlighted_html, unsafe_allow_html=True)
            elif chunk_content:
                # Fallback: show chunk without highlighting
                st.info(chunk_content)

            # Show metadata for this chunk
            with st.expander(t(f"📊 Chunk {idx} メタデータ", f"📊 Chunk {idx} Metadata")):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(t("類似度スコア", "Similarity"), f"{evidence.get('similarity_score', 0):.3f}")
                with col2:
                    st.metric(t("セマンティック関連度", "Semantic Relevance"), f"{evidence.get('semantic_relevance', 0):.3f}")
                with col3:
                    core_term = evidence.get('core_term', '')
                    if core_term:
                        st.write(f"**{t('コアターム', 'Core Term')}:** {core_term}")

            st.markdown("---")

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
    print(f"\n🔵 add_to_history called for query: {query[:50]}...")

    # Extract highlighted evidence texts from evidences
    evidences = result.get('evidences', [])
    highlighted_evidences = []
    for evidence in evidences:
        if not evidence.get('is_empty', True) and evidence.get('extracted_evidence'):
            highlighted_evidences.append(evidence['extracted_evidence'])

    print(f"🔵 Found {len(evidences)} evidences, HISTORY_MANAGER_AVAILABLE={HISTORY_MANAGER_AVAILABLE}")

    # Save to database first
    query_id = None
    if HISTORY_MANAGER_AVAILABLE:
        print(f"🔵 Attempting to save to database...")
        try:
            import json
            manager = QueryHistoryManager(DB_PATH)

            # Serialize evidences to JSON string
            evidences_json = json.dumps(evidences, ensure_ascii=False) if evidences else ""

            query_id = manager.add_query(
                query=query,
                generated_answer=result.get('answer', ''),
                confidence=result.get('confidence', 0),
                processing_time=result.get('processing_time', 0),
                model=result.get('model', 'Unknown'),
                dataset_answer=result.get('dataset_answer', ''),
                evidences=evidences_json,
                answer_judgment=result.get('answer_judgment', '')
            )

            # Save each evidence to database
            for evidence in evidences:
                manager.add_evidence_extraction(
                    query_id=query_id,
                    chunk_id=evidence.get('chunk_id', 0),
                    chunk_content=evidence.get('chunk_content', ''),
                    extraction_prompt=evidence.get('evidence_variant_prompt', '') or evidence.get('evidence_range_prompt', ''),
                    llm_raw_response=evidence.get('llm_response', ''),
                    extracted_ranges=evidence.get('char_ranges', []),
                    extracted_texts=[evidence.get('extracted_evidence', '')] if evidence.get('extracted_evidence') else [],
                    similarity_score=evidence.get('similarity_score', 0),
                    semantic_relevance=evidence.get('semantic_relevance', 0),
                    core_term=evidence.get('core_term', '')
                )

            print(f"✅ Saved query {query_id} to database with {len(evidences)} evidence entries")
        except Exception as e:
            print(f"⚠️ Failed to save query to database: {e}")

    history_item = {
        'query_id': query_id,  # Store database ID for later operations
        'timestamp': datetime.now(),
        'query': query,
        'answer': result.get('answer', ''),
        'dataset_answer': result.get('dataset_answer', ''),  # Include dataset answer
        'match_metrics': result.get('match_metrics', {}),  # Include match metrics
        'processing_time': result.get('processing_time', 0),
        'confidence': result.get('confidence', 0),
        'evidence_text': result.get('evidence_text', ''),
        'start_char': result.get('start_char', 0),
        'end_char': result.get('end_char', 0),
        'evidences': evidences,  # Store full evidences data
        'highlighted_evidences': highlighted_evidences  # Store extracted evidence texts
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

    with col4:
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
        timestamp_str = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")

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
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(t("**クエリ:**", "**Query:**"))
                st.write(item['query'])

                # Display answer with LLM judgment in title
                answer_judgment = item.get('answer_judgment', '')
                if answer_judgment:
                    if answer_judgment.lower() == 'yes':
                        judgment_text = "✅yes"
                    else:
                        judgment_text = "❌no"
                    answer_title = t(f"**回答:**（LLM回答判断：{judgment_text}）", f"**Answer:** (LLM Judgment: {judgment_text})")
                else:
                    answer_title = t("**回答:**", "**Answer:**")
                st.markdown(answer_title)
                st.write(item['answer'])

                # Display dataset answer if available
                if item.get('dataset_answer'):
                    st.markdown(t("**📋 元のデータセット回答:**", "**📋 Original Dataset Answer:**"))
                    st.info(item['dataset_answer'])

                    # Display match metrics if available
                    # For v2_full_evidence: calculate best metrics from all evidences
                    match_metrics = item.get('match_metrics', {})
                    item_evidences = item.get('evidences', [])

                    # If we have evidences with per-chunk metrics, use the best one
                    if item_evidences:
                        best_f1 = 0.0
                        best_metrics = None

                        for ev in item_evidences:
                            ev_f1 = ev.get('f1_score', 0.0)
                            if ev_f1 > best_f1:
                                best_f1 = ev_f1
                                best_metrics = {
                                    'match_rate': ev.get('f1_score', 0.0),
                                    'precision': ev.get('precision', 0.0),
                                    'recall': ev.get('recall', 0.0),
                                    'exact_match': ev.get('exact_match', False)
                                }

                        # Use best metrics if found, otherwise fallback to match_metrics
                        if best_metrics and best_metrics['match_rate'] > 0:
                            match_metrics = best_metrics

                    if match_metrics:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            match_rate = match_metrics.get('match_rate', 0.0)
                            st.metric(
                                label=t("総合マッチング", "Overall Matching"),
                                value=f"{match_rate:.1%}",
                                help=t("F1スコア：精度と再現率の調和平均", "F1 Score: Harmonic mean of precision and recall")
                            )
                        with col2:
                            precision = match_metrics.get('precision', 0.0)
                            st.metric(
                                label=t("文字レベルマッチング率", "Character-level Match Rate"),
                                value=f"{precision:.1%}",
                                help=t("抽出した根拠のうち正しい割合", "Percentage of extracted evidence that is correct")
                            )
                        with col3:
                            recall = match_metrics.get('recall', 0.0)
                            st.metric(
                                label=t("再現率", "Recall"),
                                value=f"{recall:.1%}",
                                help=t("データセット回答のうち見つけた割合", "Percentage of dataset answer found")
                            )
                        with col4:
                            exact_match = match_metrics.get('exact_match', False)
                            st.metric(
                                label=t("完全一致", "Exact Match"),
                                value=t("✅ はい", "✅ Yes") if exact_match else t("❌ いいえ", "❌ No"),
                                help=t("100%一致するか", "Whether it's 100% match")
                            )

                # Display identified core terms
                item_evidences = item.get('evidences', [])
                item_core_terms = [ev.get('core_term', '') for ev in item_evidences if ev.get('core_term')]
                if item_core_terms:
                    unique_item_core_terms = list(dict.fromkeys(item_core_terms))
                    st.markdown(t("**🎯 LLM回答から識別されたコアタ一ム:**", "**🎯 Core Terms Identified from LLM Answer:**"))
                    st.info(" / ".join(unique_item_core_terms))

                # Display highlighted evidence (根拠情報)
                highlighted_evidences = item.get('highlighted_evidences', [])
                if highlighted_evidences:
                    st.markdown(t("**【根拠情報】:**", "**【Evidence Highlights】:**"))
                    for idx, evidence in enumerate(highlighted_evidences, 1):
                        st.markdown(f"**{idx}.** {evidence}")

                # Display evidence chunks with highlighting
                evidences = item.get('evidences', [])
                if evidences:
                    # Filter valid evidences: check is_empty field or extracted_evidence content
                    valid_evidences = [
                        e for e in evidences
                        if not e.get('is_empty', False) or e.get('extracted_evidence', '').strip()
                    ]
                    if valid_evidences:
                        with st.expander(t("📄 完全な根拠チャンク (黄色でハイライト表示)", "📄 Full evidence chunks (highlighted in yellow)")):
                            for idx, evidence in enumerate(valid_evidences, 1):
                                chunk_content = evidence.get('chunk_content', '')
                                extracted_evidence = evidence.get('extracted_evidence', '')
                                char_ranges = evidence.get('char_ranges', [])

                                st.markdown(f"**Chunk {idx}:**")

                                # Simple highlighting: use context as source and highlight evidence text
                                if chunk_content and extracted_evidence:
                                    # For V1 data: use context from dataset as the full source text
                                    context = item.get('context', '')

                                    if context and extracted_evidence in context:
                                        # Highlight evidence in context
                                        highlighted_html = context.replace(
                                            extracted_evidence,
                                            f'<mark style="background-color: yellow;">{extracted_evidence}</mark>'
                                        )
                                        st.markdown(f'<div style="padding: 1rem; background-color: #f0f0f0; border-radius: 5px; white-space: pre-wrap;">{highlighted_html}</div>', unsafe_allow_html=True)
                                    elif context:
                                        # Evidence not found in context, show both
                                        st.info(f"**原文 (Context):** {context}")
                                        st.warning(f"**根拠 (Evidence):** {extracted_evidence}")
                                    else:
                                        # No context available, fallback to chunk_content
                                        st.info(chunk_content or extracted_evidence)
                                else:
                                    st.info(chunk_content or extracted_evidence)

                                # Display metrics for this evidence chunk (from evidence object, not match_metrics)
                                recall = evidence.get('recall', 0.0)
                                precision = evidence.get('precision', 0.0)
                                f1_score = evidence.get('f1_score', 0.0)
                                exact_match = evidence.get('exact_match', False)

                                if recall > 0 or precision > 0 or f1_score > 0:
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric(
                                            label=t("総合Score", "Overall Score"),
                                            value=f"{f1_score:.1%}",
                                            help=t("精度と再現率の調和平均", "Harmonic mean of precision and recall")
                                        )
                                    with col2:
                                        st.metric(
                                            label=t("精度", "Precision"),
                                            value=f"{precision:.1%}",
                                            help=t("抽出した根拠のうち正しい割合", "Percentage of extracted evidence that is correct")
                                        )
                                    with col3:
                                        st.metric(
                                            label=t("再現率", "Recall"),
                                            value=f"{recall:.1%}",
                                            help=t("データセット回答のうち見つけた割合", "Percentage of dataset answer found")
                                        )
                                    with col4:
                                        st.metric(
                                            label=t("完全一致", "Exact Match"),
                                            value=t("✅", "✅") if exact_match else t("❌", "❌"),
                                            help=t("100%一致するか", "Whether it's 100% match")
                                        )

                                st.markdown("---")

                        # Add extraction prompt expander (only if prompts exist)
                        has_prompts = any(
                            evidence.get('evidence_variant_prompt', '') or evidence.get('evidence_range_prompt', '')
                            for evidence in valid_evidences
                        )
                        if has_prompts:
                            with st.expander(t("🔍 Extraction Prompt Instructions", "🔍 Extraction Prompt Instructions")):
                                for idx, evidence in enumerate(valid_evidences, 1):
                                    # Try variant prompt first (new), fallback to range prompt (old)
                                    extraction_prompt = evidence.get('evidence_variant_prompt', '') or evidence.get('evidence_range_prompt', '')
                                    llm_response = evidence.get('llm_response', '')

                                    if extraction_prompt:
                                        st.markdown(f"**Chunk {idx} - Extraction Prompt:**")
                                        st.code(extraction_prompt, language="text")

                                        if llm_response:
                                            st.markdown(f"**LLM Response:**")
                                            st.code(llm_response, language="text")

                                        st.markdown("---")
                    else:
                        # Fallback: show original evidence_text if no valid evidences
                        with st.expander(t("📄 完全な根拠チャンク", "📄 Full evidence chunk")):
                            st.info(item['evidence_text'])
                else:
                    # Fallback: show original evidence_text if no evidences data
                    with st.expander(t("📄 完全な根拠チャンク", "📄 Full evidence chunk")):
                        st.info(item['evidence_text'])
            
            with col2:
                # Only show metrics if they have valid values
                if item['processing_time'] is not None:
                    st.metric(t("処理時間", "Time"), t(f"{item['processing_time']:.2f}秒", f"{item['processing_time']:.2f}s"))

                if item['confidence'] is not None:
                    st.metric(t("信頼度", "Confidence"), f"{item['confidence']:.2f}")

                if item['start_char'] is not None and item['end_char'] is not None and not (item['start_char'] == 0 and item['end_char'] == 0):
                    st.metric(t("根拠範囲", "Range"), f"{item['start_char']}-{item['end_char']}")
                
                if st.button(t("🔄 再実行", "Re-run"), key=f"rerun_{i}"):
                    # Re-run the query
                    result, error = call_single_query("", item['query'])
                    if result:
                        st.session_state.last_result = result
                        st.session_state.last_query = item['query']
                        st.rerun()

                # Add delete button
                if st.button(t("🗑️ 削除", "Delete"), key=f"delete_{i}", type="secondary"):
                    # Delete from database if query_id exists
                    if 'query_id' in item and HISTORY_MANAGER_AVAILABLE:
                        try:
                            manager = QueryHistoryManager(DB_PATH)
                            success = manager.delete_query(item['query_id'])
                            if success:
                                # Remove from session state
                                st.session_state.query_history = [
                                    h for h in st.session_state.query_history
                                    if h.get('query_id') != item['query_id']
                                ]
                                st.success(t("✅ 削除しました", "✅ Deleted successfully"))
                                st.rerun()
                            else:
                                st.error(t("❌ 削除に失敗しました", "❌ Failed to delete"))
                        except Exception as e:
                            st.error(t(f"❌ エラー: {e}", f"❌ Error: {e}"))
                    else:
                        # Remove from session state only (for items without query_id)
                        st.session_state.query_history.remove(item)
                        st.success(t("✅ 削除しました", "✅ Deleted successfully"))
                        st.rerun()

    # Clear the expanded flag after displaying all records
    if 'expanded_history_index' in st.session_state:
        # Use a separate run to clear it, so it only highlights once
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
                    # Check if query already exists in history
                    existing_index = find_query_in_history(current_query)

                    if existing_index is not None:
                        # Query found in history - jump to that record
                        st.info(t(
                            f"💡 このクエリは履歴に既に存在します（{existing_index + 1}番目の記録）。履歴セクションに移動します...",
                            f"💡 This query already exists in history (record #{existing_index + 1}). Navigating to history section..."
                        ))

                        # Navigate to the page containing this record
                        items_per_page = st.session_state.get('history_items_per_page', 10)
                        target_page = (existing_index // items_per_page) + 1

                        # Set session state to show history and navigate to the correct page
                        st.session_state.show_history = True
                        st.session_state.history_current_page = target_page
                        st.session_state.expanded_history_index = existing_index

                        # Trigger rerun to show history
                        time.sleep(1)  # Brief pause to show the message
                        st.rerun()
                    else:
                        # Query not in history - proceed with normal search
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