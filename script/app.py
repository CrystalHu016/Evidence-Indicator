#!/usr/bin/env python3
"""
Evidence Indicator RAG System - Streamlit Frontend
A beautiful and interactive web interface for the RAG system
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_QUERIES = [
    "コンバインとは何ですか",
    "音位転倒について説明してください",
    "どのような農業機械がありますか",
    "What is a combine harvester?",
    "日本語の言語現象はどんなものがありますか"
]

def init_session_state():
    """Initialize session state variables"""
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = []

def call_rag_api(query: str) -> dict:
    """Call the RAG API and return results"""
    try:
        # For now, we'll simulate the API call
        # In production, this would call the actual RAG backend
        response = {
            "answer": f"回答: {query}に関する情報です。",
            "source_document": f"検索ヒットのチャンクを含む文書: {query}に関する詳細な文書内容がここに表示されます。",
            "evidence_range": "1文字目〜50文字目",
            "evidence_text": f"根拠情報: {query}に関する具体的な根拠情報がここに表示されます。",
            "processing_time": round(time.time() % 3 + 0.5, 2),
            "timestamp": datetime.now().isoformat()
        }
        return response
    except Exception as e:
        st.error(f"API呼び出しエラー: {str(e)}")
        return None

def display_query_result(result: dict):
    """Display the query result in a beautiful format"""
    if not result:
        return
    
    # Create columns for better layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Answer section
        st.markdown("### 【回答】")
        st.info(result["answer"])
        
        # Source document section
        st.markdown("### 【検索ヒットのチャンクを含む文書】")
        with st.expander("📄 完全なソース文書を表示", expanded=False):
            st.text(result["source_document"])
        
        # Evidence section
        st.markdown("### 【根拠情報】")
        st.markdown(f"**文字列範囲:** {result['evidence_range']}")
        st.success(result["evidence_text"])
    
    with col2:
        # Performance metrics
        st.markdown("### 📊 パフォーマンス")
        st.metric("処理時間", f"{result['processing_time']}秒")
        
        # Query timestamp
        st.markdown(f"**実行時刻:** {result['timestamp'][:19]}")

def create_performance_chart():
    """Create a performance visualization chart"""
    if not st.session_state.performance_metrics:
        return
    
    df = pd.DataFrame(st.session_state.performance_metrics)
    
    fig = px.line(df, x='timestamp', y='processing_time', 
                  title='クエリ処理時間の推移',
                  labels={'processing_time': '処理時間 (秒)', 'timestamp': '時刻'})
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Evidence Indicator RAG System",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.title("🔍 Evidence Indicator RAG System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # API Configuration
        st.subheader("API設定")
        api_url = st.text_input("API Base URL", value=API_BASE_URL)
        
        # Query History
        st.subheader("📝 クエリ履歴")
        if st.session_state.query_history:
            for i, query in enumerate(st.session_state.query_history[-5:]):
                if st.button(f"🔍 {query[:30]}...", key=f"hist_{i}"):
                    st.session_state.current_query = query
                    st.rerun()
        else:
            st.info("まだクエリ履歴がありません")
        
        # Quick Actions
        st.subheader("⚡ クイックアクション")
        if st.button("🗑️ 履歴クリア"):
            st.session_state.query_history = []
            st.session_state.performance_metrics = []
            st.rerun()
        
        if st.button("📊 パフォーマンスリセット"):
            st.session_state.performance_metrics = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input section
        st.header("💬 クエリ入力")
        
        # Query input
        query = st.text_area(
            "質問を入力してください:",
            value=st.session_state.get('current_query', ''),
            height=100,
            placeholder="例: コンバインとは何ですか？"
        )
        
        # Quick query buttons
        st.subheader("🚀 クイッククエリ")
        cols = st.columns(len(DEFAULT_QUERIES))
        for i, (col, default_query) in enumerate(zip(cols, DEFAULT_QUERIES)):
            if col.button(default_query[:15] + "...", key=f"quick_{i}"):
                st.session_state.current_query = default_query
                st.rerun()
        
        # Submit button
        if st.button("🔍 検索実行", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("検索中..."):
                    # Call API
                    result = call_rag_api(query)
                    
                    if result:
                        # Add to history
                        if query not in st.session_state.query_history:
                            st.session_state.query_history.append(query)
                        
                        # Add to performance metrics
                        st.session_state.performance_metrics.append({
                            'query': query,
                            'processing_time': result['processing_time'],
                            'timestamp': result['timestamp']
                        })
                        
                        # Display result
                        st.success("✅ 検索完了!")
                        display_query_result(result)
                    else:
                        st.error("❌ 検索に失敗しました")
            else:
                st.warning("⚠️ クエリを入力してください")
    
    with col2:
        # Statistics panel
        st.header("📈 統計情報")
        
        # Query count
        st.metric("総クエリ数", len(st.session_state.query_history))
        
        # Average processing time
        if st.session_state.performance_metrics:
            avg_time = sum(m['processing_time'] for m in st.session_state.performance_metrics) / len(st.session_state.performance_metrics)
            st.metric("平均処理時間", f"{avg_time:.2f}秒")
        
        # Performance chart
        if st.session_state.performance_metrics:
            st.subheader("📊 パフォーマンス推移")
            create_performance_chart()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>🔍 Evidence Indicator RAG System | Powered by Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 