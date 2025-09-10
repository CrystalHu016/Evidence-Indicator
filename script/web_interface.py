#!/usr/bin/env python3
"""
Web Interface for LLM-driven Intelligent RAG System
LLM智能RAG系统的Web界面
"""

import os
import time
import streamlit as st
import json
from dotenv import load_dotenv
from enhanced_rag_system import EnhancedRAGSystem

# 页面配置
st.set_page_config(
    page_title="🧠 LLM智能RAG系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 加载环境变量
load_dotenv()

def initialize_system():
    """初始化RAG系统"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY 环境变量未设置")
        return None
    
    try:
        with st.spinner("🚀 正在初始化LLM智能RAG系统..."):
            rag_system = EnhancedRAGSystem(
                openai_api_key=api_key,
                chroma_path="./chroma",
                model="gpt-4o-mini"
            )
        st.success("✅ 系统初始化完成！")
        return rag_system
    except Exception as e:
        st.error(f"❌ 系统初始化失败: {e}")
        return None

def display_chunk_details(chunk, index):
    """显示chunk详细信息"""
    with st.expander(f"📄 Chunk {index} (评分: {chunk.get('final_score', 0):.3f})", 
                     expanded=(index == 1)):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🎯 最终评分", f"{chunk.get('final_score', 0):.3f}")
        
        with col2:
            st.metric("📊 向量评分", f"{chunk.get('similarity_score', 0):.3f}")
        
        with col3:
            st.metric("🧠 LLM评分", f"{chunk.get('llm_score', 0):.3f}")
        
        # 原始内容
        st.subheader("📝 原始内容")
        st.text_area("", chunk['content'], height=100, key=f"content_{index}")
        
        # 高亮内容
        st.subheader("🔦 LLM高亮内容")
        highlighted = chunk.get('highlighted_content', chunk['content'])
        st.markdown(highlighted)
        
        # 相关性分析
        st.subheader("💭 LLM相关性分析")
        reason = chunk.get('relevance_reason', '无分析')
        st.text_area("", reason, height=80, key=f"reason_{index}")

def main():
    """主界面"""
    
    # 标题
    st.title("🧠 LLM驱动的智能RAG系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # 搜索参数
        initial_k = st.slider("🔍 初始检索数量", 3, 20, 8)
        final_k = st.slider("📊 最终返回数量", 1, 10, 3)
        use_llm_ranking = st.checkbox("🧠 启用LLM智能排序", value=True)
        
        # 系统信息
        st.header("📋 系统信息")
        st.info("""
        **数据集**: single_20240229.json (9,103条)
        **向量模型**: text-embedding-ada-002  
        **LLM模型**: GPT-4o-mini
        **特性**: 智能ranking + 高亮显示
        """)
    
    # 初始化系统
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = initialize_system()
    
    if st.session_state.rag_system is None:
        st.error("系统初始化失败，请检查配置")
        return
    
    # 查询界面
    st.header("🔍 智能查询")
    
    # 预设查询示例
    example_queries = [
        "コンバインとは何ですか？",
        "音位転倒について説明してください",
        "農業機械の種類について教えて",
        "慣用句の間違いを指摘する方法",
        "待ち合わせのマナーについて"
    ]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("📝 質問を入力してください:", 
                             placeholder="例: コンバインについて教えて")
    
    with col2:
        if st.selectbox("💡 例文を選択:", [""] + example_queries, key="examples"):
            query = st.session_state.examples
    
    # 検索実行
    if st.button("🔍 検索実行", type="primary") and query:
        
        with st.spinner("🧠 LLM智能分析中..."):
            start_time = time.time()
            
            try:
                # RAG検索実行
                result = st.session_state.rag_system.query(
                    query_text=query,
                    initial_k=initial_k,
                    final_k=final_k,
                    use_llm_ranking=use_llm_ranking
                )
                
                processing_time = time.time() - start_time
                
                # 結果表示
                st.success(f"✅ 処理完了! (総処理時間: {result['processing_time']:.2f}秒)")
                
                # メトリクス表示
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("⏱️ 総処理時間", f"{result['processing_time']:.2f}s")
                
                with col2:
                    st.metric("📊 向量检索", f"{result['vector_search_time']:.2f}s")
                
                with col3:
                    st.metric("🧠 LLM排序", f"{result['llm_ranking_time']:.2f}s")
                
                with col4:
                    st.metric("💬 答案生成", f"{result['answer_generation_time']:.2f}s")
                
                # 回答表示
                st.header("💬 AI回答")
                st.markdown(f"**質問**: {query}")
                st.markdown("**回答**:")
                st.info(result['answer'])
                
                # 高亮証拠表示
                if result.get('highlighted_evidence'):
                    st.header("🔦 高亮显示的証拠")
                    st.markdown(result['highlighted_evidence'])
                
                # 詳細な検索結果
                if result.get('chunks'):
                    st.header("📊 詳細検索結果")
                    st.markdown(f"検索到 **{len(result['chunks'])}** 個相関chunks:")
                    
                    for i, chunk in enumerate(result['chunks'], 1):
                        display_chunk_details(chunk, i)
                
                # 排序摘要
                summary = result.get('ranking_summary', {})
                if isinstance(summary, dict) and summary.get('ranking_summary'):
                    st.header("📈 LLM排序分析")
                    st.text(summary['ranking_summary'])
                    
                    if 'llm_improvement' in summary:
                        improvement = summary['llm_improvement']
                        if improvement > 0:
                            st.success(f"🎯 LLM改進: +{improvement:.3f} (LLM排序效果更好)")
                        elif improvement < 0:
                            st.warning(f"📉 LLM改進: {improvement:.3f} (向量排序更優)")
                        else:
                            st.info(f"🟰 LLM改進: {improvement:.3f} (効果相当)")
                
            except Exception as e:
                st.error(f"❌ 検索失敗: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # フッター
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🧠 LLM智能RAG系统 | 搭载GPT-4o-mini智能ranking和highlighting功能
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()