#!/usr/bin/env python3
"""
LLM-driven Intelligent RAG System - Streamlit Frontend
LLM智能RAG系统的Streamlit前端界面
"""

import streamlit as st
import os
import time
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our LLM systems
try:
    from enhanced_rag_system import EnhancedRAGSystem
    from ultra_fast_rag import UltraFastRAG
    LLM_SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ LLM系统导入失败: {e}")
    LLM_SYSTEM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="🧠 LLM智能RAG系统",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def initialize_rag_systems():
    """Initialize RAG systems with caching"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        st.error("❌ OPENAI_API_KEY 环境变量未设置")
        return None, None
    
    try:
        # Initialize Enhanced RAG System
        enhanced_rag = EnhancedRAGSystem(
            openai_api_key=api_key,
            chroma_path="./chroma",
            model="gpt-4o-mini"
        )
        
        # Initialize Ultra Fast RAG (LLM mode)
        ultra_fast_rag = UltraFastRAG(
            openai_api_key=api_key,
            chroma_path="./chroma",
            use_llm_ranking=True
        )
        
        return enhanced_rag, ultra_fast_rag
        
    except Exception as e:
        st.error(f"❌ 系统初始化失败: {e}")
        return None, None

def display_chunk_analysis(chunks):
    """Display detailed chunk analysis"""
    if not chunks:
        return
    
    st.subheader("📊 详细检索结果分析")
    
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"📄 Chunk {i} - 评分: {chunk.get('final_score', 0):.3f}", 
                         expanded=(i == 1)):
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 最终评分", f"{chunk.get('final_score', 0):.3f}")
            with col2:
                st.metric("📊 向量评分", f"{chunk.get('similarity_score', 0):.3f}")
            with col3:
                st.metric("🧠 LLM评分", f"{chunk.get('llm_score', 0):.3f}")
            
            # Content
            st.write("**📝 原始内容:**")
            st.text_area("", chunk['content'], height=100, key=f"content_{i}")
            
            # Highlighted content
            st.write("**🔦 LLM高亮内容:**")
            highlighted = chunk.get('highlighted_content', chunk['content'])
            st.markdown(highlighted)
            
            # Relevance analysis
            st.write("**💭 LLM相关性分析:**")
            reason = chunk.get('relevance_reason', '无分析')
            st.text_area("", reason, height=80, key=f"reason_{i}")

def main():
    """Main application"""
    
    # Title and header
    st.title("🧠 LLM驱动的智能RAG系统")
    st.markdown("**搭载GPT-4o-mini智能ranking和highlighting功能**")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ 系统配置")
        
        # System mode selection
        system_mode = st.selectbox(
            "🚀 选择系统模式",
            ["Enhanced RAG (完整功能)", "Ultra Fast RAG (LLM模式)"],
            help="Enhanced RAG提供完整功能，Ultra Fast RAG更快速"
        )
        
        # Search parameters
        st.subheader("🔍 搜索参数")
        initial_k = st.slider("初始检索数量", 3, 20, 8)
        final_k = st.slider("最终返回数量", 1, 10, 3)
        use_llm_ranking = st.checkbox("启用LLM智能排序", value=True)
        
        # System info
        st.subheader("📋 系统信息")
        st.info("""
        **数据集**: single_20240229.json  
        **记录数**: 9,103条  
        **向量模型**: text-embedding-ada-002  
        **LLM模型**: GPT-4o-mini  
        **特性**: 智能ranking + 高亮显示
        """)
        
        # Performance tip
        st.info("💡 首次查询可能需要15-20秒进行LLM分析，后续查询会更快")
    
    # Initialize systems
    if not LLM_SYSTEM_AVAILABLE:
        st.error("❌ LLM系统不可用，请检查模块导入")
        return
    
    enhanced_rag, ultra_fast_rag = initialize_rag_systems()
    
    if enhanced_rag is None or ultra_fast_rag is None:
        st.error("❌ 系统初始化失败，请检查配置")
        return
    
    st.success("✅ LLM智能RAG系统已就绪")
    
    # Query interface
    st.header("🔍 智能查询")
    
    # Example queries
    example_queries = [
        "コンバインとは何ですか？",
        "音位転倒について説明してください",
        "農業機械の種類について教えて",
        "慣用句の間違いを指摘する方法",
        "待ち合わせのマナーについて"
    ]
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "📝 您的问题:",
            placeholder="例: コンバインについて詳しく教えてください",
            height=100
        )
    
    with col2:
        st.write("💡 **例文选择:**")
        for i, example in enumerate(example_queries):
            if st.button(f"{i+1}. {example[:15]}...", key=f"example_{i}"):
                query = example
                st.rerun()
    
    # Search execution
    if st.button("🔍 开始智能检索", type="primary") and query.strip():
        
        with st.spinner("🧠 LLM智能分析中...请稍候..."):
            start_time = time.time()
            
            try:
                if system_mode == "Enhanced RAG (完整功能)":
                    # Use Enhanced RAG System
                    result = enhanced_rag.query(
                        query_text=query.strip(),
                        initial_k=initial_k,
                        final_k=final_k,
                        use_llm_ranking=use_llm_ranking
                    )
                    
                    # Display results
                    st.success(f"✅ Enhanced RAG处理完成! (耗时: {result['processing_time']:.2f}秒)")
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("⏱️ 总时间", f"{result['processing_time']:.2f}s")
                    with col2:
                        st.metric("📊 向量检索", f"{result['vector_search_time']:.2f}s")
                    with col3:
                        st.metric("🧠 LLM排序", f"{result['llm_ranking_time']:.2f}s")
                    with col4:
                        st.metric("💬 答案生成", f"{result['answer_generation_time']:.2f}s")
                    
                    # Answer
                    st.header("💬 AI智能回答")
                    st.info(result['answer'])
                    
                    # Highlighted evidence
                    if result.get('highlighted_evidence'):
                        st.header("🔦 高亮显示证据")
                        st.markdown(result['highlighted_evidence'])
                    
                    # Detailed analysis
                    if result.get('chunks'):
                        display_chunk_analysis(result['chunks'])
                    
                    # Ranking summary
                    summary = result.get('ranking_summary', {})
                    if isinstance(summary, dict) and summary.get('ranking_summary'):
                        st.header("📈 LLM排序分析")
                        st.text(summary['ranking_summary'])
                        
                        if 'llm_improvement' in summary:
                            improvement = summary['llm_improvement']
                            if improvement > 0:
                                st.success(f"🎯 LLM改进: +{improvement:.3f} (LLM排序效果更好)")
                            elif improvement < 0:
                                st.warning(f"📉 LLM改进: {improvement:.3f} (向量排序更优)")
                            else:
                                st.info(f"🟰 LLM改进: {improvement:.3f} (效果相当)")
                
                else:  # Ultra Fast RAG mode
                    # Use Ultra Fast RAG
                    answer, source, evidence, start_pos, end_pos = ultra_fast_rag.query(query.strip(), k=initial_k)
                    processing_time = time.time() - start_time
                    
                    st.success(f"✅ Ultra Fast RAG处理完成! (耗时: {processing_time:.2f}秒)")
                    
                    # Results
                    st.header("💬 AI回答")
                    st.info(answer)
                    
                    st.header("🔍 检索证据")
                    st.markdown(evidence)
                    
                    st.header("📄 源文档")
                    st.text_area("源文档内容", source, height=200)
                    
                    if start_pos > 0 and end_pos > 0:
                        st.info(f"📍 证据位置: {start_pos}-{end_pos} 字符")
                
            except Exception as e:
                st.error(f"❌ 查询失败: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        🧠 LLM智能RAG系统 v2.0 | 搭载GPT-4o-mini智能ranking和highlighting功能<br>
        基于single_20240229.json数据集 (9,103条记录)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()