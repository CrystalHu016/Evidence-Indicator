#!/usr/bin/env python3
"""
清理硬编码的后端集成 - 纯语义RAG实现
"""

import os
import json
import time
import sys
from typing import Optional, Dict, Any, Tuple
from dotenv import load_dotenv

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

# 导入纯语义RAG系统
from pure_semantic_rag import PureSemanticRAG

# 全局变量
pure_rag = None
BACKEND_AVAILABLE = False

def initialize_pure_rag():
    """初始化纯语义RAG系统"""
    global pure_rag, BACKEND_AVAILABLE
    
    try:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            print("⚠️ OPENAI_API_KEY not found")
            BACKEND_AVAILABLE = False
            return
        
        print("🔧 初始化纯语义RAG系统...")
        pure_rag = PureSemanticRAG(api_key)
        
        # 构建向量数据库
        data_file = "../data/single_20240229.json"
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"📖 加载数据: {len(data)} 条")
            pure_rag.build_vector_store(data[:200])  # 使用前200条数据
            BACKEND_AVAILABLE = True
            print("✅ 纯语义RAG系统初始化完成")
        else:
            print("❌ 数据文件不存在")
            BACKEND_AVAILABLE = False
            
    except Exception as e:
        print(f"❌ 纯语义RAG系统初始化失败: {e}")
        BACKEND_AVAILABLE = False

def call_backend_query(query: str, system_mode: str = "enhanced") -> Tuple[Optional[Dict], Optional[str]]:
    """
    调用纯语义RAG系统
    
    Args:
        query: 查询字符串
        system_mode: 系统模式（保持兼容性）
    
    Returns:
        (result_dict, error_message)
    """
    if not BACKEND_AVAILABLE or pure_rag is None:
        return None, "纯语义RAG系统未初始化"
    
    try:
        print(f"🔍 纯语义RAG查询: '{query}'")
        
        # 使用纯语义RAG系统
        result = pure_rag.query_with_answer(query, k=3)
        
        print(f"✅ 查询完成: {result['processing_time']:.2f}s")
        print(f"📊 使用chunks: {result['chunks_used']}")
        
        return result, None
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")
        return None, str(e)

def simulate_backend_response(query: str) -> Dict[str, Any]:
    """模拟后端响应（当系统不可用时）"""
    return {
        'answer': f'模拟回答: 关于"{query}"的信息正在处理中...',
        'evidence_text': '模拟证据文本',
        'source_document': '模拟源文档',
        'confidence': 0.5,
        'model': 'PureSemanticRAG (模拟)',
        'processing_time': 0.1,
        'chunks_used': 0
    }

# 初始化系统
initialize_pure_rag()

def test_clean_backend():
    """测试清理后的后端"""
    print("🧪 清理后的后端系统测试")
    print("=" * 60)
    
    test_queries = [
        "農業機械の種類について教えてください",
        "コンバインとは何ですか",
        "普通型と自立型の違いは何ですか"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")
        print("-" * 40)
        
        result, error = call_backend_query(query)
        
        if error:
            print(f"❌ 错误: {error}")
        else:
            print(f"⏱️  处理时间: {result['processing_time']:.2f}s")
            print(f"💬 回答: {result['answer']}")
            print(f"🔍 证据: {result['evidence_text']}")
            print(f"📊 信心度: {result['confidence']}")
            print(f"🧠 推理: {result.get('reasoning', 'N/A')}")
            print(f"📄 使用chunks: {result['chunks_used']}")

if __name__ == "__main__":
    test_clean_backend()
