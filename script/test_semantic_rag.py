#!/usr/bin/env python3
"""
测试纯语义RAG系统
"""

import os
import json
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("❌ OPENAI_API_KEY 未设置")
    exit(1)

# 导入纯语义RAG系统
from ultra_fast_rag_semantic import PureSemanticRAG

def test_semantic_rag():
    """测试纯语义RAG系统"""
    print("🧪 纯语义RAG系统测试")
    print("=" * 60)
    
    # 初始化系统
    rag = PureSemanticRAG(api_key, "./chroma_semantic_test")
    
    # 构建向量数据库
    data_file = "../data/single_20240229.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    print("🏗️ 构建向量数据库...")
    success = rag.build_vector_store(data_file)
    
    if not success:
        print("❌ 向量数据库构建失败")
        return
    
    # 测试查询
    test_query = "農業機械の種類について教えてください"
    print(f"\n🔍 测试查询: {test_query}")
    print("-" * 40)
    
    # 执行查询
    result = rag.query_with_answer(test_query, k=3)
    
    print(f"⏱️  处理时间: {result['processing_time']:.2f}s")
    print(f"💬 回答: {result['answer']}")
    print(f"🔍 证据: {result['evidence_text']}")
    print(f"📊 信心度: {result['confidence']:.2f}")
    print(f"🧠 推理: {result['reasoning']}")
    print(f"📄 使用chunks: {result.get('chunks_used', 0)}")

if __name__ == "__main__":
    test_semantic_rag()
