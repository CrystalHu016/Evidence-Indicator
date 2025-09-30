#!/usr/bin/env python3
"""
测试纯语义RAG系统的多个查询
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

def test_multiple_queries():
    """测试多个不同类型的查询"""
    print("🧪 纯语义RAG系统 - 多查询测试")
    print("=" * 80)
    
    # 初始化系统
    rag = PureSemanticRAG(api_key, "./chroma_semantic_test")
    
    # 测试查询列表
    test_queries = [
        {
            "query": "コンバインとは何ですか",
            "type": "定义查询",
            "expected": "应该解释コンバイン的定义和功能"
        },
        {
            "query": "普通型と自立型の違いは何ですか",
            "type": "比较查询", 
            "expected": "应该比较两种类型的差异"
        },
        {
            "query": "音位転倒について説明してください",
            "type": "解释查询",
            "expected": "应该解释音位転倒的概念和例子"
        },
        {
            "query": "日本の農業について教えてください",
            "type": "概述查询",
            "expected": "应该提供日本农业的概述信息"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🔍 测试 {i}: {test_case['type']}")
        print(f"查询: {test_case['query']}")
        print(f"期望: {test_case['expected']}")
        print("-" * 60)
        
        # 执行查询
        result = rag.query_with_answer(test_case['query'], k=3)
        
        print(f"⏱️  处理时间: {result['processing_time']:.2f}s")
        print(f"💬 回答: {result['answer']}")
        print(f"🔍 证据: {result['evidence_text'][:100]}...")
        print(f"📊 信心度: {result['confidence']:.2f}")
        print(f"🧠 推理: {result['reasoning']}")
        print(f"📄 使用chunks: {result.get('chunks_used', 0)}")
        
        print("\n" + "=" * 80)

if __name__ == "__main__":
    test_multiple_queries()
