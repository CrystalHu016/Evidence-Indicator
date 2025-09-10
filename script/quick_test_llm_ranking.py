#!/usr/bin/env python3
"""
Quick Test of LLM Ranking System
快速测试LLM ranking系统
"""

import os
import time
from dotenv import load_dotenv
from ultra_fast_rag import UltraFastRAG

def test_llm_vs_original():
    """测试LLM模式 vs 原始模式"""
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设定")
        return
    
    print("🧪 LLM Ranking vs 原始方法对比测试")
    print("=" * 60)
    
    # 测试查询
    test_queries = [
        "コンバインとは何ですか",
        "音位転倒について説明してください", 
        "作物について教えて",
        "農業の機械について"
    ]
    
    chroma_path = "./chroma"
    
    # 初始化两个系统
    print("🚀 初始化系统...")
    
    try:
        # LLM智能模式
        llm_rag = UltraFastRAG(
            openai_api_key=api_key,
            chroma_path=chroma_path,
            use_llm_ranking=True
        )
        print("✅ LLM智能模式初始化完成")
        
        # 原始快速模式  
        fast_rag = UltraFastRAG(
            openai_api_key=api_key,
            chroma_path=chroma_path,
            use_llm_ranking=False
        )
        print("✅ 原始快速模式初始化完成")
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 对比测试
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"📋 测试 {i}/{len(test_queries)}: {query}")
        print(f"{'='*40}")
        
        # 测试LLM模式
        print(f"\n🧠 LLM智能模式:")
        print("-" * 25)
        
        try:
            start_time = time.time()
            llm_answer, llm_source, llm_evidence, llm_start, llm_end = llm_rag.query(query, k=5)
            llm_time = time.time() - start_time
            
            print(f"⏱️ 耗时: {llm_time:.2f}s")
            print(f"💬 答案: {llm_answer}")
            print(f"🔍 证据: {llm_evidence[:100]}...")
            print(f"📄 位置: {llm_start}-{llm_end}")
            
        except Exception as e:
            print(f"❌ LLM模式失败: {e}")
            llm_time = 0
        
        # 测试原始模式
        print(f"\n⚡ 原始快速模式:")
        print("-" * 25)
        
        try:
            start_time = time.time()
            fast_answer, fast_source, fast_evidence, fast_start, fast_end = fast_rag.query(query)
            fast_time = time.time() - start_time
            
            print(f"⏱️ 耗时: {fast_time:.2f}s")
            print(f"💬 答案: {fast_answer}")
            print(f"🔍 证据: {fast_evidence[:100]}...")
            print(f"📄 位置: {fast_start}-{fast_end}")
            
        except Exception as e:
            print(f"❌ 原始模式失败: {e}")
            fast_time = 0
        
        # 对比分析
        if llm_time > 0 and fast_time > 0:
            speed_ratio = llm_time / fast_time
            print(f"\n📊 性能对比:")
            print(f"  - 速度比: LLM模式 {speed_ratio:.1f}x 耗时 ({llm_time:.2f}s vs {fast_time:.2f}s)")
            print(f"  - LLM额外耗时: +{llm_time - fast_time:.2f}s")
        
        print(f"\n🔄 等待下个测试...")
        time.sleep(1)  # 避免API限制
    
    print(f"\n🎉 对比测试完成!")
    print(f"\n📋 总结:")
    print(f"✅ LLM智能模式: 更准确的相关性判断和高亮显示")
    print(f"⚡ 原始快速模式: 更快的响应速度")
    print(f"🎯 建议: 根据应用场景选择合适的模式")


def demo_highlighting():
    """演示高亮功能"""
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设定")
        return
    
    print(f"\n🔦 高亮功能演示")
    print("=" * 40)
    
    try:
        from enhanced_rag_system import EnhancedRAGSystem
        
        rag = EnhancedRAGSystem(
            openai_api_key=api_key,
            chroma_path="./chroma"
        )
        
        demo_query = "コンバインの特徴について"
        print(f"🔍 演示查询: {demo_query}")
        
        result = rag.query(demo_query, initial_k=3, final_k=2, use_llm_ranking=True)
        
        print(f"\n📊 检索到 {len(result.get('chunks', []))} 个相关chunks:")
        
        for i, chunk in enumerate(result.get('chunks', []), 1):
            print(f"\n--- Chunk {i} ---")
            print(f"🎯 最终评分: {chunk.get('final_score', 0):.3f}")
            print(f"📊 向量评分: {chunk.get('similarity_score', 0):.3f}")
            print(f"🧠 LLM评分: {chunk.get('llm_score', 0):.3f}")
            
            print(f"\n📝 原始内容 (前100字符):")
            print(f"   {chunk['content'][:100]}...")
            
            print(f"\n🔦 LLM高亮内容:")
            highlighted = chunk.get('highlighted_content', '无高亮')
            print(f"   {highlighted}")
            
            print(f"\n💭 相关性分析:")
            reason = chunk.get('relevance_reason', '无分析')
            print(f"   {reason[:150]}...")
            
            print("-" * 30)
    
    except Exception as e:
        print(f"❌ 高亮演示失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行对比测试
    test_llm_vs_original()
    
    # 演示高亮功能
    demo_highlighting()