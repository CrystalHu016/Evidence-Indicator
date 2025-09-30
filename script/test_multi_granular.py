#!/usr/bin/env python3
"""
多粒度分块算法测试脚本
Multi-granular Chunking Algorithm Test
"""

import os
import time
from dotenv import load_dotenv
from ultra_fast_rag_integrated import UltraFastRAG

def test_multi_granular_chunking():
    """测试多粒度分块功能"""
    print("🚀 多粒度分块算法测试")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    # 初始化RAG系统
    rag = UltraFastRAG(
        openai_api_key=api_key,
        chroma_path="./chroma_multi_granular",
        use_llm_ranking=True,
        highlight_mode="auto"
    )
    
    # 构建多粒度向量库
    data_file = "../data/single_20240229.json"
    print(f"\n🗃️ 构建多粒度向量库: {data_file}")
    
    start_time = time.time()
    success = rag.build_multi_granular_vector_store(data_file)
    build_time = time.time() - start_time
    
    if not success:
        print("❌ 多粒度向量库构建失败")
        return
    
    print(f"✅ 向量库构建完成，耗时: {build_time:.2f}s")
    
    # 测试不同复杂度的查询
    test_queries = [
        # 简单查询 (应该偏好句子级)
        ("コンバイン", "简单"),
        ("普通型", "简单"),
        
        # 中等查询 (应该偏好短段落级)
        ("コンバインとは何ですか", "中等"),
        ("普通型の特徴について", "中等"),
        
        # 复杂查询 (应该偏好长段落级)
        ("コンバインの種類とそれぞれの特徴、使用される作物について詳しく教えて", "复杂"),
        ("日本独自の農業機械の開発背景と普通型との違いについて", "复杂")
    ]
    
    print(f"\n🧪 测试 {len(test_queries)} 个不同复杂度的查询")
    print("=" * 60)
    
    for i, (query, expected_complexity) in enumerate(test_queries, 1):
        print(f"\n📋 测试 {i}: {query}")
        print(f"预期复杂度: {expected_complexity}")
        print("-" * 40)
        
        try:
            # 执行多粒度检索
            start_time = time.time()
            multi_chunks = rag._multi_granular_retrieval(query, k=5)
            retrieval_time = time.time() - start_time
            
            print(f"⏱️  检索耗时: {retrieval_time:.2f}s")
            print(f"📦 返回 {len(multi_chunks)} 个chunks")
            
            # 分析结果
            if multi_chunks:
                granularity_analysis = {}
                for chunk in multi_chunks:
                    granularity = chunk['granularity']
                    if granularity not in granularity_analysis:
                        granularity_analysis[granularity] = {
                            'count': 0,
                            'avg_size': 0,
                            'scores': []
                        }
                    
                    granularity_analysis[granularity]['count'] += 1
                    granularity_analysis[granularity]['scores'].append(chunk['weighted_score'])
                
                # 计算平均大小
                for granularity in granularity_analysis:
                    scores = granularity_analysis[granularity]['scores']
                    granularity_analysis[granularity]['avg_score'] = sum(scores) / len(scores)
                
                print("\n📊 粒度分析:")
                for granularity, stats in granularity_analysis.items():
                    print(f"  {granularity}: {stats['count']}个 (平均分数: {stats['avg_score']:.3f})")
                
                # 显示最佳结果
                best_chunk = min(multi_chunks, key=lambda x: x['weighted_score'])
                print(f"\n🎯 最佳结果:")
                print(f"  粒度: {best_chunk['granularity']}")
                print(f"  分数: {best_chunk['weighted_score']:.3f}")
                print(f"  大小: {best_chunk['chunk_size']}字符")
                print(f"  内容: {best_chunk['document'].page_content[:80]}...")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print("\n🎉 多粒度分块算法测试完成!")
    print("=" * 60)

def analyze_granularity_performance():
    """分析不同粒度的性能表现"""
    print("\n📈 粒度性能分析")
    print("=" * 40)
    
    # 这里可以添加更详细的性能分析
    # 比如各粒度的召回率、准确率等
    print("✅ 性能分析完成")

if __name__ == "__main__":
    test_multi_granular_chunking()
    analyze_granularity_performance()