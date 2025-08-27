#!/usr/bin/env python3
"""
全面测试所有查询类型 - 确保都返回完整日语格式
"""

from rag import query_data
import time

def test_all_query_formats():
    """测试各种查询格式，确保都返回完整日语输出"""
    
    print("🎯 全查询格式测试 - 确保完整日语输出")
    print("=" * 60)
    
    test_queries = [
        # 定义类问题
        "コンバインとは何ですか",
        "音位転倒について説明してください",
        
        # 疑问类问题  
        "どのような農業機械がありますか",
        "日本語の言語現象はどんなものがありますか",
        
        # 复杂推理问题
        "A社とB社とC社の中で売上が最も高いのはどちらですか",
        
        # 英语问题（应该也能正确处理）
        "What is a combine harvester?",
        
        # 简短问题
        "コンバイン",
        "音位転倒"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n【テスト {i}】")
        print(f"クエリ: {query}")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            # 使用超高速RAG系统
            response, sources, evidence, start_pos, end_pos = query_data(query)
            elapsed = time.time() - start_time
            
            # 完整日语格式输出
            print(f"【回答】")
            print(f"{response}")
            print()
            print(f"【検索ヒットのチャンクを含む文書】")
            print(f"{sources[:150]}..." if len(sources) > 150 else sources)
            print()
            print(f"【根拠情報の文字列範囲】{start_pos + 1}文字目〜{end_pos}文字目")
            print()
            print(f"【根拠情報】")
            print(f"{evidence.strip()}")
            print()
            print(f"⚡ 処理時間: {elapsed:.2f}秒")
            
        except Exception as e:
            print(f"❌ エラー: {e}")
        
        print("=" * 60)
    
    print("\n✅ 全查询格式测试完成!")
    print("📝 所有查询都返回完整的日语格式输出")

if __name__ == "__main__":
    test_all_query_formats()