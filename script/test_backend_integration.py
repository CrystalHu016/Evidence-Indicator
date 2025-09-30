#!/usr/bin/env python3
"""
测试修改后的backend_integration.py
"""

import sys
import os

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

def test_backend_integration():
    """测试backend_integration的LLM回答生成"""
    print("🧪 测试Backend Integration LLM回答生成")
    print("=" * 50)

    try:
        from backend_integration import call_backend_query

        test_queries = [
            "農業機械の種類について教えてください",
            "コンバインとは何ですか"
        ]

        for query in test_queries:
            print(f"\n🔍 测试查询: {query}")
            print("-" * 40)

            result, error = call_backend_query(query, "enhanced")

            if error:
                print(f"❌ 错误: {error}")
            elif result:
                print(f"💬 回答: {result.get('answer', '')}")
                print(f"🔍 evidence_text: {result.get('evidence_text', '')}")
                print(f"⏱️  处理时间: {result.get('processing_time', 0.0):.2f}秒")
                print(f"📊 信心度: {result.get('confidence', 0.0)}")
                print(f"🏷️  模型: {result.get('model', 'Unknown')}")

                # 检查回答是否是LLM生成的而不是简单抽取
                answer = result.get('answer', '')
                evidence = result.get('evidence_text', '')

                if answer == evidence:
                    print("⚠️  回答和evidence相同，可能仍在使用抽取式方法")
                else:
                    print("✅ 回答和evidence不同，使用了生成式方法")
            else:
                print("❌ 无结果返回")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backend_integration()