#!/usr/bin/env python3
"""
快速测试后端回答
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

def test_specific_query():
    """测试特定问题的回答"""
    print("🔍 测试查询: 農業機械の種類について教えてください")
    print("=" * 60)

    try:
        from backend_integration import call_backend_query

        query = "農業機械の種類について教えてください"

        start_time = time.time()
        result, error = call_backend_query(query, "enhanced")
        elapsed = time.time() - start_time

        if error:
            print(f"❌ 错误: {error}")
            return

        if result:
            print(f"⏱️  处理时间: {elapsed:.2f}秒")
            print(f"💬 回答: {result.get('answer', 'No answer')}")
            print(f"📄 源文档: {result.get('source_document', '')[:100]}...")
            print(f"🔍 证据文本: {result.get('evidence_text', '')[:100]}...")
            print(f"📊 信心度: {result.get('confidence', 0.0)}")
            print(f"🏷️  模型: {result.get('model', 'Unknown')}")

            # 检查是否使用了生成式回答
            answer = result.get('answer', '')
            evidence = result.get('evidence_text', '')

            print(f"\n🔍 分析:")
            print(f"回答长度: {len(answer)} 字符")
            print(f"证据长度: {len(evidence)} 字符")

            if answer == evidence:
                print("⚠️  回答和证据完全相同 - 可能仍在使用抽取式")
            elif len(answer) > len(evidence):
                print("✅ 回答比证据更长 - 使用了生成式方法")
            else:
                print("🤔 回答比证据更短 - 需要进一步检查")

            # 检查是否回答了"种类"问题
            if "種類" in answer or "分類" in answer or ("普通型" in answer and "自立型" in answer):
                print("✅ 正确回答了种类相关问题")
            else:
                print("⚠️  没有明确回答种类问题")

        else:
            print("❌ 无结果返回")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_specific_query()