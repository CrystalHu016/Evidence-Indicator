#!/usr/bin/env python3
"""
打印三个具体输出
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

def test_actual_backend():
    """测试实际的后端输出"""
    print("🔍 实际后端测试 - 農業機械の種類について教えてください")
    print("=" * 80)

    try:
        from backend_integration import call_backend_query

        query = "農業機械の種類について教えてください"

        print(f"正在调用后端查询: {query}")
        print("请稍等...")

        result, error = call_backend_query(query, "enhanced")

        if error:
            print(f"❌ 错误: {error}")
            return

        if not result:
            print("❌ 无结果返回")
            return

        print("\n" + "=" * 80)
        print("📋 实际后端返回的三个输出:")
        print("=" * 80)

        # 1. 【回答】
        answer = result.get('answer', '')
        print("\n📝 1. 【回答】")
        print("-" * 50)
        print(answer)

        # 2. 【検索ヒットのチャンクを含む文書】 - source_document
        source_document = result.get('source_document', '')
        print("\n📄 2. 【検索ヒットのチャンクを含む文書】")
        print("-" * 50)
        print(source_document)

        # 3. 【根拠情報】 - evidence_text
        evidence_text = result.get('evidence_text', '')
        print("\n🔍 3. 【根拠情報】")
        print("-" * 50)
        print(evidence_text)

        # 额外信息
        print("\n" + "=" * 80)
        print("📊 详细分析:")
        print("=" * 80)

        print(f"• 回答长度: {len(answer)} 字符")
        print(f"• 源文档长度: {len(source_document)} 字符")
        print(f"• 证据文本长度: {len(evidence_text)} 字符")
        print(f"• 处理时间: {result.get('processing_time', 0):.2f}秒")
        print(f"• 置信度: {result.get('confidence', 0):.2f}")
        print(f"• 模型: {result.get('model', 'Unknown')}")

        # 检查是否使用了生成式回答
        print(f"\n🔍 回答类型分析:")
        if answer == evidence_text:
            print("⚠️  回答 = 证据文本 (可能仍在使用抽取式)")
        else:
            print("✅ 回答 ≠ 证据文本 (使用生成式)")

        if "種類" in answer or "2種類" in answer:
            print("✅ 正确回答了种类问题")
        else:
            print("⚠️  没有明确回答种类问题")

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("尝试使用模拟数据...")
        simulate_outputs()
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def simulate_outputs():
    """模拟输出（如果后端不可用）"""
    print("\n📋 模拟输出（基于我们的修改）:")
    print("=" * 80)

    # 模拟的三个输出
    answer = "農業機械にはさまざまな種類がありますが、特にコンバインについて説明します。コンバインは、穀物の収穫、脱穀、選別を一台で行うことができる自走式の農業機械です。日本で使用されるコンバインは、主に普通型と自立型の2種類に分けられます。普通型は、アメリカやヨーロッパの大規模農業で使用され、稲や麦、大豆のほか、小豆、菜種、トウモロコシなど多様な作物に対応しています。一方、自立型は、日本特有の農業機械であり、特に水分含有率が高い稲の収穫に適しています。"

    source_document = "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"

    evidence_text = "日本で使われているコンバインは普通型と自立型の2種類に大別されます"

    print("\n📝 1. 【回答】")
    print("-" * 50)
    print(answer)

    print("\n📄 2. 【検索ヒットのチャンクを含む文書】")
    print("-" * 50)
    print(source_document)

    print("\n🔍 3. 【根拠情報】")
    print("-" * 50)
    print(evidence_text)

if __name__ == "__main__":
    test_actual_backend()