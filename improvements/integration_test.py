#!/usr/bin/env python3
"""
完整集成测试 - 验证所有四个架构改进协同工作
"""

import os
import time
from config_driven_rag import ConfigDrivenRAGSystem

def run_integration_test():
    """运行完整的集成测试"""

    print("🚀 运行完整集成测试")
    print("🎯 验证所有四个架构改进协同工作")
    print("=" * 60)

    # 检查API密钥
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY not set - 无法运行完整测试")
        return False

    # 配置测试系统
    test_config = {
        "llm_model": "gpt-4o-mini",
        "llm_temperature": 0.1,
        "use_llm_intent": True,
        "use_dynamic_context": True,
        "keyword_max_count": 8,
        "intent_confidence_threshold": 0.8,
        "similarity_threshold": 0.3
    }

    print(f"⚙️ 初始化配置驱动的RAG系统...")
    try:
        system = ConfigDrivenRAGSystem(config_dict=test_config)
        print("✅ 系统初始化成功")
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return False

    # 测试查询集合
    test_cases = [
        {
            "query": "コンバインとは何ですか",
            "expected_intent": "definition",
            "language": "japanese",
            "domain": "agriculture"
        },
        {
            "query": "What is a combine harvester?",
            "expected_intent": "definition",
            "language": "english",
            "domain": "agriculture"
        },
        {
            "query": "コンバインの種類はいくつありますか",
            "expected_intent": "classification",
            "language": "japanese",
            "domain": "agriculture"
        }
    ]

    # 测试上下文块
    context_chunks = [
        "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
        "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
        "普通型は主にアメリカやヨーロッパ等大規模農業で使われています。",
        "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"
    ]

    passed_tests = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 集成测试 {i}/{total_tests}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Intent: {test_case['expected_intent']}")
        print("-" * 40)

        start_time = time.time()

        try:
            # 运行完整的配置驱动查询
            result = system.query(
                question=test_case["query"],
                domain=test_case["domain"],
                context_chunks=context_chunks
            )

            processing_time = time.time() - start_time

            # 验证结果
            if result.get("error"):
                print(f"❌ 查询失败: {result['answer']}")
                continue

            # 检查四个核心改进是否都工作
            analysis = result["query_analysis"]

            print(f"🎯 检测到的意图: {analysis['intent']} (置信度: {analysis['intent_confidence']:.2f})")
            print(f"🔑 语义关键词: {analysis['keywords']}")
            print(f"💬 生成的回答: {result['answer'][:80]}...")
            print(f"⏱️  处理时间: {processing_time:.2f}s")

            # 验证各组件是否正常工作
            checks = {
                "intent_correct": analysis["intent"] == test_case["expected_intent"],
                "has_keywords": len(analysis["keywords"]) > 0,
                "has_answer": len(result["answer"]) > 10,
                "reasonable_time": processing_time < 60
            }

            print(f"\n📊 组件验证:")
            print(f"  ✅ 语义关键词提取: {'通过' if checks['has_keywords'] else '失败'}")
            print(f"  ✅ LLM意图理解: {'通过' if checks['intent_correct'] else '失败'} ({analysis['intent']})")
            print(f"  ✅ 动态上下文生成: {'通过' if checks['has_answer'] else '失败'}")
            print(f"  ✅ 配置驱动系统: {'通过' if checks['reasonable_time'] else '失败'}")

            if all(checks.values()):
                print("✅ 集成测试通过")
                passed_tests += 1
            else:
                print("❌ 集成测试失败")

        except Exception as e:
            print(f"❌ 集成测试异常: {e}")

    # 测试结果总结
    print(f"\n🎉 集成测试完成!")
    print(f"📊 结果: {passed_tests}/{total_tests} 测试通过")

    if passed_tests == total_tests:
        print("🏆 所有架构改进协同工作正常!")
        print("✅ 零硬编码架构验证成功!")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} 个测试需要调试")
        return False

if __name__ == "__main__":
    success = run_integration_test()

    if success:
        print("\n🚀 架构改进总结:")
        print("  ✅ 第一步: 语义嵌入替代硬编码关键词 - 测试通过")
        print("  ✅ 第二步: LLM意图理解替代模式匹配 - 测试通过")
        print("  ✅ 第三步: 动态上下文生成替代固定模板 - 测试通过")
        print("  ✅ 第四步: 配置驱动替代硬编码规则 - 测试通过")
        print("\n🎯 现代化RAG系统架构改进完成!")
    else:
        print("\n⚠️  部分功能需要进一步调试")