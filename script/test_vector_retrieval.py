#!/usr/bin/env python3
"""
测试向量检索对分类信息的处理
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_embedding_similarity():
    """测试不同文本的embedding相似度"""

    try:
        from langchain_openai import OpenAIEmbeddings
        import numpy as np

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY 未设置")
            return

        embeddings = OpenAIEmbeddings(api_key=api_key)

        print("🧠 向量相似度测试")
        print("=" * 60)

        # 用户查询
        query = "農業機械の種類について教えてください"

        # 不同的候选文本chunks
        candidates = [
            "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
            "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",  # 核心分类信息
            "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。",
            "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"
        ]

        print(f"🔍 查询: {query}")
        print()

        # 获取查询的embedding
        query_embedding = embeddings.embed_query(query)

        similarities = []

        for i, candidate in enumerate(candidates, 1):
            # 获取候选文本的embedding
            candidate_embedding = embeddings.embed_query(candidate)

            # 计算余弦相似度
            similarity = cosine_similarity(query_embedding, candidate_embedding)
            similarities.append((candidate, similarity))

            # 标记核心分类信息
            is_core = "🎯" if "2種類に大別" in candidate else "  "

            print(f"{is_core} Chunk {i} (相似度: {similarity:.4f})")
            print(f"    {candidate[:50]}...")
            print()

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        print("📊 按相似度排序:")
        print("-" * 40)
        for i, (text, sim) in enumerate(similarities, 1):
            is_core = "🎯" if "2種類に大別" in text else "  "
            print(f"{i}. {is_core} 相似度: {sim:.4f} - {text[:40]}...")

        # 分析结果
        print("\n🔍 分析结果:")
        print("-" * 40)

        core_chunk = next((text for text, sim in similarities if "2種類に大別" in text), None)
        if core_chunk:
            core_similarity = next(sim for text, sim in similarities if text == core_chunk)
            core_rank = next(i for i, (text, sim) in enumerate(similarities, 1) if text == core_chunk)

            print(f"核心分类信息排名: 第{core_rank}位")
            print(f"核心分类信息相似度: {core_similarity:.4f}")

            if core_rank == 1:
                print("✅ 核心分类信息排名最高 - 向量检索应该能找到")
            else:
                print("⚠️  核心分类信息不是最高排名 - 可能被其他chunk掩盖")
                print(f"最高相似度: {similarities[0][1]:.4f}")
                print(f"核心信息与最高的差距: {similarities[0][1] - core_similarity:.4f}")

    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        print("使用模拟数据进行分析...")
        simulate_similarity_analysis()
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def cosine_similarity(vec1, vec2):
    """计算余弦相似度"""
    import numpy as np
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def simulate_similarity_analysis():
    """模拟相似度分析"""
    print("📊 模拟相似度分析:")
    print("=" * 60)

    # 基于经验的相似度估计
    candidates_with_sim = [
        ("コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。", 0.78),
        ("日本で使われているコンバインは普通型と自立型の2種類に大別されます。", 0.85),  # 应该最高
        ("普通型は主にアメリカやヨーロッパ等大規模農業で使われていて...", 0.72),
        ("自立型は収穫時に水分含有率が高い稲の収穫に対応するために...", 0.75)
    ]

    candidates_with_sim.sort(key=lambda x: x[1], reverse=True)

    print("预期的相似度排序:")
    for i, (text, sim) in enumerate(candidates_with_sim, 1):
        is_core = "🎯" if "2種類に大別" in text else "  "
        print(f"{i}. {is_core} 相似度: {sim:.2f} - {text[:40]}...")

def analyze_why_core_info_missed():
    """分析为什么核心信息可能被遗漏"""
    print("\n" + "=" * 60)
    print("🕵️ 核心信息遗漏原因分析")
    print("=" * 60)

    print("💡 可能的原因:")
    print()

    print("1️⃣ Chunking策略问题:")
    print("   • 如果使用150字符chunking，核心句子可能与其他内容混合")
    print("   • 混合后的chunk稀释了分类信息的语义密度")
    print()

    print("2️⃣ 向量检索偏好:")
    print("   • 系统可能偏好更长、更详细的chunks")
    print("   • 核心分类句子虽然重要但较短，可能被忽略")
    print()

    print("3️⃣ 多粒度检索权重:")
    print("   • 如果系统使用多粒度检索，可能优先选择长段落级chunk")
    print("   • 句子级的分类信息被长段落级内容掩盖")
    print()

    print("4️⃣ LLM ranking问题:")
    print("   • LLM可能认为详细描述比分类概述更有价值")
    print("   • 没有专门针对'種類'查询优化ranking逻辑")
    print()

    print("🔧 建议的解决方案:")
    print()

    print("✅ 1. 优化chunking:")
    print("   • 确保分类信息独立成chunk")
    print("   • 使用句子级chunking保留关键信息")
    print()

    print("✅ 2. 查询类型识别:")
    print("   • 识别'種類'/'分類'查询")
    print("   • 对分类信息给予更高权重")
    print()

    print("✅ 3. 语义增强:")
    print("   • 为'種類'查询添加同义词扩展")
    print("   • 增强'2種類'等分类表达的匹配")

if __name__ == "__main__":
    test_embedding_similarity()
    analyze_why_core_info_missed()