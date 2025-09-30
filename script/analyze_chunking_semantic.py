#!/usr/bin/env python3
"""
分析chunking和语义匹配问题
"""

import os
import sys
from typing import List
from dotenv import load_dotenv

load_dotenv()

def analyze_chunking_and_semantic():
    """分析chunking和语义匹配问题"""

    print("🔍 Chunking和语义匹配分析")
    print("=" * 80)

    # 原始文本
    original_text = "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"

    query = "農業機械の種類について教えてください"
    core_content = "日本で使われているコンバインは普通型と自立型の2種類に大別されます"

    print(f"📝 原始文本: {original_text[:60]}...")
    print(f"❓ 用户查询: {query}")
    print(f"🎯 核心内容: {core_content}")
    print()

    # 1. 分析chunking策略
    print("📊 1. Chunking策略分析")
    print("-" * 50)

    # 模拟不同的chunking策略
    analyze_different_chunking_strategies(original_text, core_content)

    # 2. 语义匹配分析
    print("\n🧠 2. 语义匹配分析")
    print("-" * 50)

    analyze_semantic_matching(query, core_content, original_text)

    # 3. 向量检索分析
    print("\n🔍 3. 向量检索问题分析")
    print("-" * 50)

    analyze_vector_retrieval_issues(query, original_text, core_content)

def analyze_different_chunking_strategies(original_text, core_content):
    """分析不同chunking策略"""

    # 策略1：按句子分割
    sentences = split_by_sentences(original_text)
    print("🔸 策略1 - 按句子分割:")
    for i, sentence in enumerate(sentences, 1):
        is_core = "✅" if core_content.strip() in sentence else "  "
        print(f"  {is_core} Chunk {i}: {sentence[:50]}...")

    # 策略2：按字符长度分割（150字符）
    print("\n🔸 策略2 - 按150字符分割:")
    char_chunks = split_by_characters(original_text, 150)
    for i, chunk in enumerate(char_chunks, 1):
        is_core = "✅" if core_content.strip() in chunk else "  "
        print(f"  {is_core} Chunk {i}: {chunk[:50]}...")

    # 策略3：按语义单元分割
    print("\n🔸 策略3 - 按语义单元分割:")
    semantic_chunks = split_by_semantic_units(original_text)
    for i, chunk in enumerate(semantic_chunks, 1):
        is_core = "✅" if core_content.strip() in chunk else "  "
        print(f"  {is_core} Chunk {i}: {chunk[:50]}...")

def analyze_semantic_matching(query, core_content, original_text):
    """分析语义匹配问题"""

    # 提取查询关键词
    query_keywords = extract_keywords_from_query(query)
    core_keywords = extract_keywords_from_text(core_content)

    print(f"🔤 查询关键词: {query_keywords}")
    print(f"🎯 核心内容关键词: {core_keywords}")

    # 关键词重叠分析
    overlap = set(query_keywords) & set(core_keywords)
    print(f"📊 关键词重叠: {overlap}")

    # 语义匹配度分析
    print(f"\n🧠 语义匹配分析:")
    print(f"  • 查询'種類' vs 核心'2種類': 高匹配度 ✅")
    print(f"  • 查询'農業機械' vs 核心'コンバイン': 中等匹配度 🔸")
    print(f"  • 核心内容包含具体分类信息: 高相关性 ✅")

    # 但是为什么可能被忽略
    print(f"\n❗ 可能被忽略的原因:")
    print(f"  1. 向量embedding可能没有充分捕获'種類'和'2種類'的语义关系")
    print(f"  2. 如果chunking将此句与其他内容混合，可能降低了相关性")
    print(f"  3. 其他chunk的向量相似度可能意外更高")

def analyze_vector_retrieval_issues(query, original_text, core_content):
    """分析向量检索问题"""

    print("🔍 向量检索可能的问题:")
    print()

    # 问题1：Chunking颗粒度
    print("❓ 问题1: Chunking颗粒度")
    print("  如果chunk太大，核心信息可能被稀释")
    print("  如果chunk太小，可能丢失上下文")
    print(f"  核心句子长度: {len(core_content)} 字符")
    print(f"  建议: 使用句子级chunking (30-80字符) 来保留此类关键信息")
    print()

    # 问题2：向量相似度计算
    print("❓ 问题2: 向量相似度可能的问题")
    print("  '農業機械の種類' vs '2種類に大別' - 可能相似度不够高")
    print("  可能需要:")
    print("    - 更好的embedding模型")
    print("    - 关键词增强")
    print("    - 语义扩展")
    print()

    # 问题3：多粒度检索策略
    print("❓ 问题3: 多粒度检索策略")
    print("  当前系统可能:")
    print("    - 偏好更长的chunk")
    print("    - 没有给分类信息足够权重")
    print("    - 需要专门的分类检索策略")

def split_by_sentences(text):
    """按句子分割"""
    import re
    sentences = re.split(r'[。！？.!?]', text)
    return [s.strip() + "。" for s in sentences if s.strip()]

def split_by_characters(text, chunk_size):
    """按字符长度分割"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def split_by_semantic_units(text):
    """按语义单元分割（手动定义）"""
    # 基于内容结构手动分割
    units = [
        "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
        "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
        "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。",
        "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"
    ]
    return units

def extract_keywords_from_query(query):
    """从查询中提取关键词"""
    import re
    # 移除疑问词和助词
    clean_query = re.sub(r'[について教えてください？とは何ですか]', '', query)
    keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', clean_query)
    return [kw for kw in keywords if len(kw) > 1]

def extract_keywords_from_text(text):
    """从文本中提取关键词"""
    import re
    keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', text)
    return [kw for kw in keywords if len(kw) > 1]

if __name__ == "__main__":
    analyze_chunking_and_semantic()