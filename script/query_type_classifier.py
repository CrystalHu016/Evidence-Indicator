#!/usr/bin/env python3
"""
查询类型识别和分类信息加权优化
实现RAG系统的第一个高优先级优化
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class QueryType(Enum):
    CLASSIFICATION = "classification"  # 分类查询
    DEFINITION = "definition"          # 定义查询
    COMPARISON = "comparison"          # 比较查询
    PROCEDURE = "procedure"            # 流程查询
    GENERAL = "general"                # 一般查询

@dataclass
class ChunkScore:
    """Chunk评分数据结构"""
    content: str
    original_score: float
    boost_factor: float = 1.0
    final_score: float = None
    boost_reason: str = ""

    def __post_init__(self):
        self.final_score = self.original_score * self.boost_factor

class QueryTypeClassifier:
    """查询类型分类器"""

    def __init__(self):
        self.classification_patterns = [
            r'種類.*について',      # 種類について
            r'.*の種類',           # ～の種類
            r'分類.*について',      # 分類について
            r'.*分類',            # ～分類
            r'何種類',            # 何種類
            r'いくつ.*種類',       # いくつの種類
            r'どんな.*種類',       # どんな種類
            r'.*にはどのような.*種類', # ～にはどのような種類
        ]

        self.definition_patterns = [
            r'とは何',            # とは何
            r'何ですか',          # 何ですか
            r'とは',             # とは
            r'について説明',       # について説明
            r'どのような.*ものか',  # どのようなものか
        ]

        self.comparison_patterns = [
            r'違い',             # 違い
            r'差.*は',           # 差は
            r'比較',             # 比較
            r'.*と.*の違い',      # AとBの違い
        ]

        self.procedure_patterns = [
            r'手順',             # 手順
            r'方法',             # 方法
            r'やり方',           # やり方
            r'どのように',        # どのように
            r'ステップ',          # ステップ
        ]

    def classify(self, query: str) -> QueryType:
        """查询分类"""

        # 分类查询检测
        if any(re.search(pattern, query) for pattern in self.classification_patterns):
            return QueryType.CLASSIFICATION

        # 比较查询检测
        if any(re.search(pattern, query) for pattern in self.comparison_patterns):
            return QueryType.COMPARISON

        # 流程查询检测
        if any(re.search(pattern, query) for pattern in self.procedure_patterns):
            return QueryType.PROCEDURE

        # 定义查询检测
        if any(re.search(pattern, query) for pattern in self.definition_patterns):
            return QueryType.DEFINITION

        return QueryType.GENERAL

    def get_confidence(self, query: str, query_type: QueryType) -> float:
        """获取分类置信度"""
        if query_type == QueryType.CLASSIFICATION:
            patterns = self.classification_patterns
        elif query_type == QueryType.DEFINITION:
            patterns = self.definition_patterns
        elif query_type == QueryType.COMPARISON:
            patterns = self.comparison_patterns
        elif query_type == QueryType.PROCEDURE:
            patterns = self.procedure_patterns
        else:
            return 0.5

        matches = sum(1 for pattern in patterns if re.search(pattern, query))
        return min(0.95, 0.6 + matches * 0.15)

class ClassificationContentDetector:
    """分类内容检测器"""

    def __init__(self):
        self.classification_markers = [
            r'\d+種類',           # 2種類
            r'\d+つに.*分',       # 2つに分け
            r'大別される',         # 大別される
            r'大別.*される',       # 大別される
            r'分類される',         # 分類される
            r'型と.*型',          # 普通型と自立型
            r'.*型.*型',          # 複数の型
            r'種類.*分け',         # 種類に分け
            r'分けられる',         # 分けられる
            r'タイプ.*ある',       # タイプがある
        ]

    def detect(self, text: str) -> bool:
        """检测是否包含分类信息"""
        return any(re.search(marker, text) for marker in self.classification_markers)

    def get_classification_strength(self, text: str) -> float:
        """获取分类信息强度"""
        matches = sum(1 for marker in self.classification_markers
                     if re.search(marker, text))
        return min(1.0, matches * 0.4)

class ChunkRanker:
    """Chunk排序器"""

    def __init__(self):
        self.query_classifier = QueryTypeClassifier()
        self.content_detector = ClassificationContentDetector()

    def rank_chunks(self, chunks: List[Dict[str, Any]], query: str) -> List[ChunkScore]:
        """对chunks进行智能排序"""

        query_type = self.query_classifier.classify(query)
        confidence = self.query_classifier.get_confidence(query, query_type)

        print(f"🎯 查询类型: {query_type.value} (置信度: {confidence:.2f})")

        chunk_scores = []

        for chunk in chunks:
            content = chunk.get('content', '') or chunk.get('page_content', '')
            original_score = chunk.get('score', 0.5)  # 原始相似度分数

            chunk_score = ChunkScore(
                content=content,
                original_score=original_score
            )

            # 根据查询类型应用加权
            if query_type == QueryType.CLASSIFICATION:
                chunk_score = self._boost_classification_chunk(chunk_score, query)
            elif query_type == QueryType.DEFINITION:
                chunk_score = self._boost_definition_chunk(chunk_score, query)
            elif query_type == QueryType.COMPARISON:
                chunk_score = self._boost_comparison_chunk(chunk_score, query)

            # 确保final_score被正确计算
            chunk_score.final_score = chunk_score.original_score * chunk_score.boost_factor
            chunk_scores.append(chunk_score)

        # 按最终分数排序
        chunk_scores.sort(key=lambda x: x.final_score, reverse=True)

        return chunk_scores

    def _boost_classification_chunk(self, chunk_score: ChunkScore, query: str) -> ChunkScore:
        """分类查询的chunk加权"""

        # 检测是否包含分类信息
        if self.content_detector.detect(chunk_score.content):
            classification_strength = self.content_detector.get_classification_strength(chunk_score.content)

            # 基础分类加权
            base_boost = 1.5

            # 根据分类信息强度调整
            strength_boost = 1.0 + classification_strength * 0.3

            chunk_score.boost_factor = base_boost * strength_boost
            chunk_score.boost_reason = f"分类信息加权 (强度: {classification_strength:.2f})"

            print(f"✅ 发现分类信息: {chunk_score.content[:30]}... (加权: {chunk_score.boost_factor:.2f})")

        return chunk_score

    def _boost_definition_chunk(self, chunk_score: ChunkScore, query: str) -> ChunkScore:
        """定义查询的chunk加权"""

        # 检测是否为定义性内容
        if self._is_definition_content(chunk_score.content):
            chunk_score.boost_factor = 1.3
            chunk_score.boost_reason = "定义信息加权"

        return chunk_score

    def _boost_comparison_chunk(self, chunk_score: ChunkScore, query: str) -> ChunkScore:
        """比较查询的chunk加权"""

        # 检测是否包含比较信息
        if self._is_comparison_content(chunk_score.content):
            chunk_score.boost_factor = 1.4
            chunk_score.boost_reason = "比较信息加权"

        return chunk_score

    def _is_definition_content(self, content: str) -> bool:
        """检测是否为定义内容"""
        definition_indicators = [r'とは', r'である', r'です', r'を指す', r'を意味する']
        return any(re.search(indicator, content) for indicator in definition_indicators)

    def _is_comparison_content(self, content: str) -> bool:
        """检测是否为比较内容"""
        comparison_indicators = [r'一方', r'他方', r'違い', r'比べて', r'対して']
        return any(re.search(indicator, content) for indicator in comparison_indicators)

def test_query_classification():
    """测试查询分类系统"""

    print("🧪 查询类型分类测试")
    print("=" * 60)

    classifier = QueryTypeClassifier()
    ranker = ChunkRanker()

    # 测试查询
    test_queries = [
        "農業機械の種類について教えてください",
        "コンバインとは何ですか",
        "普通型と自立型の違いは何ですか",
        "農業機械を使う手順を教えてください",
        "今日の天気はどうですか"
    ]

    for query in test_queries:
        query_type = classifier.classify(query)
        confidence = classifier.get_confidence(query, query_type)

        print(f"🔍 '{query}'")
        print(f"   → 类型: {query_type.value}, 置信度: {confidence:.2f}")
        print()

    # 测试chunk排序
    print("\n📊 Chunk排序测试")
    print("=" * 60)

    query = "農業機械の種類について教えてください"
    test_chunks = [
        {
            'content': "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
            'score': 0.85
        },
        {
            'content': "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
            'score': 0.82  # 原本分数较低
        },
        {
            'content': "普通型は主にアメリカやヨーロッパ等大規模農業で使われています。",
            'score': 0.88
        },
        {
            'content': "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された機械です。",
            'score': 0.87
        }
    ]

    print(f"🔍 查询: {query}")
    print("\n原始排序:")
    for i, chunk in enumerate(sorted(test_chunks, key=lambda x: x['score'], reverse=True), 1):
        is_classification = "🎯" if "2種類に大別" in chunk['content'] else "  "
        print(f"{i}. {is_classification} {chunk['score']:.2f} - {chunk['content'][:40]}...")

    # 应用智能排序
    ranked_chunks = ranker.rank_chunks(test_chunks, query)

    print("\n优化后排序:")
    for i, chunk_score in enumerate(ranked_chunks, 1):
        is_classification = "🎯" if "2種類に大別" in chunk_score.content else "  "
        boost_info = f"(×{chunk_score.boost_factor:.2f})" if chunk_score.boost_factor > 1.0 else ""

        # 确保使用正确计算的final_score
        calculated_final = chunk_score.original_score * chunk_score.boost_factor

        print(f"{i}. {is_classification} {chunk_score.original_score:.2f}→{calculated_final:.2f} {boost_info} - {chunk_score.content[:40]}...")
        if chunk_score.boost_reason:
            print(f"     理由: {chunk_score.boost_reason}")

if __name__ == "__main__":
    test_query_classification()