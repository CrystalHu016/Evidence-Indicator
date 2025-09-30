#!/usr/bin/env python3
"""
多粒度检索策略优化
实现RAG系统的第二个高优先级优化：确保分类信息不被长文档掩盖
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# 导入第一个优化的查询分类器
from query_type_classifier import QueryType, QueryTypeClassifier, ClassificationContentDetector

class GranularityLevel(Enum):
    """粒度级别枚举"""
    SENTENCE = "sentence"           # 句子级 (30-80字符)
    SHORT_PASSAGE = "short_passage" # 短段落级 (80-200字符)
    LONG_PASSAGE = "long_passage"   # 长段落级 (200-500字符)

@dataclass
class GranularChunk:
    """多粒度chunk数据结构"""
    content: str
    granularity: GranularityLevel
    original_score: float
    granularity_weight: float = 1.0
    content_boost: float = 1.0
    final_score: float = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.final_score = self.original_score * self.granularity_weight * self.content_boost

class MultiGranularRetriever:
    """多粒度检索器"""

    def __init__(self):
        self.query_classifier = QueryTypeClassifier()
        self.content_detector = ClassificationContentDetector()

    def get_adaptive_granularity_weights(self, query: str) -> Dict[GranularityLevel, float]:
        """根据查询类型自适应调整粒度权重"""

        query_type = self.query_classifier.classify(query)

        if query_type == QueryType.CLASSIFICATION:
            # 分类查询：优先句子级，避免被长文档掩盖
            return {
                GranularityLevel.SENTENCE: 1.0,      # 句子级最高权重
                GranularityLevel.SHORT_PASSAGE: 0.7, # 短段落中等权重
                GranularityLevel.LONG_PASSAGE: 0.4   # 长段落低权重
            }
        elif query_type == QueryType.DEFINITION:
            # 定义查询：偏好短段落，需要一定上下文
            return {
                GranularityLevel.SENTENCE: 0.8,      # 句子级高权重
                GranularityLevel.SHORT_PASSAGE: 1.0, # 短段落最高权重
                GranularityLevel.LONG_PASSAGE: 0.6   # 长段落中等权重
            }
        elif query_type == QueryType.COMPARISON:
            # 比较查询：需要更多上下文
            return {
                GranularityLevel.SENTENCE: 0.6,      # 句子级中等权重
                GranularityLevel.SHORT_PASSAGE: 1.0, # 短段落最高权重
                GranularityLevel.LONG_PASSAGE: 0.8   # 长段落高权重
            }
        elif query_type == QueryType.PROCEDURE:
            # 流程查询：偏好长段落
            return {
                GranularityLevel.SENTENCE: 0.5,      # 句子级低权重
                GranularityLevel.SHORT_PASSAGE: 0.8, # 短段落高权重
                GranularityLevel.LONG_PASSAGE: 1.0   # 长段落最高权重
            }
        else:
            # 一般查询：平衡权重
            return {
                GranularityLevel.SENTENCE: 0.7,
                GranularityLevel.SHORT_PASSAGE: 0.9,
                GranularityLevel.LONG_PASSAGE: 1.0
            }

    def layered_retrieval(self, query: str, chunks_by_granularity: Dict[GranularityLevel, List[Dict]], k: int = 5) -> List[GranularChunk]:
        """分层检索策略"""

        query_type = self.query_classifier.classify(query)
        granularity_weights = self.get_adaptive_granularity_weights(query)

        print(f"🎯 查询类型: {query_type.value}")
        print(f"📊 粒度权重: {dict((g.value, w) for g, w in granularity_weights.items())}")

        all_granular_chunks = []

        # 处理每个粒度级别的chunks
        for granularity, chunks in chunks_by_granularity.items():
            granularity_weight = granularity_weights.get(granularity, 1.0)

            for chunk_data in chunks:
                chunk_content = chunk_data.get('content', '') or chunk_data.get('page_content', '')
                original_score = chunk_data.get('score', 0.5)

                # 创建多粒度chunk
                granular_chunk = GranularChunk(
                    content=chunk_content,
                    granularity=granularity,
                    original_score=original_score,
                    granularity_weight=granularity_weight,
                    metadata=chunk_data.get('metadata', {})
                )

                # 应用内容增强（基于第一个优化的分类信息加权）
                granular_chunk = self._apply_content_boost(granular_chunk, query, query_type)

                all_granular_chunks.append(granular_chunk)

        # 排序并返回top-k
        all_granular_chunks.sort(key=lambda x: x.final_score, reverse=True)

        return all_granular_chunks[:k]

    def _apply_content_boost(self, chunk: GranularChunk, query: str, query_type: QueryType) -> GranularChunk:
        """应用内容增强权重"""

        if query_type == QueryType.CLASSIFICATION:
            # 对分类查询，检测分类信息并加权
            if self.content_detector.detect(chunk.content):
                classification_strength = self.content_detector.get_classification_strength(chunk.content)

                # 分类信息加权，句子级别加权更高
                if chunk.granularity == GranularityLevel.SENTENCE:
                    chunk.content_boost = 1.5 + classification_strength * 0.3  # 句子级分类信息最高加权
                elif chunk.granularity == GranularityLevel.SHORT_PASSAGE:
                    chunk.content_boost = 1.3 + classification_strength * 0.2  # 短段落级适中加权
                else:
                    chunk.content_boost = 1.2 + classification_strength * 0.1  # 长段落级较低加权

                chunk.metadata['classification_boost'] = True
                chunk.metadata['classification_strength'] = classification_strength

                print(f"✅ 发现分类信息 ({chunk.granularity.value}): {chunk.content[:30]}... (内容加权: {chunk.content_boost:.2f})")

        elif query_type == QueryType.DEFINITION:
            # 定义查询的内容增强
            if self._is_definition_content(chunk.content):
                chunk.content_boost = 1.3
                chunk.metadata['definition_boost'] = True

        # 重新计算最终分数
        chunk.final_score = chunk.original_score * chunk.granularity_weight * chunk.content_boost

        return chunk

    def _is_definition_content(self, content: str) -> bool:
        """检测定义内容"""
        definition_indicators = [r'とは', r'である', r'です', r'を指す', r'を意味する']
        return any(re.search(indicator, content) for indicator in definition_indicators)

    def prioritized_granularity_selection(self, query: str, chunks_by_granularity: Dict[GranularityLevel, List[Dict]], k: int = 5) -> List[GranularChunk]:
        """优先级导向的粒度选择"""

        query_type = self.query_classifier.classify(query)

        # 特殊处理分类查询
        if query_type == QueryType.CLASSIFICATION:
            return self._classification_priority_selection(query, chunks_by_granularity, k)
        else:
            return self.layered_retrieval(query, chunks_by_granularity, k)

    def _classification_priority_selection(self, query: str, chunks_by_granularity: Dict[GranularityLevel, List[Dict]], k: int) -> List[GranularChunk]:
        """分类查询的优先级选择策略"""

        print("🎯 分类查询优先级策略")

        # 第一步：从句子级寻找分类信息
        sentence_classification_chunks = []
        if GranularityLevel.SENTENCE in chunks_by_granularity:
            for chunk_data in chunks_by_granularity[GranularityLevel.SENTENCE]:
                chunk_content = chunk_data.get('content', '') or chunk_data.get('page_content', '')

                if self.content_detector.detect(chunk_content):
                    granular_chunk = GranularChunk(
                        content=chunk_content,
                        granularity=GranularityLevel.SENTENCE,
                        original_score=chunk_data.get('score', 0.5),
                        granularity_weight=1.0,  # 最高权重
                        content_boost=2.0,       # 强力加权
                        metadata={'priority_selection': 'classification_sentence'}
                    )
                    granular_chunk.final_score = granular_chunk.original_score * granular_chunk.granularity_weight * granular_chunk.content_boost
                    sentence_classification_chunks.append(granular_chunk)

        print(f"🔍 找到 {len(sentence_classification_chunks)} 个句子级分类信息")

        # 第二步：如果找到足够的句子级分类信息，优先使用
        if len(sentence_classification_chunks) >= 2:
            # 补充一些上下文信息
            context_chunks = self._get_context_chunks(chunks_by_granularity, k - len(sentence_classification_chunks))
            result = sentence_classification_chunks + context_chunks
            return result[:k]

        # 第三步：如果句子级分类信息不足，使用混合策略
        return self._mixed_granularity_selection(query, chunks_by_granularity, sentence_classification_chunks, k)

    def _get_context_chunks(self, chunks_by_granularity: Dict[GranularityLevel, List[Dict]], needed: int) -> List[GranularChunk]:
        """获取上下文chunks"""

        context_chunks = []

        # 优先从短段落获取上下文
        if GranularityLevel.SHORT_PASSAGE in chunks_by_granularity:
            for chunk_data in chunks_by_granularity[GranularityLevel.SHORT_PASSAGE][:needed]:
                chunk_content = chunk_data.get('content', '') or chunk_data.get('page_content', '')

                granular_chunk = GranularChunk(
                    content=chunk_content,
                    granularity=GranularityLevel.SHORT_PASSAGE,
                    original_score=chunk_data.get('score', 0.5),
                    granularity_weight=0.8,
                    content_boost=1.0,
                    metadata={'context_chunk': True}
                )
                granular_chunk.final_score = granular_chunk.original_score * granular_chunk.granularity_weight * granular_chunk.content_boost
                context_chunks.append(granular_chunk)

        return context_chunks

    def _mixed_granularity_selection(self, query: str, chunks_by_granularity: Dict[GranularityLevel, List[Dict]], priority_chunks: List[GranularChunk], k: int) -> List[GranularChunk]:
        """混合粒度选择策略"""

        # 使用标准分层检索补充剩余slots
        remaining_k = k - len(priority_chunks)

        if remaining_k > 0:
            # 从非句子级获取补充chunks
            remaining_chunks_data = {}
            for granularity, chunks in chunks_by_granularity.items():
                if granularity != GranularityLevel.SENTENCE:
                    remaining_chunks_data[granularity] = chunks

            additional_chunks = self.layered_retrieval(query, remaining_chunks_data, remaining_k)
            return priority_chunks + additional_chunks

        return priority_chunks

def test_multi_granular_optimization():
    """测试多粒度检索优化"""

    print("🚀 多粒度检索优化测试")
    print("=" * 80)

    retriever = MultiGranularRetriever()

    # 模拟不同粒度的chunks
    test_chunks = {
        GranularityLevel.SENTENCE: [
            {
                'content': "コンバインは農業機械です。",
                'score': 0.75
            },
            {
                'content': "日本で使われているコンバインは普通型と自立型の2種類に大別されます。", # 核心分类信息
                'score': 0.82
            },
            {
                'content': "自立型は日本独自の農業機械です。",
                'score': 0.78
            }
        ],
        GranularityLevel.SHORT_PASSAGE: [
            {
                'content': "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本では広く使用されています。",
                'score': 0.85
            },
            {
                'content': "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。",
                'score': 0.88
            }
        ],
        GranularityLevel.LONG_PASSAGE: [
            {
                'content': "農業機械の発展により、現代の農業は大きく変化しました。特にコンバインハーベスターの導入は、穀物収穫の効率を飛躍的に向上させました。これらの機械は、収穫、脱穀、選別という複数の工程を一台で処理することができ、農作業の省力化に大きく貢献しています。",
                'score': 0.90
            }
        ]
    }

    # 测试不同查询类型
    test_queries = [
        ("農業機械の種類について教えてください", "分类查询"),
        ("コンバインとは何ですか", "定义查询"),
        ("農業機械の使い方を教えてください", "流程查询")
    ]

    for query, query_desc in test_queries:
        print(f"\n🔍 测试查询: {query} ({query_desc})")
        print("-" * 60)

        # 标准多粒度检索
        print("📊 标准多粒度检索结果:")
        standard_results = retriever.layered_retrieval(query, test_chunks, k=5)

        for i, chunk in enumerate(standard_results, 1):
            is_classification = "🎯" if "2種類に大別" in chunk.content else "  "
            print(f"{i}. {is_classification} [{chunk.granularity.value}] {chunk.original_score:.2f}→{chunk.final_score:.2f}")
            print(f"     权重: 粒度×{chunk.granularity_weight:.2f}, 内容×{chunk.content_boost:.2f}")
            print(f"     内容: {chunk.content[:50]}...")

        # 优先级导向检索（专门处理分类查询）
        if "種類" in query:
            print(f"\n🎯 优先级导向检索结果:")
            priority_results = retriever.prioritized_granularity_selection(query, test_chunks, k=5)

            for i, chunk in enumerate(priority_results, 1):
                is_classification = "🎯" if "2種類に大別" in chunk.content else "  "
                print(f"{i}. {is_classification} [{chunk.granularity.value}] {chunk.original_score:.2f}→{chunk.final_score:.2f}")
                print(f"     策略: {chunk.metadata.get('priority_selection', '标准')}")
                print(f"     内容: {chunk.content[:50]}...")

        print("=" * 80)

if __name__ == "__main__":
    test_multi_granular_optimization()