#!/usr/bin/env python3
"""
RAG系统优化建议和实现方案
"""

def print_optimization_recommendations():
    """详细的优化建议"""

    print("🚀 RAG系统优化建议")
    print("=" * 80)

    print("📋 基于分析结果的具体优化方案:")
    print()

    # 优化建议 1
    print("🔥 1. 查询类型智能识别和分类信息加权")
    print("-" * 60)
    print("""
目标: 识别分类查询，给分类信息优先权

实现方案:

A. 查询类型分类器:
```python
def classify_query_type(query):
    classification_patterns = [
        r'種類.*について',
        r'.*の種類',
        r'分類.*について',
        r'.*分類',
        r'何種類',
        r'いくつ.*種類'
    ]

    if any(re.search(pattern, query) for pattern in classification_patterns):
        return "classification"
    elif any(word in query for word in ['とは', '何ですか']):
        return "definition"
    else:
        return "general"
```

B. 分类信息检测器:
```python
def detect_classification_content(text):
    classification_markers = [
        r'\d+種類',        # 2種類
        r'\d+つに.*分',    # 2つに分け
        r'大別される',      # 大別される
        r'分類される',      # 分類される
        r'型と.*型',       # 普通型と自立型
    ]

    return any(re.search(marker, text) for marker in classification_markers)
```

C. 动态加权策略:
```python
def apply_classification_boost(chunks, query):
    query_type = classify_query_type(query)

    if query_type == "classification":
        for chunk in chunks:
            if detect_classification_content(chunk.content):
                chunk.score *= 1.5  # 分类信息加权50%
                chunk.metadata['boosted'] = 'classification'

    return sorted(chunks, key=lambda x: x.score, reverse=True)
```
""")

    # 优化建议 2
    print("\n🔥 2. 多粒度检索策略优化")
    print("-" * 60)
    print("""
目标: 确保分类信息在多粒度检索中不被长文本掩盖

实现方案:

A. 粒度权重动态调整:
```python
def adaptive_granularity_weights(query):
    query_type = classify_query_type(query)

    if query_type == "classification":
        return {
            'sentence': 1.0,      # 句子级优先
            'short_passage': 0.7,
            'long_passage': 0.4
        }
    elif query_type == "definition":
        return {
            'sentence': 0.8,
            'short_passage': 1.0, # 短段落优先
            'long_passage': 0.6
        }
    else:
        return {
            'sentence': 0.6,
            'short_passage': 0.8,
            'long_passage': 1.0   # 长段落优先
        }
```

B. 分层检索策略:
```python
def layered_retrieval(query, k=5):
    # 第一层：句子级检索分类信息
    sentence_results = retrieve_by_granularity(query, 'sentence', k=3)

    # 第二层：段落级检索上下文
    passage_results = retrieve_by_granularity(query, 'passage', k=3)

    # 合并并重新排序
    combined_results = merge_and_rerank(
        sentence_results,
        passage_results,
        query_type=classify_query_type(query)
    )

    return combined_results[:k]
```
""")

    # 优化建议 3
    print("\n🔥 3. 语义匹配增强")
    print("-" * 60)
    print("""
目标: 增强查询与分类信息的语义匹配能力

实现方案:

A. 查询扩展:
```python
def expand_classification_query(query):
    if "種類" in query:
        expanded_terms = [
            "種類", "分類", "タイプ", "型", "大別", "分け",
            "2種類", "二種類", "複数", "いくつか"
        ]
        return query + " " + " ".join(expanded_terms)
    return query
```

B. 语义相似词匹配:
```python
SEMANTIC_GROUPS = {
    "classification": ["種類", "分類", "大別", "型", "タイプ"],
    "agriculture": ["農業", "農機", "機械", "コンバイン"],
    "types": ["普通型", "自立型", "2種類", "二種類"]
}

def semantic_matching_boost(query, chunk_text):
    boost_score = 1.0

    for group, terms in SEMANTIC_GROUPS.items():
        query_matches = sum(1 for term in terms if term in query)
        chunk_matches = sum(1 for term in terms if term in chunk_text)

        if query_matches > 0 and chunk_matches > 0:
            boost_score *= (1.0 + 0.2 * min(query_matches, chunk_matches))

    return boost_score
```
""")

    # 优化建议 4
    print("\n🔥 4. Chunking策略精细化")
    print("-" * 60)
    print("""
目标: 确保关键分类信息独立成chunk

实现方案:

A. 智能句子边界检测:
```python
def smart_sentence_boundary_split(text):
    # 优先保护分类信息的完整性
    classification_sentences = []
    remaining_text = text

    # 提取分类句子
    classification_patterns = [
        r'[^。]*\d+種類[^。]*。',
        r'[^。]*大別される[^。]*。',
        r'[^。]*型と.*型[^。]*。'
    ]

    for pattern in classification_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            classification_sentences.append(match.strip())
            remaining_text = remaining_text.replace(match, "")

    # 处理剩余文本
    regular_sentences = re.split(r'[。！？]', remaining_text)
    regular_sentences = [s.strip() + "。" for s in regular_sentences if s.strip()]

    return classification_sentences + regular_sentences
```

B. 上下文感知分块:
```python
def context_aware_chunking(sentences, max_chunk_size=150):
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # 分类信息优先独立成chunk
        if detect_classification_content(sentence):
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            chunks.append(sentence)  # 分类信息独立
        else:
            if len(current_chunk + sentence) <= max_chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks
```
""")

    # 优化建议 5
    print("\n🔥 5. 混合检索策略")
    print("-" * 60)
    print("""
目标: 结合向量检索和关键词检索的优势

实现方案:

A. 双路径检索:
```python
def hybrid_retrieval(query, k=5):
    # 路径1：向量检索
    vector_results = vector_search(query, k=10)

    # 路径2：关键词检索
    keyword_results = keyword_search(query, k=10)

    # 路径3：分类专门检索
    if classify_query_type(query) == "classification":
        classification_results = classification_specific_search(query, k=5)
        return merge_results([vector_results, keyword_results, classification_results], k)

    return merge_results([vector_results, keyword_results], k)
```

B. 结果融合算法:
```python
def merge_results(result_sets, k):
    # 使用RRF (Reciprocal Rank Fusion)
    document_scores = {}

    for results in result_sets:
        for rank, doc in enumerate(results, 1):
            doc_id = doc.id
            if doc_id not in document_scores:
                document_scores[doc_id] = {'doc': doc, 'score': 0}

            document_scores[doc_id]['score'] += 1.0 / (rank + 60)

    # 排序并返回top-k
    sorted_results = sorted(
        document_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )

    return [item['doc'] for item in sorted_results[:k]]
```
""")

    # 优化建议 6
    print("\n🔥 6. LLM Ranking优化")
    print("-" * 60)
    print("""
目标: 训练LLM更好地理解查询意图和内容相关性

实现方案:

A. 查询类型感知的Ranking Prompt:
```python
def create_ranking_prompt(query, chunks, query_type):
    base_prompt = f"查询: {query}\\n"

    if query_type == "classification":
        instruction = '''
优先考虑以下因素进行排序:
1. 包含明确分类信息的内容 (如"2种类"、"大别"等)
2. 直接回答分类问题的内容
3. 提供具体类别名称的内容
        '''
    else:
        instruction = "按照内容与查询的相关性进行排序"

    return base_prompt + instruction + format_chunks(chunks)
```

B. 多阶段Ranking:
```python
def multi_stage_ranking(query, chunks):
    # 阶段1：快速过滤
    filtered = fast_filter(chunks, query)

    # 阶段2：LLM精排
    if len(filtered) <= 5:
        return llm_ranking(query, filtered)
    else:
        top_candidates = filtered[:10]
        return llm_ranking(query, top_candidates)[:5]
```
""")

    # 实施建议
    print("\n📝 实施优先级建议")
    print("-" * 60)
    print("""
🥇 高优先级 (立即实施):
1. 查询类型识别和分类信息加权 - 直接提升分类查询效果
2. 多粒度检索权重调整 - 确保分类信息不被掩盖

🥈 中优先级 (短期实施):
3. 语义匹配增强 - 提升整体匹配准确度
4. Chunking策略优化 - 改善信息保留质量

🥉 低优先级 (长期优化):
5. 混合检索策略 - 系统性能整体提升
6. LLM Ranking深度优化 - 高级功能完善

📊 预期效果:
- 分类查询准确率提升: 60% → 85%+
- 核心信息遗漏率降低: 30% → 10%以下
- 整体用户满意度提升: 25%+
""")

    print("\n" + "=" * 80)
    print("💡 下一步行动建议: 从查询类型识别开始，逐步实施优化方案")
    print("=" * 80)

if __name__ == "__main__":
    print_optimization_recommendations()