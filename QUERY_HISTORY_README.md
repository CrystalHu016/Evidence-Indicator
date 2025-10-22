# クエリ履歴（Query History）持久化存储系统

## 概述

这个系统会自动将每次查询的所有信息保存到SQLite数据库中，即使重启前端也能保留所有历史记录。

## 数据库位置

**数据库文件：** `./query_history.db`

即使删除前端缓存或重启系统，这个数据库文件都会保留所有历史记录。

## 存储的信息

### 1. 主查询表 (query_history)
- ✅ **timestamp** - 查询时间戳
- ✅ **query** - 用户的问题
- ✅ **generated_answer** - 系统生成的完整答案
- ✅ **processing_time** - 处理时间（秒）
- ✅ **model** - 使用的模型名称
- ✅ **confidence** - 置信度分数
- ✅ **num_chunks** - 使用的chunk数量

### 2. 证据提取详情表 (evidence_extraction)
- ✅ **query_id** - 关联的查询ID
- ✅ **chunk_id** - Chunk编号
- ✅ **chunk_content** - 原始chunk内容
- ✅ **extraction_prompt** - 完整的evidence extraction prompt
- ✅ **llm_raw_response** - LLM返回的原始响应
- ✅ **extracted_ranges** - 提取的字符范围 JSON
- ✅ **extracted_texts** - 高亮的文本内容 JSON
- ✅ **similarity_score** - 向量相似度分数
- ✅ **semantic_relevance** - 语义相关性分数

### 3. 数据集对比表 (dataset_comparison)
- ✅ **dataset_answer** - 原始数据集答案
- ✅ **dataset_answer_start/end** - 数据集答案位置
- ✅ **rag_extracted_text** - RAG提取的文本
- ✅ **rag_start/end** - RAG提取位置
- ✅ **match_score** - 匹配分数

## 使用方法

### 1. 查看最近的查询

```bash
# 进入Python环境
python3

from query_history_manager import QueryHistoryManager

manager = QueryHistoryManager("./query_history.db")

# 查看最近10条查询
recent = manager.get_recent_queries(limit=10)
for q in recent:
    print(f"Query: {q['query']}")
    print(f"Answer: {q['generated_answer'][:100]}...")
    print(f"Time: {q['processing_time']}s")
    print("-" * 80)
```

### 2. 查看某个查询的完整详情

```python
# 获取query_id=5的完整详情
details = manager.get_query_details(query_id=5)

print(f"Query: {details['query']}")
print(f"Answer: {details['generated_answer']}")

# 查看所有evidence extractions
for evidence in details['evidences']:
    print(f"\nChunk {evidence['chunk_id']}:")
    print(f"  Extracted text: {evidence['extracted_texts']}")
    print(f"  Ranges: {evidence['extracted_ranges']}")
    print(f"  Similarity: {evidence['similarity_score']}")
    print(f"  Relevance: {evidence['semantic_relevance']}")
```

### 3. 搜索查询

```python
# 搜索包含"亜熱帯ジェット気流"的查询
results = manager.search_queries("亜熱帯ジェット気流", limit=20)

for r in results:
    print(f"[{r['timestamp']}] {r['query']}")
```

### 4. 导出为JSON

```python
# 导出所有历史到JSON文件
manager.export_to_json("query_history_export.json")
```

### 5. 查看统计信息

```python
stats = manager.get_statistics()
print("📊 Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")
```

输出示例：
```
📊 Statistics:
  total_queries: 150
  avg_processing_time: 3.2
  avg_confidence: 0.78
  total_evidence_extractions: 450
  avg_similarity_score: 0.342
  avg_semantic_relevance: 0.725
```

## 直接查询数据库

```bash
# 使用SQLite命令行工具
sqlite3 query_history.db

# 查看所有表
.tables

# 查看最近10条查询
SELECT id, query, processing_time FROM query_history ORDER BY created_at DESC LIMIT 10;

# 查看某个查询的所有evidence extractions
SELECT chunk_id, extracted_texts, similarity_score, semantic_relevance
FROM evidence_extraction
WHERE query_id = 5;

# 退出
.quit
```

## 数据库维护

### 备份数据库

```bash
cp query_history.db query_history_backup_$(date +%Y%m%d).db
```

### 清理旧数据

```python
import sqlite3

conn = sqlite3.connect("query_history.db")
cursor = conn.cursor()

# 删除30天前的记录
cursor.execute("""
    DELETE FROM query_history
    WHERE created_at < datetime('now', '-30 days')
""")

conn.commit()
conn.close()
```

## 集成到后端

后端代码已经自动集成了历史记录功能。每次调用 `query_with_answer()` 时，系统会自动：

1. 保存查询和答案到 `query_history` 表
2. 保存每个chunk的evidence extraction详情到 `evidence_extraction` 表
3. 返回 `query_id` 供后续使用

## 分析示例

### 分析prompt效果

```python
# 找出所有提取失败的案例（extracted_texts为空）
conn = sqlite3.connect("query_history.db")
cursor = conn.cursor()

cursor.execute("""
    SELECT q.query, e.extraction_prompt, e.llm_raw_response
    FROM evidence_extraction e
    JOIN query_history q ON e.query_id = q.id
    WHERE e.extracted_texts = '[]'
    LIMIT 20
""")

for row in cursor.fetchall():
    print(f"Query: {row[0]}")
    print(f"LLM Response: {row[2]}")
    print("-" * 80)
```

### 比较不同query的表现

```python
# 找出处理时间最长的查询
cursor.execute("""
    SELECT query, processing_time, num_chunks
    FROM query_history
    ORDER BY processing_time DESC
    LIMIT 10
""")

for query, time, chunks in cursor.fetchall():
    print(f"{time:.2f}s | {chunks} chunks | {query[:50]}...")
```

## 数据永久保存

✅ **即使重启前端，数据也会保留**
✅ **SQLite数据库文件持久化存储**
✅ **支持导出为JSON进行备份**
✅ **支持SQL查询进行高级分析**

