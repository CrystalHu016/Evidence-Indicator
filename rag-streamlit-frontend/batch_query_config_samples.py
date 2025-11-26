#!/usr/bin/env python3
"""
Batch Query Config Sample Queries
Processes all 50 sample queries from config.py and saves to database with evidences
"""

import sys
import os
import json
import time

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)  # Insert current_dir first to prioritize local config.py
sys.path.insert(0, parent_dir)

from database.query_history_manager import QueryHistoryManager
from backend_integration import call_backend_query

# Import from local config.py in rag-streamlit-frontend directory
import importlib.util
config_path = os.path.join(current_dir, 'config.py')
spec = importlib.util.spec_from_file_location("config_local", config_path)
config_local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_local)
SAMPLE_QUERIES_BY_CATEGORY = config_local.SAMPLE_QUERIES_BY_CATEGORY

# Flatten config queries with category info
config_queries = []
for category, queries in SAMPLE_QUERIES_BY_CATEGORY.items():
    for q in queries:
        config_queries.append({
            'question': q,
            'category': category
        })

print("=" * 70)
print("Batch Query Config Sample Queries")
print("=" * 70)
print(f"Total queries from config.py: {len(config_queries)}")

# Show categories
category_counts = {}
for item in config_queries:
    cat = item['category']
    category_counts[cat] = category_counts.get(cat, 0) + 1

print("\n📂 Categories:")
for cat, count in category_counts.items():
    print(f"  {cat}: {count} queries")

# Load dataset for answer lookup
dataset_lookup = {}
try:
    dataset_path = os.path.join(parent_dir, 'data', 'merged_qa_dataset.json')
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        # Create lookup dictionary: question -> answer
        for item in dataset:
            if 'answers' in item and 'text' in item['answers'] and item['answers']['text']:
                dataset_lookup[item['question']] = item['answers']['text'][0]
        print(f"\n✅ Loaded dataset with {len(dataset_lookup)} Q&A pairs for lookup")
except Exception as e:
    print(f"⚠️ Could not load dataset for lookup: {e}")
    dataset_lookup = {}

# Initialize database
DB_PATH = os.path.join(parent_dir, 'query_history.db')
manager = QueryHistoryManager(DB_PATH)
print(f"✅ Query history database initialized: {DB_PATH}")

# Check existing queries in database
print("\n📊 データベース内の既存クエリを確認中...")
existing_queries = set()
recent_queries = manager.get_recent_queries(limit=2000)
for record in recent_queries:
    existing_queries.add(record['query'])

print(f"✅ データベース内の既存クエリ: {len(existing_queries)} 件")

# Filter new queries
new_queries = []
for item in config_queries:
    query = item['question']
    if query not in existing_queries:
        new_queries.append(item)

print(f"\n📋 総クエリ数: {len(config_queries)}")
print(f"🆕 処理対象の新規クエリ: {len(new_queries)}")
print(f"✅ すでにデータベースに存在: {len(config_queries) - len(new_queries)}")

if len(new_queries) == 0:
    print("\n✅ すべてのクエリがすでにデータベースに存在します。")
    sys.exit(0)

# Show warning
print("\n" + "=" * 70)
print("⚠️  警告: RAGクエリを実行します")
print("=" * 70)
print(f"クエリ数: {len(new_queries)}")
print(f"推定時間: 約{len(new_queries) * 25}秒 ({len(new_queries) * 25 / 60:.1f}分)")

print("\n実行するクエリ (カテゴリ別):")
for cat in category_counts.keys():
    cat_queries = [q for q in new_queries if q['category'] == cat]
    if cat_queries:
        print(f"\n  📁 {cat}: {len(cat_queries)} queries")
        for q in cat_queries[:2]:
            print(f"     - {q['question'][:50]}...")
        if len(cat_queries) > 2:
            print(f"     ... and {len(cat_queries) - 2} more")

print("\n⏳ 2秒後に開始...")
time.sleep(2)

# Process queries
print("\n" + "=" * 70)
print("🚀 バッチクエリ開始...")
print("=" * 70)

results = []
success_count = 0
no_info_count = 0
error_count = 0

for i, item in enumerate(new_queries, 1):
    query = item['question']
    category = item['category']

    print(f"\n[{i}/{len(new_queries)}] 処理中: {query[:60]}...")
    print(f"カテゴリ: {category}")

    start_time = time.time()

    try:
        # Call backend
        result, _ = call_backend_query(query)
        elapsed = time.time() - start_time

        # Check if answer indicates no relevant information
        answer = result.get('answer', '')
        if 'no relevant information' in answer.lower() or '関連情報' in answer:
            status = '⚠️ 関連情報なし'
            no_info_count += 1
        else:
            status = '✅ 成功'
            success_count += 1

        print(f"{status} (took {elapsed:.1f}s)")
        print(f"   回答プレビュー: {answer[:80]}...")

        # Get dataset_answer from merged_qa_dataset.json lookup
        dataset_answer = dataset_lookup.get(query, "")
        if dataset_answer:
            print(f"   📋 データセット回答: {dataset_answer[:50]}...")

        # Display match metrics if available
        match_metrics = result.get('match_metrics', {})
        if match_metrics:
            print(f"   📊 マッチ率: {match_metrics.get('match_rate', 0):.1%}")

        # Save to database
        chunks = result.get('chunks', [])
        num_chunks = len(chunks) if isinstance(chunks, list) else (chunks if isinstance(chunks, int) else 0)

        # Serialize evidences to JSON for database storage
        evidences = result.get('evidences', [])
        evidences_json = json.dumps(evidences, ensure_ascii=False) if evidences else ""

        manager.add_query(
            query=query,
            generated_answer=answer,
            processing_time=elapsed,
            model=result.get('model', 'PureSemanticRAG'),
            confidence=result.get('confidence', 0.0),
            num_chunks=num_chunks,
            dataset_answer=dataset_answer,
            evidences=evidences_json
        )
        print(f"   💾 データベースに保存済み (evidences: {len(evidences)} chunks)")

        results.append({
            'query': query,
            'category': category,
            'status': status,
            'processing_time': elapsed,
            'answer': answer,
            'dataset_answer': dataset_answer
        })

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ エラー (took {elapsed:.1f}s): {str(e)}")
        error_count += 1
        results.append({
            'query': query,
            'category': category,
            'status': '❌ エラー',
            'processing_time': elapsed,
            'error': str(e)
        })

# Summary
print("\n" + "=" * 70)
print("📊 バッチクエリ完了 - サマリー")
print("=" * 70)
print(f"✅ 成功: {success_count}/{len(new_queries)} ({success_count/len(new_queries)*100:.1f}%)")
print(f"⚠️ 関連情報なし: {no_info_count}/{len(new_queries)} ({no_info_count/len(new_queries)*100:.1f}%)")
print(f"❌ エラー: {error_count}/{len(new_queries)} ({error_count/len(new_queries)*100:.1f}%)")

# Category summary
print(f"\n📂 カテゴリ別サマリー:")
category_stats = {}
for r in results:
    cat = r['category']
    if cat not in category_stats:
        category_stats[cat] = {'success': 0, 'no_info': 0, 'error': 0, 'total': 0}
    category_stats[cat]['total'] += 1
    if '✅' in r['status']:
        category_stats[cat]['success'] += 1
    elif '⚠️' in r['status']:
        category_stats[cat]['no_info'] += 1
    else:
        category_stats[cat]['error'] += 1

for cat, stats in sorted(category_stats.items()):
    print(f"  {cat:30s}: {stats['success']}/{stats['total']} 成功, {stats['no_info']} 情報なし")

print(f"\n✅ すべてのクエリがデータベースに保存されました")
print("=" * 70)
