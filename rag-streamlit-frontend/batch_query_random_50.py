#!/usr/bin/env python3
"""
Batch Query Random 50 Questions
Query 50 random questions from data/random_50_queries.json and save to database
"""

import os
import sys
import time
from datetime import datetime
import json
import importlib.util

# Import necessary modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Insert parent path for other imports
sys.path.insert(0, parent_dir)

# Import from parent directory
from query_history_manager import QueryHistoryManager
from backend_integration import call_backend_query

print("=" * 70)
print("Batch Query Random 50 Questions")
print("=" * 70)

# Initialize query history manager
db_path = os.path.join(parent_dir, "query_history.db")
manager = QueryHistoryManager(db_path)

# Load random 50 queries
random_queries_file = os.path.join(parent_dir, "data", "random_50_queries.json")
with open(random_queries_file, 'r', encoding='utf-8') as f:
    random_queries = json.load(f)

print(f"✅ ランダム質問ロード完了: {len(random_queries)} 件")

# Check existing queries in database
print("\n📊 データベース内の既存クエリを確認中...")
existing_queries = set()
recent_queries = manager.get_recent_queries(limit=2000)
for record in recent_queries:
    existing_queries.add(record['query'])

print(f"✅ データベース内の既存クエリ: {len(existing_queries)} 件")

# Filter new queries
new_queries = []
for item in random_queries:
    query = item['question']
    if query not in existing_queries:
        new_queries.append(item)

print(f"\n📋 総クエリ数: {len(random_queries)}")
print(f"🆕 処理対象の新規クエリ: {len(new_queries)}")
print(f"✅ すでにデータベースに存在: {len(random_queries) - len(new_queries)}")

if len(new_queries) == 0:
    print("\n✅ すべてのクエリがすでにデータベースに存在します。")
    sys.exit(0)

# Show warning
print("\n" + "=" * 70)
print("⚠️  警告: RAGクエリを実行します")
print("=" * 70)
print(f"クエリ数: {len(new_queries)}")
print(f"推定時間: 約{len(new_queries) * 25}秒 ({len(new_queries) * 25 / 60:.1f}分)")

print("\n実行するクエリ:")
for i, item in enumerate(new_queries[:5], 1):
    print(f"  {i}. [{item['title'][:15]}...] {item['question'][:45]}...")
if len(new_queries) > 5:
    print(f"  ... 他 {len(new_queries) - 5} 件")

print("\n⏳ 2秒後に開始...")
time.sleep(2)

# Proceed with batch queries
print("\n" + "=" * 70)
print("🚀 バッチクエリ開始...")
print("=" * 70)

success_count = 0
error_count = 0
no_info_count = 0
results = []

for i, item in enumerate(new_queries, 1):
    query = item['question']
    title = item['title']
    dataset_answer = item['answer']

    print(f"\n[{i}/{len(new_queries)}] 処理中: {query[:60]}...")
    print(f"トピック: {title}")

    try:
        # Execute query
        start_time = time.time()
        result, error = call_backend_query(query, system_mode="enhanced")
        elapsed = time.time() - start_time

        if error:
            raise Exception(error)

        if result and result.get('answer'):
            answer = result.get('answer', '')

            # Check if it's a "no info" response
            if "Sorry, no relevant information is currently available" in answer:
                no_info_count += 1
                print(f"⚠️ 関連情報なし (took {elapsed:.1f}s)")
                status = 'no_info'
            else:
                success_count += 1
                print(f"✅ 成功 (took {elapsed:.1f}s)")
                print(f"   回答プレビュー: {answer[:80]}...")
                status = 'success'

            # Show dataset answer
            if dataset_answer:
                print(f"   📋 データセット回答: {dataset_answer[:50]}...")

            # Check match metrics
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
                'title': title,
                'status': status,
                'processing_time': elapsed
            })
        else:
            error_count += 1
            print(f"⚠️ 回答なし")
            results.append({
                'query': query,
                'title': title,
                'status': 'no_answer',
                'processing_time': elapsed
            })

    except Exception as e:
        error_count += 1
        print(f"❌ エラー: {str(e)}")
        results.append({
            'query': query,
            'title': title,
            'status': 'error',
            'error': str(e)
        })

    # Small delay between queries
    if i < len(new_queries):
        time.sleep(1)

# Print summary
print("\n" + "=" * 70)
print("📊 バッチクエリ完了 - サマリー")
print("=" * 70)
print(f"✅ 成功: {success_count}/{len(new_queries)} ({success_count/len(new_queries)*100:.1f}%)")
print(f"⚠️ 関連情報なし: {no_info_count}/{len(new_queries)} ({no_info_count/len(new_queries)*100:.1f}%)")
print(f"❌ エラー: {error_count}/{len(new_queries)} ({error_count/len(new_queries)*100:.1f}%)")

# Group by topic
print(f"\n📂 トピック別サマリー:")
from collections import defaultdict
by_topic = defaultdict(lambda: {'success': 0, 'no_info': 0, 'error': 0, 'total': 0})

for result in results:
    topic = result['title']
    status = result['status']
    by_topic[topic]['total'] += 1
    if status == 'success':
        by_topic[topic]['success'] += 1
    elif status == 'no_info':
        by_topic[topic]['no_info'] += 1
    else:
        by_topic[topic]['error'] += 1

for topic, stats in sorted(by_topic.items(), key=lambda x: x[1]['total'], reverse=True)[:10]:
    print(f"  {topic[:30]:30s}: {stats['success']}/{stats['total']} 成功, {stats['no_info']} 情報なし")

print("\n✅ すべてのクエリがデータベースに保存されました")
print("=" * 70)
