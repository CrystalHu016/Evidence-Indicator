#!/usr/bin/env python3
"""
Batch Query Additional 50 Questions from Merged Dataset
Processes 50 randomly selected questions from merged_qa_dataset.json (excluding existing sample queries)
"""

import sys
import os
import json
import time

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from database.query_history_manager import QueryHistoryManager
from backend_integration import call_backend_query

# Load additional random queries
additional_queries_path = os.path.join(parent_dir, 'data', 'random_50_additional_queries.json')
with open(additional_queries_path, 'r', encoding='utf-8') as f:
    additional_queries = json.load(f)

print("=" * 70)
print("Batch Query Additional 50 Questions from Merged Dataset")
print("=" * 70)

# Initialize database
DB_PATH = os.path.join(parent_dir, 'query_history.db')
manager = QueryHistoryManager(DB_PATH)
print(f"✅ Query history database initialized: {DB_PATH}")
print(f"✅ 追加質問ロード完了: {len(additional_queries)} 件")

# Check existing queries in database
print("\n📊 データベース内の既存クエリを確認中...")
existing_queries = set()
recent_queries = manager.get_recent_queries(limit=2000)
for record in recent_queries:
    existing_queries.add(record['query'])

print(f"✅ データベース内の既存クエリ: {len(existing_queries)} 件")

# Filter new queries
new_queries = []
for item in additional_queries:
    query = item['question']
    if query not in existing_queries:
        new_queries.append(item)

print(f"\n📋 総クエリ数: {len(additional_queries)}")
print(f"🆕 処理対象の新規クエリ: {len(new_queries)}")
print(f"✅ すでにデータベースに存在: {len(additional_queries) - len(new_queries)}")

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
    title = item.get('title', 'Unknown')[:15]
    question = item['question'][:45]
    print(f"  {i}. [{title}...] {question}...")
if len(new_queries) > 5:
    print(f"  ... 他 {len(new_queries) - 5} 件")

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
    title = item.get('title', 'Unknown')

    # Get dataset answer
    answers = item.get('answers', {})
    if isinstance(answers, dict) and 'text' in answers:
        dataset_answer = answers['text'][0] if answers['text'] else ''
    else:
        dataset_answer = ''

    print(f"\n[{i}/{len(new_queries)}] 処理中: {query[:60]}...")
    print(f"トピック: {title}")

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

        # Add dataset_answer to result
        result['dataset_answer'] = dataset_answer

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
            'title': title,
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
            'title': title,
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

# Topic summary
print(f"\n📂 トピック別サマリー:")
topic_stats = {}
for r in results:
    topic = r['title'][:30]
    if topic not in topic_stats:
        topic_stats[topic] = {'success': 0, 'no_info': 0, 'error': 0, 'total': 0}
    topic_stats[topic]['total'] += 1
    if '✅' in r['status']:
        topic_stats[topic]['success'] += 1
    elif '⚠️' in r['status']:
        topic_stats[topic]['no_info'] += 1
    else:
        topic_stats[topic]['error'] += 1

for topic, stats in sorted(topic_stats.items()):
    print(f"  {topic:30s}: {stats['success']}/{stats['total']} 成功, {stats['no_info']} 情報なし")

print(f"\n✅ すべてのクエリがデータベースに保存されました")
print("=" * 70)
