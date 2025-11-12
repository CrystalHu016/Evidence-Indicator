#!/usr/bin/env python3
"""
Batch Query Sample Questions
Query all sample questions that are not in history yet and save to database
"""

import os
import sys
import time
from datetime import datetime

# Add paths
sys.path.insert(0, 'rag-streamlit-frontend')
sys.path.insert(0, 'script')

# Import necessary modules
sys.path.insert(0, os.path.join(os.getcwd(), 'rag-streamlit-frontend'))
from backend_integration import call_backend_query
from query_history_manager import QueryHistoryManager
from config import SAMPLE_QUERIES_BY_CATEGORY

print("=" * 70)
print("Batch Query Sample Questions")
print("=" * 70)

# Initialize database
DB_PATH = "query_history.db"
manager = QueryHistoryManager(DB_PATH)

# Get all existing queries from database
print("\n📊 Checking existing queries in database...")
existing_queries = set()
try:
    recent_queries = manager.get_recent_queries(limit=1000)
    for record in recent_queries:
        existing_queries.add(record['query'])
    print(f"✅ Found {len(existing_queries)} existing queries in database")
except Exception as e:
    print(f"⚠️ Could not load existing queries: {e}")

# Collect all sample queries
all_sample_queries = []
for category, queries in SAMPLE_QUERIES_BY_CATEGORY.items():
    for query in queries:
        all_sample_queries.append({
            'category': category,
            'query': query
        })

print(f"\n📋 Total sample queries: {len(all_sample_queries)}")

# Filter queries that are not in database
new_queries = []
for item in all_sample_queries:
    if item['query'] not in existing_queries:
        new_queries.append(item)

print(f"🆕 New queries to process: {len(new_queries)}")
print(f"✅ Already in database: {len(all_sample_queries) - len(new_queries)}")

if not new_queries:
    print("\n✨ All sample queries are already in the database!")
    print("No need to run batch queries.")
    sys.exit(0)

# Ask for confirmation
print("\n" + "=" * 70)
print("⚠️  WARNING: This will execute RAG queries")
print("=" * 70)
print(f"Number of queries: {len(new_queries)}")
print(f"Estimated time: ~{len(new_queries) * 25} seconds ({len(new_queries) * 25 / 60:.1f} minutes)")
print("\nQueries to be executed:")
for i, item in enumerate(new_queries[:5], 1):
    print(f"  {i}. [{item['category']}] {item['query'][:50]}...")
if len(new_queries) > 5:
    print(f"  ... and {len(new_queries) - 5} more")

# Proceed with batch queries
print("\n" + "=" * 70)
print("🚀 Starting batch queries...")
print("=" * 70)

success_count = 0
error_count = 0
results = []

for i, item in enumerate(new_queries, 1):
    query = item['query']
    category = item['category']
    
    print(f"\n[{i}/{len(new_queries)}] Processing: {query[:60]}...")
    print(f"Category: {category}")
    
    try:
        # Execute query
        start_time = time.time()
        result, error = call_backend_query(query, system_mode="enhanced")
        elapsed = time.time() - start_time

        if error:
            raise Exception(error)
        
        if result and result.get('answer'):
            success_count += 1
            print(f"✅ Success (took {elapsed:.1f}s)")
            print(f"   Answer preview: {result['answer'][:80]}...")
            
            # Check if dataset answer was found
            if result.get('dataset_answer'):
                print(f"   📋 Dataset answer: {result['dataset_answer'][:50]}...")
            
            # Check match metrics
            match_metrics = result.get('match_metrics', {})
            if match_metrics:
                print(f"   📊 Match rate: {match_metrics.get('match_rate', 0):.1%}")
            
            results.append({
                'query': query,
                'category': category,
                'status': 'success',
                'processing_time': elapsed
            })
        else:
            error_count += 1
            print(f"⚠️ No answer returned")
            results.append({
                'query': query,
                'category': category,
                'status': 'no_answer',
                'processing_time': elapsed
            })
            
    except Exception as e:
        error_count += 1
        print(f"❌ Error: {e}")
        results.append({
            'query': query,
            'category': category,
            'status': 'error',
            'error': str(e)
        })
    
    # Brief pause between queries to avoid overwhelming the system
    if i < len(new_queries):
        time.sleep(1)

# Summary
print("\n" + "=" * 70)
print("📊 Batch Query Summary")
print("=" * 70)
print(f"Total queries: {len(new_queries)}")
print(f"✅ Successful: {success_count}")
print(f"❌ Errors: {error_count}")
print(f"Success rate: {success_count/len(new_queries)*100:.1f}%")

# Category breakdown
print("\n📁 Results by category:")
from collections import defaultdict
category_stats = defaultdict(lambda: {'success': 0, 'error': 0})
for r in results:
    if r['status'] == 'success':
        category_stats[r['category']]['success'] += 1
    else:
        category_stats[r['category']]['error'] += 1

for category in sorted(category_stats.keys()):
    stats = category_stats[category]
    total = stats['success'] + stats['error']
    print(f"  {category}: {stats['success']}/{total} successful")

print("\n✨ Batch query completed!")
print("💡 Open the Streamlit app to see all queries in history")
print(f"🌐 URL: http://localhost:8501")

