#!/usr/bin/env python3
"""
Batch Add Answer Judgments to Existing Query History
Adds Gemini-based yes/no judgment for all existing queries in database
"""

import sys
import os
from dotenv import load_dotenv

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

load_dotenv()

from query_history_manager import QueryHistoryManager
from calculate_match_metrics import judge_answer_relevance
import sqlite3
import time

def batch_add_judgments(db_path: str = "../query_history.db"):
    """Add answer judgments to all existing queries"""

    print("=" * 80)
    print("📊 Batch Adding Answer Judgments to Query History")
    print("=" * 80)

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all queries without answer_judgment
    cursor.execute("""
        SELECT id, query, generated_answer, answer_judgment, dataset_answer
        FROM query_history
        ORDER BY created_at DESC
    """)

    all_queries = cursor.fetchall()
    total = len(all_queries)

    print(f"\n📋 Found {total} queries in database")

    # Count queries that need judgment
    queries_needing_judgment = [q for q in all_queries if not q['answer_judgment']]
    print(f"📝 Queries needing judgment: {len(queries_needing_judgment)}")
    print(f"✅ Queries already have judgment: {total - len(queries_needing_judgment)}")

    if not queries_needing_judgment:
        print("\n✅ All queries already have answer judgments!")
        conn.close()
        return

    # Ask for confirmation
    print(f"\n⚠️  This will call OpenAI API {len(queries_needing_judgment)} times")
    print(f"⏱️  Estimated time: ~{len(queries_needing_judgment) * 2} seconds")
    response = input("\n🤔 Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled by user")
        conn.close()
        return

    # Process each query
    print("\n" + "=" * 80)
    print("🚀 Starting batch processing...")
    print("=" * 80 + "\n")

    success_count = 0
    error_count = 0

    for i, query_record in enumerate(queries_needing_judgment, 1):
        query_id = query_record['id']
        query = query_record['query']
        generated_answer = query_record['generated_answer']
        dataset_answer = query_record['dataset_answer'] or ""

        print(f"[{i}/{len(queries_needing_judgment)}] Processing query ID: {query_id}")
        print(f"   Query: {query[:60]}...")

        try:
            # Get judgment from OpenAI with dataset answer
            judgment = judge_answer_relevance(query, generated_answer, dataset_answer)

            # Update database
            cursor.execute("""
                UPDATE query_history
                SET answer_judgment = ?
                WHERE id = ?
            """, (judgment, query_id))

            conn.commit()

            success_count += 1
            print(f"   ✅ Judgment: {judgment}")

            # Small delay to avoid rate limiting
            if i < len(queries_needing_judgment):
                time.sleep(0.5)

        except Exception as e:
            error_count += 1
            print(f"   ❌ Error: {e}")
            continue

    conn.close()

    # Print summary
    print("\n" + "=" * 80)
    print("📊 Batch Processing Complete!")
    print("=" * 80)
    print(f"✅ Successfully processed: {success_count}/{len(queries_needing_judgment)}")
    print(f"❌ Errors: {error_count}")
    print(f"📝 Total queries in database: {total}")
    print("=" * 80 + "\n")

    # Show statistics
    cursor = sqlite3.connect(db_path).cursor()
    cursor.execute("""
        SELECT answer_judgment, COUNT(*) as count
        FROM query_history
        WHERE answer_judgment IS NOT NULL AND answer_judgment != ''
        GROUP BY answer_judgment
    """)

    judgment_stats = cursor.fetchall()
    if judgment_stats:
        print("📈 Answer Judgment Statistics:")
        for judgment, count in judgment_stats:
            percentage = (count / total) * 100
            print(f"   {judgment}: {count} ({percentage:.1f}%)")

    print()

if __name__ == "__main__":
    batch_add_judgments()
