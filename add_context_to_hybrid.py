#!/usr/bin/env python3
"""
Add context field to v1_v2_hybrid records by matching queries with dataset
"""

import sqlite3
import json

def add_context_to_hybrid():
    """Load dataset contexts and add to hybrid version records"""

    # Load dataset
    print("📂 Loading dataset...")
    with open('data/merged_qa_dataset.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Create mapping: question -> context
    question_to_context = {}
    for item in dataset:
        question = item['question']
        context = item['context']
        question_to_context[question] = context

    print(f"✅ Loaded {len(question_to_context)} questions from dataset")

    # Connect to database
    conn = sqlite3.connect('query_history.db')
    cursor = conn.cursor()

    # Check if context column exists
    cursor.execute("PRAGMA table_info(query_history)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'context' not in columns:
        print("➕ Adding 'context' column to query_history table...")
        cursor.execute("ALTER TABLE query_history ADD COLUMN context TEXT")
        conn.commit()

    # Get all v1_v2_hybrid records
    print("\n🔄 Updating v1_v2_hybrid records with context...")
    cursor.execute("""
        SELECT id, query FROM query_history
        WHERE version = 'v1_v2_hybrid'
    """)

    records = cursor.fetchall()
    print(f"✅ Found {len(records)} v1_v2_hybrid records")

    updated = 0
    not_found = 0

    for record_id, query in records:
        if query in question_to_context:
            context = question_to_context[query]
            cursor.execute("""
                UPDATE query_history
                SET context = ?
                WHERE id = ?
            """, (context, record_id))
            updated += 1
        else:
            print(f"⚠️  Query not found in dataset: {query[:50]}...")
            not_found += 1

    conn.commit()
    conn.close()

    # Summary
    print("\n" + "="*60)
    print("✅ CONTEXT DATA ADDED!")
    print("="*60)
    print(f"Total records processed:  {len(records)}")
    print(f"Successfully updated:     {updated}")
    print(f"Not found in dataset:     {not_found}")
    print("="*60)

    return updated

if __name__ == '__main__':
    try:
        count = add_context_to_hybrid()
        print(f"\n✅ Successfully added context to {count} records")
        print(f"\n💡 Frontend can now display context with highlighted evidence!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
