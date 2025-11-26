#!/usr/bin/env python3
"""
Create Hybrid Version: V1 Retrieval + V2 Judgment
This combines V1's query results with V2's improved judgment logic
"""

import json
import sqlite3
import sys
from pathlib import Path

def create_hybrid_version():
    """Merge V1 and V2 data to create hybrid records"""

    # Load V1 and V2 JSON results
    print("📂 Loading evaluation results...")
    with open('evaluation_results.json', 'r', encoding='utf-8') as f:
        v1_data = json.load(f)

    with open('evaluation_results_v2_original_bm25.json', 'r', encoding='utf-8') as f:
        v2_data = json.load(f)

    print(f"✅ V1: {len(v1_data['details'])} records")
    print(f"✅ V2: {len(v2_data['details'])} records")

    # Create mapping: query -> judgment for V2
    v2_judgments = {}
    for record in v2_data['details']:
        query = record['query']
        v2_judgments[query] = {
            'judgment': record['judgment'],
            'is_correct': record['is_correct']
        }

    print(f"\n🔗 Created V2 judgment mapping for {len(v2_judgments)} queries")

    # Connect to database
    db_path = 'query_history.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get V1 records from database
    print(f"\n🗄️ Fetching V1 records from database...")
    cursor.execute("""
        SELECT id, query, generated_answer, dataset_answer,
               answer_judgment, evidences, processing_time,
               confidence, num_chunks
        FROM query_history
        WHERE version = 'v1'
        ORDER BY id
    """)

    v1_records = cursor.fetchall()
    print(f"✅ Found {len(v1_records)} V1 records in database")

    # Create hybrid records
    print(f"\n🔀 Creating hybrid version (V1 data + V2 judgment)...")

    hybrid_count = 0
    updated_count = 0

    for record in v1_records:
        (record_id, query, generated_answer, dataset_answer,
         v1_judgment, evidences, processing_time, confidence, num_chunks) = record

        # Get V2 judgment for this query
        if query in v2_judgments:
            v2_judgment_data = v2_judgments[query]
            v2_judgment = v2_judgment_data['judgment']

            # Check if judgment changed
            judgment_changed = (v1_judgment != v2_judgment)

            if judgment_changed:
                print(f"\n  Query: {query[:50]}...")
                print(f"    V1 judgment: {v1_judgment}")
                print(f"    V2 judgment: {v2_judgment}")
                updated_count += 1

            # Insert hybrid record with V2 judgment
            cursor.execute("""
                INSERT INTO query_history
                (timestamp, query, generated_answer, processing_time,
                 model, confidence, num_chunks, dataset_answer, evidences,
                 answer_judgment, version, created_at)
                VALUES (datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, 'v1_v2_hybrid', datetime('now'))
            """, (
                query,
                generated_answer,
                processing_time,
                'gpt-4o-mini',
                confidence,
                num_chunks,
                dataset_answer,
                evidences,  # Will be empty, but structure preserved
                v2_judgment  # Use V2's judgment!
            ))

            hybrid_count += 1
        else:
            print(f"⚠️  Warning: Query not found in V2: {query[:50]}...")

    # Commit changes
    conn.commit()

    # Verify results
    cursor.execute("SELECT COUNT(*) FROM query_history WHERE version = 'v1_v2_hybrid'")
    final_count = cursor.fetchone()[0]

    cursor.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN answer_judgment = 'yes' THEN 1 ELSE 0 END) as correct
        FROM query_history
        WHERE version = 'v1_v2_hybrid'
    """)
    total, correct = cursor.fetchone()

    conn.close()

    # Summary
    print("\n" + "="*60)
    print("✅ HYBRID VERSION CREATED!")
    print("="*60)
    print(f"Total hybrid records:     {final_count}")
    print(f"Judgments updated:        {updated_count}")
    print(f"Correct answers:          {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"\nVersion: v1_v2_hybrid")
    print(f"Description: V1 retrieval + V2 judgment logic")
    print("="*60)

    return hybrid_count

if __name__ == '__main__':
    try:
        count = create_hybrid_version()
        print(f"\n✅ Successfully created {count} hybrid records")
        print(f"\n💡 You can now select 'v1_v2_hybrid' in the UI to see:")
        print(f"   - V1's retrieval results and answers")
        print(f"   - V2's improved judgment (temporal logic)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
