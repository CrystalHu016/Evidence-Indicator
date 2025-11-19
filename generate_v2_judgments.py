#!/usr/bin/env python3
"""
Generate Version 2 judgments by re-evaluating v1 answers with improved prompt
使用改进的prompt重新评判v1的答案，生成v2判断结果
"""

import sys
import os
import sqlite3
import json
from dotenv import load_dotenv

# Add path
current_dir = os.path.dirname(os.path.abspath(__file__))
rag_frontend_dir = os.path.join(current_dir, "rag-streamlit-frontend")
sys.path.insert(0, rag_frontend_dir)

load_dotenv()

from calculate_match_metrics import judge_answer_relevance

def generate_v2_judgments(db_path: str):
    """
    Re-evaluate v1 answers with improved LLM judgment prompt
    This represents Version 2: same RAG answers, but better judgment
    """

    print("=" * 80)
    print("🔄 Generating Version 2 Judgments")
    print("=" * 80)
    print("\nVersion 2 = Version 1 RAG answers + Improved LLM judgment prompt")
    print("(with temporal matching logic)")
    print("=" * 80)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all v1 records
    cursor.execute("""
        SELECT id, query, generated_answer, dataset_answer
        FROM query_history
        WHERE version = 'v1'
    """)

    v1_records = cursor.fetchall()
    print(f"\n📊 Found {len(v1_records)} Version 1 records")

    # Re-judge each one with improved prompt
    print("\n🔄 Re-judging with improved prompt...")

    yes_count = 0
    no_count = 0

    for v1_id, query, generated_answer, dataset_answer in v1_records:
        # Use current improved judge_answer_relevance
        # (which includes temporal matching logic)
        new_judgment = judge_answer_relevance(query, generated_answer, dataset_answer)

        if new_judgment == 'yes':
            yes_count += 1
        else:
            no_count += 1

        # Update v2 record with new judgment
        cursor.execute("""
            UPDATE query_history
            SET answer_judgment = ?
            WHERE version = 'v2' AND query = ?
        """, (new_judgment, query))

    conn.commit()

    print(f"\n✅ Re-judged all {len(v1_records)} records")
    print(f"   YES: {yes_count} ({yes_count/len(v1_records)*100:.1f}%)")
    print(f"   NO: {no_count} ({no_count/len(v1_records)*100:.1f}%)")

    # Calculate system errors
    cursor.execute("""
        SELECT COUNT(*)
        FROM query_history
        WHERE version = 'v2'
        AND (generated_answer LIKE '%sorry%' OR generated_answer LIKE '%no relevant%')
    """)
    error_count = cursor.fetchone()[0]

    accuracy = yes_count / (len(v1_records) - error_count) * 100 if len(v1_records) > error_count else 0

    print(f"   System Errors: {error_count} ({error_count/len(v1_records)*100:.1f}%)")
    print(f"   Accuracy (excluding errors): {accuracy:.1f}%")

    conn.close()

    print("\n" + "=" * 80)
    print("✅ Version 2 generation complete!")
    print("=" * 80)


def main():
    """Main function"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "query_history.db")

    if not os.path.exists(db_path):
        print(f"❌ Error: {db_path} not found")
        return False

    generate_v2_judgments(db_path)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
