#!/usr/bin/env python3
"""
Add LLM answerability scores to existing v2_full_evidence records
Evaluates if the extracted evidence can answer the original question
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.query_history_manager import QueryHistoryManager
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'rag-streamlit-frontend'))
from calculate_match_metrics import evaluate_evidence_answerability
import json

def add_answerability_scores():
    """Add answerability scores to all v2_full_evidence records"""

    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'query_history.db')
    manager = QueryHistoryManager(db_path)

    # Get all v2_full_evidence records
    conn = manager.conn
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, query, evidences
        FROM query_history
        WHERE version = 'v2_full_evidence'
    """)

    records = cursor.fetchall()
    print(f"📊 Found {len(records)} v2_full_evidence records")
    print()

    updated_count = 0

    for record_id, query, evidences_json in records:
        evidences = json.loads(evidences_json) if evidences_json else []

        if not evidences:
            continue

        # Update each evidence with answerability score
        updated_evidences = []
        for ev in evidences:
            extracted = ev.get('extracted_evidence', '')

            if extracted:
                # Calculate answerability score
                answerability_score = evaluate_evidence_answerability(query, extracted)
                ev['answerability_score'] = answerability_score
                print(f"✅ Query: {query[:50]}...")
                print(f"   Evidence: {extracted[:50]}...")
                print(f"   Answerability Score: {answerability_score*100:.1f}%")
                print()
            else:
                ev['answerability_score'] = 0.0

            updated_evidences.append(ev)

        # Update record in database
        cursor.execute("""
            UPDATE query_history
            SET evidences = ?
            WHERE id = ?
        """, (json.dumps(updated_evidences, ensure_ascii=False), record_id))

        updated_count += 1

        if updated_count % 10 == 0:
            conn.commit()
            print(f"💾 Committed {updated_count} records...")
            print()

    conn.commit()
    print(f"✅ Successfully updated {updated_count} records with answerability scores!")

if __name__ == "__main__":
    add_answerability_scores()
