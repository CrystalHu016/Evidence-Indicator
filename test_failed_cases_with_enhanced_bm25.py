#!/usr/bin/env python3
"""
Test all NO judgment queries with enhanced BM25
使用增强的BM25测试所有NO判断的查询
"""

import sys
import os
import sqlite3
from typing import List, Dict
from dotenv import load_dotenv

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "script")
rag_frontend_dir = os.path.join(current_dir, "rag-streamlit-frontend")
sys.path.insert(0, script_dir)
sys.path.insert(0, rag_frontend_dir)

load_dotenv()

from ultra_fast_rag_semantic import PureSemanticRAG
from calculate_match_metrics import judge_answer_relevance

def get_no_judgment_queries() -> List[Dict]:
    """Get all queries with NO judgment from database"""
    db_path = os.path.join(current_dir, "query_history.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT query, generated_answer, dataset_answer, answer_judgment
        FROM query_history
        WHERE answer_judgment = 'no'
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    queries = []
    for row in rows:
        query, old_generated_answer, dataset_answer, judgment = row

        # Skip system errors (no relevant information)
        if 'sorry' in old_generated_answer.lower() or 'no relevant' in old_generated_answer.lower():
            continue

        queries.append({
            'query': query,
            'old_answer': old_generated_answer,
            'expected_answer': dataset_answer,
            'old_judgment': judgment
        })

    return queries


def test_failed_cases():
    """Test all failed cases with enhanced BM25"""

    print("=" * 80)
    print("🧪 Testing Failed Cases with Enhanced BM25")
    print("=" * 80)

    # Get NO judgment queries (exclude system errors)
    failed_queries = get_no_judgment_queries()

    print(f"\n📊 Total failed cases to test: {len(failed_queries)}")
    print("=" * 80)

    # Initialize RAG system with enhanced BM25
    api_key = os.getenv("OPENAI_API_KEY")
    chroma_path = os.path.join(current_dir, "chroma")

    print(f"\n🔄 Initializing RAG with Enhanced BM25...")
    rag = PureSemanticRAG(openai_api_key=api_key, chroma_path=chroma_path)
    print("✅ RAG system initialized\n")

    # Results tracking
    results = {
        'total': len(failed_queries),
        'fixed': 0,
        'still_wrong': 0,
        'system_error': 0,
        'details': []
    }

    # Test each failed case
    for i, query_data in enumerate(failed_queries, 1):
        query = query_data['query']
        old_answer = query_data['old_answer']
        expected_answer = query_data['expected_answer']

        print(f"\n{'=' * 80}")
        print(f"Test Case {i}/{len(failed_queries)}")
        print(f"{'=' * 80}")
        print(f"📝 Query: {query}")
        print(f"✅ Expected: {expected_answer}")
        print(f"❌ Old Answer: {old_answer}")

        try:
            # Query with new enhanced BM25
            result = rag.query_with_answer(query, k=10)
            new_answer = result.get('answer', '')
            evidences = result.get('evidences', [])

            print(f"🆕 New Answer: {new_answer}")

            # Check if it's a system error
            is_error = 'sorry' in new_answer.lower() or 'no relevant' in new_answer.lower()

            if is_error:
                results['system_error'] += 1
                status = "⚠️ SYSTEM ERROR"
                new_judgment = "error"
            else:
                # Judge with LLM
                new_judgment = judge_answer_relevance(query, new_answer, expected_answer)

                if new_judgment.lower() == 'yes':
                    results['fixed'] += 1
                    status = "✅ FIXED!"
                else:
                    results['still_wrong'] += 1
                    status = "❌ STILL WRONG"

            print(f"📊 Status: {status}")
            print(f"🎯 New Judgment: {new_judgment}")
            print(f"📈 Evidences: {len(evidences)}")

            # Store result
            results['details'].append({
                'query': query,
                'expected_answer': expected_answer,
                'old_answer': old_answer,
                'new_answer': new_answer,
                'old_judgment': 'no',
                'new_judgment': new_judgment,
                'status': status,
                'num_evidences': len(evidences)
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            results['system_error'] += 1
            results['details'].append({
                'query': query,
                'expected_answer': expected_answer,
                'old_answer': old_answer,
                'new_answer': f'ERROR: {e}',
                'old_judgment': 'no',
                'new_judgment': 'error',
                'status': '⚠️ ERROR',
                'num_evidences': 0
            })

    # Print summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY - Enhanced BM25 Performance")
    print("=" * 80)

    total = results['total']
    fixed = results['fixed']
    still_wrong = results['still_wrong']
    errors = results['system_error']

    print(f"\n✅ Total Failed Cases Tested: {total}")
    print(f"✅ FIXED (No → Yes): {fixed} ({fixed/total*100:.1f}%)")
    print(f"❌ Still Wrong (No → No): {still_wrong} ({still_wrong/total*100:.1f}%)")
    print(f"⚠️  System Errors: {errors} ({errors/total*100:.1f}%)")

    print("\n" + "=" * 80)
    print("📝 Detailed Results")
    print("=" * 80)

    # Show fixed cases
    print(f"\n✅ FIXED Cases ({fixed} total):")
    fixed_cases = [d for d in results['details'] if d['new_judgment'].lower() == 'yes']
    for i, case in enumerate(fixed_cases, 1):
        print(f"\n{i}. {case['query']}")
        print(f"   Expected: {case['expected_answer']}")
        print(f"   Old: {case['old_answer']}")
        print(f"   New: {case['new_answer']}")

    # Show still wrong cases
    print(f"\n\n❌ Still Wrong Cases ({still_wrong} total):")
    wrong_cases = [d for d in results['details'] if d['new_judgment'].lower() == 'no']
    for i, case in enumerate(wrong_cases, 1):
        print(f"\n{i}. {case['query']}")
        print(f"   Expected: {case['expected_answer']}")
        print(f"   Old: {case['old_answer']}")
        print(f"   New: {case['new_answer']}")

    print("\n" + "=" * 80)
    print("📊 Impact Assessment")
    print("=" * 80)
    print(f"\n🎯 Fix Rate: {fixed/total*100:.1f}%")
    print(f"📈 Error Rate: {errors/total*100:.1f}%")

    if fixed > 0:
        print(f"\n✅ Enhanced BM25 successfully fixed {fixed} out of {total} failed cases!")
        print(f"   This is a {fixed/total*100:.1f}% improvement rate.")
    else:
        print(f"\n⚠️  Enhanced BM25 did not fix any cases.")
        print(f"   Further improvements may be needed.")

    print("=" * 80)

    return results


if __name__ == "__main__":
    results = test_failed_cases()
    sys.exit(0)
