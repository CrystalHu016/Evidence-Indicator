#!/usr/bin/env python3
"""
Re-evaluate 100 samples with Enhanced BM25
使用增强的BM25重新评测100条样本
"""

import sys
import os
import sqlite3
import json
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Add script directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "script")
rag_frontend_dir = os.path.join(current_dir, "rag-streamlit-frontend")
sys.path.insert(0, script_dir)
sys.path.insert(0, rag_frontend_dir)

load_dotenv()

from ultra_fast_rag_semantic import PureSemanticRAG
from calculate_match_metrics import (
    calculate_char_match_rate,
    judge_answer_relevance
)


def load_query_history(db_path: str, limit: int = 100) -> List[Dict]:
    """Load query history from database"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, query, generated_answer, dataset_answer, evidences
        FROM query_history
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    queries = []
    for row in rows:
        query_id, query, generated_answer, dataset_answer, evidences_json = row

        # Parse evidences JSON
        evidences = []
        if evidences_json:
            try:
                evidences = json.loads(evidences_json)
            except:
                evidences = []

        queries.append({
            'id': query_id,
            'query': query,
            'generated_answer': generated_answer,
            'dataset_answer': dataset_answer,
            'evidences': evidences
        })

    return queries


def calculate_best_recall(evidences: List[Dict], dataset_answer: str) -> Tuple[float, Dict]:
    """
    Calculate the best recall from all evidences
    返回再现率最高的evidence
    """
    if not evidences or not dataset_answer:
        return 0.0, {}

    best_recall = 0.0
    best_evidence = {}

    for evidence in evidences:
        # Try both 'text' and 'extracted_evidence' fields
        evidence_text = evidence.get('text', '') or evidence.get('extracted_evidence', '')
        if not evidence_text:
            continue

        # Calculate character-level metrics
        metrics = calculate_char_match_rate(evidence_text, dataset_answer)
        recall = metrics.get('recall', 0.0)

        if recall > best_recall:
            best_recall = recall
            best_evidence = {
                'text': evidence_text,
                'recall': recall,
                'precision': metrics.get('precision', 0.0),
                'f1_score': metrics.get('f1_score', 0.0)
            }

    return best_recall, best_evidence


def evaluate_rag_system(rag: PureSemanticRAG, queries: List[Dict]) -> Dict:
    """
    Evaluate RAG system on query samples with Enhanced BM25
    使用Enhanced BM25评测RAG系统
    """
    print("=" * 80)
    print("🧪 Re-Evaluating RAG System on 100 Samples with Enhanced BM25")
    print("=" * 80)
    print("\n📊 Using Enhanced BM25 with ordinal number boosting\n")
    print("=" * 80 + "\n")

    results = {
        'total': len(queries),
        'correct_answers': 0,
        'incorrect_answers': 0,
        'system_errors': 0,
        'total_recall': 0.0,
        'details': []
    }

    for i, query_data in enumerate(queries, 1):
        query = query_data['query']
        dataset_answer = query_data['dataset_answer']
        old_generated_answer = query_data['generated_answer']

        print(f"\n{'=' * 80}")
        print(f"Query {i}/{len(queries)}")
        print(f"{'=' * 80}")
        print(f"📝 Query: {query}")
        print(f"✅ Dataset Answer: {dataset_answer}")
        print(f"🔄 Querying RAG system with Enhanced BM25...")

        try:
            # Query RAG system with Enhanced BM25
            result = rag.query_with_answer(query, k=10)

            new_generated_answer = result.get('answer', '')
            new_evidences = result.get('evidences', [])

            print(f"🤖 RAG Answer: {new_generated_answer}")

            # Judge answer correctness using LLM
            judgment = judge_answer_relevance(query, new_generated_answer, dataset_answer)

            # Calculate best recall from evidences
            best_recall, best_evidence = calculate_best_recall(new_evidences, dataset_answer)

            is_correct = judgment.lower() == 'yes'
            is_error = 'sorry' in new_generated_answer.lower() or 'no relevant' in new_generated_answer.lower()

            if is_error:
                results['system_errors'] += 1
                status = "⚠️ SYSTEM ERROR"
            elif is_correct:
                results['correct_answers'] += 1
                status = "✅ CORRECT"
            else:
                results['incorrect_answers'] += 1
                status = "❌ INCORRECT"

            results['total_recall'] += best_recall

            print(f"📊 Status: {status}")
            print(f"🎯 LLM Judgment: {judgment}")
            print(f"📈 Best Evidence Recall: {best_recall:.2%}")

            results['details'].append({
                'query': query,
                'dataset_answer': dataset_answer,
                'old_answer': old_generated_answer,
                'new_answer': new_generated_answer,
                'judgment': judgment,
                'is_correct': is_correct,
                'is_error': is_error,
                'best_recall': best_recall,
                'best_evidence': best_evidence,
                'num_evidences': len(new_evidences)
            })

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            results['system_errors'] += 1
            results['details'].append({
                'query': query,
                'dataset_answer': dataset_answer,
                'old_answer': old_generated_answer,
                'new_answer': 'ERROR',
                'judgment': 'no',
                'is_correct': False,
                'is_error': True,
                'best_recall': 0.0,
                'best_evidence': {},
                'num_evidences': 0,
                'error': str(e)
            })

    return results


def print_summary(results: Dict, old_results: Dict = None):
    """Print evaluation summary with comparison to old results"""
    total = results['total']
    correct = results['correct_answers']
    incorrect = results['incorrect_answers']
    errors = results['system_errors']
    avg_recall = results['total_recall'] / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("📊 EVALUATION SUMMARY - Enhanced BM25")
    print("=" * 80)
    print(f"\n✅ Total Queries: {total}")
    print(f"✅ Correct Answers: {correct} ({correct/total*100:.1f}%)")
    print(f"❌ Incorrect Answers: {incorrect} ({incorrect/total*100:.1f}%)")
    print(f"⚠️  System Errors: {errors} ({errors/total*100:.1f}%)")
    print(f"\n📈 Average Best Evidence Recall: {avg_recall:.2%}")
    print(f"📈 Accuracy (Correct / Total): {correct/total*100:.1f}%")
    print(f"📈 Success Rate (Correct / Non-Error): {correct/(total-errors)*100:.1f}%\" if total > errors else \"N/A\"")

    # Comparison with old results
    if old_results:
        print("\n" + "=" * 80)
        print("📊 COMPARISON: Enhanced BM25 vs Original")
        print("=" * 80)

        old_correct = old_results['correct_answers']
        old_incorrect = old_results['incorrect_answers']
        old_errors = old_results['system_errors']

        improvement = correct - old_correct
        error_change = errors - old_errors

        print(f"\n✅ Correct Answers:")
        print(f"   Original: {old_correct} ({old_correct/total*100:.1f}%)")
        print(f"   Enhanced: {correct} ({correct/total*100:.1f}%)")
        print(f"   Change: {'+' if improvement > 0 else ''}{improvement} ({improvement/total*100:+.1f}%)")

        print(f"\n❌ Incorrect Answers:")
        print(f"   Original: {old_incorrect} ({old_incorrect/total*100:.1f}%)")
        print(f"   Enhanced: {incorrect} ({incorrect/total*100:.1f}%)")
        print(f"   Change: {'+' if (incorrect-old_incorrect) > 0 else ''}{incorrect-old_incorrect}")

        print(f"\n⚠️  System Errors:")
        print(f"   Original: {old_errors} ({old_errors/total*100:.1f}%)")
        print(f"   Enhanced: {errors} ({errors/total*100:.1f}%)")
        print(f"   Change: {'+' if error_change > 0 else ''}{error_change}")

    print("\n" + "=" * 80)


def main():
    """Main evaluation function"""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment")
        return False

    # Initialize RAG system with Enhanced BM25
    chroma_path = os.path.join(current_dir, "chroma")
    print("🔄 Initializing RAG system with Enhanced BM25...")
    rag = PureSemanticRAG(openai_api_key=api_key, chroma_path=chroma_path)
    print("✅ RAG system initialized\n")

    # Load query history
    db_path = os.path.join(current_dir, "query_history.db")
    print(f"📁 Loading query history from: {db_path}")
    queries = load_query_history(db_path, limit=100)
    print(f"✅ Loaded {len(queries)} queries\n")

    # Load old results for comparison
    old_results_file = os.path.join(current_dir, "evaluation_results.json")
    old_results = None
    if os.path.exists(old_results_file):
        with open(old_results_file, 'r', encoding='utf-8') as f:
            old_results = json.load(f)
        print(f"📂 Loaded old results for comparison\n")

    # Evaluate with Enhanced BM25
    results = evaluate_rag_system(rag, queries)

    # Print summary with comparison
    print_summary(results, old_results)

    # Save detailed results to file
    output_file = os.path.join(current_dir, "evaluation_results_enhanced_bm25.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Detailed results saved to: {output_file}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
