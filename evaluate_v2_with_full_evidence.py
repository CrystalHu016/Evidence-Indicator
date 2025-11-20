#!/usr/bin/env python3
"""
V2 with Full Evidence: Complete re-evaluation with ALL details saved
V2完整版：重新评估并保存所有Evidence和Chunk细节
"""

import sys
import os
import sqlite3
import json
from typing import Dict, List
from dotenv import load_dotenv

# Add script directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "script")
rag_frontend_dir = os.path.join(current_dir, "rag-streamlit-frontend")
sys.path.insert(0, script_dir)
sys.path.insert(0, rag_frontend_dir)

load_dotenv()

# Import RAG system
from ultra_fast_rag_semantic import PureSemanticRAG
from calculate_match_metrics import judge_answer_relevance, calculate_char_match_rate


def load_dataset(limit=100) -> List[Dict]:
    """Load dataset matching v1_v2_hybrid queries"""
    dataset_path = os.path.join(current_dir, "data", "merged_qa_dataset.json")
    queries_path = os.path.join(current_dir, "v1_v2_hybrid_queries.json")

    # Load full dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)

    # Load v1_v2_hybrid queries
    with open(queries_path, 'r', encoding='utf-8') as f:
        target_queries = json.load(f)

    # Filter dataset to match v1_v2_hybrid queries
    dataset = []
    for item in full_dataset:
        if item['question'] in target_queries:
            dataset.append(item)
            if len(dataset) >= limit:
                break

    print(f"✅ Loaded {len(dataset)} samples matching v1_v2_hybrid queries")
    return dataset


def evaluate_with_full_evidence(rag: PureSemanticRAG, dataset: List[Dict]) -> Dict:
    """Evaluate and save COMPLETE evidence information"""

    print("=" * 80)
    print("🧪 V2 Evaluation with FULL Evidence & Context")
    print("=" * 80)
    print(f"\n📊 Total queries: {len(dataset)}")
    print("=" * 80 + "\n")

    results = {
        'total': len(dataset),
        'correct_answers': 0,
        'incorrect_answers': 0,
        'system_errors': 0,
        'details': []
    }

    for i, item in enumerate(dataset, 1):
        query = item['question']
        # Handle both formats: 'answer' (simple) or 'answers' (SQuAD format)
        if 'answer' in item:
            dataset_answer = item['answer']
        elif 'answers' in item and isinstance(item['answers'], dict):
            # SQuAD format: answers.text is a list
            dataset_answer = item['answers']['text'][0] if item['answers']['text'] else ''
        else:
            dataset_answer = ''
        context = item.get('context', '')

        print(f"\n{'=' * 80}")
        print(f"Query {i}/{len(dataset)}")
        print(f"{'=' * 80}")
        print(f"📝 Query: {query}")
        print(f"✅ Expected: {dataset_answer}")

        try:
            # Query RAG system - get FULL result with evidences
            result = rag.query_with_answer(query, k=10)

            generated_answer = result.get('answer', '')
            evidences = result.get('evidences', [])
            processing_time = result.get('processing_time', 0)
            model = result.get('model', 'gpt-4o-mini')
            confidence = result.get('confidence', 0.0)

            print(f"🤖 Generated: {generated_answer}")
            print(f"📦 Evidences: {len(evidences)} chunks")

            # Use improved prompt to judge
            judgment = judge_answer_relevance(query, generated_answer, dataset_answer)

            is_error = 'sorry' in generated_answer.lower() or 'no relevant' in generated_answer.lower()
            is_correct = judgment.lower() == 'yes'

            if is_error:
                results['system_errors'] += 1
                status = "⚠️ ERROR"
            elif is_correct:
                results['correct_answers'] += 1
                status = "✅ CORRECT"
            else:
                results['incorrect_answers'] += 1
                status = "❌ INCORRECT"

            print(f"📊 Status: {status}")
            print(f"🎯 Judgment: {judgment}")

            # Prepare evidence details for storage with calculated metrics
            evidence_list = []
            for idx, evidence in enumerate(evidences):
                extracted_evidence = evidence.get('extracted_evidence', '')

                # Calculate metrics by comparing extracted evidence with dataset answer
                metrics = calculate_char_match_rate(extracted_evidence, dataset_answer)

                evidence_detail = {
                    'chunk_id': idx + 1,
                    'chunk_content': evidence.get('chunk_content', ''),
                    'extracted_evidence': extracted_evidence,
                    'score': evidence.get('score', 0.0),
                    'text': evidence.get('text', ''),
                    'char_ranges': evidence.get('char_ranges', []),
                    'recall': metrics.get('recall', 0.0),  # Calculated metric
                    'precision': metrics.get('precision', 0.0),  # Calculated metric
                    'f1_score': metrics.get('f1_score', 0.0),  # Calculated metric
                    'exact_match': metrics.get('exact_match', False),  # Exact match flag
                    'semantic_score': evidence.get('semantic_score', 0.0),
                    'bm25_score': evidence.get('bm25_score', 0.0),
                    'evidence_variant_prompt': evidence.get('evidence_variant_prompt', ''),
                    'evidence_range_prompt': evidence.get('evidence_range_prompt', ''),
                    'llm_response': evidence.get('llm_response', '')
                }
                evidence_list.append(evidence_detail)

            results['details'].append({
                'query': query,
                'dataset_answer': dataset_answer,
                'generated_answer': generated_answer,
                'judgment': judgment,
                'is_correct': is_correct,
                'is_error': is_error,
                'evidences': evidence_list,  # FULL evidence details
                'processing_time': processing_time,
                'model': model,
                'confidence': confidence,
                'num_chunks': len(evidences),
                'context': context  # Include context from dataset
            })

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

            results['system_errors'] += 1
            results['details'].append({
                'query': query,
                'dataset_answer': dataset_answer,
                'generated_answer': 'ERROR',
                'judgment': 'no',
                'is_correct': False,
                'is_error': True,
                'evidences': [],
                'processing_time': 0,
                'model': 'gpt-4o-mini',
                'confidence': 0.0,
                'num_chunks': 0,
                'context': context,
                'error': str(e)
            })

    return results


def save_v2_full_to_database(results: Dict):
    """Save v2 with FULL evidence to database"""
    db_path = os.path.join(current_dir, "query_history.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Add context column if not exists
    cursor.execute("PRAGMA table_info(query_history)")
    columns = [col[1] for col in cursor.fetchall()]

    if 'context' not in columns:
        cursor.execute("ALTER TABLE query_history ADD COLUMN context TEXT")
        conn.commit()
        print("✅ Added 'context' column to database")

    # Delete existing v2_full_evidence data
    cursor.execute("DELETE FROM query_history WHERE version = 'v2_full_evidence'")
    print("🗑️  Deleted existing v2_full_evidence records")

    # Insert new v2_full_evidence data with COMPLETE information
    from datetime import datetime

    for detail in results['details']:
        cursor.execute("""
            INSERT INTO query_history (
                timestamp, query, generated_answer, processing_time,
                model, confidence, num_chunks, dataset_answer, evidences,
                answer_judgment, version, context, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            detail['query'],
            detail['generated_answer'],
            detail['processing_time'],
            detail['model'],
            detail['confidence'],
            detail['num_chunks'],
            detail['dataset_answer'],
            json.dumps(detail['evidences'], ensure_ascii=False),  # FULL evidences!
            detail['judgment'],
            'v2_full_evidence',
            detail['context'],  # Include context
            datetime.now().isoformat()
        ))

    conn.commit()
    conn.close()

    print(f"\n✅ Saved {len(results['details'])} v2_full_evidence records to database")


def print_summary(results: Dict):
    """Print evaluation summary"""
    total = results['total']
    correct = results['correct_answers']
    incorrect = results['incorrect_answers']
    errors = results['system_errors']

    print("\n" + "=" * 80)
    print("📊 V2 FULL EVIDENCE EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\nTotal Queries: {total}")
    print(f"✅ Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"❌ Incorrect: {incorrect} ({incorrect/total*100:.1f}%)")
    print(f"⚠️  System Errors: {errors} ({errors/total*100:.1f}%)")
    if total > errors:
        print(f"\n📈 Accuracy (excluding errors): {correct/(total-errors)*100:.1f}%")
    print("=" * 80)


def main():
    """Main function"""
    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found")
        return False

    # Initialize RAG
    chroma_path = os.path.join(current_dir, "chroma")
    print("🔄 Initializing RAG system...")
    rag = PureSemanticRAG(openai_api_key=api_key, chroma_path=chroma_path)
    print("✅ RAG initialized\n")

    # Load complete dataset
    print("📁 Loading dataset...")
    dataset = load_dataset()
    print(f"✅ Loaded {len(dataset)} samples\n")

    # Evaluate with FULL evidence collection
    results = evaluate_with_full_evidence(rag, dataset)

    # Print summary
    print_summary(results)

    # Save to database with version 'v2_full_evidence'
    save_v2_full_to_database(results)

    # Save to JSON file
    output_file = os.path.join(current_dir, "evaluation_results_v2_full_evidence.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("✅ V2 Full Evidence Evaluation Complete!")
    print("=" * 80)
    print(f"\n💡 Use version 'v2_full_evidence' in the UI to view:")
    print("   - Complete evidence details with chunks")
    print("   - Context highlighting from original dataset")
    print("   - All match metrics (recall, precision, F1)")
    print("   - LLM extraction prompts and responses")
    print("=" * 80)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
