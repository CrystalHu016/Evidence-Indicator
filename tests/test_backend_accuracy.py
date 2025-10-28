#!/usr/bin/env python3
"""
Test Backend RAG System Accuracy
Compare RAG answers with ground truth dataset answers
Calculate matching metrics (Exact Match, F1, etc.)
"""

import os
import sys
import json
import re
from typing import List, Tuple, Dict
from dotenv import load_dotenv

# Add script directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "script")
sys.path.insert(0, script_dir)

from ultra_fast_rag_semantic import PureSemanticRAG


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    # Remove spaces and special characters
    text = re.sub(r'\s+', '', text)
    # Convert to lowercase for case-insensitive comparison
    return text.lower()


def calculate_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Check if prediction exactly matches any ground truth"""
    pred_normalized = normalize_text(prediction)

    for gt in ground_truths:
        gt_normalized = normalize_text(gt)
        if pred_normalized == gt_normalized:
            return True
        # Also check if ground truth is contained in prediction
        if gt_normalized in pred_normalized:
            return True

    return False


def calculate_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate F1 score between prediction and ground truths"""
    pred_normalized = normalize_text(prediction)

    max_f1 = 0.0
    for gt in ground_truths:
        gt_normalized = normalize_text(gt)

        # Character-level F1
        pred_chars = set(pred_normalized)
        gt_chars = set(gt_normalized)

        if len(pred_chars) == 0 or len(gt_chars) == 0:
            f1 = 0.0
        else:
            common = pred_chars & gt_chars
            if len(common) == 0:
                f1 = 0.0
            else:
                precision = len(common) / len(pred_chars)
                recall = len(common) / len(gt_chars)
                f1 = 2 * precision * recall / (precision + recall)

        max_f1 = max(max_f1, f1)

    return max_f1


def calculate_answer_overlap(prediction: str, ground_truths: List[str]) -> float:
    """Calculate how much of ground truth is in prediction"""
    pred_normalized = normalize_text(prediction)

    max_overlap = 0.0
    for gt in ground_truths:
        gt_normalized = normalize_text(gt)

        if len(gt_normalized) == 0:
            continue

        # Check substring overlap
        if gt_normalized in pred_normalized:
            overlap = 1.0
        else:
            # Calculate longest common substring ratio
            overlap = 0.0
            for i in range(len(gt_normalized)):
                for j in range(i + 1, len(gt_normalized) + 1):
                    substr = gt_normalized[i:j]
                    if substr in pred_normalized:
                        overlap = max(overlap, len(substr) / len(gt_normalized))

        max_overlap = max(max_overlap, overlap)

    return max_overlap


def test_backend_accuracy():
    """Test backend accuracy with dataset questions"""
    print("=" * 80)
    print("🧪 Backend RAG System Accuracy Test")
    print("=" * 80)

    # Load environment
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return

    # Initialize RAG system
    data_file = os.path.join(current_dir, "data", "jsquad_validation_100.json")
    chroma_path = os.path.join(script_dir, "chroma_squad_multi_paragraph")

    print(f"\n📁 Data file: {data_file}")
    print(f"🗄️ Vector DB: {chroma_path}")

    # Load ground truth data
    with open(data_file, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    print(f"✅ Loaded {len(ground_truth_data)} ground truth entries\n")

    # Initialize RAG
    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)

    # Check if vector store exists
    if not os.path.exists(chroma_path):
        print("🏗️ Building vector store...")
        rag.build_vector_store(data_file, chunk_size=200, chunk_overlap=50)
    else:
        print("✅ Using existing vector store\n")

    # Select test questions - sample from different topics
    test_questions = [
        # Original 3 failing cases
        "梅雨とは何季の一種か?",
        "日本で梅雨がないのは北海道とどこか。",
        "梅雨がみられるのはどの期間？",
        # Additional diverse questions
        "入梅は何の目安の時期か？",
        "梅雨明けの別名を何というか。",
    ]

    # Find matching entries in ground truth
    test_cases = []
    for question in test_questions:
        for item in ground_truth_data:
            if item['question'] == question:
                test_cases.append({
                    'id': item['id'],
                    'question': question,
                    'context': item['context'],
                    'ground_truth_answers': item['answers']['text'],
                    'answer_start': item['answers']['answer_start']
                })
                break

    if len(test_cases) == 0:
        print("❌ No matching test cases found")
        return

    print(f"🔍 Testing {len(test_cases)} questions\n")
    print("=" * 80)

    # Run tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(test_cases)}: {test_case['id']}")
        print(f"{'=' * 80}")

        question = test_case['question']
        gt_answers = test_case['ground_truth_answers']

        print(f"❓ Question: {question}")
        print(f"✅ Ground Truth: {gt_answers}")

        try:
            # Query RAG system
            result = rag.query_with_answer(question, k=10)

            prediction = result['answer']
            print(f"\n🤖 RAG Answer: {prediction}")

            # Calculate metrics
            exact_match = calculate_exact_match(prediction, gt_answers)
            f1_score = calculate_f1_score(prediction, gt_answers)
            overlap = calculate_answer_overlap(prediction, gt_answers)

            # Check if evidence was extracted
            evidences = result.get('evidences', [])
            valid_evidences = [e for e in evidences if not e.get('is_empty', True)]
            has_evidence = len(valid_evidences) > 0

            # Check if ground truth is in retrieved context
            gt_in_context = any(
                any(normalize_text(gt) in normalize_text(e['chunk_content'])
                    for gt in gt_answers)
                for e in evidences
            )

            print(f"\n📊 Metrics:")
            print(f"   Exact Match: {'✅ YES' if exact_match else '❌ NO'}")
            print(f"   F1 Score: {f1_score:.3f}")
            print(f"   Answer Overlap: {overlap:.3f} ({overlap*100:.1f}%)")
            print(f"   Evidence Extracted: {'✅ YES' if has_evidence else '❌ NO'} ({len(valid_evidences)}/{len(evidences)} chunks)")
            print(f"   GT in Retrieved Context: {'✅ YES' if gt_in_context else '❌ NO'}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")

            # Detailed analysis
            if not exact_match:
                print(f"\n🔍 Analysis:")
                if overlap > 0.5:
                    print(f"   ⚠️  Partial match: {overlap*100:.1f}% of ground truth found in answer")
                elif gt_in_context:
                    print(f"   ⚠️  Ground truth in retrieved context but not in generated answer")
                else:
                    print(f"   ❌ Ground truth not found in retrieved context")

            results.append({
                'id': test_case['id'],
                'question': question,
                'ground_truth': gt_answers,
                'prediction': prediction,
                'exact_match': exact_match,
                'f1_score': f1_score,
                'overlap': overlap,
                'has_evidence': has_evidence,
                'gt_in_context': gt_in_context,
                'confidence': result['confidence'],
                'processing_time': result['processing_time']
            })

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'id': test_case['id'],
                'question': question,
                'ground_truth': gt_answers,
                'prediction': '',
                'exact_match': False,
                'f1_score': 0.0,
                'overlap': 0.0,
                'has_evidence': False,
                'gt_in_context': False,
                'confidence': 0.0,
                'processing_time': 0.0,
                'error': str(e)
            })

    # Final Summary
    print("\n" + "=" * 80)
    print("📊 ACCURACY SUMMARY")
    print("=" * 80)

    total = len(results)
    exact_matches = sum(1 for r in results if r['exact_match'])
    avg_f1 = sum(r['f1_score'] for r in results) / total if total > 0 else 0
    avg_overlap = sum(r['overlap'] for r in results) / total if total > 0 else 0
    avg_confidence = sum(r['confidence'] for r in results) / total if total > 0 else 0
    avg_time = sum(r['processing_time'] for r in results) / total if total > 0 else 0
    with_evidence = sum(1 for r in results if r['has_evidence'])
    gt_retrieved = sum(1 for r in results if r['gt_in_context'])

    print(f"\n📈 Overall Metrics:")
    print(f"   Total Questions: {total}")
    print(f"   Exact Match (EM): {exact_matches}/{total} ({exact_matches/total*100:.1f}%)")
    print(f"   Average F1 Score: {avg_f1:.3f}")
    print(f"   Average Answer Overlap: {avg_overlap:.3f} ({avg_overlap*100:.1f}%)")
    print(f"   Evidence Extracted: {with_evidence}/{total} ({with_evidence/total*100:.1f}%)")
    print(f"   GT in Retrieved Context: {gt_retrieved}/{total} ({gt_retrieved/total*100:.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Average Processing Time: {avg_time:.2f}s")

    print(f"\n📋 Detailed Results:")
    for i, result in enumerate(results, 1):
        status = "✅" if result['exact_match'] else "⚠️" if result['overlap'] > 0.5 else "❌"
        print(f"\n{status} Test {i}: {result['id']}")
        print(f"   Q: {result['question']}")
        print(f"   GT: {result['ground_truth'][0]}")
        print(f"   Pred: {result['prediction'][:80]}...")
        print(f"   EM: {result['exact_match']} | F1: {result['f1_score']:.3f} | Overlap: {result['overlap']:.3f}")

    # Grade the system
    print(f"\n🎯 System Grade:")
    if exact_matches == total:
        grade = "A+ (Perfect)"
        emoji = "🏆"
    elif exact_matches / total >= 0.8:
        grade = "A (Excellent)"
        emoji = "🎉"
    elif exact_matches / total >= 0.6:
        grade = "B (Good)"
        emoji = "👍"
    elif avg_overlap >= 0.7:
        grade = "C (Partial Matches)"
        emoji = "⚠️"
    else:
        grade = "D (Needs Improvement)"
        emoji = "❌"

    print(f"   {emoji} Grade: {grade}")
    print(f"   EM Rate: {exact_matches/total*100:.1f}%")
    print(f"   Avg Overlap: {avg_overlap*100:.1f}%")

    return results


if __name__ == "__main__":
    test_backend_accuracy()
