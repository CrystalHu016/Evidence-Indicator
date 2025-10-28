#!/usr/bin/env python3
"""
Evaluate RAG system with deduplicated dataset
使用去重后的数据集评估 RAG 系统

Key differences from previous evaluation:
1. Use ALL answer texts as ground truth (not just first one)
2. Evaluate if RAG answer contains ANY of the ground truth answers
3. Support multiple answer positions for highlighting
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add script directory to path
script_dir = os.path.join(os.path.dirname(__file__), "script")
sys.path.insert(0, script_dir)

from ultra_fast_rag_semantic import PureSemanticRAG

def calculate_f1_score(prediction: str, ground_truths: list) -> float:
    """
    Calculate F1 score between prediction and any of the ground truth answers
    计算预测答案和任意一个真实答案之间的 F1 分数
    """
    best_f1 = 0.0

    for gt in ground_truths:
        # Tokenize (simple character-level for Japanese)
        pred_tokens = set(prediction)
        gt_tokens = set(gt)

        common = pred_tokens & gt_tokens

        if len(common) == 0:
            f1 = 0.0
        else:
            precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = len(common) / len(gt_tokens) if len(gt_tokens) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        best_f1 = max(best_f1, f1)

    return best_f1

def exact_match(prediction: str, ground_truths: list) -> bool:
    """
    Check if prediction exactly matches ANY of the ground truth answers
    检查预测答案是否精确匹配任意一个真实答案
    """
    pred_clean = prediction.strip().lower()
    for gt in ground_truths:
        if gt.strip().lower() in pred_clean:
            return True
    return False

def answer_overlap(prediction: str, ground_truths: list) -> float:
    """
    Calculate the maximum overlap ratio with any ground truth answer
    计算与任意真实答案的最大重叠率
    """
    best_overlap = 0.0

    for gt in ground_truths:
        if gt in prediction:
            overlap = 1.0
        else:
            # Check partial overlap
            overlap = 0.0
            for i in range(len(gt)):
                for j in range(i+1, len(gt)+1):
                    substring = gt[i:j]
                    if substring in prediction:
                        overlap = max(overlap, len(substring) / len(gt))

        best_overlap = max(best_overlap, overlap)

    return best_overlap

def evaluate_rag_system():
    """Evaluate RAG system with deduplicated dataset"""

    print("="*80)
    print("🧪 RAG System Evaluation - Deduplicated Dataset")
    print("="*80)

    # Load environment
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return

    # Load test data
    data_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100.json")
    with open(data_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    print(f"\n📖 Loaded {len(test_data)} test questions")

    # Initialize RAG system with new database
    chroma_path = os.path.join(script_dir, "chroma_squad_dedup")
    print(f"🗄️  Using database: {chroma_path}")

    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)
    print("✅ RAG system initialized\n")

    # Evaluate subset (5 questions for quick test)
    test_subset = test_data[:5]

    results = []
    total_em = 0
    total_f1 = 0.0
    total_overlap = 0.0
    total_evidence = 0
    total_time = 0.0

    print("="*80)
    print("📊 Running Evaluation")
    print("="*80)

    for idx, item in enumerate(test_subset, 1):
        question = item['question']
        # Use ALL answer texts as ground truth
        ground_truth_texts = item['answers']['text']
        item_id = item.get('id', f'q{idx}')

        print(f"\n{'='*80}")
        print(f"Test {idx}/{len(test_subset)}: {item_id}")
        print(f"{'='*80}")
        print(f"❓ Question: {question}")
        print(f"📚 Ground Truth Answers ({len(ground_truth_texts)}):")
        for i, gt in enumerate(ground_truth_texts, 1):
            print(f"   {i}. {gt}")

        # Query RAG system
        print(f"\n🔍 Querying RAG system...")
        result = rag.query_with_answer(question, k=5)

        rag_answer = result['answer']
        confidence = result['confidence']
        processing_time = result['processing_time']
        evidences = result.get('evidences', [])
        valid_evidences = [e for e in evidences if not e['is_empty']]

        print(f"🤖 RAG Answer: {rag_answer}")

        # Calculate metrics
        em = exact_match(rag_answer, ground_truth_texts)
        f1 = calculate_f1_score(rag_answer, ground_truth_texts)
        overlap = answer_overlap(rag_answer, ground_truth_texts)
        has_evidence = len(valid_evidences) > 0

        total_em += 1 if em else 0
        total_f1 += f1
        total_overlap += overlap
        total_evidence += 1 if has_evidence else 0
        total_time += processing_time

        # Store result
        results.append({
            'id': item_id,
            'question': question,
            'ground_truth': ground_truth_texts,
            'prediction': rag_answer,
            'exact_match': em,
            'f1_score': f1,
            'overlap': overlap,
            'has_evidence': has_evidence,
            'num_valid_evidences': len(valid_evidences),
            'confidence': confidence,
            'processing_time': processing_time
        })

        print(f"\n📊 Metrics:")
        print(f"   Exact Match: {'✅ YES' if em else '❌ NO'}")
        print(f"   F1 Score: {f1:.3f}")
        print(f"   Answer Overlap: {overlap:.3f} ({overlap*100:.1f}%)")
        print(f"   Evidence Extracted: {'✅ YES' if has_evidence else '❌ NO'} ({len(valid_evidences)}/{len(evidences)} chunks)")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Processing Time: {processing_time:.2f}s")

        # Show evidence positions
        if valid_evidences:
            print(f"\n   📍 Evidence Positions:")
            for i, ev in enumerate(valid_evidences[:3], 1):
                print(f"      Chunk {i}: {ev['char_ranges']}")
                print(f"         Text: {ev['extracted_evidence'][:50]}...")

    # Calculate averages
    avg_em = total_em / len(test_subset)
    avg_f1 = total_f1 / len(test_subset)
    avg_overlap = total_overlap / len(test_subset)
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_time = total_time / len(test_subset)
    evidence_rate = total_evidence / len(test_subset)

    # Final summary
    print(f"\n{'='*80}")
    print("📊 EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n📈 Overall Metrics:")
    print(f"   Total Questions: {len(test_subset)}")
    print(f"   Exact Match (EM): {total_em}/{len(test_subset)} ({avg_em*100:.1f}%)")
    print(f"   Average F1 Score: {avg_f1:.3f}")
    print(f"   Average Answer Overlap: {avg_overlap:.3f} ({avg_overlap*100:.1f}%)")
    print(f"   Evidence Extraction Rate: {total_evidence}/{len(test_subset)} ({evidence_rate*100:.1f}%)")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Average Processing Time: {avg_time:.2f}s")

    # Detailed results
    print(f"\n📋 Detailed Results:\n")
    for r in results:
        em_icon = "✅" if r['exact_match'] else "❌"
        print(f"{em_icon} {r['id']}")
        print(f"   Q: {r['question'][:50]}...")
        print(f"   GT: {r['ground_truth']}")
        print(f"   Pred: {r['prediction'][:80]}...")
        print(f"   EM: {r['exact_match']} | F1: {r['f1_score']:.3f} | Overlap: {r['overlap']:.3f}")
        print()

    # Grade
    if avg_em >= 0.9:
        grade = "A+ (Excellent)"
    elif avg_em >= 0.8:
        grade = "A (Very Good)"
    elif avg_em >= 0.7:
        grade = "B+ (Good)"
    elif avg_em >= 0.6:
        grade = "B (Satisfactory)"
    else:
        grade = "C (Needs Improvement)"

    print(f"🎯 System Grade: {grade}")
    print(f"   EM Rate: {avg_em*100:.1f}%")
    print(f"   Avg Overlap: {avg_overlap*100:.1f}%")
    print(f"={'='*80}")

if __name__ == "__main__":
    evaluate_rag_system()
