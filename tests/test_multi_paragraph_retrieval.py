#!/usr/bin/env python3
"""
Test Multi-Paragraph Retrieval System
Verify improvements for the three failing examples from SQuAD dataset
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add script directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "script")
sys.path.insert(0, script_dir)

from ultra_fast_rag_semantic import PureSemanticRAG


def test_multi_paragraph_retrieval():
    """Test the three failing examples with multi-paragraph retrieval"""
    print("=" * 80)
    print("🧪 Testing Multi-Paragraph Retrieval System")
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

    # Create lookup by question
    ground_truth_map = {}
    for item in ground_truth_data:
        question = item['question']
        ground_truth_map[question] = {
            'context': item['context'],
            'answers': item['answers']['text'],
            'answer_start': item['answers']['answer_start']
        }

    print(f"✅ Loaded {len(ground_truth_data)} ground truth entries\n")

    # Initialize RAG
    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)

    # Build or load vector store
    print("\n🏗️ Building/Loading vector store with enhanced metadata...")
    rag.build_vector_store(data_file, chunk_size=200, chunk_overlap=50)

    # Test cases - the three failing examples
    test_cases = [
        {
            "question": "梅雨とは何季の一種か?",
            "expected_answer": "雨季",
            "issue": "Retrieved irrelevant fragment 'みられる' instead of '雨季'"
        },
        {
            "question": "日本で梅雨がないのは北海道とどこか。",
            "expected_answer": "小笠原諸島",
            "issue": "Retrieved fragment about where 梅雨 exists, not where it doesn't"
        },
        {
            "question": "梅雨がみられるのはどの期間？",
            "expected_answer": "5月から7月にかけて",
            "issue": "Retrieved from wrong paragraph (general duration vs specific period)"
        }
    ]

    print("\n" + "=" * 80)
    print("🧪 Testing Three Failing Examples")
    print("=" * 80)

    results = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}/3")
        print(f"{'=' * 80}")

        question = test_case['question']
        expected = test_case['expected_answer']
        issue = test_case['issue']

        print(f"❓ Question: {question}")
        print(f"✅ Expected Answer: {expected}")
        print(f"⚠️  Previous Issue: {issue}\n")

        # Get ground truth context
        if question in ground_truth_map:
            gt = ground_truth_map[question]
            print(f"📖 Ground Truth Context ({len(gt['context'])} chars):")
            print(f"   {gt['context'][:200]}...\n")

        # Query RAG system
        try:
            result = rag.query_with_answer(question, k=3)

            print(f"\n📊 RAG Results:")
            print(f"   Answer: {result['answer']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Chunks Used: {result['chunks_used']}")
            print(f"   Processing Time: {result['processing_time']:.2f}s")

            # Check if evidence contains expected answer
            evidences = result.get('evidences', [])
            print(f"\n🔍 Evidence Extraction Results:")
            print(f"   Total Chunks: {len(evidences)}")

            valid_evidences = [e for e in evidences if not e['is_empty']]
            print(f"   Valid Evidences: {len(valid_evidences)}")

            # Check if answer is in the retrieved evidence
            answer_found = False
            if expected in result['answer']:
                print(f"   ✅ Expected answer '{expected}' found in generated answer!")
                answer_found = True
            else:
                print(f"   ❌ Expected answer '{expected}' NOT in generated answer")

            # Check evidence quality
            evidence_quality = "NONE"
            for j, evidence in enumerate(evidences, 1):
                if not evidence['is_empty']:
                    extracted = evidence['extracted_evidence']
                    print(f"\n   Evidence {j}:")
                    print(f"      Similarity: {evidence['similarity_score']:.3f}")
                    print(f"      Extracted: {extracted[:150]}...")
                    print(f"      Ranges: {evidence.get('char_ranges', [])}")

                    if expected in extracted:
                        print(f"      ✅ Contains expected answer '{expected}'")
                        evidence_quality = "PERFECT"
                    elif expected in evidence['chunk_content']:
                        print(f"      ⚠️  Answer in chunk but not extracted")
                        if evidence_quality != "PERFECT":
                            evidence_quality = "PARTIAL"

            # Summary for this test case
            print(f"\n📝 Test Case {i} Summary:")
            print(f"   Answer Match: {'✅ YES' if answer_found else '❌ NO'}")
            print(f"   Evidence Quality: {evidence_quality}")
            print(f"   Status: {'✅ FIXED' if answer_found and evidence_quality in ['PERFECT', 'PARTIAL'] else '❌ STILL FAILING'}")

            results.append({
                'question': question,
                'expected': expected,
                'answer': result['answer'],
                'answer_match': answer_found,
                'evidence_quality': evidence_quality,
                'success': answer_found and evidence_quality in ['PERFECT', 'PARTIAL']
            })

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'question': question,
                'expected': expected,
                'answer': '',
                'answer_match': False,
                'evidence_quality': 'ERROR',
                'success': False,
                'error': str(e)
            })

    # Final Summary
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)

    successful = sum(1 for r in results if r['success'])
    print(f"\nResults: {successful}/{len(results)} test cases passed")

    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"\n{status} Test {i}:")
        print(f"   Q: {result['question']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Got: {result['answer'][:100]}...")
        print(f"   Evidence Quality: {result['evidence_quality']}")

    if successful == len(results):
        print("\n🎉 All tests passed! Multi-paragraph retrieval is working correctly.")
    else:
        print(f"\n⚠️  {len(results) - successful} test(s) still failing. Further improvements needed.")

    return results


if __name__ == "__main__":
    test_multi_paragraph_retrieval()
