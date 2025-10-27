#!/usr/bin/env python3
"""
Test script to evaluate the accuracy of character marker-based evidence extraction
Compares the old method vs new method
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.ultra_fast_rag_semantic_with_char_markers import ImprovedSemanticRAG


def test_evidence_extraction_accuracy():
    """Test evidence extraction accuracy with multiple test cases"""

    print("="*80)
    print("🧪 Evidence Extraction Accuracy Test")
    print("="*80)

    # Load environment
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    # Initialize RAG system
    print("\n📦 Initializing RAG system...")
    rag = ImprovedSemanticRAG(api_key)

    if not rag.build_vector_store(""):
        print("❌ Failed to load vector store")
        return

    # Test cases with expected answers
    test_cases = [
        {
            "id": 1,
            "query": "梅雨とは何季の一種か?",
            "expected_answer": "雨季",
            "expected_context": "雨の多い期間のこと。雨季の一種である。",
            "description": "Question about what kind of season - should extract only '雨季'"
        },
        {
            "id": 2,
            "query": "造語とは何か?",
            "expected_answer": "造語",
            "expected_context": "新たに語（単語）を造ること",
            "description": "Definition question - should extract the term being defined"
        },
        {
            "id": 3,
            "query": "梅雨がみられるのはどの期間?",
            "expected_answer": "5月から7月",
            "expected_context": "5月から7月にかけて来る曇りや雨の多い期間",
            "description": "Time period question - should extract time range"
        }
    ]

    results = []

    for test_case in test_cases:
        print(f"\n{'='*80}")
        print(f"📝 Test Case {test_case['id']}: {test_case['description']}")
        print(f"{'='*80}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Answer: {test_case['expected_answer']}")

        # Run query
        result = rag.query_with_answer(test_case['query'], k=3)

        # Extract evidence
        evidences = result.get('evidences', [])

        # Check if any evidence matches expected answer
        found_match = False
        extracted_texts = []

        for evidence in evidences:
            if not evidence.get('is_empty', True):
                extracted_text = evidence.get('extracted_evidence', '')
                extracted_texts.append(extracted_text)

                # Check if extracted text matches or contains expected answer
                if test_case['expected_answer'] in extracted_text or extracted_text in test_case['expected_answer']:
                    found_match = True
                    print(f"\n✅ SUCCESS: Extracted '{extracted_text}'")
                    print(f"   Character Range: {evidence.get('char_ranges', [])}")
                    print(f"   Core Term: {evidence.get('core_term', 'N/A')}")
                    break

        if not found_match:
            print(f"\n❌ FAILED: Expected '{test_case['expected_answer']}', but extracted: {extracted_texts}")

        # Store result
        results.append({
            'test_id': test_case['id'],
            'query': test_case['query'],
            'expected': test_case['expected_answer'],
            'extracted': extracted_texts,
            'success': found_match,
            'answer': result.get('answer', ''),
            'processing_time': result.get('processing_time', 0.0)
        })

    # Print summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")

    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    accuracy = (success_count / total_count * 100) if total_count > 0 else 0

    print(f"\nTotal Tests: {total_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {total_count - success_count}")
    print(f"Accuracy: {accuracy:.1f}%")

    print(f"\n{'='*80}")
    print("Detailed Results:")
    print(f"{'='*80}")

    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} Test {result['test_id']}: {result['query']}")
        print(f"   Expected: {result['expected']}")
        print(f"   Extracted: {result['extracted']}")
        print(f"   Answer: {result['answer']}")
        print(f"   Time: {result['processing_time']:.2f}s")

    # Write results to file
    output_file = "/tmp/char_marker_accuracy_test.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total': total_count,
                'success': success_count,
                'failed': total_count - success_count,
                'accuracy': accuracy
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n📄 Detailed results saved to: {output_file}")

    return accuracy


if __name__ == "__main__":
    accuracy = test_evidence_extraction_accuracy()

    print(f"\n{'='*80}")
    print(f"🎯 Final Accuracy: {accuracy:.1f}%")
    print(f"{'='*80}")
