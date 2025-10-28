#!/usr/bin/env python3
"""
Test unified core term extraction - ensures all chunks use the same core term
"""
import sys
import os
sys.path.insert(0, './script')

from ultra_fast_rag_semantic import PureSemanticRAG

def test_unified_core_term():
    """Test that all chunks use the same unified core term"""

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    # Initialize RAG
    print("Initializing RAG system...")
    rag = PureSemanticRAG(openai_api_key=api_key, chroma_path='./chroma_semantic')

    # Test query
    query = '初夏に入った5月ごろ、北上する気流は何か？'
    print(f"\n{'='*80}")
    print(f"Test Query: {query}")
    print(f"{'='*80}\n")

    print("Expected behavior:")
    print("  - All chunks should use the SAME unified core term")
    print("  - Core term should be: '亜熱帯ジェット気流'")
    print("  - NOT '熱帯モンスーン気団' or other terms\n")

    # Run query
    result = rag.query_with_answer(query, k=3)

    if not result.get('evidences'):
        print("\n⚠️ No evidences found - database may be empty")
        return

    # Collect all core terms
    core_terms = []
    print(f"\n{'='*80}")
    print("Results:")
    print(f"{'='*80}")
    print(f"\nGenerated Answer: {result.get('answer', 'N/A')}")
    print(f"\nFound {len(result.get('evidences', []))} evidence chunks:\n")

    for i, ev in enumerate(result.get('evidences', []), 1):
        core_term = ev.get('core_term', 'N/A')
        core_terms.append(core_term)

        print(f"Chunk {i}:")
        print(f"  Core Term: '{core_term}'")
        print(f"  Evidence: '{ev.get('extracted_evidence', 'N/A')[:80]}...'")
        print()

    # Verify all core terms are the same
    print(f"{'='*80}")
    print("Verification:")
    print(f"{'='*80}")

    unique_core_terms = list(set(core_terms))
    print(f"\nUnique core terms found: {len(unique_core_terms)}")
    print(f"Core terms: {unique_core_terms}")

    if len(unique_core_terms) == 1:
        print(f"\n✅ SUCCESS: All chunks use the same unified core term: '{unique_core_terms[0]}'")
        return True
    else:
        print(f"\n❌ FAILURE: Multiple different core terms found!")
        for i, term in enumerate(unique_core_terms, 1):
            print(f"  {i}. '{term}'")
        return False

if __name__ == "__main__":
    success = test_unified_core_term()
    sys.exit(0 if success else 1)
