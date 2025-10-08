#!/usr/bin/env python3
"""
Test script for the grain harvest query
"""
import os
import sys
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra_fast_rag_semantic import PureSemanticRAG

def main():
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    # Initialize RAG system
    print("🚀 Initializing RAG system...")
    rag = PureSemanticRAG(api_key)

    # Test query
    query = "コンバインで収穫できる穀物は何ですか？"

    print(f"\n{'='*80}")
    print(f"🔍 Testing query: {query}")
    print(f"{'='*80}\n")

    # Execute query
    result = rag.query_with_answer(query, k=8, relevance_threshold=0.3)

    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}\n")

    print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
    print(f"\n💬 Answer:\n{result['answer']}")
    print(f"\n📊 Confidence: {result['confidence']:.2f}")
    print(f"📄 Chunks used: {result['chunks_used']}")
    print(f"🧠 Model: {result['model']}")

    # Evidence details
    evidences = result.get('evidences', [])
    print(f"\n{'='*80}")
    print(f"📋 EVIDENCE EXTRACTION ({len(evidences)} chunks)")
    print(f"{'='*80}\n")

    for i, evidence in enumerate(evidences, 1):
        print(f"\n--- Chunk {i} ---")
        print(f"Similarity: {evidence.get('similarity_score', 0):.3f}")
        print(f"Semantic relevance: {evidence.get('semantic_relevance', 0):.3f}")
        print(f"Is empty: {evidence.get('is_empty', True)}")

        if not evidence.get('is_empty', True):
            print(f"Char ranges: {evidence.get('char_ranges', [])}")
            print(f"Extracted evidence:\n{evidence.get('extracted_evidence', '')[:200]}...")

        print(f"Original chunk ({len(evidence.get('chunk_content', ''))} chars):")
        print(f"{evidence.get('chunk_content', '')[:200]}...")

    # Check debug log
    print(f"\n{'='*80}")
    print("📝 Debug log available at: /tmp/rag_evidence_debug.log")
    print(f"{'='*80}\n")

    if os.path.exists('/tmp/rag_evidence_debug.log'):
        with open('/tmp/rag_evidence_debug.log', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Show last 100 lines
            print("Last debug entries:")
            print(''.join(lines[-100:]))

if __name__ == "__main__":
    main()
