#!/usr/bin/env python3
"""
Test the 入梅 query to see evidence extraction
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script'))
load_dotenv()

from ultra_fast_rag_semantic import PureSemanticRAG

def test_nyubai_query():
    print("🧪 Testing 入梅 Query")
    print("=" * 80)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    chroma_path = os.path.join(os.path.dirname(__file__), "script", "chroma_squad_dedup")
    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)

    test_query = "入梅は何の目安の時期か？"
    print(f"\n🔍 Test Query: {test_query}")
    print("-" * 80)

    result = rag.query_with_answer(test_query, k=3)

    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\n💬 Answer: {result['answer']}")
    print(f"📄 Chunks Used: {result['chunks_used']}")

    evidences = result.get('evidences', [])
    print(f"\n📌 Total Evidences: {len(evidences)}")

    for idx, evidence in enumerate(evidences, 1):
        print("\n" + "=" * 80)
        print(f"Evidence #{idx}")
        print("=" * 80)

        core_term = evidence.get('core_term', '')
        print(f"🎯 Core Term: '{core_term}'")

        char_ranges = evidence.get('char_ranges', [])
        if char_ranges:
            ranges_str = ', '.join([f"{s}～{e}" for s, e in char_ranges])
            print(f"📍 Character Ranges: {ranges_str}")

        extracted = evidence.get('extracted_evidence', '')
        print(f"📝 Extracted Text ({len(extracted)} chars):")
        print(f"'{extracted}'")

        print(f"\n🎯 Similarity: {evidence.get('similarity_score', 0):.3f}")
        print(f"🧠 Semantic Relevance: {evidence.get('semantic_relevance', 0):.3f}")

        chunk_content = evidence.get('chunk_content', '')
        print(f"\n📄 Chunk Content ({len(chunk_content)} chars, first 200):")
        print(chunk_content[:200] + "...")

        # Show LLM response
        llm_response = evidence.get('llm_response', '')
        if llm_response:
            print(f"\n🤖 LLM Raw Response:")
            print(llm_response[:300])
            if len(llm_response) > 300:
                print("...")

if __name__ == "__main__":
    test_nyubai_query()
