#!/usr/bin/env python3
"""
Test enhanced BM25 with ordinal number boosting
测试增强的BM25序数词匹配
"""

import sys
import os
from dotenv import load_dotenv

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.join(current_dir, "script")
rag_frontend_dir = os.path.join(current_dir, "rag-streamlit-frontend")
sys.path.insert(0, script_dir)
sys.path.insert(0, rag_frontend_dir)

load_dotenv()

from ultra_fast_rag_semantic import PureSemanticRAG

def test_thailand_pm_query():
    """Test Thailand PM query with enhanced BM25"""

    query = "タイ王国第31代首相は誰か？"
    correct_answer = "タクシン・チナワット"

    print("=" * 80)
    print("🧪 Testing Enhanced BM25 with Ordinal Number Boosting")
    print("=" * 80)
    print(f"\n❓ Query: {query}")
    print(f"✅ Expected Answer: {correct_answer}")

    # Initialize RAG system
    api_key = os.getenv("OPENAI_API_KEY")
    chroma_path = os.path.join(current_dir, "chroma")

    print(f"\n🔄 Initializing RAG system (this will rebuild BM25 with new tokenizer)...")
    rag = PureSemanticRAG(openai_api_key=api_key, chroma_path=chroma_path)

    print(f"\n🔍 Querying with k=10...\n")

    # Query with k=10
    result = rag.query_with_answer(query, k=10)

    generated_answer = result.get('answer', '')
    evidences = result.get('evidences', [])

    print("\n" + "=" * 80)
    print("📊 Results")
    print("=" * 80)
    print(f"\n🤖 Generated Answer: {generated_answer}")
    print(f"\n📊 Total Evidences: {len(evidences)}")

    # Check if correct chunk is in results
    correct_chunk_found = False
    correct_chunk_index = -1

    print("\n" + "=" * 80)
    print("📝 Retrieved Chunks Analysis")
    print("=" * 80)

    for i, evidence in enumerate(evidences, 1):
        text = evidence.get('text', '') or evidence.get('extracted_evidence', '')
        score = evidence.get('semantic_score', 0.0)

        # Check if this chunk contains the correct answer
        contains_correct = "タクシン・チナワット" in text
        contains_31st = "第31代首相" in text
        is_correct_chunk = contains_correct and contains_31st

        if is_correct_chunk:
            correct_chunk_found = True
            correct_chunk_index = i

        status = "✅ CORRECT CHUNK!" if is_correct_chunk else ""
        contains_inluck = "インラック" in text

        print(f"\n{i}. {status}")
        print(f"   Semantic Score: {score:.4f}")
        print(f"   Contains '第31代首相': {'✅ Yes' if contains_31st else '❌ No'}")
        print(f"   Contains 'タクシン・チナワット': {'✅ Yes' if contains_correct else '❌ No'}")
        print(f"   Contains 'インラック': {'⚠️ Yes' if contains_inluck else '❌ No'}")
        print(f"   Text preview: {text[:200]}...")

    print("\n" + "=" * 80)
    print("📊 Final Assessment")
    print("=" * 80)

    if correct_chunk_found:
        print(f"\n✅ SUCCESS! Correct chunk found at position #{correct_chunk_index}")
        print(f"   Enhanced BM25 tokenizer is working!")

        # Check if answer is also correct
        if correct_answer in generated_answer or "タクシン" in generated_answer:
            print(f"✅ Generated answer is CORRECT!")
            return True
        else:
            print(f"⚠️ Correct chunk retrieved but answer is still wrong")
            print(f"   This means LLM answer generation needs improvement")
            return False
    else:
        print(f"\n❌ FAILED! Correct chunk NOT found in top {len(evidences)} results")
        print(f"   Enhanced BM25 tokenizer needs further tuning")
        return False


if __name__ == "__main__":
    success = test_thailand_pm_query()
    sys.exit(0 if success else 1)
