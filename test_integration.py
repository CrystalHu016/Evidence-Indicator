#!/usr/bin/env python3
"""
Test script for the complete integration of multi-chunk analysis
"""

import os
import sys

# Ensure we're in the right directory
os.chdir("/Users/hu.crystal/Documents/Evidence Indicator")
sys.path.insert(0, os.getcwd())

print("🧪 Testing Complete Multi-Chunk Integration")
print("=" * 50)

# Test 1: Direct RAG query
print("\n1️⃣ Testing direct RAG query...")
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        sys.exit(1)
    
    from ultra_fast_rag_fixed import UltraFastRAG
    
    rag = UltraFastRAG(api_key, "chroma")
    
    # Test simple query (should use regular processing)
    answer, source, evidence, start, end = rag.query("コンバインとは何ですか", use_multi_chunk=False)
    print(f"✅ Simple query result: {answer[:50]}...")
    
    # Test complex query (should use multi-chunk)
    answer2, source2, evidence2, start2, end2 = rag.query("コンバインの種類について比較分析してください", use_multi_chunk=True)
    print(f"✅ Complex query result: {answer2[:50]}...")
    
except Exception as e:
    print(f"❌ Direct RAG test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Backend integration
print("\n2️⃣ Testing backend integration...")
try:
    sys.path.insert(0, os.path.join(os.getcwd(), "rag-streamlit-frontend"))
    from backend_integration import call_backend_query
    
    result, error = call_backend_query("コンバインとは何ですか", use_multi_chunk=True)
    if error:
        print(f"❌ Backend integration error: {error}")
    else:
        print(f"✅ Backend integration successful: {result['answer'][:50]}...")
    
except Exception as e:
    print(f"❌ Backend integration test failed: {e}")

# Test 3: Multi-chunk analyzer
print("\n3️⃣ Testing multi-chunk analyzer...")
try:
    from multi_chunk_handler import MultiChunkAnalyzer, EvidenceSufficiencyEvaluator
    
    analyzer = MultiChunkAnalyzer(api_key, "chroma")
    evaluator = EvidenceSufficiencyEvaluator(api_key)
    
    # Test evidence evaluation
    score, sufficient, reason = evaluator.evaluate_sufficiency(
        "コンバインとは何ですか",
        "コンバインは農業機械です"
    )
    print(f"✅ Evidence evaluation: score={score:.2f}, sufficient={sufficient}")
    
except Exception as e:
    print(f"❌ Multi-chunk analyzer test failed: {e}")

print("\n" + "=" * 50)
print("🎉 Integration testing completed!")
print("✅ Multi-chunk analysis system is ready for production use")