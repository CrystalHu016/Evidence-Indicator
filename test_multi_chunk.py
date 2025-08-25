#!/usr/bin/env python3
"""
Test script for Multi-Chunk Analysis System
Tests the new progressive chunking and evidence sufficiency evaluation
"""

import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from ultra_fast_rag import UltraFastRAG
    from multi_chunk_handler import MultiChunkAnalyzer
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_multi_chunk_system():
    """Test the multi-chunk analysis system"""
    
    # Check environment
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        return False
    
    chroma_path = os.environ.get("CHROMA_PATH", "chroma")
    if not os.path.exists(os.path.join(chroma_path, "chroma.sqlite3")):
        print(f"❌ Vector database not found at {chroma_path}")
        print("Please run 'python rag.py generate' first")
        return False
    
    print("🧪 Testing Multi-Chunk Analysis System")
    print("=" * 50)
    
    try:
        # Initialize RAG system
        rag = UltraFastRAG(api_key, chroma_path)
        print("✅ RAG system initialized")
        
        # Test queries of varying complexity
        test_queries = [
            {
                "query": "コンバインとは何ですか",
                "description": "Simple definition query (should use regular processing)",
                "expected_multi_chunk": False
            },
            {
                "query": "コンバインの種類はいくつありますか？それぞれの特徴と用途の違いを説明してください",
                "description": "Complex comparison query (should use multi-chunk)",
                "expected_multi_chunk": True
            },
            {
                "query": "稲作の手順を詳しく説明してください。田植えから収穫、精米までのすべての工程について",
                "description": "Complex procedural query (should use multi-chunk)",
                "expected_multi_chunk": True
            },
            {
                "query": "農業機械の分類について、種類と用途を比較分析してください",
                "description": "Analytical comparison query (should use multi-chunk)",
                "expected_multi_chunk": True
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n🔍 Test {i}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print("-" * 40)
            
            start_time = time.time()
            
            # Execute query
            answer, source_doc, evidence, start_char, end_char = rag.query(test_case['query'])
            
            processing_time = time.time() - start_time
            
            print(f"⚡ Processing time: {processing_time:.2f}s")
            print(f"📝 Answer: {answer[:100]}...")
            print(f"🔍 Evidence: {evidence[:80]}...")
            print(f"📊 Evidence range: {start_char}-{end_char}")
            
            # Check if multi-chunk was used (based on complexity)
            complexity_used = rag._should_use_multi_chunk(test_case['query'])
            expected = test_case['expected_multi_chunk']
            
            if complexity_used == expected:
                print(f"✅ Complexity detection: {'Multi-chunk' if complexity_used else 'Regular'} (as expected)")
            else:
                print(f"⚠️  Complexity detection mismatch: Expected {'Multi-chunk' if expected else 'Regular'}, got {'Multi-chunk' if complexity_used else 'Regular'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_evidence_sufficiency():
    """Test evidence sufficiency evaluation specifically"""
    print("\n🧪 Testing Evidence Sufficiency Evaluation")
    print("=" * 50)
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        from multi_chunk_handler import EvidenceSufficiencyEvaluator
        
        evaluator = EvidenceSufficiencyEvaluator(api_key)
        
        # Test cases for evidence sufficiency
        test_cases = [
            {
                "query": "コンバインとは何ですか",
                "evidence": "コンバインは農業機械です",
                "expected_sufficient": False,
                "description": "Insufficient evidence (too brief)"
            },
            {
                "query": "コンバインとは何ですか", 
                "evidence": "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
                "expected_sufficient": True,
                "description": "Sufficient evidence (complete definition)"
            },
            {
                "query": "ABC 3社の売上が最も高いのはどちらですか？",
                "evidence": "A社の売上がB社より高い",
                "expected_sufficient": False,
                "description": "Insufficient evidence (missing C社 information)"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Sufficiency Test {i}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print(f"Evidence: {test_case['evidence']}")
            
            score, is_sufficient, reason = evaluator.evaluate_sufficiency(
                test_case['query'], 
                test_case['evidence']
            )
            
            print(f"📊 Score: {score:.2f}")
            print(f"✅ Sufficient: {is_sufficient}")
            print(f"💭 Reason: {reason}")
            
            if is_sufficient == test_case['expected_sufficient']:
                print("✅ Evaluation matches expectation")
            else:
                print("⚠️  Evaluation differs from expectation")
        
        return True
        
    except Exception as e:
        print(f"❌ Sufficiency test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Multi-Chunk Analysis Tests")
    print("=" * 60)
    
    success = True
    
    # Test 1: Multi-chunk system
    if not test_multi_chunk_system():
        success = False
    
    # Test 2: Evidence sufficiency
    if not test_evidence_sufficiency():
        success = False
    
    # Summary
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ Multi-chunk analysis system is ready to use")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    main()