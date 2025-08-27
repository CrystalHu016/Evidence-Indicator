#!/usr/bin/env python3
"""
Comprehensive test script for Ultra Fast RAG System
"""

import os
import time
import json
from dotenv import load_dotenv
from ultra_fast_rag import UltraFastRAG

def load_test_data():
    """Load test data from the test_sample.json file"""
    try:
        with open('data/test_sample.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} test samples")
        return data
    except Exception as e:
        print(f"❌ Failed to load test data: {e}")
        return []

def test_basic_functionality(rag):
    """Test basic RAG functionality"""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 50)
    
    test_queries = [
        "コンバインとは何ですか",
        "音位転倒について説明してください",
        "どのような農業機械がありますか"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        start_time = time.time()
        
        try:
            answer, source, evidence, start, end = rag.query(query)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Processing time: {elapsed:.2f}s")
            print(f"💡 Answer: {answer[:200]}...")
            print(f"📄 Source length: {len(source)} chars")
            print(f"🔍 Evidence: {evidence[:150]}...")
            print(f"📍 Position: {start+1}-{end}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")

def test_multi_chunk_analysis(rag):
    """Test multi-chunk analysis functionality"""
    print("\n🧪 Testing Multi-Chunk Analysis")
    print("=" * 50)
    
    complex_queries = [
        "稲作の手順と工程について詳しく説明してください",
        "農業機械の種類とそれぞれの特徴を比較してください",
        "作物の栽培方法と収穫時期について分析してください"
    ]
    
    for query in complex_queries:
        print(f"\n📝 Complex Query: {query}")
        start_time = time.time()
        
        try:
            answer, source, evidence, start, end = rag.query(query, use_multi_chunk=True)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Processing time: {elapsed:.2f}s")
            print(f"💡 Answer: {answer[:200]}...")
            print(f"📄 Source length: {len(source)} chars")
            print(f"🔍 Evidence: {evidence[:150]}...")
            
        except Exception as e:
            print(f"❌ Error processing complex query: {e}")

def test_performance_benchmark(rag):
    """Test performance with multiple queries"""
    print("\n🧪 Performance Benchmark")
    print("=" * 50)
    
    # Load test queries from data
    test_data = load_test_data()
    if not test_data:
        return
    
    # Take first 10 queries for performance testing
    test_queries = [item['text'] for item in test_data[:10]]
    
    total_time = 0
    successful_queries = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}/{len(test_queries)}: {query[:50]}...")
        
        start_time = time.time()
        try:
            answer, source, evidence, start, end = rag.query(query)
            elapsed = time.time() - start_time
            total_time += elapsed
            successful_queries += 1
            
            print(f"⏱️  Time: {elapsed:.2f}s")
            print(f"💡 Answer length: {len(answer)} chars")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if successful_queries > 0:
        avg_time = total_time / successful_queries
        print(f"\n📊 Performance Summary:")
        print(f"   Total queries: {len(test_queries)}")
        print(f"   Successful: {successful_queries}")
        print(f"   Average time: {avg_time:.2f}s")
        print(f"   Total time: {total_time:.2f}s")

def test_edge_cases(rag):
    """Test edge cases and error handling"""
    print("\n🧪 Testing Edge Cases")
    print("=" * 50)
    
    edge_queries = [
        "",  # Empty query
        "a" * 1000,  # Very long query
        "稲作のレシピについて教えてください",  # Recipe query for agriculture
        "What is a combine harvester?",  # English query
        "稲作の手順と工程について詳しく説明してください。また、それぞれの工程で必要な機械や道具についても教えてください。収穫後の処理方法についても詳しく説明してください。"  # Very complex query
    ]
    
    for i, query in enumerate(edge_queries):
        print(f"\n📝 Edge Case {i+1}: {query[:50]}...")
        
        try:
            start_time = time.time()
            answer, source, evidence, start, end = rag.query(query)
            elapsed = time.time() - start_time
            
            print(f"⏱️  Time: {elapsed:.2f}s")
            print(f"💡 Answer: {answer[:100]}...")
            
        except Exception as e:
            print(f"❌ Error (expected for edge case): {e}")

def main():
    """Main test function"""
    print("🚀 Ultra Fast RAG System - Comprehensive Test Suite")
    print("=" * 60)
    
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment")
        print("Please create a .env file with your OpenAI API key")
        return
    
    print("✅ OpenAI API key loaded")
    
    # Initialize RAG system
    try:
        rag = UltraFastRAG(api_key, "chroma")
        print("✅ RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return
    
    # Run tests
    test_basic_functionality(rag)
    test_multi_chunk_analysis(rag)
    test_performance_benchmark(rag)
    test_edge_cases(rag)
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
