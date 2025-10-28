#!/usr/bin/env python3
"""
Build new vector database from deduplicated squad_test_100.json
使用去重后的 squad_test_100.json 构建新的向量数据库
"""

import os
import sys
from dotenv import load_dotenv

# Add script directory to path
script_dir = os.path.join(os.path.dirname(__file__), "script")
sys.path.insert(0, script_dir)

from ultra_fast_rag_semantic import PureSemanticRAG

def main():
    print("="*80)
    print("🏗️  Building new vector database from deduplicated dataset")
    print("="*80)

    # Load environment
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False

    # Paths
    data_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100.json")
    chroma_path = os.path.join(script_dir, "chroma_squad_dedup")

    print(f"\n📁 Data file: {data_file}")
    print(f"🗄️  Database path: {chroma_path}")

    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False

    # Initialize RAG system
    print(f"\n🔧 Initializing RAG system...")
    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)

    # Build vector store with multi-paragraph support
    print(f"\n📊 Building vector store with multi-paragraph retrieval...")
    print(f"   Settings: chunk_size=300, chunk_overlap=50")

    success = rag.build_vector_store(
        data_file,
        chunk_size=300,
        chunk_overlap=50
    )

    if success:
        print("\n" + "="*80)
        print("✅ Database built successfully!")
        print("="*80)

        # Test the database with a sample query
        print("\n🧪 Testing the new database...")
        test_query = "梅雨とは何季の一種か?"

        print(f"\n🔍 Test query: {test_query}")
        result = rag.query_with_answer(test_query, k=5)

        print(f"\n📊 Test Results:")
        print(f"   Answer: {result['answer'][:100]}...")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Evidence extracted: {len([e for e in result.get('evidences', []) if not e['is_empty']])}/{len(result.get('evidences', []))} chunks")
        print(f"   Processing time: {result['processing_time']:.2f}s")

        # Show evidence details
        evidences = result.get('evidences', [])
        valid_evidences = [e for e in evidences if not e['is_empty']]
        if valid_evidences:
            print(f"\n   📍 Evidence samples:")
            for i, ev in enumerate(valid_evidences[:2], 1):
                print(f"      {i}. Char ranges: {ev['char_ranges']}")
                print(f"         Text: {ev['extracted_evidence'][:60]}...")

        print("\n" + "="*80)
        print("🎉 Database setup complete!")
        print(f"   Database location: {chroma_path}")
        print(f"   Ready for use with frontend and evaluation scripts")
        print("="*80)

        return True
    else:
        print("\n❌ Failed to build database")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
