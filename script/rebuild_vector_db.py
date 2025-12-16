#!/usr/bin/env python3
"""Rebuild vector database with correct data"""

from ultra_fast_rag_semantic import PureSemanticRAG
import os

# Get API key from environment
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("❌ OPENAI_API_KEY not found in environment")
    exit(1)

print("🏗️ Rebuilding vector database with chunk_size=150...")
rag = PureSemanticRAG(api_key, chroma_path="./chroma_semantic_chunk150")

data_file = "../data/single_20240229.json"
success = rag.build_vector_store(data_file, chunk_size=150, chunk_overlap=30)

if success:
    print("\n✅ Vector database rebuilt successfully!")
    print("\n🧪 Testing search for 'コンバイン'...")

    results = rag.db.similarity_search_with_score("コンバイン", k=3)
    print(f"\nFound {len(results)} results:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Content: {doc.page_content[:150]}...")
else:
    print("\n❌ Failed to rebuild vector database")
