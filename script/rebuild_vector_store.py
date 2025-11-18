#!/usr/bin/env python3
"""
Rebuild Vector Store with Improved Chunking Strategy
重建向量数据库以使用改进的分块策略
"""

import sys
import os

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ultra_fast_rag_semantic import PureSemanticRAG
from dotenv import load_dotenv

load_dotenv()

def main():
    """Rebuild vector store with improved chunking parameters"""

    print("=" * 80)
    print("🔧 Rebuilding Vector Store with Improved Chunking")
    print("=" * 80)

    # Configuration
    data_file = os.path.join(parent_dir, "data", "merged_qa_dataset.json")
    chroma_path = os.path.join(parent_dir, "chroma")

    print(f"\n📁 Data file: {data_file}")
    print(f"🗄️ Vector store path: {chroma_path}")

    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Error: Data file not found: {data_file}")
        return False

    # Ask for confirmation
    print("\n⚠️  This will delete the existing vector store and rebuild it.")
    print("⏱️  Estimated time: 5-10 minutes for ~10,000 documents")
    print("\n📏 New chunking parameters:")
    print("   - Chunk size: 300 characters (was: 200)")
    print("   - Chunk overlap: 100 characters (was: 50)")
    print("\n✨ Benefits:")
    print("   - Better context preservation")
    print("   - Improved retrieval for time-specific queries")
    print("   - More complete information in each chunk")

    response = input("\n🤔 Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cancelled by user")
        return False

    print("\n" + "=" * 80)
    print("🚀 Starting rebuild process...")
    print("=" * 80 + "\n")

    try:
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ Error: OPENAI_API_KEY not found in environment")
            return False

        # Initialize RAG system
        rag = PureSemanticRAG(openai_api_key=api_key, chroma_path=chroma_path)

        # Build vector store with improved parameters
        success = rag.build_vector_store(
            data_file=data_file,
            chunk_size=300,  # Increased from 200
            chunk_overlap=100,  # Increased from 50
            force_rebuild=True  # Force rebuild even if database exists
        )

        if success:
            print("\n" + "=" * 80)
            print("✅ Vector Store Rebuild Complete!")
            print("=" * 80)
            print("\n📊 Next steps:")
            print("1. Test the improved system with sample queries")
            print("2. Compare retrieval quality with previous version")
            print("3. Monitor performance on time-specific queries")
            return True
        else:
            print("\n❌ Rebuild failed")
            return False

    except Exception as e:
        print(f"\n❌ Error during rebuild: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
