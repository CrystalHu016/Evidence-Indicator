#!/usr/bin/env python3
"""
Rebuild ChromaDB vector database with merged dataset
"""
import os
import sys
from pathlib import Path

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent / 'script'))

from ultra_fast_rag_semantic import PureSemanticRAG

def rebuild_database():
    """Rebuild ChromaDB with merged_qa_dataset.json"""
    print("🚀 Rebuilding ChromaDB Vector Database")
    print("=" * 70)

    # Configuration
    chroma_path = "chroma"
    dataset_path = "data/merged_qa_dataset.json"

    # Get API key from environment
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return False

    # Check if dataset exists
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False

    print(f"\n📂 Dataset: {dataset_path}")
    print(f"📂 ChromaDB path: {chroma_path}")
    print(f"🔑 Using OpenAI API")

    # Initialize RAG system
    print(f"\n🔄 Initializing PureSemanticRAG...")
    rag = PureSemanticRAG(
        openai_api_key=api_key,
        chroma_path=chroma_path
    )

    # Build vector store
    print(f"\n🔄 Building vector store from {dataset_path}...")
    print("   This may take several minutes for 10,100 Q&A pairs...")
    print()

    success = rag.build_vector_store(dataset_path)

    if success:
        print("\n" + "=" * 70)
        print("✅ ChromaDB vector database rebuilt successfully!")
        print("=" * 70)

        # Check database size
        if Path(chroma_path).exists():
            import subprocess
            result = subprocess.run(['du', '-sh', chroma_path], capture_output=True, text=True)
            if result.returncode == 0:
                size = result.stdout.split()[0]
                print(f"\n💾 Database size: {size}")

        # Count chunks (if available)
        try:
            collection = rag.chroma_collection
            count = collection.count()
            print(f"📊 Total chunks in database: {count:,}")
        except:
            pass

        print(f"\n🎉 Ready to use with Evidence Indicator RAG system!")
        return True
    else:
        print("\n❌ Failed to rebuild vector database")
        return False

if __name__ == "__main__":
    import time
    start_time = time.time()

    success = rebuild_database()

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total time: {elapsed:.2f} seconds")

    sys.exit(0 if success else 1)
