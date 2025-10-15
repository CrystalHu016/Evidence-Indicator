#!/usr/bin/env python3
"""
Build new vector database using squad_test_100.json
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add script directory to path
script_dir = os.path.join(os.path.dirname(__file__), "script")
sys.path.insert(0, script_dir)

from ultra_fast_rag_semantic import PureSemanticRAG

def main():
    """Build new vector database with squad_test_100.json"""
    
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment variables")
        return False
    
    # Paths
    data_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100.json")
    chroma_path = os.path.join(script_dir, "chroma_squad")
    
    print(f"🏗️ Building new vector database...")
    print(f"📁 Data file: {data_file}")
    print(f"🗄️ Vector database path: {chroma_path}")
    
    # Check if data file exists
    if not os.path.exists(data_file):
        print(f"❌ Data file does not exist: {data_file}")
        return False
    
    # Load and validate data
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📖 Loaded {len(data)} data entries from squad_test_100.json")
    
    # Convert SQuAD format to our format
    converted_data = []
    for item in data:
        # Extract context and question-answer pairs
        context = item.get('context', '')
        question = item.get('question', '')
        answers = item.get('answers', {})
        
        if context and question:
            # Create a document with context and Q&A
            content = f"文脈: {context}\n\n質問: {question}\n\n回答: {answers.get('text', [''])[0] if answers.get('text') else '回答なし'}"
            
            converted_data.append({
                'output': content,
                'context': context,
                'question': question,
                'answers': answers,
                'id': item.get('id', ''),
                'title': item.get('title', '')
            })
    
    print(f"📄 Converted {len(converted_data)} documents")
    
    # Save converted data temporarily
    temp_data_file = os.path.join(os.path.dirname(__file__), "data", "squad_converted.json")
    with open(temp_data_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved converted data to: {temp_data_file}")
    
    # Initialize RAG system
    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)
    
    # Build vector store
    success = rag.build_vector_store(temp_data_file, chunk_size=300, chunk_overlap=50)
    
    if success:
        print("✅ Vector database built successfully!")
        
        # Test the database
        print("\n🧪 Testing the new database...")
        test_queries = [
            "梅雨とは何ですか",
            "梅雨の期間はいつですか",
            "梅雨前線はどのように形成されますか"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Test query: {query}")
            result = rag.query_with_answer(query, k=3)
            print(f"💬 Answer: {result['answer'][:100]}...")
            print(f"📊 Confidence: {result['confidence']:.2f}")
            print(f"⏱️ Processing time: {result['processing_time']:.2f}s")
        
        # Clean up temporary file
        os.remove(temp_data_file)
        print(f"\n🗑️ Cleaned up temporary file: {temp_data_file}")
        
        return True
    else:
        print("❌ Failed to build vector database")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Database setup complete! You can now run the frontend.")
    else:
        print("\n❌ Database setup failed!")
