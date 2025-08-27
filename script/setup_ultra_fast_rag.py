#!/usr/bin/env python3
"""
Setup script for Ultra Fast RAG System
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def create_env_file():
    """Create .env file if it doesn't exist"""
    if not os.path.exists('.env'):
        print("📝 Creating .env file...")
        with open('.env', 'w') as f:
            f.write("""# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Test Data Configuration (set to 'true' for faster development)
USE_TEST_DATA=false

# Data and Storage Paths
CHROMA_PATH=chroma
DATA_PATH=./data/single_20240229.json
""")
        print("✅ .env file created. Please add your OpenAI API key.")
        return False
    else:
        print("✅ .env file already exists")
        return True

def clean_metadata(metadata):
    """Clean metadata to be compatible with Chroma"""
    cleaned = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            # Convert lists to comma-separated strings
            cleaned[key] = ', '.join(str(item) for item in value)
        elif isinstance(value, (dict, bool)):
            # Convert complex types to strings
            cleaned[key] = str(value)
        else:
            # Keep simple types as is
            cleaned[key] = value
    return cleaned

def load_and_process_data(api_key: str):
    """Load and process data for the RAG system"""
    print("📚 Loading and processing data...")
    
    # Load test data first (smaller file for testing)
    test_data_path = 'data/test_sample.json'
    if not os.path.exists(test_data_path):
        print(f"❌ Test data not found at {test_data_path}")
        return False
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"📖 Loaded {len(data)} documents from test data")
        
        # Prepare documents for Chroma
        documents = []
        for item in data:
            # Clean metadata to be Chroma-compatible
            metadata = {
                'id': item['ID'],
                'task': item['meta'].get('task', []),
                'domain': item['meta'].get('domain', []),
                'source': 'test_sample'
            }
            cleaned_metadata = clean_metadata(metadata)
            
            # Create document with cleaned metadata
            doc = Document(
                page_content=item['text'],
                metadata=cleaned_metadata
            )
            documents.append(doc)
        
        # Initialize embeddings and Chroma
        embeddings = OpenAIEmbeddings(api_key=api_key)
        
        # Create Chroma database
        chroma_path = "chroma"
        if os.path.exists(chroma_path):
            print("🗑️  Removing existing Chroma database...")
            import shutil
            shutil.rmtree(chroma_path)
        
        print("🔧 Creating new Chroma database...")
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=chroma_path
        )
        
        # Persist the database (not needed in newer versions)
        # db.persist()
        print("✅ Chroma database created and populated successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False

def verify_setup():
    """Verify that the setup is working correctly"""
    print("🔍 Verifying setup...")
    
    try:
        # Check if Chroma database exists
        if not os.path.exists("chroma"):
            print("❌ Chroma database not found")
            return False
        
        # Check if we can load the database
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("❌ OpenAI API key not properly set")
            return False
        
        embeddings = OpenAIEmbeddings(api_key=api_key)
        db = Chroma(persist_directory="chroma", embedding_function=embeddings)
        
        # Try a simple search
        results = db.similarity_search("コンバイン", k=1)
        if results:
            print("✅ Setup verification successful!")
            return True
        else:
            print("❌ No search results found")
            return False
            
    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Ultra Fast RAG System Setup")
    print("=" * 40)
    
    # Check if .env file exists
    env_ready = create_env_file()
    
    if not env_ready:
        print("\n⚠️  Please edit the .env file and add your OpenAI API key")
        print("Then run this script again.")
        return
    
    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ Please set your OpenAI API key in the .env file")
        return
    
    print("✅ OpenAI API key loaded")
    
    # Process data
    if load_and_process_data(api_key):
        print("✅ Data processing completed")
        
        # Verify setup
        if verify_setup():
            print("\n🎉 Setup completed successfully!")
            print("\nYou can now run:")
            print("  python3 test_ultra_fast_rag.py")
            print("  python3 ultra_fast_rag.py")
        else:
            print("❌ Setup verification failed")
    else:
        print("❌ Data processing failed")

if __name__ == "__main__":
    main()
