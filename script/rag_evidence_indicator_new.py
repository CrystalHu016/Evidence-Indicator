#!/usr/bin/env python3
"""
RAG Evidence Indicator - New Dataset Version
Enhanced RAG system using the cleaned Ichikara dataset
"""

import os
import json
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr

# Load environment variables
load_dotenv()

class RAGEvidenceIndicator:
    """Enhanced RAG Evidence Indicator using new dataset"""
    
    def __init__(self, dataset_path: str, chroma_path: str = "./chroma_new"):
        self.dataset_path = dataset_path
        self.chroma_path = chroma_path
        self.collection_name = "evidence_indicator_collection"
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.documents = []
        
        # Configuration
        self.chunk_size = 300
        self.chunk_overlap = 100
        self.search_k = 3
        
    def initialize_embeddings(self):
        """Initialize OpenAI embeddings"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=SecretStr(api_key),
                model="text-embedding-ada-002"
            )
            print("✅ OpenAI embeddings initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize embeddings: {e}")
            return False
    
    def load_dataset(self) -> bool:
        """Load the new dataset"""
        try:
            print(f"📖 Loading dataset from: {self.dataset_path}")
            
            # Load JSON dataset
            loader = JSONLoader(
                file_path=self.dataset_path,
                jq_schema='.[]',
                text_content=False
            )
            
            raw_documents = loader.load()
            print(f"📊 Loaded {len(raw_documents)} raw documents")
            
            # Process documents
            self.documents = []
            for doc in raw_documents:
                # Extract content from the new dataset structure
                content = doc.page_content
                metadata = doc.metadata
                
                # Create structured document with simple metadata (ChromaDB compatible)
                structured_doc = Document(
                    page_content=content,
                    metadata={
                        "source": "ichikara_dataset",
                        "id": str(metadata.get("ID", "unknown")),
                        "text": str(metadata.get("text", "")),
                        "dataset": "ichikara_rebuilt"
                    }
                )
                self.documents.append(structured_doc)
            
            print(f"✅ Processed {len(self.documents)} documents")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False
    
    def split_documents(self) -> List[Document]:
        """Split documents into chunks"""
        try:
            print("✂️ Splitting documents into chunks...")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
            )
            
            split_docs = text_splitter.split_documents(self.documents)
            print(f"✅ Split into {len(split_docs)} chunks")
            
            return split_docs
            
        except Exception as e:
            print(f"❌ Failed to split documents: {e}")
            return []
    
    def create_vectorstore(self, documents: List[Document]) -> bool:
        """Create and populate vector store"""
        try:
            print("🗄️ Creating vector store...")
            
            # Create ChromaDB vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.chroma_path,
                collection_name=self.collection_name
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            print(f"✅ Vector store created and persisted to: {self.chroma_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create vector store: {e}")
            return False
    
    def search(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        try:
            if not self.vectorstore:
                print("❌ Vector store not initialized")
                return []
            
            k = k or self.search_k
            print(f"🔍 Searching for: '{query}' (k={k})")
            
            # Perform similarity search
            results = self.vectorstore.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                    "source": doc.metadata.get("source", "unknown")
                })
            
            print(f"✅ Found {len(formatted_results)} relevant documents")
            return formatted_results
            
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def get_evidence(self, query: str, k: int = None) -> Dict[str, Any]:
        """Get evidence for a query with enhanced metadata"""
        try:
            start_time = time.time()
            
            # Search for relevant documents
            results = self.search(query, k)
            
            # Process results
            evidence = {
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results_count": len(results),
                "processing_time": time.time() - start_time,
                "evidence_sources": [],
                "summary": ""
            }
            
            # Extract evidence sources
            for result in results:
                source_info = {
                    "content": result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"],
                    "metadata": result["metadata"],
                    "similarity_score": result["similarity_score"],
                    "source_type": result["metadata"].get("source", "unknown")
                }
                evidence["evidence_sources"].append(source_info)
            
            # Generate summary
            if results:
                evidence["summary"] = f"Found {len(results)} relevant evidence sources from the Ichikara dataset. "
                evidence["summary"] += f"Top result has similarity score: {results[0]['similarity_score']:.4f}"
            else:
                evidence["summary"] = "No relevant evidence found for the query."
            
            return evidence
            
        except Exception as e:
            print(f"❌ Failed to get evidence: {e}")
            return {"error": str(e), "query": query}
    
    def initialize_system(self) -> bool:
        """Initialize the complete RAG system"""
        try:
            print("🚀 Initializing RAG Evidence Indicator System...")
            print("=" * 60)
            
            # Step 1: Initialize embeddings
            if not self.initialize_embeddings():
                return False
            
            # Step 2: Load dataset
            if not self.load_dataset():
                return False
            
            # Step 3: Split documents
            split_docs = self.split_documents()
            if not split_docs:
                return False
            
            # Step 4: Create vector store
            if not self.create_vectorstore(split_docs):
                return False
            
            print("🎉 RAG Evidence Indicator System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and statistics"""
        try:
            info = {
                "system_name": "RAG Evidence Indicator - New Dataset Version",
                "dataset_path": self.dataset_path,
                "chroma_path": self.chroma_path,
                "collection_name": self.collection_name,
                "documents_loaded": len(self.documents),
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "search_k": self.search_k,
                "embeddings_model": "text-embedding-ada-002",
                "status": "initialized" if self.vectorstore else "not_initialized"
            }
            
            if self.vectorstore:
                info["vectorstore_status"] = "active"
                info["collection_size"] = self.vectorstore._collection.count()
            else:
                info["vectorstore_status"] = "inactive"
            
            return info
            
        except Exception as e:
            return {"error": str(e), "status": "error"}

def main():
    """Main function to demonstrate the system"""
    
    # Initialize the system
    rag_system = RAGEvidenceIndicator(
        dataset_path="./data/ichikara-rag-sampleToMF-rebuilt.json",
        chroma_path="./chroma_new"
    )
    
    # Initialize the system
    if not rag_system.initialize_system():
        print("❌ Failed to initialize RAG system")
        return
    
    # Get system information
    print("\n📊 System Information:")
    print("=" * 40)
    system_info = rag_system.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Test search functionality
    print("\n🔍 Testing Search Functionality:")
    print("=" * 40)
    
    test_queries = [
        "上高地について",
        "長野県の観光",
        "日本の山岳地帯"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        evidence = rag_system.get_evidence(query, k=2)
        
        if "error" not in evidence:
            print(f"  Results: {evidence['results_count']}")
            print(f"  Processing time: {evidence['processing_time']:.4f}s")
            print(f"  Summary: {evidence['summary']}")
        else:
            print(f"  Error: {evidence['error']}")
    
    print("\n🎉 RAG Evidence Indicator System test completed!")

if __name__ == "__main__":
    main()
