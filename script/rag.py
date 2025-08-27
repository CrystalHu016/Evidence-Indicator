#!/usr/bin/env python3
"""
Evidence Indicator RAG System - Implementation
Ultra-fast RAG system with Japanese format support and precise character positioning
"""

import os
import time
import shutil
from dotenv import load_dotenv
from typing import List
import argparse

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr

from ultra_fast_rag import UltraFastRAG

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma")
DATA_PATH = os.environ.get("DATA_PATH", "./data/single_20240229.json")
TEST_DATA_PATH = os.environ.get("TEST_DATA_PATH", "./data/test_sample.json")

def generate_data_store(use_test_data=False):
    """Generate and save the vector data store from documents."""
    start = time.time()
    documents = load_documents(use_test_data)
    print(f"Loaded documents in {time.time() - start:.2f} seconds")
    
    start = time.time()
    chunks = split_text(documents)
    print(f"Split text in {time.time() - start:.2f} seconds")
    
    start = time.time()
    save_to_chroma(chunks)
    print(f"Saved to Chroma in {time.time() - start:.2f} seconds")

def load_documents(use_test_data=False):
    """Load documents from JSON file."""
    file_path = TEST_DATA_PATH if use_test_data else DATA_PATH
    loader = JSONLoader(
        file_path=file_path,
        jq_schema='.[ ]',  # Each object in the top-level array
        content_key='output'  # Extract content from 'output' field
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents from {file_path}")
    return documents

def split_text(documents: List[Document]):
    """Split documents into chunks for vector storage."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks: List[Document]):
    """Save document chunks to ChromaDB vector store."""
    print(f"Starting Chroma save for {len(chunks)} chunks...")
    
    # Remove existing directory for fresh start
    if os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    # Get OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    # Create embeddings and vector store
    embedding_function = OpenAIEmbeddings(api_key=SecretStr(api_key))
    db = Chroma.from_documents(
        chunks, 
        embedding_function, 
        persist_directory=CHROMA_PATH
    )
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def query_data(query_text: str):
    """
    Query the RAG system using UltraFastRAG.
    Returns: (answer, source_document, evidence_text, start_char, end_char)
    """
    # More robust check: ensure the ChromaDB SQLite file exists.
    # This prevents errors if the directory exists but is empty or incomplete.
    chroma_db_file = os.path.join(CHROMA_PATH, "chroma.sqlite3")
    if not os.path.exists(chroma_db_file):
        raise FileNotFoundError(
            f"❌ Vector store is missing or invalid in '{CHROMA_PATH}'.\n"
            "Please run the 'generate' command to create it."
        )

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    
    # Use UltraFastRAG system (75% performance improvement)
    ultra_fast_rag = UltraFastRAG(api_key, CHROMA_PATH)
    answer, source_document, evidence_text, start_char, end_char = ultra_fast_rag.query(query_text)
    
    return answer, source_document, evidence_text, start_char, end_char

def print_results(answer, source_document, evidence_text, start_char, end_char, processing_time=None):
    """Helper function to print query results in a consistent format."""
    if processing_time is not None:
        print(f"\n⚡ Processing time: {processing_time:.2f} seconds")
    
    print("\n【回答】")
    print(answer)
    
    if source_document and evidence_text:
        print(f"\n【検索ヒットのチャンクを含む文書】")
        print(source_document)
        print(f"\n【根拠情報の文字列範囲】{start_char + 1}文字目～{end_char}文字目")
        print(f"\n【根拠情報】")
        print(evidence_text)
    else:
        print("\nNo relevant sources found.")

def main_default_flow(use_test_data=False):
    """The default flow: generate data store and start interactive mode."""
    print("--- Generating Data Store ---")
    generate_data_store(use_test_data=use_test_data)
    print("--- Data Store Generation Complete ---")
    interactive_query_loop()

def interactive_query_loop():
    """Interactive query loop for real-time testing."""
    print("\n--- Evidence Indicator RAG - Interactive Mode ---")
    print("Type your query and press Enter. Type 'exit' to quit.\n")
    
    while True:
        query = input("Your query: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting interactive query mode.")
            break
        if not query:
            continue
            
        try:
            start_time = time.time()
            answer, source_document, evidence_text, start_char, end_char = query_data(query)
            processing_time = time.time() - start_time
            print_results(answer, source_document, evidence_text, start_char, end_char, processing_time)
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="Evidence Indicator RAG System")
    parser.add_argument("action", nargs='?', default=None,
                        choices=["generate", "query", "interactive"],
                        help="Action to perform. If no action is specified, "
                             "the script will generate the data store and then "
                             "start interactive mode.")
    parser.add_argument("--query", type=str, help="Query text to search for.")
    parser.add_argument("--test", action="store_true", help="Use test data for faster development.")
    
    args = parser.parse_args()
    
    if args.action == "generate":
        print("--- Generating Data Store ---")
        generate_data_store(use_test_data=args.test)
        print("--- Data Store Generation Complete ---")
        
    elif args.action == "query":
        if not args.query:
            print("Error: --query argument is required for the 'query' action.")
            return
            
        try:
            answer, source, evidence, start, end = query_data(args.query)
            print_results(answer, source, evidence, start, end)
        except Exception as e:
            print(f"Error: {e}")
            
    elif args.action == "interactive":
        interactive_query_loop()
        
    elif args.action is None:
        # Default behavior when no action is specified
        main_default_flow(use_test_data=args.test)

if __name__ == "__main__":
    main()