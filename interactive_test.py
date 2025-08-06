#!/usr/bin/env python3
"""
Interactive Testing Script for Evidence Indicator RAG System
Simple interactive prompt for testing queries
"""

import os
import sys
import time
from rag import query_data

def main():
    """Interactive testing interface"""
    print("🎯 Evidence Indicator RAG System - Interactive Testing")
    print("=" * 60)
    print("Type your queries and see the complete Japanese format output.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'help' for usage examples.")
    print("=" * 60)
    
    while True:
        try:
            # Get user input
            query = input("\n🔍 Your query: ").strip()
            
            # Handle exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("👋 Goodbye! Thanks for testing the Evidence Indicator RAG system.")
                break
            
            # Handle help command
            if query.lower() == 'help':
                print("\n📚 Example queries:")
                print("  • コンバインとは何ですか")
                print("  • 音位転倒について説明してください")
                print("  • どのような農業機械がありますか")
                print("  • What is a combine harvester?")
                print("  • コンバイン")
                print("  • 音位転倒")
                continue
            
            # Skip empty queries
            if not query:
                continue
            
            # Process the query
            print(f"\n⚡ Processing: {query}")
            start_time = time.time()
            
            try:
                answer, source_document, evidence_text, start_char, end_char = query_data(query)
                processing_time = time.time() - start_time
                
                # Display results in the required format
                print(f"\n⏱️  Processing time: {processing_time:.2f} seconds")
                print("\n" + "="*60)
                print("【回答】")
                print(answer)
                
                if source_document and evidence_text:
                    print(f"\n【検索ヒットのチャンクを含む文書】")
                    print(source_document)
                    print(f"\n【根拠情報の文字列範囲】")
                    print(f"{start_char + 1}文字目〜{end_char}文字目")
                    print(f"\n【根拠情報】")
                    print(evidence_text)
                else:
                    print("\n❌ No relevant sources found.")
                
                print("="*60)
                
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("💡 Make sure the data store is generated first with: python3 rag.py generate --test")
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 End of input. Goodbye!")
            break

if __name__ == "__main__":
    main() 