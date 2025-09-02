#!/usr/bin/env python3
"""
Ichikara Dataset Integration for Evidence Indicator RAG System
Enhanced support for Japanese instruction-following datasets with rich metadata
"""

import os
import json
import time
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr

# Load environment variables
load_dotenv()

class IchikaraDatasetIntegrator:
    """Enhanced dataset integrator for Ichikara RAG datasets"""
    
    def __init__(self, chroma_path: str = "chroma"):
        self.chroma_path = chroma_path
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(self.api_key))
    
    def load_ichikara_dataset(self, file_path: str) -> List[Document]:
        """Load Ichikara dataset with enhanced metadata processing"""
        print(f"Loading Ichikara dataset from {file_path}...")
        
        # Load the JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        
        for item in data:
            # Create document from output (response) content
            output_doc = Document(
                page_content=item['output'],
                metadata={
                    'id': item['ID'],
                    'type': 'response',
                    'instruction': item['text'],
                    'references': item['meta'].get('output-reference', []),
                    'timestamp': item['meta'].get('output-reference', [{}])[0].get('timestamp', ''),
                    'misc_tags': item['meta'].get('misc', []),
                    'source_type': 'ichikara'
                }
            )
            documents.append(output_doc)
            
            # Create document from instruction (query) content
            instruction_doc = Document(
                page_content=item['text'],
                metadata={
                    'id': item['ID'],
                    'type': 'instruction',
                    'response': item['output'],
                    'references': item['meta'].get('output-reference', []),
                    'timestamp': item['meta'].get('output-reference', [{}])[0].get('timestamp', ''),
                    'misc_tags': item['meta'].get('misc', []),
                    'source_type': 'ichikara'
                }
            )
            documents.append(instruction_doc)
        
        print(f"Created {len(documents)} documents from Ichikara dataset")
        return documents
    
    def create_enhanced_chunks(self, documents: List[Document], 
                              chunk_size: int = 300, 
                              chunk_overlap: int = 100) -> List[Document]:
        """Create enhanced chunks with metadata preservation"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Enhance chunks with additional metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['chunk_size'] = len(chunk.page_content)
            chunk.metadata['dataset'] = 'ichikara'
        
        print(f"Created {len(chunks)} enhanced chunks")
        return chunks
    
    def save_to_chroma(self, chunks: List[Document], collection_name: str = "ichikara_collection"):
        """Save chunks to ChromaDB with collection management"""
        print(f"Saving {len(chunks)} chunks to ChromaDB...")
        
        # Create or get collection
        db = Chroma.from_documents(
            chunks,
            self.embedding_function,
            persist_directory=self.chroma_path,
            collection_name=collection_name
        )
        
        print(f"Successfully saved to ChromaDB collection: {collection_name}")
        return db
    
    def query_ichikara_dataset(self, query_text: str, collection_name: str = "ichikara_collection") -> Tuple[str, str, str, int, int]:
        """Query the Ichikara dataset with enhanced capabilities"""
        db = Chroma(persist_directory=self.chroma_path, 
                    embedding_function=self.embedding_function,
                    collection_name=collection_name)
        
        # Search with relevance scores
        search_results = db.similarity_search_with_relevance_scores(query_text, k=3)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        # Get best result
        best_doc = search_results[0][0]
        confidence = search_results[0][1]
        
        source_text = best_doc.page_content
        metadata = best_doc.metadata
        
        # Extract evidence
        evidence_text, start_pos, end_pos = self._extract_evidence_enhanced(source_text, query_text)
        
        # Generate answer with metadata context
        answer = self._generate_enhanced_answer(evidence_text, query_text, metadata)
        
        return answer, source_text, evidence_text, start_pos, end_pos
    
    def _extract_evidence_enhanced(self, text: str, query: str) -> Tuple[str, int, int]:
        """Enhanced evidence extraction for Japanese content"""
        # Japanese-specific text splitting
        sentences = text.split('。')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            snippet = text[:100]
            return snippet, 0, len(snippet)
        
        # Find best matching sentence
        best_sentence = max(sentences, key=lambda s: self._calculate_relevance_score(s, query))
        
        # Find position in original text
        start_pos = text.find(best_sentence)
        end_pos = start_pos + len(best_sentence)
        
        return best_sentence, start_pos, end_pos
    
    def _calculate_relevance_score(self, sentence: str, query: str) -> float:
        """Calculate relevance score for Japanese text"""
        score = 0.0
        
        # Keyword matching
        query_words = query.split()
        for word in query_words:
            if word in sentence:
                score += 1.0
        
        # Length penalty for very long sentences
        if len(sentence) > 200:
            score *= 0.8
        
        return score
    
    def _generate_enhanced_answer(self, evidence: str, query: str, metadata: Dict) -> str:
        """Generate enhanced answer using metadata context"""
        answer = f"【回答】\n{evidence}\n\n"
        
        # Add metadata information
        if metadata.get('references'):
            answer += f"【参考情報】\n"
            for ref in metadata['references']:
                if isinstance(ref, str):
                    answer += f"• {ref}\n"
                elif isinstance(ref, dict) and 'url' in ref:
                    answer += f"• {ref['url']}\n"
        
        if metadata.get('timestamp'):
            answer += f"【更新日時】\n{metadata['timestamp']}\n"
        
        return answer

def main():
    """Main integration function"""
    print("🚀 Starting Ichikara Dataset Integration...")
    
    # Initialize integrator
    integrator = IchikaraDatasetIntegrator()
    
    # Dataset path
    dataset_path = "./data/ichikara-rag-sampleToMF.json"
    
    try:
        # Load dataset
        documents = integrator.load_ichikara_dataset(dataset_path)
        
        # Create chunks
        chunks = integrator.create_enhanced_chunks(documents)
        
        # Save to ChromaDB
        db = integrator.save_to_chroma(chunks)
        
        print("✅ Ichikara dataset integration completed successfully!")
        
        # Test query
        test_query = "上高地について教えて"
        print(f"\n🧪 Testing query: {test_query}")
        
        answer, source, evidence, start, end = integrator.query_ichikara_dataset(test_query)
        print(f"Answer: {answer[:200]}...")
        
    except Exception as e:
        print(f"❌ Error during integration: {str(e)}")
        raise

if __name__ == "__main__":
    main()
