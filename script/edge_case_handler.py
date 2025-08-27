import os
import time
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import SecretStr
import openai

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class ChunkInfo:
    """Information about a chunk including its content and metadata"""
    content: str
    metadata: Dict[str, Any]
    score: float
    chunk_id: str
    start_char: int = 0
    end_char: int = 0

@dataclass
class EvidenceResult:
    """Result of evidence extraction from multiple chunks"""
    evidence_text: str
    source_chunks: List[ChunkInfo]
    confidence_score: float
    strategy_used: str
    is_complete: bool

class EdgeCaseHandler:
    """Handles edge cases for RAG systems, particularly multi-chunk information retrieval"""
    
    def __init__(self, openai_api_key: str, chroma_path: str):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        
    def strategy_1_full_document_context(self, query: str, top_k: int = 5) -> EvidenceResult:
        """
        Strategy 1: Retrieve the full document context containing the search hits
        """
        print("--- Using Strategy 1: Full Document Context ---")
        
        # Get top chunks
        results = self.db.similarity_search_with_relevance_scores(query, k=top_k)
        
        if not results:
            return EvidenceResult("", [], 0.0, "strategy_1", False)
        
        # Group chunks by source document
        document_chunks = {}
        for doc, score in results:
            source = doc.metadata.get('source', 'unknown')
            if source not in document_chunks:
                document_chunks[source] = []
            document_chunks[source].append(ChunkInfo(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                chunk_id=doc.metadata.get('chunk_id', 'unknown')
            ))
        
        # Find the document with the most relevant chunks
        best_source = max(document_chunks.keys(), 
                         key=lambda x: sum(chunk.score for chunk in document_chunks[x]))
        
        # Get all chunks from the best source document
        all_chunks_from_source = self.db.similarity_search_with_relevance_scores(
            query, k=50  # Get more chunks to find all from the same document
        )
        
        source_chunks = []
        full_context = ""
        for doc, score in all_chunks_from_source:
            if doc.metadata.get('source') == best_source:
                source_chunks.append(ChunkInfo(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    chunk_id=doc.metadata.get('chunk_id', 'unknown')
                ))
                full_context += doc.page_content + "\n\n"
        
        # Use LLM to extract relevant evidence from full context
        evidence_text = self._extract_evidence_from_context(query, full_context)
        
        return EvidenceResult(
            evidence_text=evidence_text,
            source_chunks=source_chunks,
            confidence_score=sum(chunk.score for chunk in source_chunks) / len(source_chunks),
            strategy_used="strategy_1_full_document_context",
            is_complete=len(evidence_text) > 0
        )
    
    def strategy_2_adaptive_chunking(self, query: str, max_steps: int = 3) -> EvidenceResult:
        """
        Strategy 2: Adaptive chunking - start with large chunks and refine if needed
        """
        print("--- Using Strategy 2: Adaptive Chunking ---")
        
        # Step 1: Try with current chunking
        results = self.db.similarity_search_with_relevance_scores(query, k=5)
        if not results:
            return EvidenceResult("", [], 0.0, "strategy_2", False)
        
        # Check if current chunks can answer the question
        current_context = "\n\n".join([doc.page_content for doc, _ in results])
        can_answer = self._check_if_context_can_answer(query, current_context)
        
        if can_answer:
            evidence_text = self._extract_evidence_from_context(query, current_context)
            return EvidenceResult(
                evidence_text=evidence_text,
                source_chunks=[ChunkInfo(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    chunk_id=doc.metadata.get('chunk_id', 'unknown')
                ) for doc, score in results],
                confidence_score=sum(score for _, score in results) / len(results),
                strategy_used="strategy_2_adaptive_chunking",
                is_complete=True
            )
        
        # If not sufficient, try with larger context
        print("Current chunks insufficient, trying larger context...")
        larger_results = self.db.similarity_search_with_relevance_scores(query, k=10)
        larger_context = "\n\n".join([doc.page_content for doc, _ in larger_results])
        
        evidence_text = self._extract_evidence_from_context(query, larger_context)
        
        return EvidenceResult(
            evidence_text=evidence_text,
            source_chunks=[ChunkInfo(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                chunk_id=doc.metadata.get('chunk_id', 'unknown')
            ) for doc, score in larger_results],
            confidence_score=sum(score for _, score in larger_results) / len(larger_results),
            strategy_used="strategy_2_adaptive_chunking",
            is_complete=len(evidence_text) > 0
        )
    
    def strategy_3_multi_chunk_aggregation(self, query: str, top_k: int = 5) -> EvidenceResult:
        """
        Strategy 3: Aggregate information from multiple high-similarity chunks
        """
        print("--- Using Strategy 3: Multi-Chunk Aggregation ---")
        
        # Get top chunks with scores
        results = self.db.similarity_search_with_relevance_scores(query, k=top_k)
        
        if not results:
            return EvidenceResult("", [], 0.0, "strategy_3", False)
        
        # Filter chunks with high similarity scores
        high_similarity_threshold = 0.7
        high_similarity_chunks = [
            ChunkInfo(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
                chunk_id=doc.metadata.get('chunk_id', 'unknown')
            )
            for doc, score in results if score >= high_similarity_threshold
        ]
        
        # If no high similarity chunks, use all results
        if not high_similarity_chunks:
            high_similarity_chunks = [
                ChunkInfo(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score,
                    chunk_id=doc.metadata.get('chunk_id', 'unknown')
                )
                for doc, score in results
            ]
        
        # Combine all relevant chunks
        combined_context = "\n\n".join([chunk.content for chunk in high_similarity_chunks])
        
        # Extract evidence from combined context
        evidence_text = self._extract_evidence_from_context(query, combined_context)
        
        return EvidenceResult(
            evidence_text=evidence_text,
            source_chunks=high_similarity_chunks,
            confidence_score=sum(chunk.score for chunk in high_similarity_chunks) / len(high_similarity_chunks),
            strategy_used="strategy_3_multi_chunk_aggregation",
            is_complete=len(evidence_text) > 0
        )
    
    def _check_if_context_can_answer(self, query: str, context: str) -> bool:
        """Use LLM to check if the context can answer the query"""
        prompt = f"""
        Given the following context and question, determine if the context contains enough information to answer the question.
        
        Context: {context}
        Question: {query}
        
        Respond with only 'YES' if the context can answer the question, or 'NO' if it cannot.
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that determines if context can answer questions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10
            )
            return "YES" in response.choices[0].message.content.upper()
        except Exception as e:
            print(f"Error checking context: {e}")
            return False
    
    def _extract_evidence_from_context(self, query: str, context: str) -> str:
        """Extract relevant evidence from context using LLM"""
        prompt = f"""
        Given the following context and question, extract the most relevant evidence that answers the question.
        Focus on the specific information needed to answer the question accurately.
        DO NOT include the question itself in the evidence.
        DO NOT repeat the question text.
        Extract only the factual information that answers the question.
        
        IMPORTANT: If the question is asked in Japanese, respond entirely in Japanese. If the question is asked in English, respond entirely in English. Do not mix languages in your response.
        
        Context: {context}
        Question: {query}
        
        Extract and return only the relevant evidence (without repeating the question):
        """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting relevant evidence from text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error extracting evidence: {e}")
            return context[:500]  # Fallback to first 500 chars
    
    def handle_edge_case(self, query: str, strategy: str = "auto") -> EvidenceResult:
        """
        Main method to handle edge cases using the specified strategy
        Automatically selects the highest accuracy strategy when using "auto"
        """
        if strategy == "auto":
            # Try all strategies and select the one with highest confidence
            strategies = [
                ("strategy_1", self.strategy_1_full_document_context),
                ("strategy_2", self.strategy_2_adaptive_chunking),
                ("strategy_3", self.strategy_3_multi_chunk_aggregation)
            ]
            
            best_result = None
            best_score = 0.0
            best_strategy = None
            
            for strategy_name, strategy_func in strategies:
                try:
                    result = strategy_func(query)
                    if result.is_complete and result.confidence_score > best_score:
                        best_result = result
                        best_score = result.confidence_score
                        best_strategy = strategy_name
                except Exception as e:
                    print(f"Strategy {strategy_name} failed: {e}")
                    continue
            
            if best_result and best_score > 0.3:  # Minimum threshold
                print(f"Selected {best_strategy} with confidence score: {best_score:.3f}")
                return best_result
            elif best_result:
                print(f"Using best available result from {best_strategy} with confidence score: {best_score:.3f}")
                return best_result
            else:
                # Fallback to basic strategy if all fail
                print("All strategies failed, using fallback")
                return EvidenceResult("", [], 0.0, "fallback", False)
        
        elif strategy == "strategy_1":
            return self.strategy_1_full_document_context(query)
        elif strategy == "strategy_2":
            return self.strategy_2_adaptive_chunking(query)
        elif strategy == "strategy_3":
            return self.strategy_3_multi_chunk_aggregation(query)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

def create_test_scenario():
    """Create a test scenario to demonstrate the edge case"""
    test_document = """
    A社の売上がB社より高い、その差が1000万円。B社の売り上げがC社より高い、その差が500万円。A社の売上が5000万円でした。
    """
    
    # This would be split into chunks like:
    chunks = [
        "A社の売上がB社より高い",
        "その差が1000万円",
        "B社の売上がC社より高い", 
        "その差が500万円",
        "A社の売上が5000万円でした"
    ]
    
    test_query = "ABC 3社の売上が最も高いのはどれですか？"
    
    return test_document, chunks, test_query 