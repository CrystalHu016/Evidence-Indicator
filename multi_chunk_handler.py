#!/usr/bin/env python3
"""
Multi-Chunk Evidence Handler for Evidence Indicator RAG System
Implements progressive chunk analysis with evidence sufficiency evaluation
"""

import os
import re
import time
import openai
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr


@dataclass
class ChunkAnalysisResult:
    """Results from chunk analysis step"""
    step_number: int
    chunk_size: int
    chunk_overlap: int
    retrieved_chunks: List[Document]
    best_chunk: Document
    evidence_text: str
    start_char: int
    end_char: int
    sufficiency_score: float
    is_sufficient: bool
    reason: str


@dataclass
class MultiChunkResult:
    """Final result from multi-chunk analysis"""
    answer: str
    source_document: str
    evidence_text: str
    start_char: int
    end_char: int
    analysis_steps: List[ChunkAnalysisResult]
    final_step: int
    processing_time: float
    confidence: float


class EvidenceSufficiencyEvaluator:
    """Evaluates whether evidence is sufficient to answer a query"""
    
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        self.openai_client = openai.OpenAI(
            api_key=openai_api_key,
            timeout=30.0
        )
    
    def evaluate_sufficiency(self, query: str, evidence: str, chunk_context: str = "") -> Tuple[float, bool, str]:
        """
        Evaluate if evidence is sufficient to answer the query
        Returns: (sufficiency_score, is_sufficient, reason)
        """
        try:
            prompt = f"""
あなたは証拠充分性評価エキスパートです。以下の質問に対して、提供された証拠が回答に十分かどうかを評価してください。

質問: {query}
証拠: {evidence}
コンテキスト: {chunk_context}

評価基準:
1. 証拠が質問に直接関連しているか
2. 証拠が質問を完全に回答するのに十分な情報を含んでいるか
3. 証拠に矛盾や不明確な部分がないか

以下の形式で回答してください:
スコア: [0.0-1.0の数値]
十分性: [十分/不十分]
理由: [評価理由を簡潔に]
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたは証拠評価の専門家です。簡潔で正確な評価を提供してください。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1,
                timeout=30  # Add timeout to prevent hanging
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse the response
            score = 0.0
            is_sufficient = False
            reason = "評価に失敗しました"
            
            lines = result.split('\n')
            for line in lines:
                if 'スコア:' in line:
                    try:
                        score = float(re.search(r'(\d+\.?\d*)', line).group(1))
                    except:
                        pass
                elif '十分性:' in line:
                    is_sufficient = '十分' in line and '不十分' not in line
                elif '理由:' in line:
                    reason = line.replace('理由:', '').strip()
            
            return score, is_sufficient, reason
            
        except Exception as e:
            print(f"⚠️ Evidence evaluation failed: {e}")
            # Fallback to simple heuristic evaluation
            return self._heuristic_evaluation(query, evidence)
    
    def _heuristic_evaluation(self, query: str, evidence: str) -> Tuple[float, bool, str]:
        """Fallback heuristic evaluation"""
        if not evidence or len(evidence.strip()) < 10:
            return 0.0, False, "証拠が不足しています"
        
        # Extract key terms from query
        query_terms = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query)
        query_terms = [term for term in query_terms if len(term) > 1 and term not in ['とは', 'について', 'です', 'ます']]
        
        # Check term coverage in evidence
        covered_terms = sum(1 for term in query_terms if term in evidence)
        coverage_ratio = covered_terms / len(query_terms) if query_terms else 0
        
        # Length-based scoring
        length_score = min(len(evidence) / 100, 1.0)  # Normalize to 0-1
        
        # Combined score
        score = (coverage_ratio * 0.7 + length_score * 0.3)
        is_sufficient = score >= 0.6
        reason = f"用語カバー率: {coverage_ratio:.2f}, 長さスコア: {length_score:.2f}"
        
        return score, is_sufficient, reason


class MultiChunkAnalyzer:
    """Progressive multi-chunk analysis system"""
    
    def __init__(self, openai_api_key: str, chroma_path: str):
        self.api_key = openai_api_key
        self.chroma_path = chroma_path
        self.evaluator = EvidenceSufficiencyEvaluator(openai_api_key)
        
        # Initialize OpenAI client
        self.openai_client = openai.OpenAI(
            api_key=openai_api_key,
            timeout=30.0
        )
        
        # Initialize embeddings
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        
        # Progressive chunk size strategy
        self.chunk_strategies = [
            {"size": 800, "overlap": 200},   # Step 1: Large chunks
            {"size": 500, "overlap": 150},   # Step 2: Medium chunks  
            {"size": 300, "overlap": 100},   # Step 3: Small chunks (original)
            {"size": 200, "overlap": 80},    # Step 4: Very small chunks
        ]
    
    def analyze_with_progressive_chunking(self, query: str, source_documents: List[Document]) -> MultiChunkResult:
        """
        Perform progressive chunk analysis following the Step 1-i strategy
        """
        start_time = time.time()
        analysis_steps = []
        
        print(f"🔍 Starting progressive chunk analysis for query: {query}")
        
        for step_num, strategy in enumerate(self.chunk_strategies, 1):
            print(f"\n📊 Step {step_num}: Analyzing with chunk_size={strategy['size']}, overlap={strategy['overlap']}")
            
            # Create chunks with current strategy
            chunks = self._create_chunks_with_strategy(source_documents, strategy)
            
            # Analyze chunks at this level
            step_result = self._analyze_chunk_step(query, chunks, step_num, strategy)
            analysis_steps.append(step_result)
            
            print(f"   📈 Sufficiency score: {step_result.sufficiency_score:.2f}")
            print(f"   📝 Evidence: {step_result.evidence_text[:50]}...")
            
            # Check if evidence is sufficient
            if step_result.is_sufficient:
                print(f"   ✅ Evidence sufficient at Step {step_num}")
                break
            elif step_num < len(self.chunk_strategies):
                print(f"   ⚠️  Evidence insufficient, proceeding to Step {step_num + 1}")
            else:
                print(f"   ❌ Evidence insufficient even at final step")
        
        # Generate final answer based on best available evidence
        best_step = self._select_best_step(analysis_steps)
        final_answer = self._generate_comprehensive_answer(query, best_step, analysis_steps)
        
        processing_time = time.time() - start_time
        
        return MultiChunkResult(
            answer=final_answer,
            source_document=best_step.best_chunk.page_content,
            evidence_text=best_step.evidence_text,
            start_char=best_step.start_char,
            end_char=best_step.end_char,
            analysis_steps=analysis_steps,
            final_step=best_step.step_number,
            processing_time=processing_time,
            confidence=best_step.sufficiency_score
        )
    
    def _create_chunks_with_strategy(self, documents: List[Document], strategy: Dict[str, int]) -> List[Document]:
        """Create chunks using specified strategy"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["size"],
            chunk_overlap=strategy["overlap"],
            length_function=len,
            add_start_index=True,
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"   📦 Created {len(chunks)} chunks")
        return chunks
    
    def _analyze_chunk_step(self, query: str, chunks: List[Document], step_num: int, strategy: Dict[str, int]) -> ChunkAnalysisResult:
        """Analyze chunks at a specific step"""
        
        # Create temporary vector store for this chunk size
        temp_db = Chroma.from_documents(
            chunks, 
            self.embedding_function,
            persist_directory=None  # In-memory for temporary analysis
        )
        
        # Retrieve relevant chunks
        search_results = temp_db.similarity_search_with_relevance_scores(query, k=5)
        retrieved_chunks = [doc for doc, score in search_results]
        
        if not retrieved_chunks:
            return ChunkAnalysisResult(
                step_number=step_num,
                chunk_size=strategy["size"],
                chunk_overlap=strategy["overlap"],
                retrieved_chunks=[],
                best_chunk=Document(page_content="", metadata={}),
                evidence_text="",
                start_char=0,
                end_char=0,
                sufficiency_score=0.0,
                is_sufficient=False,
                reason="No relevant chunks found"
            )
        
        # Select best chunk using existing logic
        best_chunk = self._select_best_chunk(query, search_results)
        
        # Extract evidence from best chunk
        evidence_text, start_char, end_char = self._extract_evidence_from_chunk(
            best_chunk.page_content, query
        )
        
        # Evaluate evidence sufficiency
        chunk_context = self._get_surrounding_context(best_chunk, retrieved_chunks)
        sufficiency_score, is_sufficient, reason = self.evaluator.evaluate_sufficiency(
            query, evidence_text, chunk_context
        )
        
        return ChunkAnalysisResult(
            step_number=step_num,
            chunk_size=strategy["size"],
            chunk_overlap=strategy["overlap"],
            retrieved_chunks=retrieved_chunks,
            best_chunk=best_chunk,
            evidence_text=evidence_text,
            start_char=start_char,
            end_char=end_char,
            sufficiency_score=sufficiency_score,
            is_sufficient=is_sufficient,
            reason=reason
        )
    
    def _select_best_chunk(self, query: str, search_results: List[Tuple[Document, float]]) -> Document:
        """Select the best chunk from search results (reuse existing logic)"""
        if not search_results:
            return Document(page_content="", metadata={})
        
        # Simple scoring based on relevance score and content quality
        best_doc = None
        best_score = -1
        
        for doc, relevance_score in search_results:
            content = doc.page_content
            
            # Calculate content quality score
            query_terms = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query)
            term_matches = sum(1 for term in query_terms if term in content and len(term) > 1)
            
            # Combine relevance and term matching
            combined_score = relevance_score * 0.7 + (term_matches / len(query_terms) if query_terms else 0) * 0.3
            
            if combined_score > best_score:
                best_score = combined_score
                best_doc = doc
        
        return best_doc or search_results[0][0]
    
    def _extract_evidence_from_chunk(self, chunk_content: str, query: str) -> Tuple[str, int, int]:
        """Extract evidence from chunk content"""
        # Use sentence-based extraction similar to existing logic
        sentences = re.split(r'[。！？.!?]', chunk_content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return chunk_content[:100], 0, min(100, len(chunk_content))
        
        # Score sentences based on query relevance
        query_terms = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query)
        query_terms = [term for term in query_terms if len(term) > 1]
        
        best_sentence = sentences[0]
        best_score = 0
        
        for sentence in sentences:
            score = sum(1 for term in query_terms if term in sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        
        # Find position in original content
        start_pos = chunk_content.find(best_sentence)
        if start_pos < 0:
            start_pos = 0
            best_sentence = chunk_content[:100]
        
        end_pos = start_pos + len(best_sentence)
        
        return best_sentence, start_pos, end_pos
    
    def _get_surrounding_context(self, best_chunk: Document, all_chunks: List[Document]) -> str:
        """Get surrounding context from related chunks"""
        context_parts = []
        best_content = best_chunk.page_content
        
        for chunk in all_chunks[:3]:  # Use top 3 chunks for context
            if chunk.page_content != best_content:
                context_parts.append(chunk.page_content[:100])
        
        return " | ".join(context_parts)
    
    def _select_best_step(self, analysis_steps: List[ChunkAnalysisResult]) -> ChunkAnalysisResult:
        """Select the best analysis step based on sufficiency and quality"""
        # Prefer sufficient evidence from earlier (larger) steps
        for step in analysis_steps:
            if step.is_sufficient:
                return step
        
        # If no sufficient evidence, return the one with highest score
        return max(analysis_steps, key=lambda x: x.sufficiency_score)
    
    def _generate_comprehensive_answer(self, query: str, best_step: ChunkAnalysisResult, all_steps: List[ChunkAnalysisResult]) -> str:
        """Generate comprehensive answer using the best available evidence"""
        if not best_step.evidence_text:
            return "申し訳ありませんが、適切な回答を見つけることができませんでした。"
        
        # For definition questions, return the evidence directly
        if any(pattern in query for pattern in ['とは何', 'とは', '何ですか', '何でしょうか']):
            return best_step.evidence_text
        
        # For complex questions, try to use LLM for better answer generation
        try:
            # Collect evidence from multiple steps for richer context
            all_evidence = []
            for step in all_steps:
                if step.evidence_text and step.evidence_text not in all_evidence:
                    all_evidence.append(step.evidence_text)
            
            combined_evidence = " ".join(all_evidence[:3])  # Use top 3 evidence pieces
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "以下の証拠を基に、質問に対して簡潔で正確な日本語回答を提供してください。"},
                    {"role": "user", "content": f"証拠: {combined_evidence}\n\n質問: {query}\n\n回答:"}
                ],
                max_tokens=150,
                temperature=0.1,
                timeout=30  # Add timeout to prevent hanging
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"⚠️ LLM answer generation failed: {e}")
            return best_step.evidence_text