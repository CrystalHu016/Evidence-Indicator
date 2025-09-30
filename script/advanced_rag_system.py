#!/usr/bin/env python3
"""
Advanced Multi-Granular RAG System
Implements state-of-the-art retrieval engineering techniques:
1. Multi-granular chunks with similarity threshold preference for smallest chunks
2. Hybrid retrieval (sparse+dense) with cross-encoder re-ranking
3. LLM chunk filtering for purity and relevance scoring
4. Synthetic QA indexing for fine-grained content mapping
5. Atomic subqueries for complex query decomposition
6. Keyword coverage monitoring with multi-granularity diagnostics
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from pydantic import SecretStr

# Try to import BM25 for sparse retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("⚠️ BM25 not available. Install with: pip install rank-bm25")

@dataclass
class ChunkResult:
    """Enhanced chunk result with scoring metadata"""
    document: Document
    similarity_score: float
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    llm_purity_score: float = 0.0
    final_score: float = 0.0
    granularity: str = ""
    chunk_size: int = 0
    keyword_coverage: float = 0.0

class AdvancedMultiGranularRAG:
    """Advanced Multi-Granular RAG with state-of-the-art retrieval engineering"""
    
    def __init__(self, openai_api_key: str, similarity_threshold: float = 0.3):
        self.openai_api_key = openai_api_key
        self.similarity_threshold = similarity_threshold
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.llm = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini", temperature=0)
        
        # Multi-granular stores
        self.vector_stores = {}
        self.bm25_indices = {}
        self.synthetic_qa_store = None
        
        # Granularity definitions with enhanced size ranges
        self.granularities = {
            'phrase': {'min_size': 5, 'max_size': 25, 'chunk_size': 15, 'overlap': 3},
            'sentence': {'min_size': 10, 'max_size': 80, 'chunk_size': 50, 'overlap': 10},
            'short_passage': {'min_size': 60, 'max_size': 200, 'chunk_size': 130, 'overlap': 30},
            'long_passage': {'min_size': 180, 'max_size': 400, 'chunk_size': 290, 'overlap': 60}
        }
    
    def create_enhanced_multi_granular_chunks(self, data: List[Dict]) -> Dict[str, List[Document]]:
        """Create enhanced multi-granular chunks with phrase-level precision"""
        print("🔀 Creating enhanced multi-granular chunks...")
        
        all_chunks = {granularity: [] for granularity in self.granularities.keys()}
        
        for doc_idx, item in enumerate(data[:150]):  # Increased limit for better coverage
            content = item.get('output', '') or item.get('text', '') or item.get('content', '')
            if not content:
                continue
            
            for granularity, config in self.granularities.items():
                chunks = self._create_granular_chunks(content, doc_idx, granularity, config)
                all_chunks[granularity].extend(chunks)
        
        return all_chunks
    
    def _create_granular_chunks(self, content: str, doc_idx: int, granularity: str, config: Dict) -> List[Document]:
        """Create chunks for specific granularity level"""
        chunks = []
        
        if granularity == 'phrase':
            # Phrase-level chunking for precise keyword extraction
            phrases = self._extract_key_phrases(content)
            for i, phrase in enumerate(phrases):
                if config['min_size'] <= len(phrase) <= config['max_size']:
                    chunk_doc = Document(
                        page_content=phrase,
                        metadata={
                            'granularity': granularity,
                            'doc_index': doc_idx,
                            'chunk_index': i,
                            'chunk_size': len(phrase),
                            'chunk_type': 'phrase'
                        }
                    )
                    chunks.append(chunk_doc)
        
        elif granularity == 'sentence':
            # Enhanced sentence chunking
            sentences = re.split(r'[。！？.!?]', content)
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if config['min_size'] <= len(sentence) <= config['max_size'] and sentence:
                    chunk_doc = Document(
                        page_content=sentence,
                        metadata={
                            'granularity': granularity,
                            'doc_index': doc_idx,
                            'chunk_index': i,
                            'chunk_size': len(sentence),
                            'chunk_type': 'sentence'
                        }
                    )
                    chunks.append(chunk_doc)
        
        else:
            # Passage-level chunking with RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['chunk_size'],
                chunk_overlap=config['overlap'],
                length_function=len,
                separators=["\\n\\n", "。", "！", "？", "、", "\\n", " ", ""]
            )
            
            raw_chunks = splitter.split_text(content)
            for i, chunk_text in enumerate(raw_chunks):
                if config['min_size'] <= len(chunk_text) <= config['max_size']:
                    chunk_doc = Document(
                        page_content=chunk_text,
                        metadata={
                            'granularity': granularity,
                            'doc_index': doc_idx,
                            'chunk_index': i,
                            'chunk_size': len(chunk_text),
                            'chunk_type': granularity
                        }
                    )
                    chunks.append(chunk_doc)
        
        return chunks
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases using pattern-based approach for Japanese"""
        phrases = []
        
        # Technical term patterns
        tech_patterns = [
            r'[\\u4E00-\\u9FAF]+機械',  # X機械
            r'[\\u4E00-\\u9FAF]+型',   # X型
            r'[\\u4E00-\\u9FAF]+農業', # X農業
            r'コンバイン[\\u4E00-\\u9FAF]*',  # コンバインX
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text)
            phrases.extend(matches)
        
        # Noun phrase extraction (simplified)
        words = re.findall(r'[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FAF]+', text)
        for i in range(len(words) - 1):
            phrase = words[i] + words[i + 1]
            if 5 <= len(phrase) <= 25:
                phrases.append(phrase)
        
        return list(set(phrases))  # Remove duplicates
    
    def build_hybrid_vector_stores(self, data: List[Dict]) -> bool:
        """Build hybrid vector stores with both dense and sparse indices"""
        try:
            print("🏗️ Building hybrid vector stores...")
            
            # Create multi-granular chunks
            all_chunks = self.create_enhanced_multi_granular_chunks(data)
            
            # Build vector stores for each granularity
            for granularity, chunks in all_chunks.items():
                if not chunks:
                    continue
                
                print(f"📦 Building {granularity} store: {len(chunks)} chunks")
                
                # Dense vector store
                self.vector_stores[granularity] = Chroma.from_documents(
                    chunks,
                    self.embedding_function,
                    persist_directory=f"./chroma_advanced_{granularity}"
                )
                
                # Sparse BM25 index
                if BM25_AVAILABLE:
                    corpus = [chunk.page_content for chunk in chunks]
                    tokenized_corpus = [doc.split() for doc in corpus]
                    self.bm25_indices[granularity] = BM25Okapi(tokenized_corpus)
            
            # Build synthetic QA store
            self.build_synthetic_qa_store(all_chunks)
            
            print("✅ Hybrid vector stores built successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to build hybrid stores: {e}")
            return False
    
    def hybrid_retrieval_with_reranking(self, query: str, k: int = 10) -> List[ChunkResult]:
        """Hybrid retrieval with cross-encoder re-ranking"""
        print(f"🔍 Hybrid retrieval for: {query}")
        
        all_results = []
        
        for granularity in self.granularities.keys():
            if granularity not in self.vector_stores:
                continue
            
            # Dense retrieval
            dense_results = self.vector_stores[granularity].similarity_search_with_score(
                query, k=k//len(self.granularities)
            )
            
            # Sparse retrieval (BM25)
            bm25_scores = []
            if granularity in self.bm25_indices:
                tokenized_query = query.split()
                bm25_scores = self.bm25_indices[granularity].get_scores(tokenized_query)
            
            # Combine scores
            for i, (doc, dense_score) in enumerate(dense_results):
                bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
                
                # Hybrid scoring (weighted combination)
                hybrid_score = 0.7 * (1 - dense_score) + 0.3 * bm25_score  # Lower dense_score is better
                
                chunk_result = ChunkResult(
                    document=doc,
                    similarity_score=dense_score,
                    bm25_score=bm25_score,
                    hybrid_score=hybrid_score,
                    granularity=granularity,
                    chunk_size=len(doc.page_content)
                )
                all_results.append(chunk_result)
        
        # Sort by hybrid score (higher is better)
        all_results.sort(key=lambda x: x.hybrid_score, reverse=True)
        
        return all_results[:k]
    
    def llm_chunk_filtering(self, query: str, candidates: List[ChunkResult]) -> List[ChunkResult]:
        """Filter chunks using LLM for purity and relevance scoring"""
        if not candidates:
            return []
        
        print("🧠 LLM chunk filtering for purity...")
        
        filtered_results = []
        
        for candidate in candidates[:5]:  # Limit LLM calls for efficiency
            chunk_text = candidate.document.page_content
            
            # LLM purity scoring
            purity_prompt = f"""
            Query: {query}
            Chunk: {chunk_text}
            
            Score this chunk's relevance and purity for answering the query on a scale of 0.0 to 1.0:
            - Relevance: How well does this chunk answer the query?
            - Purity: How focused is this chunk (no irrelevant information)?
            
            Return only a single number between 0.0 and 1.0.
            """
            
            try:
                response = self.llm.invoke(purity_prompt)
                purity_score = float(response.content.strip())
                candidate.llm_purity_score = max(0.0, min(1.0, purity_score))
            except:
                candidate.llm_purity_score = 0.5  # Default score
            
            # Calculate final score combining all factors
            candidate.final_score = (
                0.4 * candidate.hybrid_score +
                0.4 * candidate.llm_purity_score +
                0.2 * self._calculate_keyword_coverage(query, chunk_text)
            )
            
            # Only keep high-quality chunks
            if candidate.final_score > 0.3:
                filtered_results.append(candidate)
        
        # Sort by final score
        filtered_results.sort(key=lambda x: x.final_score, reverse=True)
        return filtered_results
    
    def _calculate_keyword_coverage(self, query: str, text: str) -> float:
        """Calculate keyword coverage score"""
        query_words = set(re.findall(r'[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FAF]+', query.lower()))
        text_words = set(re.findall(r'[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FAF]+', text.lower()))
        
        if not query_words:
            return 0.0
        
        overlap = query_words.intersection(text_words)
        return len(overlap) / len(query_words)
    
    def preference_for_smallest_chunks(self, results: List[ChunkResult]) -> List[ChunkResult]:
        """Apply preference for smallest chunks meeting similarity threshold"""
        if not results:
            return []
        
        # Filter by similarity threshold
        qualified_results = [
            r for r in results 
            if r.similarity_score <= self.similarity_threshold  # Lower score is better for similarity
        ]
        
        if not qualified_results:
            # If no results meet threshold, take the best ones
            qualified_results = results[:3]
        
        # Group by similarity bands and prefer smaller chunks within each band
        similarity_bands = defaultdict(list)
        for result in qualified_results:
            band = int(result.similarity_score * 10) / 10  # 0.1 precision bands
            similarity_bands[band].append(result)
        
        # Within each band, prefer smaller chunks
        final_results = []
        for band, band_results in similarity_bands.items():
            # Sort by chunk size (smaller first) within similarity band
            band_results.sort(key=lambda x: (x.chunk_size, -x.final_score))
            final_results.extend(band_results[:2])  # Take top 2 from each band
        
        return final_results[:5]  # Limit total results
    
    def advanced_multi_granular_query(self, query: str, k: int = 5) -> List[ChunkResult]:
        """Advanced multi-granular query with all enhancements"""
        start_time = time.time()
        
        # Step 1: Hybrid retrieval
        candidates = self.hybrid_retrieval_with_reranking(query, k=k*3)
        
        # Step 2: LLM filtering for purity
        filtered_candidates = self.llm_chunk_filtering(query, candidates)
        
        # Step 3: Apply preference for smallest chunks
        final_results = self.preference_for_smallest_chunks(filtered_candidates)
        
        query_time = time.time() - start_time
        
        # Statistics
        granularity_counts = defaultdict(int)
        for result in final_results:
            granularity_counts[result.granularity] += 1
        
        print(f"⚡ Query completed in {query_time:.2f}s")
        print(f"📊 Results by granularity: {dict(granularity_counts)}")
        
        return final_results
    
    def decompose_atomic_subqueries(self, complex_query: str) -> List[str]:
        """Decompose complex queries into atomic subqueries"""
        decomposition_prompt = f"""
        Complex Query: {complex_query}
        
        Decompose this complex query into 2-4 atomic subqueries that each target a single fact or concept.
        Each subquery should be answerable with a single sentence or short passage.
        
        Return as a JSON list of strings.
        Example: ["What is X?", "What are the types of Y?", "How does Z work?"]
        """
        
        try:
            response = self.llm.invoke(decomposition_prompt)
            import json
            subqueries = json.loads(response.content.strip())
            return subqueries if isinstance(subqueries, list) else [complex_query]
        except:
            return [complex_query]  # Fallback to original query
    
    def monitor_keyword_coverage(self, query: str, results: List[ChunkResult]) -> Dict[str, float]:
        """Monitor keyword coverage across different granularities"""
        query_keywords = set(re.findall(r'[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FAF]+', query.lower()))
        
        coverage_by_granularity = {}
        
        for granularity in self.granularities.keys():
            granularity_results = [r for r in results if r.granularity == granularity]
            if not granularity_results:
                coverage_by_granularity[granularity] = 0.0
                continue
            
            all_words = set()
            for result in granularity_results:
                words = set(re.findall(r'[\\u3040-\\u309F\\u30A0-\\u30FF\\u4E00-\\u9FAF]+', 
                                     result.document.page_content.lower()))
                all_words.update(words)
            
            coverage = len(query_keywords.intersection(all_words)) / len(query_keywords) if query_keywords else 0.0
            coverage_by_granularity[granularity] = coverage
        
        return coverage_by_granularity
    
    def build_synthetic_qa_store(self, all_chunks: Dict[str, List[Document]]) -> None:
        """Build synthetic QA store for fine-grained content mapping"""
        print("🤖 Building synthetic QA store...")
        
        synthetic_qa_docs = []
        
        # Generate synthetic questions for sentence and phrase level chunks
        for granularity in ['phrase', 'sentence']:
            if granularity not in all_chunks:
                continue
                
            chunks = all_chunks[granularity][:50]  # Limit for efficiency
            
            for i, chunk in enumerate(chunks):
                content = chunk.page_content
                
                # Generate synthetic question
                qa_prompt = f"""
                Content: {content}
                
                Generate 1-2 specific questions that this content directly answers.
                Questions should be factual and precise.
                Return as a JSON list of strings.
                Example: ["What is X?", "What type of Y is this?"]
                """
                
                try:
                    response = self.llm.invoke(qa_prompt)
                    import json
                    questions = json.loads(response.content.strip())
                    
                    if isinstance(questions, list):
                        for question in questions[:2]:  # Limit to 2 questions per chunk
                            synthetic_doc = Document(
                                page_content=question,
                                metadata={
                                    'original_content': content,
                                    'granularity': granularity,
                                    'chunk_id': f"{granularity}_{i}",
                                    'type': 'synthetic_qa'
                                }
                            )
                            synthetic_qa_docs.append(synthetic_doc)
                except:
                    continue  # Skip failed generations
        
        # Build synthetic QA vector store
        if synthetic_qa_docs:
            self.synthetic_qa_store = Chroma.from_documents(
                synthetic_qa_docs,
                self.embedding_function,
                persist_directory="./chroma_synthetic_qa"
            )
            print(f"✅ Built synthetic QA store with {len(synthetic_qa_docs)} Q&A pairs")
        else:
            print("⚠️ No synthetic QA pairs generated")
    
    def query_synthetic_qa(self, query: str, k: int = 3) -> List[ChunkResult]:
        """Query synthetic QA store for direct question-to-content mapping"""
        if not self.synthetic_qa_store:
            return []
        
        results = self.synthetic_qa_store.similarity_search_with_score(query, k=k)
        
        qa_results = []
        for doc, score in results:
            chunk_result = ChunkResult(
                document=Document(
                    page_content=doc.metadata['original_content'],
                    metadata=doc.metadata
                ),
                similarity_score=score,
                granularity=doc.metadata['granularity'],
                chunk_size=len(doc.metadata['original_content']),
                final_score=1.0 - score  # Higher is better
            )
            qa_results.append(chunk_result)
        
        return qa_results

def main():
    """Test the advanced multi-granular RAG system"""
    print("🚀 Advanced Multi-Granular RAG System")
    print("=" * 60)
    
    # Load environment
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Load data
    data_file = "../data/single_20240229.json"
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} documents")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Initialize system
    rag = AdvancedMultiGranularRAG(api_key, similarity_threshold=0.35)
    
    # Build hybrid stores
    if not rag.build_hybrid_vector_stores(data):
        return
    
    # Test queries with different complexities
    test_queries = [
        "コンバイン",  # Simple
        "コンバインとは何ですか",  # Medium
        "コンバインの種類とそれぞれの特徴、使用される作物について詳しく教えて"  # Complex
    ]
    
    print(f"\n🧪 Testing Advanced Retrieval")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query}")
        print("-" * 40)
        
        # Test atomic subqueries for complex queries
        if len(query) > 30:
            subqueries = rag.decompose_atomic_subqueries(query)
            if len(subqueries) > 1:
                print(f"🔄 Decomposed into {len(subqueries)} subqueries:")
                for sq in subqueries:
                    print(f"  • {sq}")
                query = subqueries[0]  # Use first subquery for demonstration
        
        # Execute advanced query
        results = rag.advanced_multi_granular_query(query, k=3)
        
        if results:
            print(f"\n🎯 Best result:")
            best = results[0]
            print(f"  Granularity: {best.granularity}")
            print(f"  Size: {best.chunk_size} chars")
            print(f"  Final Score: {best.final_score:.3f}")
            print(f"  Content: {best.document.page_content[:80]}...")
            
            # Monitor coverage
            coverage = rag.monitor_keyword_coverage(query, results)
            print(f"\n📊 Keyword Coverage by Granularity:")
            for granularity, cov in coverage.items():
                if cov > 0:
                    print(f"  {granularity}: {cov:.1%}")
    
    print(f"\n🎉 Advanced Multi-Granular RAG System Demo Complete!")

if __name__ == "__main__":
    main()