#!/usr/bin/env python3
"""
Pure Semantic RAG System - Fully based on LLM semantic understanding, no hardcoded rules
Refactored from ultra_fast_rag_integrated.py, removing all hardcoded algorithms
"""

import os
import re
import json
import time
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pydantic import SecretStr
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from dotenv import load_dotenv


@dataclass
class SemanticChunk:
    """Semantic chunk result"""
    content: str
    similarity_score: float
    semantic_relevance: float
    final_score: float
    granularity: str
    reasoning: str
    metadata: Dict[str, Any]


class SemanticLLMRanker:
    """Pure Semantic LLM Ranking System - No hardcoded rules"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
    
    def rank_chunks_semantically(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
        """Pure semantic ranking - Fully based on LLM understanding"""
        if not chunks:
            return []

        print(f"🧠 Pure Semantic LLM Evaluation: {len(chunks)} candidate chunks")

        # Only evaluate the top 1 chunk with best vector match using LLM
        chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        top_chunk = chunks[0]
        
        try:
            llm_score, relevance_reason, generated_answer = self._evaluate_chunk_semantically(
                query, top_chunk["content"], top_chunk.get("similarity_score", 0.0)
            )
            
            enhanced_chunk = {
                **top_chunk,
                "llm_score": llm_score,
                "relevance_reason": relevance_reason,
                "generated_answer": generated_answer,
                "final_score": llm_score,  # Fully based on LLM score
                "rank_order": 1
            }

            print(f"✅ Semantic evaluation complete: LLM score {llm_score:.3f}")
            return [enhanced_chunk]

        except Exception as e:
            print(f"❌ Semantic evaluation failed: {e}")
            # Fallback to vector similarity
            enhanced_chunk = {
                **top_chunk,
                "llm_score": top_chunk.get("similarity_score", 0.0),
                "relevance_reason": "LLM evaluation failed, using vector similarity",
                "generated_answer": top_chunk["content"][:100] + "...",
                "final_score": top_chunk.get("similarity_score", 0.0),
                "rank_order": 1
            }
            return [enhanced_chunk]

    def _evaluate_chunk_semantically(self, query: str, content: str, vector_score: float) -> Tuple[float, str, str]:
        """Pure semantic evaluation - Fully based on LLM understanding with lenient scoring"""

        evaluation_prompt = f"""
        Question: {query}
        Reference Text: {content}

        Evaluate the semantic relevance between this reference text and the question, then generate an answer.

        Important Evaluation Criteria:
        1. First, verify if the reference text is related to the question's topic
           - The main keywords of the question must match the topic of the reference text
           - If topics are completely different, score as 0.0
           - Empty summary statements should be scored as 0.0

        2. Only if topics match, evaluate using these criteria:
           - If the question contains multiple aspects, give high score even if text can partially answer
           - Partial but useful information should score 0.5 or higher
           - Complete answer capability should score 0.8 or higher

        Evaluation Requirements:
        1. Evaluate relevance score (between 0-1)
        2. Explain the reasoning
        3. Generate an answer based on the reference text

        Return in JSON format:
        {{
            "relevance_score": <score between 0-1, must be 0.0 if topics differ>,
            "reason": "<reasoning for this score>",
            "generated_answer": "<answer generated based on reference text>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an intelligent question-answering assistant. You can accurately evaluate text relevance and generate complete answers."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            # Debug: write to file
            with open('/tmp/rag_debug.log', 'a') as f:
                f.write(f"\n=== LLM Evaluation ===\n")
                f.write(f"Query: {query}\n")
                f.write(f"Content: {content[:100]}...\n")
                f.write(f"Raw response: {result_text}\n")
            relevance_score, reason, generated_answer = self._parse_semantic_response(result_text)
            with open('/tmp/rag_debug.log', 'a') as f:
                f.write(f"Parsed score: {relevance_score}\n\n")

            return relevance_score, reason, generated_answer

        except Exception as e:
            print(f"LLM semantic evaluation failed: {e}")
            # Fallback to simple evaluation
            fallback_answer = self._generate_fallback_answer(query, content)
            return vector_score, "LLM evaluation failed, using vector similarity", fallback_answer

    def _parse_semantic_response(self, result_text: str) -> Tuple[float, str, str]:
        """Parse LLM semantic response"""
        try:
            # Extract JSON block
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)

                relevance_score = float(parsed.get('relevance_score', 0.0))
                reason = parsed.get('reason', 'No reason provided')
                generated_answer = parsed.get('generated_answer', '')

                if not generated_answer or generated_answer.strip() == '':
                    generated_answer = "Sorry, unable to generate answer based on reference text."

                return relevance_score, reason, generated_answer
            else:
                raise ValueError("Valid JSON format not found")

        except Exception as e:
            print(f"LLM response parsing failed: {e}")
            return 0.5, f"Parsing failed: {str(e)}", "Parsing failed, cannot generate answer"

    def _generate_fallback_answer(self, query: str, content: str) -> str:
        """Generate fallback answer"""
        # Simple answer generation based on content length
        if len(content) > 100:
            return content[:100] + "..."
        else:
            return content


class PureSemanticRAG:
    """Pure Semantic RAG System - Fully based on LLM semantic understanding"""
    
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma_semantic"):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.llm = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini", temperature=0)

        # Vector storage
        self.db = None
        if os.path.exists(chroma_path):
            try:
                self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
                print(f"✅ Loaded existing vector database: {chroma_path}")
            except Exception as e:
                print(f"⚠️ Vector database loading failed: {e}")

        # Semantic ranking system
        self.semantic_ranker = SemanticLLMRanker(openai_api_key)

        # Configuration - Balanced for accuracy and recall
        self.config = {
            'similarity_threshold': 0.2,
            'max_candidates': 8,  # Increased to 8 for better recall (find more candidate chunks)
            'use_query_expansion': True,  # Keep query expansion to ensure recall
            'use_semantic_ranking': True
        }

    def build_vector_store(self, data_file: str, chunk_size: int = 200, chunk_overlap: int = 50) -> bool:
        """Build pure semantic vector database"""
        try:
            print(f"🏗️ Building pure semantic vector database...")
            print(f"📁 Data file: {data_file}")
            print(f"🗄️ Vector database path: {self.chroma_path}")

            # Check if vector store already exists
            if os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
                print(f"✅ Loaded existing vector database: {self.chroma_path}")
                self.db = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=self.embedding_function
                )
                return True

            # Try using other available vector databases
            alternative_paths = [
                "./chroma_integrated",
                "./chroma_semantic_test",
                "./chroma_pure_semantic",
                "./chroma_improved_semantic"
            ]

            for alt_path in alternative_paths:
                if os.path.exists(alt_path) and os.listdir(alt_path):
                    print(f"✅ Using alternative vector database: {alt_path}")
                    self.chroma_path = alt_path
                    self.db = Chroma(
                        persist_directory=self.chroma_path,
                        embedding_function=self.embedding_function
                    )
                    return True

            # Check data file
            if not os.path.exists(data_file):
                print(f"❌ Data file does not exist: {data_file}")
                return False

            # Load data
            print("📖 Loading data...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ Loaded {len(data)} data entries")

            # Deduplicate contexts - SQuAD format has duplicate contexts for multiple questions
            print("🔄 Deduplicating contexts (SQuAD format has multiple Q&A pairs per context)...")
            seen_contexts = {}
            for item in data:
                content = item.get('context', '') or item.get('output', '') or item.get('text', '') or item.get('content', '')
                if not content:
                    continue

                # Use content as key for deduplication
                if content not in seen_contexts:
                    # Extract paragraph ID from ID field (e.g., "a10336p0q0" -> "a10336p0")
                    item_id = item.get('id', '')
                    # Extract paragraph identifier (everything before the last 'q')
                    if 'q' in item_id:
                        para_id = item_id.rsplit('q', 1)[0]  # e.g., "a10336p0q0" -> "a10336p0"
                    else:
                        para_id = item_id

                    title = item.get('title', 'unknown')

                    seen_contexts[content] = {
                        'content': content,
                        'para_id': para_id,
                        'title': title,
                        'item_ids': [item_id]
                    }
                else:
                    # Track all item IDs that use this context
                    item_id = item.get('id', '')
                    seen_contexts[content]['item_ids'].append(item_id)

            print(f"  📊 Deduplicated: {len(data)} entries -> {len(seen_contexts)} unique contexts")

            # Convert to Document format with enhanced metadata for multi-paragraph linking
            documents = []
            for idx, (content, info) in enumerate(seen_contexts.items()):
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'semantic_rag',
                        'original_index': idx,
                        'original_full_text': content,  # Save complete original text
                        'doc_id': info['para_id'],  # Use paragraph ID
                        'title': info['title'],  # Document title for grouping
                        'is_full_context': True,  # Mark as full context before chunking
                        'related_item_ids': ','.join(info['item_ids'])  # Join list as string for ChromaDB compatibility
                    }
                )
                documents.append(doc)

            print(f"📄 Converted {len(documents)} documents")

            # Text splitting with enhanced metadata preservation
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "。", "！", "？", "、", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents(documents)

            # Post-process: clean up and add chunk relationship metadata
            for chunk_idx, chunk in enumerate(chunks):
                # Remove leading punctuation (。！？、etc)
                chunk.page_content = chunk.page_content.lstrip('。！？、 \n')

                # Add chunk relationship metadata
                original_meta = chunk.metadata
                chunk.metadata.update({
                    'chunk_id': chunk_idx,  # Unique chunk ID
                    'chunk_position': chunk_idx,  # Position in chunk sequence
                    'parent_doc_id': original_meta.get('doc_id', f"doc_{original_meta.get('original_index', 0)}"),
                    'parent_title': original_meta.get('title', 'unknown'),
                    'is_chunk': True,  # Mark as chunk (vs full context)
                    'char_start': 0,  # Character position in original document (to be calculated)
                    'char_end': len(chunk.page_content)
                })

                # Calculate character position in original document
                original_full = original_meta.get('original_full_text', '')
                if original_full and chunk.page_content in original_full:
                    char_start = original_full.find(chunk.page_content)
                    chunk.metadata['char_start'] = char_start
                    chunk.metadata['char_end'] = char_start + len(chunk.page_content)

            print(f"📄 Created {len(chunks)} chunks with relationship metadata (cleaned)")

            # Clean up old vector store
            if os.path.exists(self.chroma_path):
                import shutil
                shutil.rmtree(self.chroma_path)
                print("🗑️ Cleaned up old vector store")

            # Build vector storage
            print("🔄 Creating pure semantic vector store...")
            start_time = time.time()

            self.db = Chroma.from_documents(
                chunks,
                self.embedding_function,
                persist_directory=self.chroma_path
            )

            build_time = time.time() - start_time
            print(f"✅ Pure semantic vector store built! Time: {build_time:.2f}s")
            print(f"📊 Stats: {len(chunks)} chunks, avg {build_time/len(chunks)*1000:.1f}ms/chunk")

            return True

        except Exception as e:
            print(f"❌ Pure semantic vector store build failed: {e}")
            return False
    
    def semantic_query(self, query: str, k: int = 3) -> List[SemanticChunk]:
        """Pure semantic query - Fully based on LLM understanding"""
        if not self.db:
            return []

        print(f"🔍 Pure semantic query: {query}")

        # 1. Query expansion
        expanded_queries = self._expand_query_semantically(query)
        print(f"📝 Query expansion: {len(expanded_queries)} variants")

        # 2. Multi-query retrieval
        all_candidates = []
        for expanded_query in expanded_queries:
            candidates = self.db.similarity_search_with_score(
                expanded_query,
                k=self.config['max_candidates']
            )
            for doc, score in candidates:
                all_candidates.append({
                    'document': doc,
                    'similarity_score': score,
                    'source_query': expanded_query
                })

        # 3. Deduplication
        unique_candidates = self._deduplicate_candidates(all_candidates)
        print(f"🎯 Candidates after deduplication: {len(unique_candidates)}")

        # 4. Combine related chunks from same parent document
        combined_candidates = self._combine_related_chunks(unique_candidates)

        # 5. Semantic evaluation
        if self.config['use_semantic_ranking']:
            semantic_chunks = self._evaluate_semantic_relevance(query, combined_candidates)
        else:
            semantic_chunks = self._convert_to_semantic_chunks(combined_candidates)

        # 6. Sort and return
        semantic_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return semantic_chunks[:k]
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-questions if applicable"""
        decomposition_prompt = f"""
        Question: {query}

        Analyze this question. If it contains multiple aspects (e.g., "what is X" and "its structure"), decompose it into independent sub-questions.

        Return in JSON format:
        {{
            "is_complex": <true/false, whether it contains multiple aspects>,
            "sub_questions": [<list of sub-questions. For simple questions, return only the original question>]
        }}

        Example 1:
        Question: What is a combine harvester and explain its structure
        Response: {{"is_complex": true, "sub_questions": ["What is a combine harvester?", "Explain the structure of a combine harvester"]}}

        Example 2:
        Question: Tell me about types of agricultural machinery
        Response: {{"is_complex": false, "sub_questions": ["Tell me about types of agricultural machinery"]}}
        """

        try:
            response = self.llm.invoke(decomposition_prompt)
            content = response.content

            # Parse JSON response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                sub_questions = result.get('sub_questions', [query])
                is_complex = result.get('is_complex', False)

                if is_complex:
                    print(f"🔀 Query decomposed into {len(sub_questions)} sub-questions")
                    for i, sq in enumerate(sub_questions, 1):
                        print(f"   {i}. {sq}")

                return sub_questions
            else:
                print("⚠️ Query decomposition failed, using original query")
                return [query]

        except Exception as e:
            print(f"⚠️ Query decomposition error: {e}, using original query")
            return [query]

    def _expand_query_semantically(self, query: str) -> List[str]:
        """Semantic query expansion with decomposition support"""
        if not self.config['use_query_expansion']:
            return [query]

        # First decompose query if complex
        sub_questions = self._decompose_query(query)

        # If query was decomposed into multiple sub-questions, return them
        if len(sub_questions) > 1:
            return sub_questions

        # Otherwise, expand the single query
        expansion_prompt = f"""
        Original Question: {query}

        Generate one semantically equivalent variation of this question. Return separated by | (total of 2 including the original):

        Example: Tell me about types of agricultural machinery|About classification of agricultural machinery
        """

        try:
            response = self.llm.invoke(expansion_prompt)
            content = response.content.strip()

            # Safe parsing
            if '|' in content:
                expanded_queries = [q.strip() for q in content.split('|') if q.strip()]
            else:
                expanded_queries = [query]

            return expanded_queries[:2]  # Limit to 2 variants (original + 1)

        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return [query]

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """Deduplicate candidate documents"""
        seen_contents = set()
        unique_candidates = []

        for candidate in candidates:
            content = candidate['document'].page_content
            if content not in seen_contents:
                seen_contents.add(content)
                unique_candidates.append(candidate)

        return unique_candidates

    def _combine_related_chunks(self, candidates: List[Dict]) -> List[Dict]:
        """
        Combine chunks from the same parent document to restore full context.
        This solves the multi-paragraph retrieval problem where answer spans multiple chunks.
        """
        print("\n🔗 Combining related chunks from same documents...")

        # Group candidates by parent document
        doc_groups = {}
        for candidate in candidates:
            doc = candidate['document']
            parent_doc_id = doc.metadata.get('parent_doc_id', 'unknown')

            if parent_doc_id not in doc_groups:
                doc_groups[parent_doc_id] = {
                    'chunks': [],
                    'parent_title': doc.metadata.get('parent_title', 'unknown'),
                    'original_full_text': doc.metadata.get('original_full_text', ''),
                    'max_similarity': 0.0
                }

            doc_groups[parent_doc_id]['chunks'].append(candidate)
            # Track highest similarity score among chunks
            doc_groups[parent_doc_id]['max_similarity'] = max(
                doc_groups[parent_doc_id]['max_similarity'],
                candidate['similarity_score']
            )

        print(f"  📊 Found {len(candidates)} chunks from {len(doc_groups)} parent documents")

        # Combine chunks for each document
        combined_candidates = []
        for parent_doc_id, group in doc_groups.items():
            chunks = group['chunks']

            if len(chunks) == 1:
                # Single chunk - use as-is
                combined_candidates.append(chunks[0])
                print(f"  📄 {parent_doc_id}: 1 chunk, kept as-is")
            else:
                # Multiple chunks from same document - combine them
                print(f"  🔗 {parent_doc_id}: {len(chunks)} chunks, combining...")

                # Sort chunks by character position
                chunks_sorted = sorted(
                    chunks,
                    key=lambda c: c['document'].metadata.get('char_start', 0)
                )

                # Use the original full text if available
                original_full = group['original_full_text']
                if original_full:
                    # Use complete original document
                    combined_content = original_full
                    print(f"    ✅ Using original full context ({len(combined_content)} chars)")
                else:
                    # Fallback: concatenate chunks in order
                    combined_content = '\n'.join([c['document'].page_content for c in chunks_sorted])
                    print(f"    ⚠️ Concatenated {len(chunks)} chunks ({len(combined_content)} chars)")

                # Create combined document with averaged scores
                avg_similarity = sum(c['similarity_score'] for c in chunks) / len(chunks)

                # Use metadata from first chunk but mark as combined
                first_chunk = chunks_sorted[0]
                combined_metadata = first_chunk['document'].metadata.copy()
                combined_metadata.update({
                    'is_combined': True,
                    'num_chunks_combined': len(chunks),
                    'combined_chunk_ids': ','.join(str(c['document'].metadata.get('chunk_id', -1)) for c in chunks),  # Join as string for ChromaDB
                    'char_start': 0,
                    'char_end': len(combined_content)
                })

                combined_doc = Document(
                    page_content=combined_content,
                    metadata=combined_metadata
                )

                combined_candidate = {
                    'document': combined_doc,
                    'similarity_score': group['max_similarity'],  # Use max score for ranking
                    'avg_similarity_score': avg_similarity,
                    'source_query': first_chunk.get('source_query', ''),
                    'is_combined': True
                }

                combined_candidates.append(combined_candidate)

        print(f"  ✅ Combined into {len(combined_candidates)} documents\n")
        return combined_candidates

    def _evaluate_semantic_relevance(self, query: str, candidates: List[Dict]) -> List[SemanticChunk]:
        """LLM semantic relevance evaluation"""
        print("🧠 LLM semantic evaluation in progress...")
        
        semantic_chunks = []

        for i, candidate in enumerate(candidates):
            doc = candidate['document']
            similarity_score = candidate['similarity_score']

            # Use semantic ranker for evaluation
            try:
                chunks_for_ranking = [{
                    'content': doc.page_content,
                    'similarity_score': similarity_score,
                    'metadata': doc.metadata
                }]

                ranked_chunks = self.semantic_ranker.rank_chunks_semantically(query, chunks_for_ranking, top_k=1)

                if ranked_chunks:
                    ranked_chunk = ranked_chunks[0]

                    semantic_chunk = SemanticChunk(
                        content=doc.page_content,
                        similarity_score=similarity_score,
                        semantic_relevance=ranked_chunk.get('llm_score', 0.5),
                        final_score=ranked_chunk.get('final_score', 0.5),
                        granularity='semantic',
                        reasoning=ranked_chunk.get('relevance_reason', ''),
                        metadata={
                            'source_query': candidate.get('source_query', query),
                            'is_direct_answer': ranked_chunk.get('llm_score', 0.5) > 0.7,
                            'confidence': ranked_chunk.get('llm_score', 0.5),
                            'original_metadata': doc.metadata
                        }
                    )

                    semantic_chunks.append(semantic_chunk)

            except Exception as e:
                print(f"⚠️ Semantic evaluation failed: {e}")
                # Fallback to basic conversion
                semantic_chunk = SemanticChunk(
                    content=doc.page_content,
                    similarity_score=similarity_score,
                    semantic_relevance=similarity_score,
                    final_score=similarity_score,
                    granularity='semantic',
                    reasoning='LLM evaluation failed, using vector similarity',
                    metadata={
                        'source_query': candidate.get('source_query', query),
                        'is_direct_answer': False,
                        'confidence': 0.5,
                        'original_metadata': doc.metadata
                    }
                )
                semantic_chunks.append(semantic_chunk)

        return semantic_chunks

    def _convert_to_semantic_chunks(self, candidates: List[Dict]) -> List[SemanticChunk]:
        """Convert to semantic chunks (used when no LLM evaluation)"""
        semantic_chunks = []

        for candidate in candidates:
            doc = candidate['document']
            similarity_score = candidate['similarity_score']

            semantic_chunk = SemanticChunk(
                content=doc.page_content,
                similarity_score=similarity_score,
                semantic_relevance=similarity_score,
                final_score=similarity_score,
                granularity='semantic',
                reasoning='Based on vector similarity',
                metadata={
                    'source_query': candidate.get('source_query', ''),
                    'is_direct_answer': False,
                    'confidence': 0.5,
                    'original_metadata': doc.metadata
                }
            )
            semantic_chunks.append(semantic_chunk)

        return semantic_chunks
    
    def generate_answer(self, query: str, semantic_chunks: List[SemanticChunk]) -> Dict[str, Any]:
        """Generate answer based on semantic chunks - Strategy 3: Extract evidence from each high-similarity chunk separately"""
        if not semantic_chunks:
            return {
                'answer': 'Sorry, no relevant information found.',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': 'No relevant documents found',
                'evidences': []
            }

        # Build context - use all filtered chunks
        context_parts = []
        chunks_details = []
        for i, chunk in enumerate(semantic_chunks, 1):
            context_parts.append(f"Document {i}: {chunk.content}")
            # Collect detailed scoring information for each chunk
            chunks_details.append({
                'chunk_id': i,
                'content': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content,
                'similarity_score': float(chunk.similarity_score),
                'semantic_relevance': float(chunk.semantic_relevance),
                'final_score': float(chunk.final_score),
                'confidence': float(chunk.metadata.get('confidence', 0.0)),
                'reasoning': chunk.reasoning,
                'granularity': chunk.granularity
            })

        context = "\n\n".join(context_parts)

        # Print detailed scoring information
        print("\n📊 Retrieved document scoring details:")
        print("=" * 80)
        for detail in chunks_details:
            print(f"\nDocument {detail['chunk_id']}:")
            print(f"  📝 Content preview: {detail['content']}")
            print(f"  🎯 Vector similarity (similarity_score): {detail['similarity_score']:.4f}")
            print(f"  🧠 Semantic relevance (semantic_relevance): {detail['semantic_relevance']:.4f}")
            print(f"  ⭐ Final score (final_score): {detail['final_score']:.4f}")
            print(f"  💯 Confidence: {detail['confidence']:.4f}")
            print(f"  📐 Granularity: {detail['granularity']}")
            print(f"  💬 Scoring reason: {detail['reasoning']}")
        print("=" * 80)

        # Prompt for generating answer - generate concise summary answer based on original text
        answer_prompt = f"""
        User's Question: {query}

        Reference Documents:
        {context}

        Answer the user's question based on the reference documents. Requirements:
        1. Carefully read the reference documents and understand the core of the user's question
        2. Based on the document content, directly answer the core point in 1-2 concise sentences
        3. The answer must be completely based on the document content; do not fabricate information
        4. Use the original document's expressions and terminology
        5. For definition questions (e.g., "what is"), provide only the core definition without detailed elaboration
        6. For classification/enumeration questions, list the main categories or items
        7. Maintain the original language's expression style and tone

        Provide a concise answer (1-2 sentences) directly, without preamble or additional explanation:
        """

        try:
            # Generate overall answer
            response = self.llm.invoke(answer_prompt)
            answer = response.content.strip()

            # Strategy 3 core: Extract evidence from each chunk separately
            evidences = []
            print("\n🔍 Starting evidence extraction from each chunk...")

            for i, chunk in enumerate(semantic_chunks, 1):
                # Pure LLM-based evidence extraction - No hardcoded rules
                import re

                # Use LLM to extract evidence with semantic relevance scoring
                evidence_range_prompt = f"""
Task: Identify the parts of the document that support the answer and evaluate their relevance.

Question: {query}
Answer: {answer}
Document: {chunk.content}

Instructions:
1. Evaluate whether the document contains evidence that supports the answer
2. If relevance is low (unrelated to question or doesn't support answer), output "empty"
3. Only if relevance is high, identify the character ranges of the evidence
4. Output format: character M～character N (multiple ranges separated by comma)
5. Keep evidence ranges as short and minimal as necessary

Judgment Criteria:
- Does the document's topic match the question's topic?
- Does the document directly support the answer's claims?
- Does it contain clear evidence, not vague associations?

Example 1 (High Relevance):
Question: What season type is the rainy season?
Answer: The rainy season is a type of wet season.
Document: Rainy season [SEP] The rainy season is a weather phenomenon seen in East Asia, occurring from May to July with cloudy and rainy periods. It is a type of wet season.
Output: character 80～character 87

Example 2 (Low Relevance):
Question: What is the structure of a combine harvester?
Answer: A combine harvester consists of a cutting section and a threshing section.
Document: Rainy season [SEP] As winter ends and spring approaches, the Siberian air mass weakens and gradually moves north.
Output: empty

Evidence Range:
"""

                llm_match_ranges = []
                llm_match_texts = []

                try:
                    evidence_response = self.llm.invoke(evidence_range_prompt)
                    range_output = evidence_response.content.strip()

                    if range_output.lower() != "empty" and range_output != "":
                        # Parse ranges - support both English and Japanese format
                        range_pattern = r'(?:character\s+)?(\d+)(?:文字目)?～(?:character\s+)?(\d+)(?:文字目)?'
                        matches = re.findall(range_pattern, range_output)

                        for start_str, end_str in matches:
                            start = int(start_str)
                            end = int(end_str)

                            if 1 <= start <= len(chunk.content) and start <= end <= len(chunk.content):
                                substring = chunk.content[start-1:end]
                                llm_match_ranges.append((start, end))
                                llm_match_texts.append(substring)
                                print(f"   ✓ LLM extraction: {start}～{end} ('{substring}')")
                except Exception as llm_e:
                    print(f"   ⚠️ LLM extraction failed: {llm_e}")

                # Use LLM results
                if llm_match_ranges:
                    char_ranges = llm_match_ranges
                    extracted_evidence = "\n".join(llm_match_texts)
                    is_empty = False
                    print(f"   ✅ LLM found {len(llm_match_ranges)} evidence range(s)")
                else:
                    char_ranges = []
                    extracted_evidence = ""
                    is_empty = True
                    print(f"   ❌ LLM: No relevant evidence in this chunk")

                # Create evidence info
                evidence_info = {
                    'chunk_id': i,
                    'chunk_content': chunk.content,
                    'extracted_evidence': "" if is_empty else extracted_evidence,
                    'char_ranges': char_ranges,  # Store the ranges for frontend
                    'similarity_score': float(chunk.similarity_score),
                    'semantic_relevance': float(chunk.semantic_relevance),
                    'is_empty': is_empty
                }

                evidences.append(evidence_info)

                status = "✅" if not is_empty else "❌"
                print(f"{status} Chunk {i} (similarity: {chunk.similarity_score:.3f})")
                if not is_empty:
                    print(f"   📍 Char ranges: {char_ranges}")
                    print(f"   📝 Extracted evidence: {extracted_evidence[:200]}...")
                print(f"   📄 Original chunk length: {len(chunk.content)} chars")

                # Write to debug file
                with open('/tmp/rag_evidence_debug.log', 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Query: {query}\n")
                    f.write(f"Generated Answer: {answer}\n")
                    f.write(f"Chunk {i} - Similarity: {chunk.similarity_score:.3f}, Semantic Relevance: {chunk.semantic_relevance:.3f}\n")
                    f.write(f"Original chunk ({len(chunk.content)} chars):\n{chunk.content}\n")
                    f.write(f"\nExtraction strategy: Pure LLM-based (no hardcoded rules)\n")
                    f.write(f"Char ranges: {char_ranges}\n")
                    f.write(f"Extracted evidence ({len(extracted_evidence)} chars):\n{extracted_evidence}\n")
                    f.write(f"Is empty: {is_empty}\n")
                    f.write(f"{'='*80}\n")

            # Find best evidence text (for backward compatibility)
            best_chunk = semantic_chunks[0]
            evidence_text = best_chunk.content

            # Get complete original document (keep "文档N: " prefix for multi-document support)
            original_full_text = best_chunk.metadata.get('original_metadata', {}).get('original_full_text', '')
            if not original_full_text:
                # Fallback: use context with document prefix (important for multi-doc results)
                original_full_text = context

            # Calculate average confidence
            avg_confidence = sum(chunk.metadata['confidence'] for chunk in semantic_chunks) / len(semantic_chunks)

            # Count valid evidences
            valid_evidences_count = sum(1 for e in evidences if not e['is_empty'])
            print(f"\n📊 Evidence extraction complete: {valid_evidences_count}/{len(evidences)} chunks contain valid evidence")

            return {
                'answer': answer,
                'evidence_text': evidence_text,  # Backward compatibility: keep best chunk as main evidence
                'source_document': original_full_text,
                'confidence': avg_confidence,
                'reasoning': f'Generated from {len(semantic_chunks)} relevant documents, {valid_evidences_count} contain valid evidence',
                'model': 'PureSemanticRAG-Strategy3',
                'processing_time': 0.0,
                'chunks_used': len(semantic_chunks),
                'chunks_details': chunks_details,
                'evidences': evidences  # Strategy 3 core: multi-evidence list
            }

        except Exception as e:
            print(f"⚠️ Answer generation failed: {e}")
            return {
                'answer': 'Sorry, an error occurred during answer generation.',
                'evidence_text': semantic_chunks[0].content if semantic_chunks else '',
                'source_document': context,
                'confidence': 0.3,
                'reasoning': f'Generation failed: {str(e)}',
                'model': 'PureSemanticRAG',
                'processing_time': 0.0,
                'chunks_used': len(semantic_chunks),
                'evidences': []
            }
    
    def query_with_answer(self, query: str, k: int = 10, relevance_threshold: float = 0.5) -> Dict[str, Any]:
        """Complete query workflow - Retrieve top k documents, filter by semantic relevance (≥ 0.5) for high precision"""
        start_time = time.time()

        # 1. Semantic retrieval - get top k candidate documents
        semantic_chunks = self.semantic_query(query, k)

        # 2. Filter: only keep documents with high semantic relevance (≥ 0.5) for precision
        filtered_chunks = [
            chunk for chunk in semantic_chunks
            if chunk.semantic_relevance >= relevance_threshold
        ]

        print(f"\n🔍 Retrieval statistics:")
        print(f"  📊 Initial retrieval: {len(semantic_chunks)} documents")
        print(f"  ✅ After filtering (relevance≥{relevance_threshold}): {len(filtered_chunks)} documents")

        # 3. If no qualifying documents found, return apology message
        if not filtered_chunks:
            processing_time = time.time() - start_time
            print(f"  ⚠️ No documents with relevance≥{relevance_threshold} found, returning empty result")
            return {
                'answer': 'Sorry, no relevant information is currently available.',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': f'No documents with semantic relevance≥{relevance_threshold} found',
                'model': 'PureSemanticRAG',
                'processing_time': processing_time,
                'chunks_used': 0,
                'chunks_details': []
            }

        # 4. Generate answer using filtered documents (support multiple partially matching chunks answering together)
        result = self.generate_answer(query, filtered_chunks)

        # 5. Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time

        return result


def test_pure_semantic_rag():
    """Test pure semantic RAG system"""
    print("🧪 Pure Semantic RAG System Test")
    print("=" * 60)

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    # Initialize system
    rag = PureSemanticRAG(api_key)

    # Load test data
    data_file = "../data/single_20240229.json"
    if not os.path.exists(data_file):
        print(f"❌ Data file does not exist: {data_file}")
        return

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📖 Loaded {len(data)} data entries")

    # Build vector database
    rag.build_vector_store(data_file)  # Use complete data file

    # Test queries
    test_queries = [
        "農業機械の種類について教えてください",
        "コンバインとは何ですか",
        "普通型と自立型の違いは何ですか"
    ]

    for query in test_queries:
        print(f"\n🔍 Test query: {query}")
        print("-" * 40)

        result = rag.query_with_answer(query, k=3)

        print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
        print(f"💬 Answer: {result['answer']}")
        print(f"🔍 Evidence: {result['evidence_text']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"🧠 Reasoning: {result['reasoning']}")
        print(f"📄 Chunks used: {result['chunks_used']}")


if __name__ == "__main__":
    test_pure_semantic_rag()
