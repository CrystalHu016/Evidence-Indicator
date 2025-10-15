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
        質問: {query}
        参考テキスト: {content}

        この参考テキストと質問の意味的関連性を評価し、回答を生成してください。

        重要な評価基準:
        1. まず、参考テキストが質問の主題と関連しているか確認してください
           - 質問の主要なキーワード（例：「コンバイン」）が参考テキストの主題と一致する必要があります
           - 主題が完全に異なる場合（例：質問が「コンバイン」で、テキストが「コンビニスイーツ」や「コンピュータ」）は0.0としてください
           - 空の要約文（「与えられた文章を要約します」のみ）は0.0としてください

        2. 主題が一致する場合のみ、以下の基準で評価してください：
           - 質問が複数の側面を含む場合（例：「〜とは何か」と「その構造」）、参考テキストが一部でも回答できれば高スコア
           - 部分的な情報でも有用であれば0.5以上
           - 完全に回答できる場合は0.8以上

        評価要件:
        1. 関連性スコアを評価 (0-1の間)
        2. 評価理由を説明
        3. 参考テキストに基づいて回答を生成

        JSON形式で返してください:
        {{
            "relevance_score": <0-1のスコア、主題が異なる場合は必ず0.0>,
            "reason": "<このスコアを付けた理由>",
            "generated_answer": "<参考テキストに基づいて生成した回答>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "あなたは知的な質問応答アシスタントです。テキストの関連性を正確に評価し、完全な回答を生成できます。"},
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
                reason = parsed.get('reason', '未提供理由')
                generated_answer = parsed.get('generated_answer', '')

                if not generated_answer or generated_answer.strip() == '':
                    generated_answer = "抱歉，无法基于参考文本生成回答。"

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
        質問: {query}

        この質問を分析してください。質問が複数の側面を含んでいる場合（例：「〜とは何か」と「その構造」）、独立したサブ質問に分解してください。

        JSON形式で返してください:
        {{
            "is_complex": <true/false、複数の側面を含むかどうか>,
            "sub_questions": [<サブ質問のリスト。単純な質問の場合は元の質問のみ>]
        }}

        例1:
        質問: コンバインとは何かとその構造を説明してください
        回答: {{"is_complex": true, "sub_questions": ["コンバインとは何ですか", "コンバインの構造を説明してください"]}}

        例2:
        質問: 農業機械の種類について教えてください
        回答: {{"is_complex": false, "sub_questions": ["農業機械の種類について教えてください"]}}
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
        元の質問: {query}

        この質問の意味的に同等な1つのバリエーションを生成してください。|で区切って返してください（元の質問を含めて合計2つ）：

        例: 農業機械の種類について教えてください|農業機械の分類について
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
                'answer': '抱歉，没有找到相关信息。',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': '没有找到相关文档',
                'evidences': []
            }

        # Build context - use all filtered chunks
        context_parts = []
        chunks_details = []
        for i, chunk in enumerate(semantic_chunks, 1):
            context_parts.append(f"文档{i}: {chunk.content}")
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
        ユーザーの質問: {query}

        参考文書:
        {context}

        参考文書に基づいてユーザーの質問に回答してください。要件：
        1. 参考文書を注意深く読み、ユーザーの質問の核心を理解する
        2. 文書の内容に基づき、簡潔な1-2文で質問の核心的なポイントを直接回答する
        3. 回答は完全に文書の内容に基づく必要があり、情報を捏造してはいけない
        4. 原文書の表現方法と用語を使用する
        5. 定義類の質問（「とは何ですか」など）の場合、核心的な定義のみを示し、詳細な展開は不要
        6. 分類・列挙類の質問の場合、主要な分類や項目を列挙する
        7. 日本語原文の表現習慣と語気を保つ

        簡潔な回答（1-2文）を直接提示してください。前置きや追加説明は不要：
        """

        try:
            # Generate overall answer
            response = self.llm.invoke(answer_prompt)
            answer = response.content.strip()

            # Strategy 3 core: Extract evidence from each chunk separately
            evidences = []
            print("\n🔍 Starting evidence extraction from each chunk...")

            for i, chunk in enumerate(semantic_chunks, 1):
                # Patent-compliant evidence extraction: LLM outputs character ranges first
                # Strategy 1: Direct substring matching (most reliable for exact matches)
                # Extract key phrases from the answer that should appear in evidence
                import re

                # Split answer into key phrases (by punctuation) and also extract key n-grams
                answer_phrases = []

                # Method 1: Split by punctuation
                phrases_from_split = [p.strip() for p in re.split(r'[、。，]', answer) if len(p.strip()) > 3]
                answer_phrases.extend(phrases_from_split)

                # Method 2: Extract key content words (4+ chars)
                content_phrases = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{4,}', answer)
                answer_phrases.extend(content_phrases)

                # Method 3: Extract key phrases with common endings (flexible matching)
                # e.g., "梅雨は雨季の一種であり" -> also try "雨季の一種である", "雨季の一種", etc.
                flexible_phrases = []
                for phrase in phrases_from_split:
                    # Try removing common verb endings
                    for ending in ['であり', 'です', 'ます', 'ました', 'である']:
                        if phrase.endswith(ending):
                            base = phrase[:-len(ending)]
                            if len(base) >= 4:
                                flexible_phrases.append(base + 'である')
                                flexible_phrases.append(base)

                    # Try removing subject prefixes (e.g., "梅雨は" from "梅雨は雨季の一種であり")
                    # Look for "Xは" pattern and extract the part after "は"
                    if 'は' in phrase:
                        parts = phrase.split('は', 1)
                        if len(parts) == 2 and len(parts[1]) >= 4:
                            suffix = parts[1]
                            flexible_phrases.append(suffix)
                            # Also try with である ending
                            for ending in ['であり', 'です', 'ます']:
                                if suffix.endswith(ending):
                                    base = suffix[:-len(ending)]
                                    if len(base) >= 4:
                                        flexible_phrases.append(base + 'である')
                                        flexible_phrases.append(base)

                answer_phrases.extend(flexible_phrases)

                # Remove duplicates while preserving order
                seen = set()
                answer_phrases = [p for p in answer_phrases if p not in seen and not seen.add(p)]

                direct_match_ranges = []
                direct_match_texts = []

                for phrase in answer_phrases:
                    # Look for this phrase (or close variations) in the chunk
                    if phrase in chunk.content:
                        phrase_start = chunk.content.find(phrase)
                        phrase_end = phrase_start + len(phrase)

                        # Avoid duplicate ranges
                        range_tuple = (phrase_start + 1, phrase_end)
                        if range_tuple not in direct_match_ranges:
                            direct_match_ranges.append(range_tuple)  # 1-based
                            direct_match_texts.append(chunk.content[phrase_start:phrase_end])
                            print(f"   ✓ Direct match found: '{phrase}' at {phrase_start + 1}～{phrase_end}")

                # Strategy 2: Sentence-level matching for answer keywords
                answer_keywords = []
                for sentence in answer.split('。'):
                    sentence = sentence.strip()
                    # Extract important terms (kanji-containing words 2+ chars)
                    words = re.findall(r'[\u4e00-\u9fff]+[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]*', sentence)
                    answer_keywords.extend([w for w in words if len(w) >= 2])

                keyword_match_ranges = []
                keyword_match_texts = []

                if not direct_match_ranges and answer_keywords:
                    # Find sentences containing answer keywords
                    sentences = chunk.content.replace('。', '。\n').split('\n')
                    for sent in sentences:
                        sent = sent.strip()
                        if not sent or sent.startswith('[SEP]'):
                            continue

                        # Check if this sentence contains answer keywords
                        keyword_count = sum(1 for kw in answer_keywords if kw in sent)
                        if keyword_count >= 1:
                            # Find position of this sentence in chunk
                            sent_start = chunk.content.find(sent)
                            if sent_start >= 0:
                                sent_end = sent_start + len(sent)
                                keyword_match_ranges.append((sent_start + 1, sent_end))
                                keyword_match_texts.append(sent)
                                print(f"   ✓ Keyword match: sentence with {keyword_count} keywords at {sent_start + 1}～{sent_end}")
                                if len(keyword_match_ranges) >= 2:  # Limit to 2 sentences
                                    break

                # Strategy 3: LLM-based extraction (as fallback for complex cases)
                llm_match_ranges = []
                llm_match_texts = []

                if not direct_match_ranges and not keyword_match_ranges:
                    evidence_range_prompt = f"""
タスク：文書から回答を支持する根拠となる部分の文字範囲を特定してください。

質問: {query}
回答: {answer}
文書: {chunk.content}

指示：
1. 回答の主張を裏付ける文書内の該当箇所を探してください
2. タイトル部分（"[SEP]"の前）は避けてください
3. 複数範囲がある場合はカンマ区切りで出力
4. 該当なしの場合は「空」と出力
5. 出力形式: M文字目～N文字目

例：
- 質問：梅雨とは何季の一種か?
  回答：梅雨は雨季の一種です。
  文書：梅雨 [SEP] 梅雨は東アジアで見られる気象現象で、5月から7月にかけて来る曇りや雨の多い期間のこと。雨季の一種である。
  正しい出力：80文字目～88文字目
  誤り例：1文字目～10文字目（タイトル部分は不可）

根拠範囲：
"""

                    try:
                        evidence_response = self.llm.invoke(evidence_range_prompt)
                        range_output = evidence_response.content.strip()

                        if range_output != "空" and range_output != "":
                            # Parse ranges
                            range_pattern = r'(\d+)文字目～(\d+)文字目'
                            matches = re.findall(range_pattern, range_output)

                            for start_str, end_str in matches:
                                start = int(start_str)
                                end = int(end_str)

                                if 1 <= start <= len(chunk.content) and start <= end <= len(chunk.content):
                                    substring = chunk.content[start-1:end]
                                    llm_match_ranges.append((start, end))
                                    llm_match_texts.append(substring)
                                    print(f"   ✓ LLM extraction: {start}～{end}")
                    except Exception as llm_e:
                        print(f"   ⚠️ LLM extraction failed: {llm_e}")

                # Choose best strategy
                if direct_match_ranges:
                    char_ranges = direct_match_ranges
                    extracted_evidence = "\n".join(direct_match_texts)
                    is_empty = False
                    print(f"   ✅ Using direct phrase matching")
                elif keyword_match_ranges:
                    char_ranges = keyword_match_ranges
                    extracted_evidence = "\n".join(keyword_match_texts)
                    is_empty = False
                    print(f"   ✅ Using keyword-based sentence matching")
                elif llm_match_ranges:
                    char_ranges = llm_match_ranges
                    extracted_evidence = "\n".join(llm_match_texts)
                    is_empty = False
                    print(f"   ✅ Using LLM extraction")
                else:
                    char_ranges = []
                    extracted_evidence = ""
                    is_empty = True
                    print(f"   ❌ No evidence found in chunk")

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
                    f.write(f"Chunk {i} - Similarity: {chunk.similarity_score:.3f}\n")
                    f.write(f"Original chunk ({len(chunk.content)} chars):\n{chunk.content}\n")
                    f.write(f"\nExtraction strategy: ")
                    if direct_match_ranges:
                        f.write("Direct phrase matching\n")
                    elif keyword_match_ranges:
                        f.write("Keyword-based sentence matching\n")
                    elif llm_match_ranges:
                        f.write("LLM extraction\n")
                    else:
                        f.write("No match\n")
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
                'answer': '抱歉，生成回答时出现错误。',
                'evidence_text': semantic_chunks[0].content if semantic_chunks else '',
                'source_document': context,
                'confidence': 0.3,
                'reasoning': f'Generation failed: {str(e)}',
                'model': 'PureSemanticRAG',
                'processing_time': 0.0,
                'chunks_used': len(semantic_chunks),
                'evidences': []
            }
    
    def query_with_answer(self, query: str, k: int = 10, relevance_threshold: float = 0.25) -> Dict[str, Any]:
        """Complete query workflow - Support partial matching: retrieve more documents (k=10), lower threshold (0.25) for better recall"""
        start_time = time.time()

        # 1. Semantic retrieval - get top k candidate documents (increased to 10 for better recall)
        semantic_chunks = self.semantic_query(query, k)

        # 2. Filter: only keep documents with semantic relevance ≥ threshold (threshold 0.25 allows more partial matches)
        filtered_chunks = [
            chunk for chunk in semantic_chunks
            if chunk.semantic_relevance >= relevance_threshold
        ]

        print(f"\n🔍 Retrieval statistics:")
        print(f"  📊 Initial retrieval: {len(semantic_chunks)} documents")
        print(f"  ✅ After filtering (relevance≥{relevance_threshold}): {len(filtered_chunks)} documents")

        # 3. If no qualifying documents found, return Japanese apology message
        if not filtered_chunks:
            processing_time = time.time() - start_time
            print(f"  ⚠️ No documents with relevance≥{relevance_threshold} found, returning empty result")
            return {
                'answer': '申し訳ございませんが、現在利用可能な関連情報が見つかりませんでした。',
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
