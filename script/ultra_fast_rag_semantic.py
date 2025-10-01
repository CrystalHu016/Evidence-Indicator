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

        # Configuration - Performance optimization: reduce candidate count for speed
        self.config = {
            'similarity_threshold': 0.2,
            'max_candidates': 3,  # Reduced from 8 to 3, expect 50% speedup
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

            # Convert to Document format
            documents = []
            for item in data:
                content = item.get('output', '') or item.get('text', '') or item.get('content', '')
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': 'semantic_rag',
                            'original_index': len(documents),
                            'original_full_text': content  # Save complete original text
                        }
                    )
                    documents.append(doc)

            print(f"📄 Converted {len(documents)} documents")

            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "。", "！", "？", "、", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents(documents)

            # Post-process: clean up chunks that start with punctuation
            for chunk in chunks:
                # Remove leading punctuation (。！？、etc)
                chunk.page_content = chunk.page_content.lstrip('。！？、 \n')

            print(f"📄 Created {len(chunks)} chunks (cleaned)")

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

        # 4. Semantic evaluation
        if self.config['use_semantic_ranking']:
            semantic_chunks = self._evaluate_semantic_relevance(query, unique_candidates)
        else:
            semantic_chunks = self._convert_to_semantic_chunks(unique_candidates)

        # 5. Sort and return
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
                # Extract evidence from each chunk individually
                evidence_extraction_prompt = f"""
                ユーザーの質問: {query}

                生成された回答: {answer}

                文書内容: {chunk.content}

                タスク：上記の文書から「生成された回答」を**直接サポートする文のみ**を抽出してください。

                重要な抽出基準：
                1. **生成された回答に直接含まれている情報または回答を直接サポートする原文の文のみ**を抽出する
                2. 回答に含まれていない情報（背景説明、例示、詳細な分類など）は除外する
                3. 抽出した文は元の文書から一字一句そのまま引用する（改変しない）
                4. 複数の文を抽出する場合は、各文を改行で区切って出力する（一行に一文）
                5. 文書に回答をサポートする文がない場合は「空」と出力

                抽出例1：
                質問：コンバインとは何かとその構造を説明してください
                生成された回答：コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。コンバインの構造は走行部・刈取部・搬送部・脱穀部・選別部・穀粒処理部・ワラ処理部から構成されています。
                文書：コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は...（詳細説明）...自立型は...（詳細説明）...コンバインの構造は走行部・刈取部・搬送部・脱穀部・選別部・穀粒処理部・ワラ処理部から構成されています。
                正しい抽出（回答に含まれる2文のみ）：
                コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。
                コンバインの構造は走行部・刈取部・搬送部・脱穀部・選別部・穀粒処理部・ワラ処理部から構成されています。

                抽出例2：
                質問：コンバインとは何ですか
                生成された回答：コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。
                文書：コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は...（詳細説明）...自立型は...（詳細説明）...コンバインの構造は走行部・刈取部・搬送部・脱穀部・選別部・穀粒処理部・ワラ処理部から構成されています。
                正しい抽出（回答に含まれる1文のみ、種類・構造は回答に含まれないため除外）：
                コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。

                抽出例3：
                質問：農業機械の種類について教えてください
                生成された回答：コンバインは普通型と自立型の2種類に大別されます。
                文書：コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は...（詳細説明）...自立型は...（詳細説明）...コンバインの構造は走行部・刈取部・搬送部・脱穀部・選別部・穀粒処理部・ワラ処理部から構成されています。
                正しい抽出（回答の分類情報をサポートする1文のみ、定義・構造は回答に含まれないため除外）：
                日本で使われているコンバインは普通型と自立型の2種類に大別されます。

                抽出した根拠テキスト（または「空」）を直接出力してください：
                """

                try:
                    evidence_response = self.llm.invoke(evidence_extraction_prompt)
                    extracted_evidence = evidence_response.content.strip()

                    # Determine if evidence is empty
                    is_empty = (extracted_evidence == "空" or
                               extracted_evidence == "" or
                               "不包含" in extracted_evidence or
                               "没有" in extracted_evidence)

                    evidence_info = {
                        'chunk_id': i,
                        'chunk_content': chunk.content,
                        'extracted_evidence': "" if is_empty else extracted_evidence,
                        'similarity_score': float(chunk.similarity_score),
                        'semantic_relevance': float(chunk.semantic_relevance),
                        'is_empty': is_empty
                    }

                    evidences.append(evidence_info)

                    status = "✅" if not is_empty else "❌"
                    print(f"{status} Chunk {i} (similarity: {chunk.similarity_score:.3f})")
                    print(f"   📝 Extracted evidence: {extracted_evidence[:200]}...")
                    print(f"   📄 Original chunk length: {len(chunk.content)} chars")

                    # Write to debug file
                    with open('/tmp/rag_evidence_debug.log', 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*80}\n")
                        f.write(f"Query: {query}\n")
                        f.write(f"Generated Answer: {answer}\n")
                        f.write(f"Chunk {i} - Similarity: {chunk.similarity_score:.3f}\n")
                        f.write(f"Original chunk ({len(chunk.content)} chars):\n{chunk.content}\n")
                        f.write(f"\nExtracted evidence ({len(extracted_evidence)} chars):\n{extracted_evidence}\n")
                        f.write(f"Is empty: {is_empty}\n")
                        f.write(f"{'='*80}\n")

                except Exception as e:
                    print(f"⚠️ Chunk {i} evidence extraction failed: {e}")
                    evidences.append({
                        'chunk_id': i,
                        'chunk_content': chunk.content,
                        'extracted_evidence': '',
                        'similarity_score': float(chunk.similarity_score),
                        'semantic_relevance': float(chunk.semantic_relevance),
                        'is_empty': True,
                        'error': str(e)
                    })

            # Find best evidence text (for backward compatibility)
            best_chunk = semantic_chunks[0]
            evidence_text = best_chunk.content

            # Get complete original document
            original_full_text = best_chunk.metadata.get('original_metadata', {}).get('original_full_text', '')
            if not original_full_text:
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
    
    def query_with_answer(self, query: str, k: int = 8, relevance_threshold: float = 0.3) -> Dict[str, Any]:
        """Complete query workflow - Support partial matching: retrieve more documents (k=8), threshold (0.3) for better recall"""
        start_time = time.time()

        # 1. Semantic retrieval - get top k candidate documents (increased to 8 for better recall)
        semantic_chunks = self.semantic_query(query, k)

        # 2. Filter: only keep documents with semantic relevance ≥ threshold (threshold 0.4 allows partial matching)
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
