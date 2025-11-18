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


class CharacterMarkedPromptStrategy:
    """
    Character Position Marking Strategy for Evidence Extraction
    Patent Implementation: チャンクの部分文字列としての根拠提示
    """

    @staticmethod
    def add_character_markers(text: str, marker_interval: int = 10) -> str:
        """Add character position markers to text every N characters"""
        marked_text = []
        for i, char in enumerate(text):
            if i % marker_interval == 0 and i > 0:
                marked_text.append(f"[{i}]")
            marked_text.append(char)
        return ''.join(marked_text)

    @staticmethod
    def create_evidence_extraction_prompt_with_markers(
        query: str,
        answer: str,
        chunk_content: str
    ) -> str:
        """
        Create evidence extraction prompt with character position markers
        Patent Step 2: Input chunk + user query + RAG answer to LLM, output character range (M～N)
        """
        marked_content = CharacterMarkedPromptStrategy.add_character_markers(chunk_content, marker_interval=10)

        prompt = f"""
Task: Extract the EXACT character range (start position ～ end position) of the evidence text from the ORIGINAL document.

IMPORTANT: The character positions must be counted in the ORIGINAL text (without markers), NOT in the marked text.

Question: {query}
Answer: {answer}

STEP 1: Reference text WITH position markers (for visual location only):
{marked_content}

STEP 2: ORIGINAL text (use THIS to count character positions):
{chunk_content}

CRITICAL INSTRUCTIONS:
1. Analyze the question to identify what it's asking for:
   - "何" (what) → looking for a NAME/TERM/CONCEPT
   - "いつ" (when) → looking for a TIME PERIOD
   - "どこ" (where) → looking for a LOCATION

2. Identify the CORE TERM from the answer that directly answers the question

3. Use the MARKED text (Step 1) to visually locate where the evidence appears

4. Then count the character positions in the ORIGINAL text (Step 2) to get the exact range

6. CRITICAL Rules for extraction:
   - Extract ONLY the core noun phrase that directly answers the question
   - MUST include ALL modifiers that are part of the noun (e.g., "亜熱帯" in "亜熱帯ジェット気流")
   - ABSOLUTELY DO NOT include:
     * Verbs or verb phrases (である、です、します、etc.)
     * Particles (も、が、は、を、の、etc.) UNLESS they are part of a compound noun
     * Punctuation (。、！？) before or after the term
     * Sentence connectors or context words
   - Extract the MINIMAL precise term that contains the answer
   - For "X の一種" questions, extract ONLY "X" (e.g., for "雨季の一種", extract "雨季")

7. Character position counting rules:
   - Count from position 1 (not 0) in the ORIGINAL text
   - Include the first and last character of the term
   - Example: "梅雨" at the start = positions 1～2 (not 0～1)

8. Output format (MUST follow exactly):
   Core Term: [the identified core term]
   Character Range: M～N
   Extracted Text: [the exact text from position M to N]

   OR if the core term does NOT exist in the document:
   Core Term: [the identified core term]
   Character Range: empty
   Extracted Text: empty

Example 1:
Question: 梅雨とは何季の一種か?
Answer: 雨季の一種である

STEP 1 - Marked text (for visual reference):
[0]梅雨 [SEP] 梅雨（つゆ[10]、ばいう）は、北海道[20]と小笠原諸島を除く日本[30]、朝鮮半島南部、中国[40]の南部から長江流域に[50]かけての沿海部、およ[60]び台湾など、東アジア[70]の広範囲においてみら[80]れる特有の気象現象で[90]、5月から7月にかけて[100]来る曇りや雨の多い期[110]間のこと。雨季の一種[120]である。

STEP 2 - Original text (for counting positions):
梅雨 [SEP] 梅雨（つゆ、ばいう）は、北海道と小笠原諸島を除く日本、朝鮮半島南部、中国の南部から長江流域にかけての沿海部、および台湾など、東アジアの広範囲においてみられる特有の気象現象で、5月から7月にかけて来る曇りや雨の多い期間のこと。雨季の一種である。

Analysis:
- In the marked text, I can see "雨季" appears around position [110]-[120]
- In the original text, I count: "...雨の多い期間のこと。雨季の一種である。"
- The word "雨季" starts after "。" at position 122
- Counting: 梅(1)雨(2)...[skip to position 122]雨(122)季(123)
- Core term is "雨季" only, not "の一種" or "である"

Core Term: 雨季
Character Range: 122～123
Extracted Text: 雨季
Reasoning: The question asks "何季の一種", so extract only "雨季", excluding particles "の一種" and verb "である".

Evidence Range:
"""
        return prompt

    @staticmethod
    def create_evidence_extraction_prompt_variant(
        query: str,
        answer: str,
        chunk_content: str
    ) -> str:
        """
        Variant method (変形例): LLM extracts the text string directly, not position numbers
        処理2': Extract evidence text string
        処理3': Check exact match
        処理4': Use edit distance to find most similar substring
        """
        prompt = f"""
Task: Extract the MINIMAL evidence text string that directly answers the question from the chunk.

Question: {query}
Answer: {answer}
Chunk: {chunk_content}

CRITICAL INSTRUCTIONS:
1. Analyze what the question is asking for:
   - "何" (what) → Extract a NAME/TERM/CONCEPT
   - "いつ" (when) → Extract a TIME PERIOD
   - "どこ" (where) → Extract a LOCATION

2. From the answer, identify the CORE TERM that directly answers the question

3. Extract ONLY that core term from the chunk
   - Extract the SHORTEST possible text that contains the answer
   - DO NOT include particles (の、が、は、を、も、に、と、で、から) UNLESS they are part of a compound noun
   - DO NOT include verbs (である、です、します、とされる、といい、etc.)
   - DO NOT include explanatory phrases or context
   - DO NOT include punctuation before/after the term
   - MAXIMUM length: prefer single terms or short phrases 
   - If the answer spans multiple parts of text, extract ONLY the most direct part

4. The extracted text MUST exist exactly in the chunk

5. CRITICAL: For complex answers, extract ONLY the core answer term, NOT the entire explanation
   - Example: If answer is "田植えの時期の目安とされている", extract ONLY "田植えの時期の目安" or "田植えの時期"
   - Example: If answer is "春の終わりであり夏の始まり", this may be too long - try to find the most direct term in chunk

Output format:
Core Term: [the identified core term]
Evidence Text: [the exact SHORT text extracted from chunk]

If the core term does NOT exist in the chunk:
Core Term: [the identified core term]
Evidence Text: empty

Example 1:
Question: 梅雨とは何季の一種か?
Answer: 雨季の一種である
Chunk: 梅雨（つゆ、ばいう）は...雨季の一種である。

Analysis:
- Question asks "何季" → looking for a TYPE OF SEASON
- Answer contains "雨季の一種" → core term is "雨季"
- Find "雨季" in chunk → exists

Core Term: 雨季
Evidence Text: 雨季


Example 2:
Question: 入梅は何の目安の時期か？
Answer: 入梅は田植えの時期の目安とされている。
Chunk: ...暦の上ではこの日を入梅とするが、これは水を必要とする田植えの時期の目安とされている。

Analysis:
- Question asks "何の目安の時期" → looking for PURPOSE/USAGE
- Answer contains "田植えの時期の目安" → this is the core term
- Find "田植えの時期の目安" in chunk → exists (9 characters)
- DO NOT extract the full sentence "入梅（にゅうばい）といい、社会通念上・気象学上は..." (too long!)
- Extract ONLY the direct answer term

Core Term: 田植えの時期の目安
Evidence Text: 田植えの時期の目安

Now extract the evidence:
"""
        return prompt

    @staticmethod
    def find_most_similar_substring(chunk: str, target: str, min_length: int = 2) -> tuple:
        """
        処理4': Find most similar substring in chunk using edit distance
        机械的に抽出する（※LLMは使わない）

        Args:
            chunk: Original chunk text
            target: Target text from LLM (may have hallucination)
            min_length: Minimum substring length to consider

        Returns:
            (best_match, (start, end), similarity_score)
        """
        from difflib import SequenceMatcher

        best_match = ""
        best_score = 0.0
        best_range = (0, 0)

        target_len = len(target)

        # Search for substrings with similar length to target
        for length in range(max(min_length, target_len - 5), min(len(chunk) + 1, target_len + 10)):
            for i in range(len(chunk) - length + 1):
                substring = chunk[i:i + length]

                # Calculate similarity using SequenceMatcher (edit distance)
                score = SequenceMatcher(None, substring, target).ratio()

                if score > best_score:
                    best_score = score
                    best_match = substring
                    best_range = (i + 1, i + length)  # 1-based indexing

        return best_match, best_range, best_score


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
        """Pure semantic evaluation - Fully based on LLM understanding using Few-shot Learning"""

        evaluation_prompt = f"""
        Evaluate how well the reference text can answer the question. Use the examples below to understand how to score:

        Example 1 (High Relevance - Direct Answer):
        Question: 新たに語（単語）を造ることや、既存の語を組み合わせて新たな意味の語を造ること
        Reference Text: 造語 [SEP] 造語（ぞうご）は、新たに語（単語）を造ることや、既存の語を組み合わせて新たな意味の語を造ること、また、そうして造られた語である。
        Score: 0.95
        Reason: The reference text directly defines and explains the concept asked in the question.

        Example 2 (High Relevance - Contains Answer):
        Question: 坂本龍一の出身地は？
        Reference Text: 坂本龍一 [SEP] 坂本 龍一（さかもと りゅういち、Sakamoto Ryūichi、1952年1月17日 - ）は、日本のミュージシャン。東京都出身。
        Score: 0.9
        Reason: The reference text contains the specific answer to the question about birthplace.

        Example 3 (Medium Relevance - Partial Information):
        Question: データベース管理システムの主な機能は？
        Reference Text: データベース [SEP] コンピュータを使用したデータベース・システムでは、データベース管理用のソフトウェアであるデータベース管理システムを使用する場合も多い。
        Score: 0.6
        Reason: Mentions database management systems but doesn't fully explain their main functions.

        Example 4 (Low Relevance - Different Topic):
        Question: 坂本龍一の出身地は？
        Reference Text: 造語 [SEP] 造語（ぞうご）は、新たに語（単語）を造ることや、既存の語を組み合わせて新たな意味の語を造ること。
        Score: 0.0
        Reason: Completely different topic - question asks about a person's birthplace but text discusses word creation.

        Now evaluate:
        Question: {query}
        Reference Text: {content}

        Return in JSON format:
        {{
            "relevance_score": <score between 0-1>,
            "reason": "<your reasoning>",
            "generated_answer": "<answer based on reference text, or empty string if cannot answer>"
        }}
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating semantic relevance between questions and reference texts. Follow the scoring patterns shown in the examples."},
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

        # BM25 retriever for keyword-based search (initialized on first use)
        self.bm25_retriever = None
        self.all_documents = None

        # Configuration - Balanced for accuracy and recall
        self.config = {
            'similarity_threshold': 0.2,
            'max_candidates': 8,  # Increased to 8 for better recall (find more candidate chunks)
            'use_query_expansion': True,  # Keep query expansion to ensure recall
            'use_semantic_ranking': True,
            'use_hybrid_search': True,  # Enable hybrid search (BM25 + Vector)
            'hybrid_alpha': 0.5  # Weight for combining BM25 (1-alpha) and Vector (alpha) scores
        }

    def build_vector_store(self, data_file: str, chunk_size: int = 300, chunk_overlap: int = 100, force_rebuild: bool = False) -> bool:
        """Build pure semantic vector database

        Args:
            data_file: Path to JSON data file
            chunk_size: Size of text chunks (default: 300, increased from 200 for better context)
            chunk_overlap: Overlap between chunks (default: 100, increased from 50 for continuity)
            force_rebuild: If True, rebuild even if database exists (default: False)
        """
        try:
            print(f"🏗️ Building pure semantic vector database...")
            print(f"📁 Data file: {data_file}")
            print(f"🗄️ Vector database path: {self.chroma_path}")
            print(f"📏 Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

            # Check if vector store already exists
            if not force_rebuild and os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
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

    def _initialize_bm25(self):
        """Initialize BM25 retriever from ChromaDB documents"""
        if self.bm25_retriever is not None:
            return  # Already initialized

        if not self.db:
            print("⚠️ Cannot initialize BM25: No database loaded")
            return

        try:
            # Get all documents from ChromaDB
            collection = self.db._collection
            all_results = collection.get()

            # Store documents for later retrieval
            self.all_documents = []
            texts = []

            for i, doc_text in enumerate(all_results['documents']):
                metadata = all_results['metadatas'][i] if all_results['metadatas'] else {}
                self.all_documents.append({
                    'text': doc_text,
                    'metadata': metadata,
                    'index': i
                })
                texts.append(doc_text)

            # Initialize BM25 with langchain's BM25Retriever
            try:
                from langchain_community.retrievers import BM25Retriever
            except ImportError:
                from langchain.retrievers import BM25Retriever
            from langchain.schema import Document as LangChainDoc

            # Japanese tokenizer for BM25
            try:
                import MeCab
                mecab = MeCab.Tagger()

                def japanese_tokenizer(text):
                    """Tokenize Japanese text using MeCab"""
                    try:
                        node = mecab.parseToNode(text)
                        tokens = []
                        while node:
                            if node.surface:
                                tokens.append(node.surface)
                            node = node.next
                        return tokens
                    except:
                        # Fallback: character-based tokenization
                        return list(text)

                print("✅ Using MeCab for Japanese tokenization")
                tokenizer_func = japanese_tokenizer
            except ImportError:
                print("⚠️ MeCab not available, using character-based tokenization")
                # Fallback: simple character-level tokenization for Japanese
                tokenizer_func = lambda text: list(text.replace(' ', ''))

            docs = [LangChainDoc(page_content=doc['text'], metadata=doc['metadata'])
                    for doc in self.all_documents]

            self.bm25_retriever = BM25Retriever.from_documents(
                docs,
                preprocess_func=tokenizer_func
            )
            self.bm25_retriever.k = self.config['max_candidates']

            print(f"✅ BM25 retriever initialized with {len(texts)} documents")

        except Exception as e:
            print(f"⚠️ BM25 initialization failed: {e}")
            self.bm25_retriever = None

    def _hybrid_search(self, query: str, k: int) -> List[Dict]:
        """Perform hybrid search combining BM25 and vector search"""
        # Initialize BM25 if not already done
        if self.bm25_retriever is None:
            self._initialize_bm25()

        alpha = self.config.get('hybrid_alpha', 0.5)

        # 1. Vector search
        vector_results = self.db.similarity_search_with_score(query, k=k*2)
        vector_candidates = {}
        for doc, score in vector_results:
            doc_id = doc.page_content[:100]  # Use first 100 chars as ID
            vector_candidates[doc_id] = {
                'document': doc,
                'vector_score': score,
                'bm25_score': 0.0
            }

        # 2. BM25 search
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.get_relevant_documents(query)[:k*2]
            for doc in bm25_results:
                doc_id = doc.page_content[:100]
                if doc_id in vector_candidates:
                    # Document found in both searches - update BM25 score
                    vector_candidates[doc_id]['bm25_score'] = 1.0
                else:
                    # Document only found in BM25
                    vector_candidates[doc_id] = {
                        'document': doc,
                        'vector_score': 0.5,  # Default moderate vector score
                        'bm25_score': 1.0
                    }

        # 3. Combine scores using weighted average
        combined_results = []
        for doc_id, scores in vector_candidates.items():
            # Normalize vector score (lower is better, so invert)
            normalized_vector = 1.0 - min(scores['vector_score'], 1.0)

            # Combined score: alpha * vector + (1-alpha) * bm25
            combined_score = alpha * normalized_vector + (1 - alpha) * scores['bm25_score']

            combined_results.append({
                'document': scores['document'],
                'similarity_score': scores['vector_score'],
                'combined_score': combined_score,
                'source_query': query
            })

        # Sort by combined score (higher is better)
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

        return combined_results[:k]

    def semantic_query(self, query: str, k: int = 3) -> List[SemanticChunk]:
        """Pure semantic query - Fully based on LLM understanding"""
        if not self.db:
            return []

        use_hybrid = self.config.get('use_hybrid_search', False)
        search_method = "🔀 Hybrid (BM25 + Vector)" if use_hybrid else "🔍 Pure semantic"
        print(f"{search_method} query: {query}")

        # 1. Query expansion
        expanded_queries = self._expand_query_semantically(query)
        print(f"📝 Query expansion: {len(expanded_queries)} variants")

        # 2. Multi-query retrieval (with optional hybrid search)
        all_candidates = []
        for expanded_query in expanded_queries:
            if use_hybrid:
                # Use hybrid search (BM25 + Vector)
                candidates = self._hybrid_search(expanded_query, k=self.config['max_candidates'])
                all_candidates.extend(candidates)
            else:
                # Use pure vector search
                vector_results = self.db.similarity_search_with_score(
                    expanded_query,
                    k=self.config['max_candidates']
                )
                for doc, score in vector_results:
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

            # 🎯 STEP 1: Extract unified core term from the generated answer (ONE TIME ONLY)
            import re
            print("\n🎯 STEP 1: Extracting unified core term from generated answer...")

            core_term_extraction_prompt = f"""
Task: Identify the CORE TERM that directly answers the question from the generated answer.

Question: {query}
Generated Answer: {answer}

CRITICAL INSTRUCTIONS:
1. Analyze what the question is asking for:
   - "何" (what) → Extract a NAME/TERM/CONCEPT
   - "いつ" (when) → Extract a TIME PERIOD
   - "どこ" (where) → Extract a LOCATION

2. From the generated answer, identify the SINGLE CORE TERM that directly answers the question
   - DO NOT include particles (の、が、は、を、も)
   - DO NOT include verbs (である、です、します、が、も)
   - Extract the MINIMAL precise term

3. Output ONLY the core term, nothing else

Example 1:
Question: 梅雨とは何季の一種か?
Answer: 雨季の一種である
Core Term: 雨季

Example 2:
Question: 初夏に入った5月ごろ、北上する気流は何か？
Answer: 亜熱帯ジェット気流が北上します。
Core Term: 亜熱帯ジェット気流

Now identify the core term:
"""

            try:
                core_term_response = self.llm.invoke(core_term_extraction_prompt)
                unified_core_term = core_term_response.content.strip()
                print(f"   ✅ Unified core term identified: '{unified_core_term}'")
            except Exception as e:
                print(f"   ⚠️ Failed to extract core term: {e}")
                unified_core_term = ""

            # Strategy 3 core: Extract evidence from each chunk separately
            evidences = []
            print("\n🔍 STEP 2: Searching for unified core term in each chunk...")

            for i, chunk in enumerate(semantic_chunks, 1):
                # 変形例: LLM extracts text string directly, then we find position
                # 処理2': Use variant prompt to extract text string
                evidence_variant_prompt = CharacterMarkedPromptStrategy.create_evidence_extraction_prompt_variant(
                    query=query,
                    answer=answer,
                    chunk_content=chunk.content
                )

                llm_match_ranges = []
                llm_match_texts = []
                core_term = unified_core_term  # Use the unified core term for all chunks

                try:
                    evidence_response = self.llm.invoke(evidence_variant_prompt)
                    variant_output = evidence_response.content.strip()

                    # Extract core term and evidence text from response
                    core_term_match = re.search(r'Core Term:\s*(.+?)(?:\n|$)', variant_output, re.IGNORECASE)
                    evidence_text_match = re.search(r'Evidence Text:\s*(.+?)(?:\n|$)', variant_output, re.IGNORECASE)

                    if core_term_match:
                        llm_identified_term = core_term_match.group(1).strip()
                        print(f"   📝 LLM identified in chunk: '{llm_identified_term}' (using unified: '{unified_core_term}')")

                    if evidence_text_match:
                        evidence_text = evidence_text_match.group(1).strip()
                    else:
                        evidence_text = ""

                    if evidence_text and evidence_text.lower() != "empty":
                        # 処理3': Check exact match in chunk
                        if evidence_text in chunk.content:
                            # Exact match found! Find position
                            match_pos = chunk.content.find(evidence_text)
                            start = match_pos + 1  # 1-based
                            end = match_pos + len(evidence_text)
                            llm_match_ranges.append((start, end))
                            llm_match_texts.append(evidence_text)
                            print(f"   ✅ 処理3': Exact match found: '{evidence_text}' ({start}～{end})")
                        else:
                            # 処理4': No exact match - use edit distance to find similar substring
                            print(f"   ⚠️ 処理3': No exact match for '{evidence_text}'")
                            print(f"   🔧 処理4': Finding most similar substring...")

                            best_match, best_range, similarity = CharacterMarkedPromptStrategy.find_most_similar_substring(
                                chunk.content,
                                evidence_text
                            )

                            if best_match and similarity > 0.6:  # Threshold for accepting match
                                llm_match_ranges.append(best_range)
                                llm_match_texts.append(best_match)
                                print(f"   ✅ 処理4': Found similar: '{best_match}' ({best_range[0]}～{best_range[1]}, similarity={similarity:.2f})")
                            else:
                                print(f"   ❌ 処理4': No similar substring found (similarity={similarity:.2f})")

                except Exception as llm_e:
                    print(f"   ⚠️ Evidence extraction failed: {llm_e}")

                # Use LLM results
                if llm_match_ranges:
                    # Deduplicate ranges
                    seen_ranges = set()
                    unique_ranges = []
                    unique_texts = []
                    for range_tuple, text in zip(llm_match_ranges, llm_match_texts):
                        if range_tuple not in seen_ranges:
                            seen_ranges.add(range_tuple)
                            unique_ranges.append(range_tuple)
                            unique_texts.append(text)

                    char_ranges = unique_ranges
                    extracted_evidence = "\n".join(unique_texts)
                    is_empty = False
                    print(f"   ✅ LLM found {len(unique_ranges)} unique evidence range(s) (deduplicated from {len(llm_match_ranges)})")
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
                    'is_empty': is_empty,
                    'evidence_variant_prompt': evidence_variant_prompt,  # Store the prompt used for extraction
                    'llm_response': variant_output,  # Store the LLM raw response
                    'core_term': core_term  # Store the identified core term
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
    
    def query_with_answer(self, query: str, k: int = 10, relevance_threshold: float = 0.6, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Complete query workflow - Retrieve top k documents, filter by semantic relevance (≥ 0.6) OR similarity score (≥ 0.7) for high precision"""
        start_time = time.time()

        # 1. Semantic retrieval - get top k candidate documents
        semantic_chunks = self.semantic_query(query, k)

        # 2. Filter: keep documents with high semantic relevance (≥ 0.5) OR similarity score (≥ 0.7) for precision
        # Using OR logic: accept chunks that are either semantically relevant OR have high vector similarity
        filtered_chunks = [
            chunk for chunk in semantic_chunks
            if chunk.semantic_relevance >= relevance_threshold or chunk.similarity_score >= similarity_threshold
        ]

        print(f"\n🔍 Retrieval statistics:")
        print(f"  📊 Initial retrieval: {len(semantic_chunks)} documents")
        print(f"  ✅ After filtering (relevance≥{relevance_threshold} OR similarity≥{similarity_threshold}): {len(filtered_chunks)} documents")

        # 3. If no qualifying documents found, return apology message
        if not filtered_chunks:
            processing_time = time.time() - start_time
            print(f"  ⚠️ No documents with relevance≥{relevance_threshold} OR similarity≥{similarity_threshold} found, returning empty result")
            return {
                'answer': 'Sorry, no relevant information is currently available.',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': f'No documents with semantic relevance≥{relevance_threshold} OR similarity≥{similarity_threshold} found',
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
