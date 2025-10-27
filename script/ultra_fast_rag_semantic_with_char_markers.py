#!/usr/bin/env python3
"""
Pure Semantic RAG System with Character Position Markers
Strategy: Add character position markers to chunks for more accurate evidence extraction
Based on patent: チャンクの部分文字列としての根拠提示 (Presenting evidence as substring of chunk)
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

    Strategy 1: Add character markers to help LLM identify exact positions
    Strategy 2: Use line-by-line format with character counts
    """

    @staticmethod
    def add_character_markers(text: str, marker_interval: int = 10) -> str:
        """
        Add character position markers to text
        Example: [0]梅雨 [SEP] 梅雨（つゆ[10]、ばいう）は、北海道[20]と小笠原諸島...
        """
        marked_text = []
        for i, char in enumerate(text):
            if i % marker_interval == 0 and i > 0:
                marked_text.append(f"[{i}]")
            marked_text.append(char)
        return ''.join(marked_text)

    @staticmethod
    def add_line_numbers_with_char_count(text: str) -> Tuple[str, List[int]]:
        """
        Add line numbers and track character positions
        Returns: (marked_text, line_start_positions)

        Example:
        [Line 1, Chars 0-50] 梅雨 [SEP] 梅雨（つゆ、ばいう）は、北海道と小笠原諸島を除く日本、
        [Line 2, Chars 51-100] 朝鮮半島南部、中国の南部から長江流域にかけての沿海部、および台湾など、
        """
        lines = text.split('\n')
        marked_lines = []
        line_start_positions = []
        char_pos = 0

        for i, line in enumerate(lines, 1):
            line_length = len(line)
            line_start_positions.append(char_pos)
            marked_lines.append(f"[Line {i}, Chars {char_pos}-{char_pos + line_length - 1}] {line}")
            char_pos += line_length + 1  # +1 for newline

        return '\n'.join(marked_lines), line_start_positions

    @staticmethod
    def create_evidence_extraction_prompt_with_markers(
        query: str,
        answer: str,
        chunk_content: str,
        use_char_markers: bool = True
    ) -> str:
        """
        Create evidence extraction prompt with character position markers
        Patent Step 2: Input chunk + user query + RAG answer to LLM, output character range (M～N)

        Strategy: Use TWO-STEP approach
        1. First show the marked text to help LLM locate the evidence visually
        2. Then ask LLM to find the evidence in the ORIGINAL unmarked text and report positions
        """

        if use_char_markers:
            # Strategy: Show both marked and unmarked versions
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
        else:
            # Strategy 2: Use line numbers with character counts
            marked_content, line_positions = CharacterMarkedPromptStrategy.add_line_numbers_with_char_count(chunk_content)

            prompt = f"""
Task: Extract the EXACT character range (start position ～ end position) of the evidence text from the document.

Question: {query}
Answer: {answer}

Document with Line Numbers and Character Positions:
{marked_content}

CRITICAL INSTRUCTIONS:
1. Analyze the question to identify what it's asking for:
   - "何" (what) → looking for a NAME/TERM/CONCEPT
   - "いつ" (when) → looking for a TIME PERIOD
   - "どこ" (where) → looking for a LOCATION

2. Identify the CORE TERM from the answer that directly answers the question

3. Find that EXACT core term in the document using the line information

4. Calculate the EXACT character positions (count from 0 in the original text)

5. Output format (MUST follow exactly):
   Core Term: [the identified core term]
   Character Range: M～N
   Extracted Text: [the exact text from position M to N]

   OR if the core term does NOT exist in the document:
   Core Term: [the identified core term]
   Character Range: empty
   Extracted Text: empty

Evidence Range:
"""

        return prompt


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
                "final_score": llm_score,
                "rank_order": 1
            }

            print(f"✅ Semantic evaluation complete: LLM score {llm_score:.3f}")
            return [enhanced_chunk]

        except Exception as e:
            print(f"❌ Semantic evaluation failed: {e}")
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
        """Pure semantic evaluation using Few-shot Learning"""

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
        Reason: Completely different topic.

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
                    {"role": "system", "content": "You are an expert at evaluating semantic relevance between questions and reference texts."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            relevance_score, reason, generated_answer = self._parse_semantic_response(result_text)

            return relevance_score, reason, generated_answer

        except Exception as e:
            print(f"LLM semantic evaluation failed: {e}")
            fallback_answer = content[:100] + "..." if len(content) > 100 else content
            return vector_score, "LLM evaluation failed, using vector similarity", fallback_answer

    def _parse_semantic_response(self, result_text: str) -> Tuple[float, str, str]:
        """Parse LLM semantic response"""
        try:
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


class ImprovedSemanticRAG:
    """
    Improved Semantic RAG with Character Position Marking
    Patent Implementation: Process 1-3 for substring evidence extraction
    """

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

        # Character marking strategy
        self.char_marker_strategy = CharacterMarkedPromptStrategy()

        # Configuration
        self.config = {
            'similarity_threshold': 0.2,
            'max_candidates': 8,
            'use_query_expansion': True,
            'use_semantic_ranking': True,
            'use_char_markers': True  # Enable character marker strategy
        }

    def build_vector_store(self, data_file: str, chunk_size: int = 200, chunk_overlap: int = 50) -> bool:
        """Build vector database - reuse existing implementation"""
        # Check if vector store already exists
        alternative_paths = [
            "./chroma",
            "./chroma_semantic",
            "./chroma_integrated",
            "./chroma_semantic_test",
            "./chroma_pure_semantic",
            "./chroma_improved_semantic"
        ]

        for alt_path in alternative_paths:
            if os.path.exists(alt_path) and os.listdir(alt_path):
                print(f"✅ Using existing vector database: {alt_path}")
                self.chroma_path = alt_path
                self.db = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=self.embedding_function
                )
                return True

        print(f"❌ No existing vector database found. Please build one first.")
        return False

    def semantic_query(self, query: str, k: int = 3) -> List[SemanticChunk]:
        """Semantic query - returns top k chunks"""
        if not self.db:
            return []

        print(f"🔍 Semantic query: {query}")

        # Simple retrieval for testing
        candidates = self.db.similarity_search_with_score(query, k=self.config['max_candidates'])

        all_candidates = []
        for doc, score in candidates:
            all_candidates.append({
                'document': doc,
                'similarity_score': score,
                'source_query': query
            })

        # Semantic evaluation
        semantic_chunks = self._evaluate_semantic_relevance(query, all_candidates)

        # Sort and return
        semantic_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return semantic_chunks[:k]

    def _evaluate_semantic_relevance(self, query: str, candidates: List[Dict]) -> List[SemanticChunk]:
        """LLM semantic relevance evaluation"""
        print("🧠 LLM semantic evaluation in progress...")

        semantic_chunks = []

        for i, candidate in enumerate(candidates):
            doc = candidate['document']
            similarity_score = candidate['similarity_score']

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

    def generate_answer_with_char_markers(
        self,
        query: str,
        semantic_chunks: List[SemanticChunk]
    ) -> Dict[str, Any]:
        """
        Generate answer with character marker-based evidence extraction
        Patent Implementation: Process 2 - Extract character range (M～N) from chunk
        """
        if not semantic_chunks:
            return {
                'answer': 'Sorry, no relevant information found.',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': 'No relevant documents found',
                'evidences': []
            }

        # Build context
        context_parts = []
        for i, chunk in enumerate(semantic_chunks, 1):
            context_parts.append(f"Document {i}: {chunk.content}")

        context = "\n\n".join(context_parts)

        # Generate overall answer
        answer_prompt = f"""
        User's Question: {query}

        Reference Documents:
        {context}

        Answer the user's question based on the reference documents. Requirements:
        1. Carefully read the reference documents and understand the core of the user's question
        2. Based on the document content, directly answer the core point in 1-2 concise sentences
        3. The answer must be completely based on the document content
        4. Use the original document's expressions and terminology

        Provide a concise answer (1-2 sentences) directly:
        """

        try:
            response = self.llm.invoke(answer_prompt)
            answer = response.content.strip()

            # Extract evidence using character marker strategy
            evidences = []
            print("\n🔍 Starting character marker-based evidence extraction...")

            for i, chunk in enumerate(semantic_chunks, 1):
                # Create prompt with character markers
                evidence_prompt = self.char_marker_strategy.create_evidence_extraction_prompt_with_markers(
                    query=query,
                    answer=answer,
                    chunk_content=chunk.content,
                    use_char_markers=self.config['use_char_markers']
                )

                try:
                    evidence_response = self.llm.invoke(evidence_prompt)
                    range_output = evidence_response.content.strip()

                    # Parse response
                    core_term = ""
                    char_ranges = []
                    extracted_text = ""
                    is_empty = False

                    # Extract core term
                    core_term_match = re.search(r'Core Term:\s*(.+?)(?:\n|$)', range_output, re.IGNORECASE)
                    if core_term_match:
                        core_term = core_term_match.group(1).strip()

                    # Extract character range
                    range_match = re.search(r'Character Range:\s*(\d+)～(\d+)', range_output, re.IGNORECASE)
                    if range_match:
                        start = int(range_match.group(1))
                        end = int(range_match.group(2))

                        # Validate range
                        if 1 <= start <= len(chunk.content) and start <= end <= len(chunk.content):
                            char_ranges.append((start, end))
                            extracted_text = chunk.content[start-1:end]
                            print(f"   ✅ Extracted: {start}～{end} = '{extracted_text}'")
                        else:
                            print(f"   ⚠️ Invalid range: {start}～{end} (text length: {len(chunk.content)})")
                            is_empty = True
                    else:
                        # Check if explicitly empty
                        if 'empty' in range_output.lower():
                            is_empty = True
                            print(f"   ❌ No evidence found in chunk {i}")

                    evidence_info = {
                        'chunk_id': i,
                        'chunk_content': chunk.content,
                        'extracted_evidence': extracted_text,
                        'char_ranges': char_ranges,
                        'similarity_score': float(chunk.similarity_score),
                        'semantic_relevance': float(chunk.semantic_relevance),
                        'is_empty': is_empty,
                        'core_term': core_term,
                        'llm_response': range_output
                    }

                    evidences.append(evidence_info)

                except Exception as e:
                    print(f"   ⚠️ Evidence extraction failed for chunk {i}: {e}")
                    evidences.append({
                        'chunk_id': i,
                        'chunk_content': chunk.content,
                        'extracted_evidence': '',
                        'char_ranges': [],
                        'is_empty': True,
                        'error': str(e)
                    })

            # Count valid evidences
            valid_evidences_count = sum(1 for e in evidences if not e.get('is_empty', True))
            print(f"\n📊 Evidence extraction complete: {valid_evidences_count}/{len(evidences)} chunks contain valid evidence")

            return {
                'answer': answer,
                'evidence_text': semantic_chunks[0].content,
                'source_document': context,
                'confidence': semantic_chunks[0].metadata.get('confidence', 0.0),
                'reasoning': f'Generated from {len(semantic_chunks)} relevant documents with character marker strategy',
                'model': 'ImprovedSemanticRAG-CharMarkers',
                'processing_time': 0.0,
                'chunks_used': len(semantic_chunks),
                'evidences': evidences,
                'valid_evidences_count': valid_evidences_count
            }

        except Exception as e:
            print(f"⚠️ Answer generation failed: {e}")
            return {
                'answer': 'Sorry, an error occurred during answer generation.',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': f'Generation failed: {str(e)}',
                'evidences': []
            }

    def query_with_answer(
        self,
        query: str,
        k: int = 10,
        relevance_threshold: float = 0.6,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Complete query workflow with character marker-based evidence extraction"""
        start_time = time.time()

        # 1. Semantic retrieval
        semantic_chunks = self.semantic_query(query, k)

        # 2. Filter by relevance
        filtered_chunks = [
            chunk for chunk in semantic_chunks
            if chunk.semantic_relevance >= relevance_threshold or chunk.similarity_score >= similarity_threshold
        ]

        print(f"\n🔍 Retrieval statistics:")
        print(f"  📊 Initial retrieval: {len(semantic_chunks)} documents")
        print(f"  ✅ After filtering: {len(filtered_chunks)} documents")

        if not filtered_chunks:
            processing_time = time.time() - start_time
            return {
                'answer': 'Sorry, no relevant information is currently available.',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': f'No documents with relevance≥{relevance_threshold} OR similarity≥{similarity_threshold} found',
                'processing_time': processing_time,
                'evidences': []
            }

        # 3. Generate answer with character marker-based evidence extraction
        result = self.generate_answer_with_char_markers(query, filtered_chunks)

        # 4. Add processing time
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time

        return result


def test_improved_rag():
    """Test improved RAG with character markers"""
    print("🧪 Testing Improved RAG with Character Position Markers")
    print("=" * 80)

    # Load environment variables
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    # Initialize system
    rag = ImprovedSemanticRAG(api_key)

    # Try to load existing vector store
    if not rag.build_vector_store(""):
        print("❌ Failed to load vector store")
        return

    # Test with the problematic query
    test_query = "梅雨とは何季の一種か?"

    print(f"\n{'='*80}")
    print(f"🔍 Test Query: {test_query}")
    print(f"{'='*80}\n")

    result = rag.query_with_answer(test_query, k=3)

    print(f"\n{'='*80}")
    print("📊 RESULTS")
    print(f"{'='*80}")
    print(f"⏱️  Processing time: {result['processing_time']:.2f}s")
    print(f"💬 Answer: {result['answer']}")
    print(f"📊 Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"🧠 Reasoning: {result['reasoning']}")
    print(f"\n📝 Evidence Extraction Results:")
    print(f"{'='*80}")

    for evidence in result.get('evidences', []):
        chunk_id = evidence.get('chunk_id', '?')
        is_empty = evidence.get('is_empty', True)
        status = "❌ No evidence" if is_empty else "✅ Evidence found"

        print(f"\n{status} - Chunk {chunk_id}:")
        print(f"  Core Term: {evidence.get('core_term', 'N/A')}")
        print(f"  Character Ranges: {evidence.get('char_ranges', [])}")
        print(f"  Extracted Text: {evidence.get('extracted_evidence', 'N/A')}")
        print(f"  Similarity Score: {evidence.get('similarity_score', 0.0):.3f}")

        if not is_empty:
            print(f"\n  Original Chunk (first 200 chars):")
            chunk_preview = evidence.get('chunk_content', '')[:200]
            print(f"  {chunk_preview}...")


if __name__ == "__main__":
    test_improved_rag()
