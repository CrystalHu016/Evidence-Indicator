#!/usr/bin/env python3
"""
优化版高级RAG系统 - 减少API调用，提升性能
"""

import os
import re
import time
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from pydantic import SecretStr
import openai
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class EvidenceRange:
    """根拠情報の文字列範囲"""
    start_char: int
    end_char: int
    evidence_text: str
    source_document: str


@dataclass
class RAGResult:
    """RAG検索結果"""
    answer: str
    hit_chunks: List[str]
    source_document: str
    evidence_range: EvidenceRange
    strategy_used: str
    confidence_score: float


class FastChunkEvaluator:
    """高速チャンク評価器（API呼出し削減版）"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def can_answer_question_fast(self, chunk: str, question: str) -> Tuple[bool, float]:
        """
        TF-IDFベースの高速評価
        """
        try:
            # TF-IDF類似度を計算
            documents = [question, chunk]
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # 基本的な判定ロジック
            if similarity > 0.3:
                return True, similarity
            else:
                return False, similarity
                
        except Exception as e:
            print(f"高速評価エラー: {e}")
            return False, 0.0


class OptimizedRAGSystem:
    """最適化されたRAGシステム"""
    
    # クラス変数でインスタンスをキャッシュ
    _instance = None
    _initialized = False
    
    def __new__(cls, openai_api_key: str, chroma_path: str, data_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, openai_api_key: str, chroma_path: str, data_path: str):
        if self._initialized:
            return
            
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.data_path = data_path
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.fast_evaluator = FastChunkEvaluator()
        
        print("🚀 初期化中...")
        
        # Chroma DB接続
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        
        # 文書データの読み込み（キャッシュ）
        self.documents = self._load_documents()
        self.document_map = self._create_document_map()
        
        print("✅ 初期化完了")
        self._initialized = True
    
    def _load_documents(self) -> List[Document]:
        """文書データを読み込み"""
        try:
            loader = JSONLoader(
                file_path=self.data_path,
                jq_schema='.[ ]',
                content_key='output'
            )
            return loader.load()
        except Exception as e:
            print(f"文書読み込みエラー: {e}")
            return []
    
    def _create_document_map(self) -> Dict[str, Document]:
        """文書内容から文書オブジェクトへのマッピングを作成"""
        doc_map = {}
        for doc in self.documents:
            # 文書の最初の100文字をキーとして使用
            key = doc.page_content[:100]
            doc_map[key] = doc
        return doc_map
    
    def strategy_1_optimized(self, query: str) -> Optional[RAGResult]:
        """
        最適化された対応策１：完全文書アプローチ
        """
        # 通常の検索
        search_results = self.db.similarity_search_with_relevance_scores(query, k=1)
        
        if not search_results:
            return None
        
        # 最も関連性の高いチャンクを選択
        hit_chunk = search_results[0][0]
        hit_score = search_results[0][1]
        
        # チャンクの元文書を効率的に特定
        source_doc = self._find_source_document_fast(hit_chunk.page_content)
        if not source_doc:
            return None
        
        full_document_text = source_doc.page_content
        
        # 簡易的な根拠範囲抽出（API呼出し削減）
        evidence_range = self._extract_evidence_range_simple(full_document_text, query)
        
        # 回答生成（1回のAPI呼出しのみ）
        answer = self._generate_answer(full_document_text, query)
        
        return RAGResult(
            answer=answer,
            hit_chunks=[hit_chunk.page_content],
            source_document=full_document_text,
            evidence_range=evidence_range,
            strategy_used="最適化完全文書アプローチ",
            confidence_score=hit_score
        )
    
    def strategy_2_optimized(self, query: str) -> Optional[RAGResult]:
        """
        最適化された対応策２：高速適応チャンキング
        """
        print("=== 最適化対応策２：高速適応チャンキング ===")
        
        # 最初に関連文書を特定
        search_results = self.db.similarity_search_with_relevance_scores(query, k=1)
        if not search_results:
            return None
        
        hit_chunk = search_results[0][0]
        source_doc = self._find_source_document_fast(hit_chunk.page_content)
        if not source_doc:
            return None
        
        document_text = source_doc.page_content
        
        # より効率的なチャンクサイズ選択
        chunk_sizes = [1000, 500]  # 段階数を削減
        
        best_result = None
        
        for step, chunk_size in enumerate(chunk_sizes, 1):
            print(f"STEP {step}: チャンクサイズ {chunk_size}")
            
            # 動的チャンキング（単一文書のみ）
            chunks = self._create_chunks_from_document(document_text, chunk_size)
            if not chunks:
                continue
            
            # 最良チャンクを選択
            best_chunk = self._select_best_chunk_fast(chunks, query)
            if not best_chunk:
                continue
            
            # 高速評価
            can_answer, confidence = self.fast_evaluator.can_answer_question_fast(
                best_chunk['content'], query
            )
            
            print(f"回答可能性: {can_answer}, 信頼度: {confidence:.3f}")
            
            if can_answer and confidence > 0.4:
                # 十分な信頼度の場合、結果を保存
                best_result = best_chunk
                best_result['confidence'] = confidence
                best_result['chunk_size'] = chunk_size
                
                # 次のレベルも試す
                continue
            else:
                # 信頼度が低い場合、前の結果があれば使用
                break
        
        if not best_result:
            return None
        
        # 回答生成
        answer = self._generate_answer(best_result['content'], query)
        evidence_range = self._extract_evidence_range_simple(best_result['content'], query)
        
        return RAGResult(
            answer=answer,
            hit_chunks=[best_result['content']],
            source_document=document_text,
            evidence_range=evidence_range,
            strategy_used=f"最適化適応チャンキング (サイズ: {best_result['chunk_size']})",
            confidence_score=best_result['confidence']
        )
    
    def _find_source_document_fast(self, chunk_content: str) -> Optional[Document]:
        """高速文書検索"""
        # まず完全一致を試す
        for doc in self.documents:
            if chunk_content in doc.page_content:
                return doc
        
        # 部分一致も試す
        chunk_start = chunk_content[:50]
        for doc in self.documents:
            if chunk_start in doc.page_content:
                return doc
        
        return None
    
    def _create_chunks_from_document(self, document: str, chunk_size: int) -> List[Dict[str, Any]]:
        """単一文書からチャンクを作成"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_size // 10,  # 10%のオーバーラップ
            length_function=len,
        )
        
        # 単一文書をチャンキング
        doc_obj = Document(page_content=document)
        chunks = text_splitter.split_documents([doc_obj])
        
        result = []
        for chunk in chunks:
            result.append({
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })
        
        return result
    
    def _select_best_chunk_fast(self, chunks: List[Dict[str, Any]], query: str) -> Optional[Dict[str, Any]]:
        """TF-IDFベースの高速チャンク選択"""
        if not chunks:
            return None
        
        try:
            # 全チャンクとクエリのTF-IDF
            texts = [query] + [chunk['content'] for chunk in chunks]
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # クエリと各チャンクの類似度
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
            
            # 最高スコアのチャンクを選択
            best_idx = np.argmax(similarities[0])
            best_chunk = chunks[best_idx].copy()
            best_chunk['similarity'] = similarities[0][best_idx]
            
            return best_chunk
            
        except Exception as e:
            print(f"チャンク選択エラー: {e}")
            return chunks[0] if chunks else None
    
    def _extract_evidence_range_simple(self, document: str, query: str) -> EvidenceRange:
        """簡易的な根拠範囲抽出（API呼出しなし）"""
        # クエリのキーワードを抽出
        query_words = set(re.findall(r'\w+', query.lower()))
        
        # 文書を文に分割
        sentences = re.split(r'[。！？.!?]', document)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return EvidenceRange(0, min(100, len(document)), document[:100], document)
        
        # 最も関連性の高い文を選択
        best_sentence = ""
        best_score = 0
        best_start = 0
        
        current_pos = 0
        for sentence in sentences:
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(query_words & sentence_words)
            
            if overlap > best_score:
                best_score = overlap
                best_sentence = sentence
                best_start = document.find(sentence, current_pos)
            
            current_pos = document.find(sentence, current_pos) + len(sentence)
        
        if best_sentence and best_start >= 0:
            return EvidenceRange(
                start_char=best_start,
                end_char=best_start + len(best_sentence),
                evidence_text=best_sentence,
                source_document=document
            )
        
        # フォールバック
        evidence_text = document[:min(100, len(document))]
        return EvidenceRange(0, len(evidence_text), evidence_text, document)
    
    def _generate_answer(self, context: str, query: str) -> str:
        """コンテキストから回答を生成"""
        # 言語検出
        def is_japanese(text):
            japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF')
            return japanese_chars > len(text) * 0.1
        
        if is_japanese(query):
            system_prompt = "あなたは日本語で正確に回答する知的なアシスタントです。質問に日本語で簡潔に答えてください。"
            user_prompt = f"""
            コンテキスト: {context}
            質問: {query}
            
            回答:
            """
        else:
            system_prompt = "You are an intelligent assistant that provides accurate and concise answers in English."
            user_prompt = f"""
            Context: {context}
            Question: {query}
            
            Answer:
            """
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"回答生成エラー: {e}")
            return "回答を生成できませんでした。"
    
    def query(self, query_text: str) -> Optional[RAGResult]:
        """
        メインクエリ処理（超高速版）
        """
        # 対応策1のみを使用（最も高速）
        result = self.strategy_1_optimized(query_text)
        
        if result and result.confidence_score > 0.5:
            return result
        
        # 低信頼度の場合のみ対応策2を試行
        if not result or result.confidence_score < 0.3:
            result_2 = self.strategy_2_optimized(query_text)
            if result_2 and result_2.confidence_score > (result.confidence_score if result else 0):
                return result_2
        
        return result
    
    def format_output(self, result: RAGResult) -> str:
        """結果を指定されたフォーマットで出力"""
        if not result:
            return "回答を生成できませんでした。"
        
        output = f"""
【回答】
{result.answer}

【検索ヒットのチャンクを含む文書】
{result.source_document}

【根拠情報の文字列範囲】
{result.evidence_range.start_char + 1}文字目〜{result.evidence_range.end_char}文字目

【根拠情報】
{result.evidence_range.evidence_text}
        """
        
        return output.strip()


def test_optimized_rag():
    """最適化RAGシステムのテスト"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未設定")
        return
    
    # システム初期化
    rag_system = OptimizedRAGSystem(
        openai_api_key=api_key,
        chroma_path="chroma",
        data_path="./data/test_sample.json"
    )
    
    # テストクエリ
    test_queries = [
        "コンバインとは何ですか",
        "音位転倒について説明してください"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"テストクエリ: {query}")
        print(f"{'='*50}")
        
        start_time = time.time()
        result = rag_system.query(query)
        total_time = time.time() - start_time
        
        if result:
            print(rag_system.format_output(result))
            print(f"\n【処理時間】{total_time:.2f}秒")
        else:
            print("❌ 回答を生成できませんでした")


if __name__ == "__main__":
    test_optimized_rag()