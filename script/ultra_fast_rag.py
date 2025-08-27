#!/usr/bin/env python3
"""
超高速RAG系统 - 专门针对性能优化
"""

import os
import re
import time
from typing import Optional, Tuple
from pydantic import SecretStr
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


class UltraFastRAG:
    """超高速RAGシステム - 最小限の機能で最大のパフォーマンス"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, openai_api_key: str, chroma_path: str):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, openai_api_key: str, chroma_path: str):
        if self._initialized:
            return
            
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        self._initialized = True
    
    def query(self, query_text: str) -> Tuple[str, str, str, int, int]:
        """
        超高速クエリ処理
        Returns: (answer, source_document, evidence, start_pos, end_pos)
        """
        # 1. 単一の最良検索結果のみ取得
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=1)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        # 2. 最初の結果のみ使用
        hit_doc = search_results[0][0]
        confidence = search_results[0][1]
        
        source_text = hit_doc.page_content
        
        # 3. 簡易根拠抽出（正規表現ベース）
        evidence_text, start_pos, end_pos = self._extract_evidence_fast(source_text, query_text)
        
        # 4. 簡易回答生成
        answer = self._generate_answer_fast(evidence_text, query_text)
        
        return answer, source_text, evidence_text, start_pos, end_pos
    
    def _extract_evidence_fast(self, text: str, query: str) -> Tuple[str, int, int]:
        """正規表現ベースの超高速根拠抽出（簡易スコアリングで精度向上）"""
        # 質問のキーワードを抽出
        keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query)
        keywords = [kw for kw in keywords if len(kw) > 1 and kw not in ['とは', '何', 'です', 'ます', 'について']]

        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s for s in sentences if s]

        if not sentences:
            snippet = text[:100]
            return snippet, 0, len(snippet)

        # 作物に関する質問の簡易ヒューリスティック
        query_has_crop = any(term in query for term in ['作物', 'どのようなもの', '何があります', '対応した'])
        crop_markers = ['作物', '稲', '麦', '大豆', '小豆', '菜種', 'トウモロコシ', 'など', '・']

        best_sentence = None
        best_score = -1

        for sentence in sentences:
            score = 0
            # キーワード一致数
            for kw in keywords:
                if kw in sentence:
                    score += 1
            # 作物系は優先
            if query_has_crop:
                for marker in crop_markers:
                    if marker in sentence:
                        score += 2
            # 長すぎる文は軽く減点（読みやすさ優先）
            if len(sentence) > 200:
                score -= 1

            if score > best_score:
                best_sentence = sentence
                best_score = score

        if best_sentence is None:
            best_sentence = sentences[0]

        start_pos = text.find(best_sentence)
        if start_pos < 0:
            best_sentence = sentences[0]
            start_pos = 0
        end_pos = start_pos + len(best_sentence)
        return best_sentence.strip(), start_pos, end_pos
    
    def _generate_answer_fast(self, evidence: str, query: str) -> str:
        """超簡易回答生成"""
        # 定義質問の場合、証拠の最初の文をそのまま返す
        if any(pattern in query for pattern in ['とは何', 'とは', '何ですか', '何でしょうか']):
            # 最初の文を抽出
            first_sentence = re.split(r'[。！？.!?]', evidence)[0]
            if first_sentence:
                return first_sentence + '。'
        
        # その他の場合、短いLLM呼び出し
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "簡潔に日本語で答えてください。"},
                    {"role": "user", "content": f"証拠: {evidence}\n質問: {query}\n回答:"}
                ],
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except:
            # API失敗時は証拠をそのまま返す
            return evidence[:100] + ('...' if len(evidence) > 100 else '')


def test_ultra_fast():
    """超高速システムのテスト"""
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未設定")
        return
    
    print("⚡ 超高速RAGシステムテスト")
    print("=" * 40)
    
    rag = UltraFastRAG(api_key, "chroma")
    
    queries = [
        "コンバインとは何ですか",
        "音位転倒について説明してください"
    ]
    
    for query in queries:
        print(f"\nクエリ: {query}")
        
        start_time = time.time()
        answer, source, evidence, start, end = rag.query(query)
        elapsed = time.time() - start_time
        
        print(f"⏱️  処理時間: {elapsed:.2f}秒")
        print(f"【回答】{answer}")
        print(f"【根拠範囲】{start+1}〜{end}文字目")
        print(f"【根拠】{evidence}")
        print("-" * 40)


if __name__ == "__main__":
    test_ultra_fast()