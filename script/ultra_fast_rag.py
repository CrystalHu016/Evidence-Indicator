#!/usr/bin/env python3
"""
超高速RAG系统 - 专门针对性能优化
现在集成LLM智能ranking功能
"""

import os
import re
import time
from typing import Optional, Tuple, List, Dict, Any
from pydantic import SecretStr
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

try:
    from llm_evidence_ranker import LLMEvidenceRanker
    LLM_RANKING_AVAILABLE = True
except ImportError:
    try:
        from script.llm_evidence_ranker import LLMEvidenceRanker
        LLM_RANKING_AVAILABLE = True
    except ImportError:
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from llm_evidence_ranker import LLMEvidenceRanker
            LLM_RANKING_AVAILABLE = True
        except ImportError:
            print("⚠️ LLM ranking 模块未找到，将使用原始规则评分")
            LLM_RANKING_AVAILABLE = False


class UltraFastRAG:
    """超高速RAGシステム - 最小限の機能で最大のパフォーマンス"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls, openai_api_key: str, chroma_path: str, use_llm_ranking: bool = True):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, openai_api_key: str, chroma_path: str, use_llm_ranking: bool = True):
        if self._initialized:
            return
            
        self.openai_api_key = openai_api_key
        self.use_llm_ranking = use_llm_ranking
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        
        # 初始化LLM ranking系统
        if use_llm_ranking and LLM_RANKING_AVAILABLE:
            try:
                self.llm_ranker = LLMEvidenceRanker(openai_api_key)
                print("✅ LLM智能ranking已启用")
            except Exception as e:
                print(f"⚠️ LLM ranking初始化失败: {e}，将使用原始方法")
                self.use_llm_ranking = False
        else:
            self.use_llm_ranking = False
            
        self._initialized = True
    
    def query(self, query_text: str, k: int = 5) -> Tuple[str, str, str, int, int]:
        """
        增强查询处理 - 现在支持LLM智能ranking
        Returns: (answer, source_document, evidence, start_pos, end_pos)
        """
        if self.use_llm_ranking:
            return self._query_with_llm_ranking(query_text, k)
        else:
            return self._query_fast_original(query_text)
    
    def _query_with_llm_ranking(self, query_text: str, k: int) -> Tuple[str, str, str, int, int]:
        """使用LLM ranking的增强查询"""
        print(f"🧠 使用LLM智能ranking处理查询: '{query_text}'")
        
        # 1. 获取更多候选结果
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        # 2. 转换为LLM ranker的格式
        chunks = []
        for doc, score in search_results:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            }
            chunks.append(chunk)
        
        # 3. 使用LLM进行智能排序
        try:
            ranked_chunks = self.llm_ranker.rank_and_highlight_chunks(query_text, chunks, top_k=1)
            
            if ranked_chunks:
                best_chunk = ranked_chunks[0]
                source_text = best_chunk["content"]
                evidence_text = best_chunk.get("highlighted_content", source_text)
                
                # 计算位置（简化处理）
                start_pos = 0
                end_pos = len(evidence_text)
                
                # 生成答案
                answer = self._generate_answer_fast(evidence_text, query_text)
                
                print(f"✅ LLM ranking完成，最佳匹配评分: {best_chunk.get('final_score', 0):.3f}")
                
                return answer, source_text, evidence_text, start_pos, end_pos
            
        except Exception as e:
            print(f"⚠️ LLM ranking失败: {e}，降级使用原始方法")
        
        # 降级到原始方法
        return self._query_fast_original(query_text)
    
    def _query_fast_original(self, query_text: str) -> Tuple[str, str, str, int, int]:
        """原始的超高速查询方法"""
        # 1. 单一的最佳检索结果
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=1)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        # 2. 使用第一个结果
        hit_doc = search_results[0][0]
        confidence = search_results[0][1]
        
        source_text = hit_doc.page_content
        
        # 3. 原始的证据抽取
        evidence_text, start_pos, end_pos = self._extract_evidence_fast(source_text, query_text)
        
        # 4. 生成答案
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