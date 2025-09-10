#!/usr/bin/env python3
"""
超高速RAG系统 - 整合版本
集成了LLM评分、精细chunking、向量搜索的完整系统
"""

import os
import re
import json
import time
from typing import Optional, Tuple, List, Dict, Any
from pydantic import SecretStr
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class IntegratedLLMEvidenceRanker:
    """内置LLM证据排序系统"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
    
    def rank_and_highlight_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
        """LLMを使用してchunksを智能ソート・ハイライト"""
        if not chunks:
            return []
        
        print(f"🧠 LLMを使用して上位2個の最高ベクトルスコアchunksを智能ソート、最適マッチを返す...")
        
        # ベクトルスコアでソート、上位2個のみLLM評価
        chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        top_chunks = chunks[:2]
        
        ranked_chunks = []
        for i, chunk in enumerate(top_chunks):
            try:
                llm_score, relevance_reason, highlighted_content = self._evaluate_chunk_with_llm(
                    query, chunk["content"], chunk.get("similarity_score", 0.0)
                )
                
                enhanced_chunk = {
                    **chunk,
                    "llm_score": llm_score,
                    "relevance_reason": relevance_reason,
                    "highlighted_content": highlighted_content,
                    "final_score": (chunk.get("similarity_score", 0.0) * 0.4) + (llm_score * 0.6),
                    "rank_order": i + 1
                }
                ranked_chunks.append(enhanced_chunk)
                
                print(f"  ✓ Chunk {i+1}: Vector={chunk.get('similarity_score', 0.0):.3f}, LLM={llm_score:.3f}, Final={enhanced_chunk['final_score']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Chunk {i+1} LLM評価失敗: {e}")
                enhanced_chunk = {
                    **chunk,
                    "llm_score": chunk.get("similarity_score", 0.0),
                    "relevance_reason": "LLM評価失敗、ベクトル類似度を使用",
                    "highlighted_content": chunk["content"],
                    "final_score": chunk.get("similarity_score", 0.0),
                    "rank_order": i + 1
                }
                ranked_chunks.append(enhanced_chunk)
        
        # 最終スコアでソート
        ranked_chunks.sort(key=lambda x: x["final_score"], reverse=True)
        
        for i, chunk in enumerate(ranked_chunks):
            chunk["rank_order"] = i + 1
        
        print(f"✅ LLMソート完了、上位{min(top_k, len(ranked_chunks))}個の最適chunkを返す")
        return ranked_chunks[:top_k]
    
    def _evaluate_chunk_with_llm(self, query: str, content: str, vector_score: float) -> Tuple[float, str, str]:
        """LLMを使用して単一chunkの関連性を評価し、ハイライトを生成"""
        evaluation_prompt = self._build_evaluation_prompt(query, content, vector_score)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "あなたは専門的な情報検索評価の専門家です。テキストとクエリの関連性を客観的かつ正確に評価し、詳細な理由とハイライト表示を提供してください。"
                    },
                    {
                        "role": "user", 
                        "content": evaluation_prompt
                    }
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_llm_response(result_text, content)
            
        except Exception as e:
            print(f"LLM評価API呼び出し失敗: {e}")
            return vector_score, f"API呼び出し失敗: {str(e)}", content
    
    def _build_evaluation_prompt(self, query: str, content: str, vector_score: float) -> str:
        """LLM評価プロンプトを構築"""
        return f"""
以下のテキスト内容とユーザークエリの関連性を評価してください：

**ユーザークエリ：**
{query}

**候補テキスト内容：**
{content}

**ベクトル類似度スコア：** {vector_score:.4f}

以下の形式で評価結果を返してください：

```json
{{
    "relevance_score": <0.0-1.0の間の関連性スコア>,
    "reason": "<詳細な関連性分析、なぜこのスコアを与えたかの説明>",
    "key_points": ["<キーマッチポイント1>", "<キーマッチポイント2>", "..."],
    "highlighted_content": "<原文から正確に抽出したクエリに最も類似する核心部分、質問に最も直接的に答える最も関連性の高い部分を選択（1-2文程度）、文字の追加や修正は行わない>"
}}
```

評価基準：
1. **直接回答性** (0-0.3): テキストがクエリに直接答えているか
2. **内容関連性** (0-0.3): テキスト内容とクエリトピックの関連度
3. **情報完全性** (0-0.2): 提供される情報が包括的で正確か
4. **意味マッチング度** (0-0.2): キーワードと概念のマッチング度

**重要な注意事項：**
- highlighted_contentは原文から正確にコピーしたテキストフラグメントである必要があります
- 質問に最も直接的に答える最も関連性の高い部分を選択してください（1-2文程度）
- 段落全体や過度に長い内容を含めないでください
- 抽出されたテキストが原文と完全に一致することを確認してください
- クエリキーワードを含む部分を優先的に選択してください

有効なJSON形式で返すことを確認してください。
"""
    
    def _parse_llm_response(self, result_text: str, original_content: str) -> Tuple[float, str, str]:
        """LLM応答を解析"""
        try:
            # JSONブロックを抽出
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                relevance_score = float(parsed.get('relevance_score', 0.0))
                reason = parsed.get('reason', '理由不明')
                highlighted_content = parsed.get('highlighted_content', original_content)
                
                # highlighted_contentを検証
                highlighted_content = self._validate_highlighted_content(highlighted_content, original_content)
                
                return relevance_score, reason, highlighted_content
            else:
                raise ValueError("有効なJSONが見つかりません")
                
        except Exception as e:
            print(f"LLM応答解析失敗: {e}")
            return 0.5, f"解析失敗: {str(e)}", original_content[:100]
    
    def _validate_highlighted_content(self, highlighted_content: str, original_content: str) -> str:
        """highlighted_contentを検証・修正"""
        clean_highlighted = highlighted_content.replace("**", "").replace("*", "").strip()
        
        if clean_highlighted in original_content:
            if len(clean_highlighted) > 200:
                sentences = clean_highlighted.split('。')
                if len(sentences) > 1:
                    return '。'.join(sentences[:2]) + '。'
                else:
                    return clean_highlighted[:150] + "..."
            return clean_highlighted
        
        # 部分マッチングを試行
        best_match = ""
        best_length = 0
        
        for i in range(len(original_content)):
            for j in range(i + 1, len(original_content) + 1):
                substring = original_content[i:j]
                if substring in clean_highlighted and len(substring) > best_length:
                    best_match = substring
                    best_length = len(substring)
        
        if best_length >= 10:
            return best_match
        
        # フォールバック: 原文の最初の50文字
        return original_content[:50] + "..." if len(original_content) > 50 else original_content


class UltraFastRAG:
    """統合された超高速RAGシステム"""
    
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma", use_llm_ranking: bool = True):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.use_llm_ranking = use_llm_ranking
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        
        # 向量データベースを初期化
        self.db = None
        if os.path.exists(chroma_path):
            try:
                self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
                print(f"✅ 既存の向量データベースを読み込み: {chroma_path}")
            except Exception as e:
                print(f"⚠️ 向量データベース読み込み失敗: {e}")
        
        # LLMランキングシステムを初期化
        if use_llm_ranking:
            try:
                self.llm_ranker = IntegratedLLMEvidenceRanker(openai_api_key)
                print("✅ LLM智能ranking已启用")
            except Exception as e:
                print(f"⚠️ LLM ranking初始化失敗: {e}，将使用原始方法")
                self.use_llm_ranking = False
        else:
            self.use_llm_ranking = False
    
    def build_vector_store(self, data_file: str) -> bool:
        """精細chunking を使用して向量データベースを構築"""
        try:
            print(f"🏗️ 精細chunking で向量データベース構築中...")
            print(f"📁 データファイル: {data_file}")
            print(f"🗄️ 向量データベースパス: {self.chroma_path}")
            
            if not os.path.exists(data_file):
                print(f"❌ データファイルが見つかりません: {data_file}")
                return False
            
            # データを読み込み
            print("📖 データ読み込み中...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ {len(data)} 件のデータを読み込み")
            
            # Documentに変換
            documents = []
            for item in data:
                content = item.get('output', '') or item.get('text', '') or item.get('content', '')
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'dataset',
                        'index': str(len(documents))
                    }
                )
                documents.append(doc)
            
            print(f"📄 {len(documents)} 個のドキュメントに変換")
            
            # 精細文本分割
            print("✂️ 精細chunking実行中...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=150,        # 150字符 (約1-2文)
                chunk_overlap=30,      # 30字符重複
                length_function=len,
                add_start_index=True,
                separators=[
                    "\n\n",           # 段落
                    "。",             # 句号  
                    "！",             # 感嘆号
                    "？",             # 疑問符
                    "、",             # 読点
                    "\n",             # 改行
                    " ",              # 空格
                    ""                # 字符级别
                ]
            )
            chunks = text_splitter.split_documents(documents)
            avg_chunk_size = sum(len(c.page_content) for c in chunks) / len(chunks)
            print(f"🔪 {len(chunks)} 個のchunksに分割 (平均 {avg_chunk_size:.1f} 文字/chunk)")
            
            # 既存の向量データベースを削除
            if os.path.exists(self.chroma_path):
                import shutil
                shutil.rmtree(self.chroma_path)
                print("🗑️ 既存の向量データベースを削除")
            
            # 新しい向量データベースを作成
            print("🔄 向量データベース作成中...")
            start_time = time.time()
            
            self.db = Chroma.from_documents(
                chunks,
                self.embedding_function,
                persist_directory=self.chroma_path
            )
            
            build_time = time.time() - start_time
            print(f"✅ 向量データベース構築完了! 時間: {build_time:.2f}s")
            print(f"📊 統計: {len(chunks)} chunks, 平均 {build_time/len(chunks)*1000:.1f}ms/chunk")
            
            return True
            
        except Exception as e:
            print(f"❌ 向量データベース構築失敗: {e}")
            return False
    
    def query(self, query_text: str, k: int = 5) -> Tuple[str, str, str, int, int]:
        """クエリ処理 - LLM ranking対応"""
        if not self.db:
            return "向量データベースが初期化されていません。", "", "", 0, 0
        
        if self.use_llm_ranking:
            return self._query_with_llm_ranking(query_text, k)
        else:
            return self._query_fast_original(query_text)
    
    def _query_with_llm_ranking(self, query_text: str, k: int) -> Tuple[str, str, str, int, int]:
        """LLM ranking を使用した拡張クエリ"""
        print(f"🧠 LLM智能ranking でクエリ処理中: '{query_text}'")
        
        # 向量類似検索
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=k)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        # LLM ranker 形式に変換
        chunks = []
        for doc, score in search_results:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            }
            chunks.append(chunk)
        
        # LLM智能排序
        try:
            ranked_chunks = self.llm_ranker.rank_and_highlight_chunks(query_text, chunks, top_k=1)
            
            if ranked_chunks:
                best_chunk = ranked_chunks[0]
                source_text = best_chunk["content"]
                evidence_text = best_chunk.get("highlighted_content", source_text)
                
                start_pos = 0
                end_pos = len(evidence_text)
                
                # 回答生成
                answer = self._generate_answer_fast(evidence_text, query_text)
                
                print(f"✅ LLM ranking完成，最佳匹配スコア: {best_chunk.get('final_score', 0):.3f}")
                
                return answer, source_text, evidence_text, start_pos, end_pos
            
        except Exception as e:
            print(f"⚠️ LLM ranking失敗: {e}，原始方法にフォールバック")
        
        # 原始方法にフォールバック
        return self._query_fast_original(query_text)
    
    def _query_fast_original(self, query_text: str) -> Tuple[str, str, str, int, int]:
        """原始の超高速クエリ方法"""
        search_results = self.db.similarity_search_with_relevance_scores(query_text, k=1)
        
        if not search_results:
            return "情報が見つかりませんでした。", "", "", 0, 0
        
        hit_doc = search_results[0][0]
        confidence = search_results[0][1]
        
        source_text = hit_doc.page_content
        evidence_text, start_pos, end_pos = self._extract_evidence_fast(source_text, query_text)
        answer = self._generate_answer_fast(evidence_text, query_text)
        
        return answer, source_text, evidence_text, start_pos, end_pos
    
    def _extract_evidence_fast(self, text: str, query: str) -> Tuple[str, int, int]:
        """高速証拠抽出"""
        keywords = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\w]+', query)
        keywords = [kw for kw in keywords if len(kw) > 1 and kw not in ['とは', '何', 'です', 'ます', 'について']]
        
        sentences = re.split(r'[。！？.!?]', text)
        sentences = [s for s in sentences if s]
        
        if not sentences:
            snippet = text[:100]
            return snippet, 0, len(snippet)
        
        best_sentence = None
        best_score = -1
        
        for sentence in sentences:
            score = 0
            for kw in keywords:
                if kw in sentence:
                    score += 1
            
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
        """高速回答生成"""
        if any(pattern in query for pattern in ['とは何', 'とは', '何ですか', '何でしょうか']):
            first_sentence = re.split(r'[。！？.!?]', evidence)[0]
            if first_sentence:
                return first_sentence + '。'
        
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
            return evidence[:100] + ('...' if len(evidence) > 100 else '')


def test_integrated_system():
    """統合システムのテスト"""
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未設定")
        return
    
    print("🚀 統合RAGシステム テスト")
    print("=" * 50)
    
    # システム初期化
    rag = UltraFastRAG(api_key, "./chroma_integrated", use_llm_ranking=True)
    
    # データベース構築をテスト (オプション)
    # rag.build_vector_store("../data/single_20240229.json")
    
    # クエリテスト
    queries = [
        "コンバインとは何ですか",
        "音位転倒について説明してください",
        "漢方薬の違いは何ですか"
    ]
    
    for query in queries:
        print(f"\n🔍 クエリ: {query}")
        
        start_time = time.time()
        answer, source, evidence, start, end = rag.query(query)
        elapsed = time.time() - start_time
        
        print(f"⏱️  処理時間: {elapsed:.2f}秒")
        print(f"💬 回答: {answer}")
        print(f"🔍 証拠: {evidence}")
        print(f"📊 証拠範囲: {start+1}〜{end}文字目")
        print("-" * 40)


if __name__ == "__main__":
    test_integrated_system()