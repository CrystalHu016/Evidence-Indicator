#!/usr/bin/env python3
"""
LLM-Based Evidence Ranker and Highlighter
使用LLM进行智能chunk ranking和highlight的系统
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
import openai
from pydantic import SecretStr


class LLMEvidenceRanker:
    """LLM驱动的证据排序和高亮系统"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        """初始化LLM排序器"""
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
    
    def rank_and_highlight_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
        """
        LLMを使用してchunksを智能ソート・ハイライト
        
        Args:
            query: ユーザークエリ
            chunks: 候補chunkリスト、形式: [{"content": str, "metadata": dict, "similarity_score": float}]
            top_k: 上位k個の最適chunkを返す
        
        Returns:
            ソート後のchunks、LLMスコアとハイライト情報を含む
        """
        if not chunks:
            return []
        
        print(f"🧠 LLMを使用して上位2個の最高ベクトルスコアchunksを智能ソート、最適マッチを返す...")
        
        # まずベクトルスコアでソート、時間節約のため上位2個のみLLM評価
        chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
        top_chunks = chunks[:2]  # 上位2個の最高ベクトルスコアchunksのみ評価
        
        # 各chunkのLLMスコアを取得
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
                    "final_score": (chunk.get("similarity_score", 0.0) * 0.4) + (llm_score * 0.6),  # 組み合わせスコア
                    "rank_order": i + 1
                }
                ranked_chunks.append(enhanced_chunk)
                
                print(f"  ✓ Chunk {i+1}: Vector={chunk.get('similarity_score', 0.0):.3f}, LLM={llm_score:.3f}, Final={enhanced_chunk['final_score']:.3f}")
                
            except Exception as e:
                print(f"  ❌ Chunk {i+1} LLM評価失敗: {e}")
                # フォールバックとして元のスコアを使用
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
        
        # ソート後のrank_orderを更新
        for i, chunk in enumerate(ranked_chunks):
            chunk["rank_order"] = i + 1
        
        print(f"✅ LLMソート完了、上位{min(top_k, len(ranked_chunks))}個の最適chunkを返す")
        
        return ranked_chunks[:top_k]
    
    def _evaluate_chunk_with_llm(self, query: str, content: str, vector_score: float) -> Tuple[float, str, str]:
        """
        LLMを使用して単一chunkの関連性を評価し、ハイライトを生成
        
        Returns:
            (llm_score, relevance_reason, highlighted_content)
        """
        
        # LLMプロンプトを構築
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
            # デフォルト値を返す
            return vector_score, f"API呼び出し失敗: {str(e)}", content
    
    def _build_evaluation_prompt(self, query: str, content: str, vector_score: float) -> str:
        """LLM評価プロンプトを構築"""
        
        prompt = f"""
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
- 例：「コンバインとは何ですか」→「コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。」

有効なJSON形式で返すことを確認してください。
"""
        return prompt
    
    def _validate_highlighted_content(self, highlighted_content: str, original_content: str) -> str:
        """highlighted_contentを検証・修正し、原文から来ていることを確認"""
        # 可能なmarkdownマークを削除
        clean_highlighted = highlighted_content.replace("**", "").replace("*", "").strip()
        
        # highlighted_contentが原文に完全に含まれている場合、LLMの選択を信頼
        if clean_highlighted in original_content:
            # LLMが選択した内容をそのまま使用（長さ制限のみ適用）
            if len(clean_highlighted) > 200:
                # 長すぎる場合のみ、文の境界で切り取る
                sentences = clean_highlighted.split('。')
                if len(sentences) > 1:
                    # 最初の2つの文まで許可
                    return '。'.join(sentences[:2]) + '。'
                else:
                    # 1つの文が長すぎる場合のみ切り取り
                    return clean_highlighted[:150] + "..."
            return clean_highlighted
        
        # highlighted_contentが原文の部分を含む場合、最長のマッチ部分を見つける
        best_match = ""
        best_length = 0
        
        # 異なる位置からマッチングを試行
        for i in range(len(original_content)):
            for j in range(i + 1, len(original_content) + 1):
                substring = original_content[i:j]
                if substring in clean_highlighted and len(substring) > best_length:
                    best_match = substring
                    best_length = len(substring)
        
        # 合理的なマッチが見つかった場合（最低10文字）、それを使用
        if best_length >= 10:
            return best_match
        
        # そうでなければ、原文の最初の100文字をフォールバックとして使用
        return original_content[:100] + ("..." if len(original_content) > 100 else "")
    
    def _parse_llm_response(self, response_text: str, original_content: str) -> Tuple[float, str, str]:
        """LLMレスポンスを解析"""
        try:
            # JSON部分を抽出
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if not json_match:
                # レスポンス全体を直接解析を試行
                json_data = json.loads(response_text)
            else:
                json_data = json.loads(json_match.group(1))
            
            score = float(json_data.get("relevance_score", 0.0))
            reason = json_data.get("reason", "評価理由なし")
            key_points = json_data.get("key_points", [])
            highlighted_content = json_data.get("highlighted_content", original_content)
            
            # スコアが有効範囲内であることを確認
            score = max(0.0, min(1.0, score))
            
            # highlighted_contentを検証・修正し、原文から来ていることを確認
            highlighted_content = self._validate_highlighted_content(highlighted_content, original_content)
            
            # 理由情報を強化
            if key_points:
                reason += f"\nキーマッチポイント: {', '.join(key_points)}"
            
            return score, reason, highlighted_content
            
        except json.JSONDecodeError as e:
            print(f"JSON解析失敗: {e}")
            print(f"レスポンス内容: {response_text}")
            # テキストからスコアを抽出を試行
            score_match = re.search(r'(?:スコア|score|評価)[:：]\s*([0-9.]+)', response_text)
            if score_match:
                try:
                    score = float(score_match.group(1))
                    return max(0.0, min(1.0, score)), response_text[:200], original_content
                except:
                    pass
            
            return 0.5, f"JSON解析失敗: {str(e)}", original_content
        
        except Exception as e:
            print(f"レスポンス解析失敗: {e}")
            return 0.5, f"解析失敗: {str(e)}", original_content
    
    def generate_ranking_summary(self, query: str, ranked_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ソート結果のサマリーを生成"""
        
        if not ranked_chunks:
            return {
                "query": query,
                "total_chunks": 0,
                "best_match": None,
                "ranking_summary": "没有找到相关内容",
                "avg_llm_score": 0.0,
                "avg_vector_score": 0.0
            }
        
        best_chunk = ranked_chunks[0]
        
        # 计算平均分数
        avg_llm_score = sum(chunk["llm_score"] for chunk in ranked_chunks) / len(ranked_chunks)
        avg_vector_score = sum(chunk.get("similarity_score", 0.0) for chunk in ranked_chunks) / len(ranked_chunks)
        
        # 生成排序摘要
        summary_lines = []
        for i, chunk in enumerate(ranked_chunks[:3], 1):  # 只显示前3个
            content_preview = chunk["content"][:100] + "..." if len(chunk["content"]) > 100 else chunk["content"]
            summary_lines.append(
                f"{i}. 最终评分: {chunk['final_score']:.3f} "
                f"(向量: {chunk.get('similarity_score', 0.0):.3f}, LLM: {chunk['llm_score']:.3f}) "
                f"- {content_preview}"
            )
        
        return {
            "query": query,
            "total_chunks": len(ranked_chunks),
            "best_match": {
                "content": best_chunk["content"],
                "highlighted_content": best_chunk["highlighted_content"],
                "final_score": best_chunk["final_score"],
                "llm_score": best_chunk["llm_score"],
                "vector_score": best_chunk.get("similarity_score", 0.0),
                "reason": best_chunk["relevance_reason"]
            },
            "ranking_summary": "\n".join(summary_lines),
            "avg_llm_score": avg_llm_score,
            "avg_vector_score": avg_vector_score,
            "llm_improvement": avg_llm_score - avg_vector_score
        }


def test_llm_ranker():
    """测试LLM排序器"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设定")
        return
    
    print("🧠 LLM Evidence Ranker 测试")
    print("=" * 50)
    
    ranker = LLMEvidenceRanker(api_key)
    
    # 模拟chunks数据
    test_chunks = [
        {
            "content": "上高地は長野県にある山岳景勝地で、毎年100万人以上の観光客が訪れる人気の場所です。標高約1,500mにあり、キャンプやウォーキングが楽しめます。",
            "metadata": {"source": "test1"},
            "similarity_score": 0.85
        },
        {
            "content": "漢方薬には当帰芍薬散、加味逍遙散、桂枝茯苓丸などがあります。それぞれ異なる効果があります。",
            "metadata": {"source": "test2"},
            "similarity_score": 0.42
        },
        {
            "content": "上高地の大正池は美しい景色で有名です。穂高連峰の雄大な景色を見ることができます。",
            "metadata": {"source": "test3"},
            "similarity_score": 0.73
        }
    ]
    
    query = "上高地について教えて"
    
    # LLM排序测试
    ranked = ranker.rank_and_highlight_chunks(query, test_chunks, top_k=3)
    
    print(f"\n📊 排序结果:")
    for i, chunk in enumerate(ranked, 1):
        print(f"\n{i}. 【最终评分: {chunk['final_score']:.3f}】")
        print(f"   向量评分: {chunk.get('similarity_score', 0.0):.3f}")
        print(f"   LLM评分: {chunk['llm_score']:.3f}")
        print(f"   相关性理由: {chunk['relevance_reason']}")
        print(f"   高亮内容: {chunk['highlighted_content']}")
        print("-" * 30)
    
    # 生成摘要
    summary = ranker.generate_ranking_summary(query, ranked)
    print(f"\n📋 排序摘要:")
    print(f"查询: {summary['query']}")
    print(f"总chunk数: {summary['total_chunks']}")
    print(f"平均LLM评分: {summary['avg_llm_score']:.3f}")
    print(f"平均向量评分: {summary['avg_vector_score']:.3f}")
    print(f"LLM改进: {summary['llm_improvement']:+.3f}")
    print(f"\n最佳匹配:")
    print(f"评分: {summary['best_match']['final_score']:.3f}")
    print(f"理由: {summary['best_match']['reason']}")


if __name__ == "__main__":
    test_llm_ranker()