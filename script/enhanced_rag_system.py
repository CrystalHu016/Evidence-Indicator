#!/usr/bin/env python3
"""
Enhanced RAG System with LLM-based Evidence Ranking
集成LLM智能ranking的增强RAG系统
"""

import os
import time
from typing import List, Dict, Any, Tuple, Optional
from pydantic import SecretStr
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

try:
    from llm_evidence_ranker import LLMEvidenceRanker
except ImportError:
    try:
        from script.llm_evidence_ranker import LLMEvidenceRanker
    except ImportError:
        import sys
        import os
        sys.path.append(os.path.dirname(__file__))
        from llm_evidence_ranker import LLMEvidenceRanker


class EnhancedRAGSystem:
    """集成LLM智能ranking的增强RAG系统"""
    
    def __init__(self, openai_api_key: str, chroma_path: str, model: str = "gpt-4o-mini"):
        """
        初始化增强RAG系统
        
        Args:
            openai_api_key: OpenAI API密钥
            chroma_path: ChromaDB数据路径
            model: LLM模型名称
        """
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
        
        # 初始化向量数据库
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
        
        # 初始化LLM排序器
        self.llm_ranker = LLMEvidenceRanker(openai_api_key, model)
        
        print("✅ 增强RAG系统初始化完成")
    
    def query(self, query_text: str, initial_k: int = 3, final_k: int = 1, use_llm_ranking: bool = True) -> Dict[str, Any]:
        """
        增强查询处理
        
        Args:
            query_text: 查询文本
            initial_k: 初始向量检索数量
            final_k: 最终返回数量
            use_llm_ranking: 是否使用LLM重新排序
            
        Returns:
            完整的查询结果，包含答案、证据、排序信息等
        """
        start_time = time.time()
        
        print(f"🔍 开始处理查询: '{query_text}'")
        print(f"📊 初始检索: {initial_k} chunks, 最终返回: {final_k} chunks")
        
        # 第一步：向量相似度检索
        vector_start = time.time()
        initial_results = self.db.similarity_search_with_relevance_scores(query_text, k=initial_k)
        vector_time = time.time() - vector_start
        
        if not initial_results:
            return {
                "query": query_text,
                "answer": "没有找到相关信息。",
                "evidence": "",
                "highlighted_evidence": "",
                "source_document": "",
                "chunks": [],
                "ranking_summary": "没有找到匹配的内容",
                "processing_time": time.time() - start_time,
                "vector_search_time": vector_time,
                "llm_ranking_time": 0,
                "answer_generation_time": 0
            }
        
        print(f"📊 向量检索完成: {len(initial_results)} 个结果 (耗时: {vector_time:.2f}s)")
        
        # 转换为标准格式
        chunks = []
        for doc, score in initial_results:
            chunk = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score
            }
            chunks.append(chunk)
        
        # 第二步：LLM智能重新排序
        llm_start = time.time()
        if use_llm_ranking:
            print("🧠 启动LLM智能排序...")
            ranked_chunks = self.llm_ranker.rank_and_highlight_chunks(query_text, chunks, top_k=final_k)
            ranking_summary = self.llm_ranker.generate_ranking_summary(query_text, ranked_chunks)
        else:
            print("⚡ 使用原始向量排序")
            # 按原始相似度排序
            chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            ranked_chunks = chunks[:final_k]
            # 为非LLM排序添加默认字段
            for chunk in ranked_chunks:
                chunk.update({
                    "llm_score": chunk["similarity_score"],
                    "relevance_reason": "使用向量相似度评分",
                    "highlighted_content": chunk["content"],
                    "final_score": chunk["similarity_score"]
                })
            ranking_summary = {
                "query": query_text,
                "total_chunks": len(ranked_chunks),
                "ranking_summary": "使用原始向量相似度排序"
            }
        
        llm_time = time.time() - llm_start
        print(f"🧠 LLM排序完成 (耗时: {llm_time:.2f}s)")
        
        # 第三步：生成最终答案
        answer_start = time.time()
        if ranked_chunks:
            best_chunk = ranked_chunks[0]
            answer = self._generate_enhanced_answer(query_text, ranked_chunks)
            evidence = best_chunk["content"]
            highlighted_evidence = best_chunk.get("highlighted_content", evidence)
            # Fix: Use the actual content as source_document, not metadata
            source_document = best_chunk["content"]
        else:
            answer = "没有找到相关信息。"
            evidence = ""
            highlighted_evidence = ""
            source_document = ""
        
        answer_time = time.time() - answer_start
        total_time = time.time() - start_time
        
        print(f"✅ 查询处理完成 (总耗时: {total_time:.2f}s)")
        
        # 返回完整结果
        return {
            "query": query_text,
            "answer": answer,
            "evidence": evidence,
            "highlighted_evidence": highlighted_evidence,
            "source_document": source_document,
            "chunks": ranked_chunks,
            "ranking_summary": ranking_summary,
            "processing_time": total_time,
            "vector_search_time": vector_time,
            "llm_ranking_time": llm_time,
            "answer_generation_time": answer_time,
            "performance_breakdown": {
                "vector_search": f"{vector_time:.2f}s ({vector_time/total_time*100:.1f}%)",
                "llm_ranking": f"{llm_time:.2f}s ({llm_time/total_time*100:.1f}%)",
                "answer_generation": f"{answer_time:.2f}s ({answer_time/total_time*100:.1f}%)"
            }
        }
    
    def _generate_enhanced_answer(self, query: str, ranked_chunks: List[Dict[str, Any]]) -> str:
        """生成增强的答案"""
        
        if not ranked_chunks:
            return "没有找到相关信息。"
        
        # 使用前3个最佳chunk生成答案
        top_chunks = ranked_chunks[:3]
        context_pieces = []
        
        for i, chunk in enumerate(top_chunks, 1):
            relevance = chunk.get("relevance_reason", "")
            content = chunk["content"][:300]  # 限制长度
            context_pieces.append(f"证据{i} (评分: {chunk.get('final_score', 0):.2f}): {content}")
        
        context = "\n\n".join(context_pieces)
        
        # 生成答案的prompt
        system_prompt = """你是一个专业的问答助手。基于提供的证据，生成准确、简洁的答案。

要求：
1. 直接回答用户的问题
2. 基于证据内容，不要添加无关信息  
3. 如果证据不足，明确说明
4. 保持回答的客观性和准确性
5. 使用自然的日语表达"""

        user_prompt = f"""
用户问题：{query}

提供的证据：
{context}

请基于以上证据回答用户的问题："""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"答案生成失败: {e}")
            # 降级使用最佳chunk的内容
            best_chunk = top_chunks[0]
            return f"根据检索到的信息：{best_chunk['content'][:200]}..."
    
    def print_detailed_results(self, result: Dict[str, Any]):
        """打印详细的查询结果"""
        
        print("\n" + "="*60)
        print("📋 详细查询结果")
        print("="*60)
        
        print(f"\n🔍 查询: {result['query']}")
        print(f"⏱️  处理时间: {result['processing_time']:.2f}s")
        
        # 性能分解
        if 'performance_breakdown' in result:
            print(f"\n⚡ 性能分解:")
            for stage, time_info in result['performance_breakdown'].items():
                print(f"  - {stage}: {time_info}")
        else:
            print(f"\n⚡ 各阶段耗时:")
            print(f"  - 向量检索: {result.get('vector_search_time', 0):.2f}s")
            print(f"  - LLM排序: {result.get('llm_ranking_time', 0):.2f}s") 
            print(f"  - 答案生成: {result.get('answer_generation_time', 0):.2f}s")
        
        # 答案
        print(f"\n💬 答案:")
        print(result['answer'])
        
        # 证据高亮
        if result.get('highlighted_evidence'):
            print(f"\n🔦 高亮证据:")
            print(result['highlighted_evidence'])
        
        # 排序结果
        print(f"\n📊 Chunk排序结果:")
        chunks = result.get('chunks', [])
        for i, chunk in enumerate(chunks[:3], 1):  # 只显示前3个
            print(f"\n  {i}. 【评分: {chunk.get('final_score', 0):.3f}】")
            print(f"     向量: {chunk.get('similarity_score', 0):.3f} | LLM: {chunk.get('llm_score', 0):.3f}")
            print(f"     理由: {chunk.get('relevance_reason', 'N/A')}")
            print(f"     内容: {chunk['content'][:100]}...")
        
        # 排序摘要
        summary = result.get('ranking_summary', {})
        if isinstance(summary, dict) and summary.get('ranking_summary'):
            print(f"\n📈 排序摘要:")
            print(summary['ranking_summary'])
            
            if 'llm_improvement' in summary:
                improvement = summary['llm_improvement']
                if improvement > 0:
                    print(f"🎯 LLM改进: +{improvement:.3f} (LLM排序效果更好)")
                elif improvement < 0:
                    print(f"📉 LLM改进: {improvement:.3f} (向量排序更优)")
                else:
                    print(f"🟰 LLM改进: {improvement:.3f} (效果相当)")
        
        print("\n" + "="*60)


def test_enhanced_rag():
    """测试增强RAG系统"""
    from dotenv import load_dotenv
    
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设定")
        return
    
    print("🚀 增强RAG系统测试")
    print("="*60)
    
    # 初始化系统
    rag = EnhancedRAGSystem(
        openai_api_key=api_key, 
        chroma_path="./chroma_new"  # 使用新的数据路径
    )
    
    # 测试查询
    test_queries = [
        "上高地について教えて",
        "漢方薬の違いは？",
        "四字熟語のスローガンとは"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")
        print("-"*40)
        
        # 执行查询
        result = rag.query(query, initial_k=5, final_k=3, use_llm_ranking=True)
        
        # 打印结果
        rag.print_detailed_results(result)
        
        # 等待用户确认
        input("\n按Enter继续下一个查询...")


if __name__ == "__main__":
    test_enhanced_rag()