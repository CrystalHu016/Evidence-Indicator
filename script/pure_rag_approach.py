#!/usr/bin/env python3
"""
纯RAG方法 - 基于语义相似度而非硬编码规则
"""

import os
import re
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from pydantic import SecretStr

class PureRAGApproach:
    """纯RAG方法 - 基于语义相似度"""
    
    def __init__(self, openai_api_key: str):
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.llm = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini")
        self.db = None
    
    def build_vector_store(self, documents: List[str]):
        """构建纯向量数据库"""
        from langchain.schema import Document
        
        docs = [Document(page_content=doc) for doc in documents]
        self.db = Chroma.from_documents(docs, self.embedding_function)
    
    def semantic_query(self, query: str, k: int = 3) -> List[Dict]:
        """纯语义查询 - 只基于向量相似度"""
        if not self.db:
            return []
        
        # 1. 向量相似度检索
        results = self.db.similarity_search_with_score(query, k=k)
        
        # 2. 返回结果（不进行任何硬编码过滤）
        return [
            {
                'content': doc.page_content,
                'similarity_score': score,
                'source': 'vector_similarity'
            }
            for doc, score in results
        ]
    
    def llm_enhanced_query(self, query: str, k: int = 3) -> List[Dict]:
        """LLM增强的语义查询"""
        if not self.db:
            return []
        
        # 1. 向量相似度检索
        results = self.db.similarity_search_with_score(query, k=k*2)  # 检索更多候选
        
        # 2. 使用LLM进行语义相关性评分（而非硬编码规则）
        enhanced_results = []
        for doc, score in results:
            # 使用LLM评估相关性，而不是硬编码规则
            relevance_prompt = f"""
            查询: {query}
            文档: {doc.page_content}
            
            请评估这个文档与查询的相关性，返回0-1之间的分数。
            只考虑语义相关性，不要使用硬编码规则。
            
            返回格式: {{"relevance_score": 0.85, "reason": "文档包含查询相关的分类信息"}}
            """
            
            try:
                response = self.llm.invoke(relevance_prompt)
                # 解析LLM响应
                import json
                llm_result = json.loads(response.content)
                enhanced_score = llm_result.get('relevance_score', score)
                reason = llm_result.get('reason', '')
            except:
                enhanced_score = score
                reason = 'LLM评分失败，使用向量相似度'
            
            enhanced_results.append({
                'content': doc.page_content,
                'vector_score': score,
                'llm_score': enhanced_score,
                'final_score': enhanced_score,  # 使用LLM评分
                'reason': reason,
                'source': 'llm_enhanced'
            })
        
        # 3. 按LLM评分排序
        enhanced_results.sort(key=lambda x: x['final_score'], reverse=True)
        return enhanced_results[:k]
    
    def query_expansion(self, query: str) -> str:
        """查询扩展 - 使用LLM生成同义词和变体"""
        expansion_prompt = f"""
        原始查询: {query}
        
        请生成这个查询的语义等价变体，包括：
        1. 同义词替换
        2. 不同的表达方式
        3. 相关的概念
        
        返回格式: 原始查询|变体1|变体2|变体3
        
        例如: 農業機械の種類について教えてください|農業機械の分類について|農業機械にはどんな種類がありますか|農業機械のタイプについて
        """
        
        try:
            response = self.llm.invoke(expansion_prompt)
            expanded_queries = response.content.strip().split('|')
            return expanded_queries[0]  # 返回原始查询
        except:
            return query
    
    def multi_query_retrieval(self, query: str, k: int = 3) -> List[Dict]:
        """多查询检索 - 生成多个查询变体进行检索"""
        # 1. 查询扩展
        expanded_queries = self.query_expansion(query).split('|')
        
        # 2. 对每个查询变体进行检索
        all_results = []
        for expanded_query in expanded_queries[:3]:  # 限制变体数量
            results = self.semantic_query(expanded_query.strip(), k=k)
            for result in results:
                result['source_query'] = expanded_query.strip()
            all_results.extend(results)
        
        # 3. 去重和排序
        seen_contents = set()
        unique_results = []
        for result in all_results:
            if result['content'] not in seen_contents:
                seen_contents.add(result['content'])
                unique_results.append(result)
        
        # 按相似度排序
        unique_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return unique_results[:k]

def demonstrate_pure_rag():
    """演示纯RAG方法"""
    print("🔍 纯RAG方法演示")
    print("=" * 60)
    
    # 示例文档
    documents = [
        "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
        "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
        "普通型は主にアメリカやヨーロッパ等大規模農業で使われています。",
        "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"
    ]
    
    # 初始化系统
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    rag = PureRAGApproach(api_key)
    rag.build_vector_store(documents)
    
    # 测试查询
    query = "農業機械の種類について教えてください"
    print(f"查询: {query}")
    print()
    
    # 1. 纯向量检索
    print("📊 1. 纯向量相似度检索:")
    vector_results = rag.semantic_query(query, k=3)
    for i, result in enumerate(vector_results, 1):
        print(f"{i}. 相似度: {result['similarity_score']:.3f}")
        print(f"   内容: {result['content']}")
        print()
    
    # 2. LLM增强检索
    print("📊 2. LLM增强检索:")
    llm_results = rag.llm_enhanced_query(query, k=3)
    for i, result in enumerate(llm_results, 1):
        print(f"{i}. 向量分数: {result['vector_score']:.3f}, LLM分数: {result['llm_score']:.3f}")
        print(f"   内容: {result['content']}")
        print(f"   理由: {result['reason']}")
        print()
    
    # 3. 多查询检索
    print("📊 3. 多查询检索:")
    multi_results = rag.multi_query_retrieval(query, k=3)
    for i, result in enumerate(multi_results, 1):
        print(f"{i}. 相似度: {result['similarity_score']:.3f}")
        print(f"   内容: {result['content']}")
        print(f"   来源查询: {result.get('source_query', 'N/A')}")
        print()

if __name__ == "__main__":
    demonstrate_pure_rag()
