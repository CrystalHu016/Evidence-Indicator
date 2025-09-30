#!/usr/bin/env python3
"""
纯语义RAG系统 - 完全基于语义理解，无硬编码规则
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from pydantic import SecretStr

@dataclass
class SemanticChunk:
    """语义chunk结果"""
    content: str
    similarity_score: float
    semantic_relevance: float
    final_score: float
    granularity: str
    reasoning: str
    metadata: Dict[str, Any]

class PureSemanticRAG:
    """纯语义RAG系统 - 完全基于LLM语义理解"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.llm = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini", temperature=0)
        
        # 向量存储
        self.vector_store = None
        self.documents = []
        
        # 语义理解配置
        self.semantic_config = {
            'similarity_threshold': 0.2,  # 降低阈值，让更多候选进入LLM评估
            'max_candidates': 10,         # 最大候选数量
            'llm_evaluation': True,       # 启用LLM评估
            'query_expansion': True,      # 启用查询扩展
            'multi_granular': True        # 启用多粒度检索
        }
    
    def build_vector_store(self, data: List[Dict], chunk_size: int = 300, chunk_overlap: int = 50):
        """构建向量数据库"""
        print("🏗️ 构建纯语义向量数据库...")
        
        # 加载数据
        documents = []
        for item in data:
            content = item.get('output', '') or item.get('text', '') or item.get('content', '')
            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'semantic_rag',
                        'original_index': len(documents)
                    }
                )
                documents.append(doc)
        
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "。", "！", "？", "、", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"📄 创建了 {len(chunks)} 个chunks")
        
        # 构建向量存储
        self.vector_store = Chroma.from_documents(
            chunks,
            self.embedding_function,
            persist_directory="./chroma_pure_semantic"
        )
        
        self.documents = chunks
        print("✅ 纯语义向量数据库构建完成")
    
    def semantic_query(self, query: str, k: int = 3) -> List[SemanticChunk]:
        """纯语义查询 - 完全基于LLM理解"""
        if not self.vector_store:
            return []
        
        print(f"🔍 纯语义查询: {query}")
        
        # 1. 查询理解和扩展
        expanded_queries = self._expand_query_semantically(query)
        print(f"📝 查询扩展: {len(expanded_queries)} 个变体")
        
        # 2. 多查询检索
        all_candidates = []
        for expanded_query in expanded_queries:
            candidates = self.vector_store.similarity_search_with_score(
                expanded_query, 
                k=self.semantic_config['max_candidates']
            )
            for doc, score in candidates:
                all_candidates.append({
                    'document': doc,
                    'similarity_score': score,
                    'source_query': expanded_query
                })
        
        # 3. 去重
        unique_candidates = self._deduplicate_candidates(all_candidates)
        print(f"🎯 去重后候选: {len(unique_candidates)} 个")
        
        # 4. LLM语义评估
        if self.semantic_config['llm_evaluation']:
            semantic_chunks = self._evaluate_semantic_relevance(query, unique_candidates)
        else:
            semantic_chunks = self._convert_to_semantic_chunks(unique_candidates)
        
        # 5. 排序和返回
        semantic_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return semantic_chunks[:k]
    
    def _expand_query_semantically(self, query: str) -> List[str]:
        """语义查询扩展"""
        if not self.semantic_config['query_expansion']:
            return [query]
        
        expansion_prompt = f"""
        原始查询: {query}
        
        请生成这个查询的语义等价变体，包括：
        1. 同义词替换
        2. 不同的表达方式  
        3. 相关的概念和术语
        4. 不同的问题形式
        
        返回格式: 原始查询|变体1|变体2|变体3|变体4
        
        例如: 農業機械の種類について教えてください|農業機械の分類について|農業機械にはどんな種類がありますか|農業機械のタイプについて|農業機械のカテゴリについて
        """
        
        try:
            response = self.llm.invoke(expansion_prompt)
            expanded_queries = [q.strip() for q in response.content.strip().split('|')]
            return expanded_queries[:5]  # 限制变体数量
        except Exception as e:
            print(f"⚠️ 查询扩展失败: {e}")
            return [query]
    
    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """去重候选文档"""
        seen_contents = set()
        unique_candidates = []
        
        for candidate in candidates:
            content = candidate['document'].page_content
            if content not in seen_contents:
                seen_contents.add(content)
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _evaluate_semantic_relevance(self, query: str, candidates: List[Dict]) -> List[SemanticChunk]:
        """LLM语义相关性评估"""
        print("🧠 LLM语义评估中...")
        
        semantic_chunks = []
        
        for i, candidate in enumerate(candidates):
            doc = candidate['document']
            similarity_score = candidate['similarity_score']
            
            # 构建评估prompt
            evaluation_prompt = f"""
            查询: {query}
            文档: {doc.page_content}
            
            请评估这个文档与查询的语义相关性，考虑：
            1. 内容是否直接回答查询
            2. 信息的相关性和有用性
            3. 文档的完整性和准确性
            
            返回JSON格式:
            {{
                "relevance_score": 0.85,
                "reasoning": "文档包含查询相关的分类信息，直接回答了种类问题",
                "is_direct_answer": true,
                "confidence": 0.9
            }}
            """
            
            try:
                response = self.llm.invoke(evaluation_prompt)
                result = json.loads(response.content)
                
                relevance_score = result.get('relevance_score', 0.5)
                reasoning = result.get('reasoning', '')
                is_direct_answer = result.get('is_direct_answer', False)
                confidence = result.get('confidence', 0.5)
                
                # 计算最终分数（结合向量相似度和语义相关性）
                final_score = (similarity_score * 0.3 + relevance_score * 0.7)
                
                # 如果是直接回答，给予额外奖励
                if is_direct_answer:
                    final_score *= 1.2
                
                semantic_chunk = SemanticChunk(
                    content=doc.page_content,
                    similarity_score=similarity_score,
                    semantic_relevance=relevance_score,
                    final_score=final_score,
                    granularity='semantic',
                    reasoning=reasoning,
                    metadata={
                        'source_query': candidate.get('source_query', query),
                        'is_direct_answer': is_direct_answer,
                        'confidence': confidence,
                        'original_metadata': doc.metadata
                    }
                )
                
                semantic_chunks.append(semantic_chunk)
                
            except Exception as e:
                print(f"⚠️ LLM评估失败: {e}")
                # 回退到基础转换
                semantic_chunk = SemanticChunk(
                    content=doc.page_content,
                    similarity_score=similarity_score,
                    semantic_relevance=0.5,
                    final_score=similarity_score,
                    granularity='semantic',
                    reasoning='LLM评估失败，使用向量相似度',
                    metadata={
                        'source_query': candidate.get('source_query', query),
                        'is_direct_answer': False,
                        'confidence': 0.3,
                        'original_metadata': doc.metadata
                    }
                )
                semantic_chunks.append(semantic_chunk)
        
        return semantic_chunks
    
    def _convert_to_semantic_chunks(self, candidates: List[Dict]) -> List[SemanticChunk]:
        """转换为语义chunk（无LLM评估时使用）"""
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
                reasoning='基于向量相似度',
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
        """基于语义chunks生成回答"""
        if not semantic_chunks:
            return {
                'answer': '抱歉，没有找到相关信息。',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': '没有找到相关文档'
            }
        
        # 构建上下文
        context_parts = []
        for i, chunk in enumerate(semantic_chunks[:3], 1):  # 使用前3个最相关的chunks
            context_parts.append(f"文档{i}: {chunk.content}")
        
        context = "\n\n".join(context_parts)
        
        # 生成回答的prompt
        answer_prompt = f"""
        查询: {query}
        
        相关文档:
        {context}
        
        请基于以上文档生成一个完整、准确的回答。要求：
        1. 直接回答用户的问题
        2. 基于文档内容，但用自己的话表达
        3. 如果查询涉及分类，请列出具体的分类
        4. 回答要完整、准确、简洁
        
        返回JSON格式:
        {{
            "answer": "生成的完整回答",
            "evidence_text": "最重要的证据文本",
            "confidence": 0.9,
            "reasoning": "回答的推理过程"
        }}
        """
        
        try:
            response = self.llm.invoke(answer_prompt)
            result = json.loads(response.content)
            
            # 找到最佳证据文本
            best_chunk = semantic_chunks[0]
            evidence_text = best_chunk.content
            
            return {
                'answer': result.get('answer', ''),
                'evidence_text': evidence_text,
                'source_document': context,
                'confidence': result.get('confidence', 0.8),
                'reasoning': result.get('reasoning', ''),
                'model': 'PureSemanticRAG',
                'processing_time': 0.0,  # 将在外部计算
                'chunks_used': len(semantic_chunks)
            }
            
        except Exception as e:
            print(f"⚠️ 回答生成失败: {e}")
            return {
                'answer': '抱歉，生成回答时出现错误。',
                'evidence_text': semantic_chunks[0].content if semantic_chunks else '',
                'source_document': context,
                'confidence': 0.3,
                'reasoning': f'生成失败: {str(e)}',
                'model': 'PureSemanticRAG',
                'processing_time': 0.0,
                'chunks_used': len(semantic_chunks)
            }
    
    def query_with_answer(self, query: str, k: int = 3) -> Dict[str, Any]:
        """完整查询流程"""
        start_time = time.time()
        
        # 1. 语义检索
        semantic_chunks = self.semantic_query(query, k)
        
        # 2. 生成回答
        result = self.generate_answer(query, semantic_chunks)
        
        # 3. 添加处理时间
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        return result

def test_pure_semantic_rag():
    """测试纯语义RAG系统"""
    print("🧪 纯语义RAG系统测试")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    # 初始化系统
    rag = PureSemanticRAG(api_key)
    
    # 加载测试数据
    data_file = "../data/single_20240229.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📖 加载了 {len(data)} 条数据")
    
    # 构建向量数据库
    rag.build_vector_store(data[:100])  # 使用前100条数据进行测试
    
    # 测试查询
    test_queries = [
        "農業機械の種類について教えてください",
        "コンバインとは何ですか",
        "普通型と自立型の違いは何ですか"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: {query}")
        print("-" * 40)
        
        result = rag.query_with_answer(query, k=3)
        
        print(f"⏱️  处理时间: {result['processing_time']:.2f}s")
        print(f"💬 回答: {result['answer']}")
        print(f"🔍 证据: {result['evidence_text']}")
        print(f"📊 信心度: {result['confidence']}")
        print(f"🧠 推理: {result['reasoning']}")
        print(f"📄 使用chunks: {result['chunks_used']}")

if __name__ == "__main__":
    test_pure_semantic_rag()
