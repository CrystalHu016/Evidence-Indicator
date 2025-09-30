#!/usr/bin/env python3
"""
改进的纯语义RAG系统 - 修复JSON解析问题，更稳定的实现
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

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

class ImprovedPureRAG:
    """改进的纯语义RAG系统"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.llm = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini", temperature=0)
        
        # 向量存储
        self.vector_store = None
        self.documents = []
        
        # 配置
        self.config = {
            'similarity_threshold': 0.2,
            'max_candidates': 8,
            'use_llm_evaluation': True,
            'use_query_expansion': True
        }
    
    def build_vector_store(self, data: List[Dict], chunk_size: int = 300, chunk_overlap: int = 50):
        """构建向量数据库"""
        print("🏗️ 构建改进的纯语义向量数据库...")
        
        # 加载数据
        documents = []
        for item in data:
            content = item.get('output', '') or item.get('text', '') or item.get('content', '')
            if content:
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'improved_semantic_rag',
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
            persist_directory="./chroma_improved_semantic"
        )
        
        self.documents = chunks
        print("✅ 改进的纯语义向量数据库构建完成")
    
    def semantic_query(self, query: str, k: int = 3) -> List[SemanticChunk]:
        """改进的语义查询"""
        if not self.vector_store:
            return []
        
        print(f"🔍 改进语义查询: {query}")
        
        # 1. 查询扩展
        expanded_queries = self._expand_query_safely(query)
        print(f"📝 查询扩展: {len(expanded_queries)} 个变体")
        
        # 2. 多查询检索
        all_candidates = []
        for expanded_query in expanded_queries:
            candidates = self.vector_store.similarity_search_with_score(
                expanded_query, 
                k=self.config['max_candidates']
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
        
        # 4. 语义评估
        if self.config['use_llm_evaluation']:
            semantic_chunks = self._evaluate_semantic_relevance_safely(query, unique_candidates)
        else:
            semantic_chunks = self._convert_to_semantic_chunks(unique_candidates)
        
        # 5. 排序和返回
        semantic_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return semantic_chunks[:k]
    
    def _expand_query_safely(self, query: str) -> List[str]:
        """安全的查询扩展"""
        if not self.config['use_query_expansion']:
            return [query]
        
        expansion_prompt = f"""
        原始查询: {query}
        
        请生成这个查询的2-3个语义等价变体，用|分隔：
        
        例如: 農業機械の種類について教えてください|農業機械の分類について|農業機械にはどんな種類がありますか
        """
        
        try:
            response = self.llm.invoke(expansion_prompt)
            content = response.content.strip()
            
            # 安全解析
            if '|' in content:
                expanded_queries = [q.strip() for q in content.split('|') if q.strip()]
            else:
                expanded_queries = [query]
            
            return expanded_queries[:3]  # 限制变体数量
            
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
    
    def _evaluate_semantic_relevance_safely(self, query: str, candidates: List[Dict]) -> List[SemanticChunk]:
        """安全的LLM语义相关性评估"""
        print("🧠 安全LLM语义评估中...")
        
        semantic_chunks = []
        
        for i, candidate in enumerate(candidates):
            doc = candidate['document']
            similarity_score = candidate['similarity_score']
            
            # 简化的评估prompt
            evaluation_prompt = f"""
            查询: {query}
            文档: {doc.page_content}
            
            这个文档与查询的相关性如何？请用0-1之间的数字表示，并简要说明原因。
            
            格式: 分数|原因
            例如: 0.85|文档包含查询相关的分类信息
            """
            
            try:
                response = self.llm.invoke(evaluation_prompt)
                content = response.content.strip()
                
                # 安全解析
                if '|' in content:
                    parts = content.split('|', 1)
                    try:
                        relevance_score = float(parts[0].strip())
                        reasoning = parts[1].strip() if len(parts) > 1 else "LLM评估"
                    except ValueError:
                        relevance_score = 0.5
                        reasoning = "解析失败，使用默认分数"
                else:
                    # 尝试提取数字
                    numbers = re.findall(r'0\.\d+', content)
                    if numbers:
                        relevance_score = float(numbers[0])
                        reasoning = content
                    else:
                        relevance_score = 0.5
                        reasoning = "无法解析分数"
                
                # 确保分数在合理范围内
                relevance_score = max(0.0, min(1.0, relevance_score))
                
            except Exception as e:
                print(f"⚠️ LLM评估失败: {e}")
                relevance_score = 0.5
                reasoning = f"评估失败: {str(e)}"
            
            # 计算最终分数
            final_score = (similarity_score * 0.4 + relevance_score * 0.6)
            
            semantic_chunk = SemanticChunk(
                content=doc.page_content,
                similarity_score=similarity_score,
                semantic_relevance=relevance_score,
                final_score=final_score,
                granularity='semantic',
                reasoning=reasoning,
                metadata={
                    'source_query': candidate.get('source_query', query),
                    'is_direct_answer': relevance_score > 0.7,
                    'confidence': relevance_score,
                    'original_metadata': doc.metadata
                }
            )
            
            semantic_chunks.append(semantic_chunk)
        
        return semantic_chunks
    
    def _convert_to_semantic_chunks(self, candidates: List[Dict]) -> List[SemanticChunk]:
        """转换为语义chunk"""
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
        for i, chunk in enumerate(semantic_chunks[:3], 1):
            context_parts.append(f"文档{i}: {chunk.content}")
        
        context = "\n\n".join(context_parts)
        
        # 简化的回答生成prompt
        answer_prompt = f"""
        查询: {query}
        
        相关文档:
        {context}
        
        请基于以上文档生成一个完整、准确的回答。要求：
        1. 直接回答用户的问题
        2. 基于文档内容，但用自己的话表达
        3. 如果查询涉及分类，请列出具体的分类
        4. 回答要完整、准确、简洁
        
        请直接返回回答内容，不需要其他格式。
        """
        
        try:
            response = self.llm.invoke(answer_prompt)
            answer = response.content.strip()
            
            # 找到最佳证据文本
            best_chunk = semantic_chunks[0]
            evidence_text = best_chunk.content
            
            # 计算平均信心度
            avg_confidence = sum(chunk.metadata['confidence'] for chunk in semantic_chunks) / len(semantic_chunks)
            
            return {
                'answer': answer,
                'evidence_text': evidence_text,
                'source_document': context,
                'confidence': avg_confidence,
                'reasoning': f'基于{len(semantic_chunks)}个相关文档生成',
                'model': 'ImprovedPureRAG',
                'processing_time': 0.0,
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
                'model': 'ImprovedPureRAG',
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

def test_improved_pure_rag():
    """测试改进的纯语义RAG系统"""
    print("🧪 改进的纯语义RAG系统测试")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    # 初始化系统
    rag = ImprovedPureRAG(api_key)
    
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
        print(f"📊 信心度: {result['confidence']:.2f}")
        print(f"🧠 推理: {result['reasoning']}")
        print(f"📄 使用chunks: {result['chunks_used']}")

if __name__ == "__main__":
    test_improved_pure_rag()
