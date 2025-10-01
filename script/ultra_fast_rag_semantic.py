#!/usr/bin/env python3
"""
纯语义RAG系统 - 完全基于LLM语义理解，无硬编码规则
基于ultra_fast_rag_integrated.py重构，删除所有硬编码算法
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
    """语义chunk结果"""
    content: str
    similarity_score: float
    semantic_relevance: float
    final_score: float
    granularity: str
    reasoning: str
    metadata: Dict[str, Any]


class SemanticLLMRanker:
    """纯语义LLM排序系统 - 无硬编码规则"""
    
    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
    
    def rank_chunks_semantically(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 1) -> List[Dict[str, Any]]:
        """纯语义排序 - 完全基于LLM理解"""
        if not chunks:
            return []
        
        print(f"🧠 纯语义LLM评估: {len(chunks)} 个候选chunks")
        
        # 只对top 1个向量匹配最好的chunk进行LLM评估
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
                "final_score": llm_score,  # 完全基于LLM评分
                "rank_order": 1
            }
            
            print(f"✅ 语义评估完成: LLM评分 {llm_score:.3f}")
            return [enhanced_chunk]
            
        except Exception as e:
            print(f"❌ 语义评估失败: {e}")
            # 回退到向量相似度
            enhanced_chunk = {
                **top_chunk,
                "llm_score": top_chunk.get("similarity_score", 0.0),
                "relevance_reason": "LLM评估失败，使用向量相似度",
                "generated_answer": top_chunk["content"][:100] + "...",
                "final_score": top_chunk.get("similarity_score", 0.0),
                "rank_order": 1
            }
            return [enhanced_chunk]
    
    def _evaluate_chunk_semantically(self, query: str, content: str, vector_score: float) -> Tuple[float, str, str]:
        """纯语义评估 - 完全基于LLM理解"""
        
        evaluation_prompt = f"""
        查询: {query}
        参考文本: {content}
        
        请评估这个参考文本与查询的语义相关性，并生成回答。要求:
        1. 评估相关性分数 (0-1之间)
        2. 说明评估理由
        3. 基于参考文本生成完整回答
        
        返回JSON格式:
        {{
            "relevance_score": <0-1分数，表示参考文本对查询的相关性>,
            "reason": "<为什么给出这个相关性分数的理由>",
            "generated_answer": "<根据参考文本生成的完整回答>"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个智能问答助手，能够准确评估文本相关性并生成完整回答。"},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            relevance_score, reason, generated_answer = self._parse_semantic_response(result_text)

            return relevance_score, reason, generated_answer

        except Exception as e:
            print(f"LLM语义评估失败: {e}")
            # 回退到简单评估
            fallback_answer = self._generate_fallback_answer(query, content)
            return vector_score, "LLM评估失败，使用向量相似度", fallback_answer
    
    def _parse_semantic_response(self, result_text: str) -> Tuple[float, str, str]:
        """解析LLM语义响应"""
        try:
            # 提取JSON块
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = result_text[json_start:json_end]
                parsed = json.loads(json_str)

                relevance_score = float(parsed.get('relevance_score', 0.0))
                reason = parsed.get('reason', '未提供理由')
                generated_answer = parsed.get('generated_answer', '')

                if not generated_answer or generated_answer.strip() == '':
                    generated_answer = "抱歉，无法基于参考文本生成回答。"

                return relevance_score, reason, generated_answer
            else:
                raise ValueError("未找到有效的JSON格式")

        except Exception as e:
            print(f"LLM响应解析失败: {e}")
            return 0.5, f"解析失败: {str(e)}", "解析失败，无法生成回答"

    def _generate_fallback_answer(self, query: str, content: str) -> str:
        """生成备用回答"""
        # 简单的基于内容长度的回答生成
        if len(content) > 100:
            return content[:100] + "..."
        else:
            return content


class PureSemanticRAG:
    """纯语义RAG系统 - 完全基于LLM语义理解"""
    
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma_semantic"):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.llm = ChatOpenAI(api_key=SecretStr(openai_api_key), model="gpt-4o-mini", temperature=0)
        
        # 向量存储
        self.db = None
        if os.path.exists(chroma_path):
            try:
                self.db = Chroma(persist_directory=chroma_path, embedding_function=self.embedding_function)
                print(f"✅ 加载现有向量数据库: {chroma_path}")
            except Exception as e:
                print(f"⚠️ 向量数据库加载失败: {e}")
        
        # 语义排序系统
        self.semantic_ranker = SemanticLLMRanker(openai_api_key)
        
        # 配置 - 优化性能：减少候选数量以提升速度
        self.config = {
            'similarity_threshold': 0.2,
            'max_candidates': 3,  # 从8减少到3，预计提速50%
            'use_query_expansion': True,  # 保留查询扩展以保证召回率
            'use_semantic_ranking': True
        }
    
    def build_vector_store(self, data_file: str, chunk_size: int = 150, chunk_overlap: int = 30) -> bool:
        """构建纯语义向量数据库"""
        try:
            print(f"🏗️ 构建纯语义向量数据库...")
            print(f"📁 数据文件: {data_file}")
            print(f"🗄️ 向量数据库路径: {self.chroma_path}")
            
            # 检查是否已存在向量库
            if os.path.exists(self.chroma_path) and os.listdir(self.chroma_path):
                print(f"✅ 加载现有向量数据库: {self.chroma_path}")
                self.db = Chroma(
                    persist_directory=self.chroma_path,
                    embedding_function=self.embedding_function
                )
                return True
            
            # 尝试使用其他可用的向量数据库
            alternative_paths = [
                "./chroma_integrated",
                "./chroma_semantic_test", 
                "./chroma_pure_semantic",
                "./chroma_improved_semantic"
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path) and os.listdir(alt_path):
                    print(f"✅ 使用替代向量数据库: {alt_path}")
                    self.chroma_path = alt_path
                    self.db = Chroma(
                        persist_directory=self.chroma_path,
                        embedding_function=self.embedding_function
                    )
                    return True
            
            # 检查数据文件
            if not os.path.exists(data_file):
                print(f"❌ 数据文件不存在: {data_file}")
                return False
            
            # 加载数据
            print("📖 加载数据...")
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"✅ 加载了 {len(data)} 条数据")
            
            # 转换为Document格式
            documents = []
            for item in data:
                content = item.get('output', '') or item.get('text', '') or item.get('content', '')
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': 'semantic_rag',
                            'original_index': len(documents),
                            'original_full_text': content  # 保存完整原文
                        }
                    )
                    documents.append(doc)
            
            print(f"📄 转换了 {len(documents)} 个文档")
            
            # 文本分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "。", "！", "？", "、", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            print(f"📄 创建了 {len(chunks)} 个chunks")
            
            # 清理旧的向量库
            if os.path.exists(self.chroma_path):
                import shutil
                shutil.rmtree(self.chroma_path)
                print("🗑️ 清理旧向量库")
            
            # 构建向量存储
            print("🔄 创建纯语义向量库...")
            start_time = time.time()
            
            self.db = Chroma.from_documents(
                chunks,
                self.embedding_function,
                persist_directory=self.chroma_path
            )
            
            build_time = time.time() - start_time
            print(f"✅ 纯语义向量库构建完成! 耗时: {build_time:.2f}s")
            print(f"📊 统计: {len(chunks)} chunks, 平均 {build_time/len(chunks)*1000:.1f}ms/chunk")
            
            return True
            
        except Exception as e:
            print(f"❌ 纯语义向量库构建失败: {e}")
            return False
    
    def semantic_query(self, query: str, k: int = 3) -> List[SemanticChunk]:
        """纯语义查询 - 完全基于LLM理解"""
        if not self.db:
            return []
        
        print(f"🔍 纯语义查询: {query}")
        
        # 1. 查询扩展
        expanded_queries = self._expand_query_semantically(query)
        print(f"📝 查询扩展: {len(expanded_queries)} 个变体")
        
        # 2. 多查询检索
        all_candidates = []
        for expanded_query in expanded_queries:
            candidates = self.db.similarity_search_with_score(
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
        if self.config['use_semantic_ranking']:
            semantic_chunks = self._evaluate_semantic_relevance(query, unique_candidates)
        else:
            semantic_chunks = self._convert_to_semantic_chunks(unique_candidates)
        
        # 5. 排序和返回
        semantic_chunks.sort(key=lambda x: x.final_score, reverse=True)
        return semantic_chunks[:k]
    
    def _expand_query_semantically(self, query: str) -> List[str]:
        """语义查询扩展"""
        if not self.config['use_query_expansion']:
            return [query]
        
        expansion_prompt = f"""
        原始查询: {query}

        请生成这个查询的1个语义等价变体，用|分隔（包含原始查询共2个）：

        例如: 農業機械の種類について教えてください|農業機械の分類について
        """

        try:
            response = self.llm.invoke(expansion_prompt)
            content = response.content.strip()

            # 安全解析
            if '|' in content:
                expanded_queries = [q.strip() for q in content.split('|') if q.strip()]
            else:
                expanded_queries = [query]

            return expanded_queries[:2]  # 限制为2个变体（原始+1个）
            
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
            
            # 使用语义排序器评估
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
                print(f"⚠️ 语义评估失败: {e}")
                # 回退到基础转换
                semantic_chunk = SemanticChunk(
                    content=doc.page_content,
                    similarity_score=similarity_score,
                    semantic_relevance=similarity_score,
                    final_score=similarity_score,
                    granularity='semantic',
                    reasoning='LLM评估失败，使用向量相似度',
                    metadata={
                        'source_query': candidate.get('source_query', query),
                        'is_direct_answer': False,
                        'confidence': 0.5,
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
        
        # 构建上下文 - 使用所有过滤后的chunks（不再限制为3个）
        context_parts = []
        chunks_details = []
        for i, chunk in enumerate(semantic_chunks, 1):  # 使用所有过滤后的chunks
            context_parts.append(f"文档{i}: {chunk.content}")
            # 收集每个chunk的详细评分信息
            chunks_details.append({
                'chunk_id': i,
                'content': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content,
                'similarity_score': float(chunk.similarity_score),
                'semantic_relevance': float(chunk.semantic_relevance),
                'final_score': float(chunk.final_score),
                'confidence': float(chunk.metadata.get('confidence', 0.0)),
                'reasoning': chunk.reasoning,
                'granularity': chunk.granularity
            })

        context = "\n\n".join(context_parts)

        # 打印详细评分信息
        print("\n📊 检索到的文档评分详情:")
        print("=" * 80)
        for detail in chunks_details:
            print(f"\n文档{detail['chunk_id']}:")
            print(f"  📝 内容预览: {detail['content']}")
            print(f"  🎯 向量相似度 (similarity_score): {detail['similarity_score']:.4f}")
            print(f"  🧠 语义相关性 (semantic_relevance): {detail['semantic_relevance']:.4f}")
            print(f"  ⭐ 最终得分 (final_score): {detail['final_score']:.4f}")
            print(f"  💯 置信度 (confidence): {detail['confidence']:.4f}")
            print(f"  📐 粒度 (granularity): {detail['granularity']}")
            print(f"  💬 评分理由: {detail['reasoning']}")
        print("=" * 80)
        
        # 生成回答的prompt - 基于原文生成简短总结答案
        answer_prompt = f"""
        用户问题: {query}

        参考文档:
        {context}

        请基于参考文档回答用户的问题。要求：
        1. 仔细阅读参考文档，理解用户问题的核心意图
        2. 根据文档内容，用简洁的1-2句话直接回答问题的核心要点
        3. 答案必须完全基于文档内容，不能编造信息
        4. 使用原文档的表达方式和术语
        5. 如果是定义类问题（如"とは何ですか"），给出核心定义即可，不需要详细展开
        6. 如果是分类/列举类问题，列出主要分类或项目即可
        7. 保持日语原文的表达习惯和语气

        请直接给出简短答案（1-2句话），不需要前缀或额外说明：
        """
        
        try:
            response = self.llm.invoke(answer_prompt)
            answer = response.content.strip()
            
            # 找到最佳证据文本
            best_chunk = semantic_chunks[0]
            evidence_text = best_chunk.content

            # 获取完整原始文档
            original_full_text = best_chunk.metadata.get('original_metadata', {}).get('original_full_text', '')
            if not original_full_text:
                # 如果没有保存完整原文，使用context作为fallback
                original_full_text = context

            # 计算平均信心度
            avg_confidence = sum(chunk.metadata['confidence'] for chunk in semantic_chunks) / len(semantic_chunks)

            return {
                'answer': answer,
                'evidence_text': evidence_text,
                'source_document': original_full_text,  # 返回完整原始文档
                'confidence': avg_confidence,
                'reasoning': f'基于{len(semantic_chunks)}个相关文档生成',
                'model': 'PureSemanticRAG',
                'processing_time': 0.0,
                'chunks_used': len(semantic_chunks),
                'chunks_details': chunks_details  # 添加详细评分信息
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
    
    def query_with_answer(self, query: str, k: int = 5, relevance_threshold: float = 0.7) -> Dict[str, Any]:
        """完整查询流程 - 先检索k个文档，然后过滤相关性≥threshold的文档"""
        start_time = time.time()

        # 1. 语义检索 - 获取top k个候选文档
        semantic_chunks = self.semantic_query(query, k)

        # 2. 过滤：只保留语义相关性≥threshold的文档
        filtered_chunks = [
            chunk for chunk in semantic_chunks
            if chunk.semantic_relevance >= relevance_threshold
        ]

        print(f"\n🔍 检索结果统计:")
        print(f"  📊 初始检索: {len(semantic_chunks)} 个文档")
        print(f"  ✅ 过滤后 (相关性≥{relevance_threshold}): {len(filtered_chunks)} 个文档")

        # 3. 如果没有符合条件的文档，返回日语抱歉信息
        if not filtered_chunks:
            processing_time = time.time() - start_time
            print(f"  ⚠️ 没有找到相关性≥{relevance_threshold}的文档，返回空结果")
            return {
                'answer': '申し訳ございませんが、現在利用可能な関連情報が見つかりませんでした。',
                'evidence_text': '',
                'source_document': '',
                'confidence': 0.0,
                'reasoning': f'没有找到语义相关性≥{relevance_threshold}的文档',
                'model': 'PureSemanticRAG',
                'processing_time': processing_time,
                'chunks_used': 0,
                'chunks_details': []
            }

        # 4. 使用过滤后的文档生成回答
        result = self.generate_answer(query, filtered_chunks)

        # 5. 添加处理时间
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
    rag.build_vector_store(data_file)  # 使用完整数据文件
    
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
    test_pure_semantic_rag()
