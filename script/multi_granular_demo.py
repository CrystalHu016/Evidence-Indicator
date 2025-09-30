#!/usr/bin/env python3
"""
多粒度分块算法演示 (Multi-granular Chunking Algorithm Demo)
"""

import os
import json
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr

class MultiGranularRAG:
    """多粒度RAG系统演示"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
        self.db = None
    
    def create_multi_granular_chunks(self, data: List[Dict]) -> Dict[str, List[Document]]:
        """创建多粒度chunks"""
        print("🔀 创建多粒度chunks...")
        
        all_chunks = {
            'sentence': [],      # 句子级 (10-60字符)
            'short_passage': [], # 短段落级 (80-200字符) 
            'long_passage': []   # 长段落级 (300-500字符)
        }
        
        for doc_idx, item in enumerate(data[:100]):  # 限制数量用于演示
            content = item.get('output', '') or item.get('text', '') or item.get('content', '')
            if not content:
                continue
                
            # 1. 句子级分块
            sentence_chunks = self._create_sentence_chunks(content, doc_idx)
            all_chunks['sentence'].extend(sentence_chunks)
            
            # 2. 短段落级分块
            short_chunks = self._create_short_passage_chunks(content, doc_idx)
            all_chunks['short_passage'].extend(short_chunks)
            
            # 3. 长段落级分块
            long_chunks = self._create_long_passage_chunks(content, doc_idx)
            all_chunks['long_passage'].extend(long_chunks)
        
        return all_chunks
    
    def _create_sentence_chunks(self, content: str, doc_idx: int) -> List[Document]:
        """创建句子级chunks (10-60字符)"""
        import re
        
        sentences = re.split(r'[。！？.!?]', content)
        chunks = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if 10 <= len(sentence) <= 60 and sentence:
                chunk_doc = Document(
                    page_content=sentence,
                    metadata={
                        'granularity': 'sentence',
                        'doc_index': doc_idx,
                        'chunk_index': i,
                        'chunk_size': len(sentence),
                        'chunk_type': 'sentence'
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def _create_short_passage_chunks(self, content: str, doc_idx: int) -> List[Document]:
        """创建短段落级chunks (80-200字符)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,
            chunk_overlap=30,
            length_function=len,
            separators=["\\n\\n", "。", "！", "？", "、", "\\n", " ", ""]
        )
        
        chunks = []
        raw_chunks = splitter.split_text(content)
        
        for i, chunk_text in enumerate(raw_chunks):
            if 80 <= len(chunk_text) <= 200:
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'granularity': 'short_passage',
                        'doc_index': doc_idx,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'chunk_type': 'short_passage'
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def _create_long_passage_chunks(self, content: str, doc_idx: int) -> List[Document]:
        """创建长段落级chunks (300-500字符)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            length_function=len,
            separators=["\\n\\n", "。", "！", "？", "\\n", " ", ""]
        )
        
        chunks = []
        raw_chunks = splitter.split_text(content)
        
        for i, chunk_text in enumerate(raw_chunks):
            if 300 <= len(chunk_text) <= 500:
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata={
                        'granularity': 'long_passage',
                        'doc_index': doc_idx,
                        'chunk_index': i,
                        'chunk_size': len(chunk_text),
                        'chunk_type': 'long_passage'
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def build_demo_vector_store(self, data: List[Dict]) -> bool:
        """构建演示向量库"""
        try:
            # 创建多粒度chunks
            all_chunks = self.create_multi_granular_chunks(data)
            
            print(f"📊 多粒度分块统计:")
            total_chunks = 0
            for granularity, chunks in all_chunks.items():
                avg_size = sum(len(c.page_content) for c in chunks) / len(chunks) if chunks else 0
                print(f"  {granularity}: {len(chunks)} chunks (平均 {avg_size:.1f} 字符)")
                total_chunks += len(chunks)
            
            # 合并所有chunks
            all_combined_chunks = []
            for granularity, chunks in all_chunks.items():
                all_combined_chunks.extend(chunks)
            
            print(f"📦 总计 {total_chunks} 个多粒度chunks")
            
            # 创建向量库
            print("🔄 创建向量库...")
            start_time = time.time()
            
            self.db = Chroma.from_documents(
                all_combined_chunks,
                self.embedding_function,
                persist_directory="./chroma_multi_demo"
            )
            
            build_time = time.time() - start_time
            print(f"✅ 向量库构建完成! 耗时: {build_time:.2f}s")
            
            return True
            
        except Exception as e:
            print(f"❌ 构建失败: {e}")
            return False
    
    def multi_granular_query(self, query: str, k: int = 5) -> List[Dict]:
        """多粒度查询演示"""
        if not self.db:
            return []
        
        print(f"🔍 执行多粒度检索: {query}")
        
        # 获取候选结果
        results = self.db.similarity_search_with_score(query, k=k*3)
        
        # 按粒度分组
        granularity_groups = {
            'sentence': [],
            'short_passage': [],
            'long_passage': []
        }
        
        for doc, score in results:
            granularity = doc.metadata.get('granularity', 'unknown')
            if granularity in granularity_groups:
                granularity_groups[granularity].append({
                    'document': doc,
                    'score': score,
                    'granularity': granularity,
                    'chunk_size': doc.metadata.get('chunk_size', len(doc.page_content))
                })
        
        # 计算查询复杂度
        query_complexity = self._calculate_query_complexity(query)
        print(f"🎯 查询复杂度: {query_complexity:.2f}")
        
        # 基于复杂度选择粒度偏好
        if query_complexity <= 0.3:
            weights = {'sentence': 1.0, 'short_passage': 0.5, 'long_passage': 0.2}
            print("📝 简单查询 - 优先句子级别")
        elif query_complexity <= 0.7:
            weights = {'sentence': 0.6, 'short_passage': 1.0, 'long_passage': 0.7}
            print("📄 中等查询 - 优先短段落级别")
        else:
            weights = {'sentence': 0.3, 'short_passage': 0.8, 'long_passage': 1.0}
            print("📚 复杂查询 - 优先长段落级别")
        
        # 重新计算加权分数
        all_weighted_chunks = []
        for granularity, chunks in granularity_groups.items():
            weight = weights.get(granularity, 0.5)
            for chunk in chunks:
                weighted_score = chunk['score'] * weight
                chunk['weighted_score'] = weighted_score
                all_weighted_chunks.append(chunk)
        
        # 排序并选择top-k
        all_weighted_chunks.sort(key=lambda x: x['weighted_score'])
        selected = all_weighted_chunks[:k]
        
        # 统计结果
        counts = {'sentence': 0, 'short_passage': 0, 'long_passage': 0}
        for chunk in selected:
            granularity = chunk['granularity']
            if granularity in counts:
                counts[granularity] += 1
        
        print(f"📊 选择结果:")
        for granularity, count in counts.items():
            if count > 0:
                print(f"  {granularity}: {count} chunks")
        
        return selected
    
    def _calculate_query_complexity(self, query: str) -> float:
        """计算查询复杂度"""
        complexity_score = 0.0
        
        # 长度因子
        if len(query) > 20:
            complexity_score += 0.3
        elif len(query) > 10:
            complexity_score += 0.1
        
        # 复杂词汇因子
        complex_patterns = ['について', 'とは何', 'どのように', '違いは', '特徴', '方法', '理由']
        for pattern in complex_patterns:
            if pattern in query:
                complexity_score += 0.2
                break
        
        # 多概念因子
        import re
        concepts = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', query)
        if len(concepts) > 3:
            complexity_score += 0.3
        elif len(concepts) > 1:
            complexity_score += 0.1
        
        return min(complexity_score, 1.0)

def main():
    """主演示函数"""
    print("🚀 多粒度分块算法演示")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return
    
    # 加载数据
    data_file = "../data/single_20240229.json"
    print(f"📖 加载数据: {data_file}")
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 加载了 {len(data)} 条数据")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 初始化系统
    rag = MultiGranularRAG(api_key)
    
    # 构建向量库
    print(f"\n🗃️ 构建多粒度向量库...")
    if not rag.build_demo_vector_store(data):
        return
    
    # 测试查询
    test_queries = [
        ("コンバイン", "简单"),
        ("コンバインとは何ですか", "中等"),
        ("コンバインの種類とそれぞれの特徴について詳しく教えて", "复杂")
    ]
    
    print(f"\n🧪 测试查询")
    print("=" * 40)
    
    for i, (query, expected) in enumerate(test_queries, 1):
        print(f"\n📋 测试 {i}: {query}")
        print(f"预期复杂度: {expected}")
        print("-" * 30)
        
        results = rag.multi_granular_query(query, k=3)
        
        if results:
            print(f"🎯 最佳结果:")
            best = results[0]
            print(f"  粒度: {best['granularity']}")
            print(f"  大小: {best['chunk_size']}字符")
            print(f"  内容: {best['document'].page_content[:60]}...")
            print(f"  分数: {best['weighted_score']:.3f}")
    
    print(f"\n🎉 多粒度分块算法演示完成!")

if __name__ == "__main__":
    main()