#!/usr/bin/env python3
"""
Build and Test LLM-driven Intelligent RAG System
构建和测试LLM智能RAG系统
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import SecretStr

# 导入我们的系统
from enhanced_rag_system import EnhancedRAGSystem
from ultra_fast_rag import UltraFastRAG


class SystemBuilder:
    """系统构建器"""
    
    def __init__(self, openai_api_key: str):
        self.openai_api_key = openai_api_key
        self.embedding_function = OpenAIEmbeddings(api_key=SecretStr(openai_api_key))
    
    def build_vector_store(self, data_file: str, chroma_path: str = "./chroma_new") -> bool:
        """构建向量数据库"""
        try:
            print(f"🏗️ 构建向量数据库...")
            print(f"📁 数据文件: {data_file}")
            print(f"🗄️ 向量库路径: {chroma_path}")
            
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
                # 原始数据格式使用output字段作为主要内容
                content = item.get('output', '')
                if not content:
                    # 如果没有output，使用其他字段
                    content = item.get('text', '') or item.get('content', '')
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': 'single_20240229',
                        'original_index': str(len(documents))  # 简化metadata
                    }
                )
                documents.append(doc)
            
            print(f"📄 转换了 {len(documents)} 个文档")
            
            # 文本分割
            print("✂️ 分割文档...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=100,
                length_function=len,
                add_start_index=True,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"🔪 分割为 {len(chunks)} 个chunks")
            
            # 清理旧的向量库
            if os.path.exists(chroma_path):
                import shutil
                shutil.rmtree(chroma_path)
                print("🗑️ 清理旧的向量库")
            
            # 创建新的向量库
            print("🔄 创建向量库...")
            start_time = time.time()
            
            db = Chroma.from_documents(
                chunks,
                self.embedding_function,
                persist_directory=chroma_path
            )
            
            build_time = time.time() - start_time
            print(f"✅ 向量库构建完成! 耗时: {build_time:.2f}s")
            print(f"📊 统计: {len(chunks)} chunks, 平均 {build_time/len(chunks)*1000:.1f}ms/chunk")
            
            return True
            
        except Exception as e:
            print(f"❌ 向量库构建失败: {e}")
            return False


class SystemTester:
    """系统测试器"""
    
    def __init__(self, openai_api_key: str, chroma_path: str = "./chroma_new"):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
    
    def run_comprehensive_test(self):
        """运行综合测试"""
        print("🧪 开始综合测试")
        print("=" * 60)
        
        # 测试查询列表
        test_queries = [
            "上高地について教えて",
            "漢方薬の違いは何ですか？",
            "四字熟語のスローガンについて",
            "恋愛相談をお願いします"
        ]
        
        # 测试1: Enhanced RAG System
        print("\n🚀 测试1: Enhanced RAG System (完整功能)")
        print("-" * 40)
        
        try:
            enhanced_rag = EnhancedRAGSystem(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path
            )
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n📋 测试 {i}/{len(test_queries)}: {query}")
                
                start_time = time.time()
                result = enhanced_rag.query(query, initial_k=5, final_k=3, use_llm_ranking=True)
                test_time = time.time() - start_time
                
                print(f"⏱️ 总耗时: {test_time:.2f}s")
                print(f"💬 答案: {result['answer'][:100]}...")
                
                if result.get('chunks'):
                    best_chunk = result['chunks'][0]
                    print(f"🎯 最佳chunk评分: {best_chunk.get('final_score', 0):.3f}")
                    print(f"🔦 高亮预览: {best_chunk.get('highlighted_content', '')[:80]}...")
                
                print("-" * 30)
                
        except Exception as e:
            print(f"❌ Enhanced RAG测试失败: {e}")
        
        # 测试2: Ultra Fast RAG (LLM mode)
        print(f"\n⚡ 测试2: Ultra Fast RAG (LLM智能模式)")
        print("-" * 40)
        
        try:
            ultra_rag_llm = UltraFastRAG(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path,
                use_llm_ranking=True
            )
            
            for i, query in enumerate(test_queries, 1):
                print(f"\n📋 测试 {i}/{len(test_queries)}: {query}")
                
                start_time = time.time()
                answer, source, evidence, start_pos, end_pos = ultra_rag_llm.query(query, k=5)
                test_time = time.time() - start_time
                
                print(f"⏱️ 耗时: {test_time:.2f}s")
                print(f"💬 答案: {answer[:80]}...")
                print(f"🔍 证据: {evidence[:80]}...")
                print("-" * 30)
                
        except Exception as e:
            print(f"❌ Ultra Fast RAG (LLM)测试失败: {e}")
        
        # 测试3: Ultra Fast RAG (原始模式对比)
        print(f"\n🏃 测试3: Ultra Fast RAG (原始快速模式)")
        print("-" * 40)
        
        try:
            ultra_rag_fast = UltraFastRAG(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path,
                use_llm_ranking=False
            )
            
            total_fast_time = 0
            for i, query in enumerate(test_queries, 1):
                print(f"\n📋 测试 {i}/{len(test_queries)}: {query}")
                
                start_time = time.time()
                answer, source, evidence, start_pos, end_pos = ultra_rag_fast.query(query)
                test_time = time.time() - start_time
                total_fast_time += test_time
                
                print(f"⏱️ 耗时: {test_time:.2f}s")
                print(f"💬 答案: {answer[:80]}...")
                print("-" * 30)
            
            print(f"\n📊 原始模式平均耗时: {total_fast_time/len(test_queries):.2f}s")
                
        except Exception as e:
            print(f"❌ Ultra Fast RAG (原始)测试失败: {e}")
        
        print("\n🎉 综合测试完成!")
    
    def test_highlighting_demo(self):
        """测试高亮功能演示"""
        print("\n🔦 高亮功能演示")
        print("=" * 40)
        
        try:
            enhanced_rag = EnhancedRAGSystem(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path
            )
            
            demo_query = "上高地の特徴を教えて"
            print(f"🔍 演示查询: {demo_query}")
            
            result = enhanced_rag.query(demo_query, initial_k=3, final_k=2, use_llm_ranking=True)
            
            if result.get('chunks'):
                print(f"\n📊 找到 {len(result['chunks'])} 个相关chunks:")
                
                for i, chunk in enumerate(result['chunks'], 1):
                    print(f"\n--- Chunk {i} ---")
                    print(f"🎯 评分: {chunk.get('final_score', 0):.3f}")
                    print(f"📝 原始内容:")
                    print(f"   {chunk['content'][:150]}...")
                    print(f"🔦 高亮内容:")
                    print(f"   {chunk.get('highlighted_content', '无高亮')}")
                    print(f"💭 相关性理由:")
                    print(f"   {chunk.get('relevance_reason', '无理由')[:100]}...")
            
        except Exception as e:
            print(f"❌ 高亮演示失败: {e}")


def main():
    """主函数"""
    print("🏗️ LLM智能RAG系统 - 构建和测试")
    print("=" * 60)
    
    # 加载环境变量
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY 未设定")
        return
    
    # 配置
    data_file = "../data/single_20240229.json"  # 使用原来的大数据集 (9,103条数据)
    chroma_path = "./chroma"  # 使用原来的路径
    
    # 步骤1: 构建系统
    print("📋 步骤1: 构建向量数据库")
    builder = SystemBuilder(api_key)
    
    if builder.build_vector_store(data_file, chroma_path):
        print("✅ 向量库构建成功!")
    else:
        print("❌ 向量库构建失败，无法继续测试")
        return
    
    # 步骤2: 系统测试
    print(f"\n📋 步骤2: 系统功能测试")
    tester = SystemTester(api_key, chroma_path)
    
    # 运行综合测试
    tester.run_comprehensive_test()
    
    # 高亮功能演示
    tester.test_highlighting_demo()
    
    print(f"\n🎯 构建和测试完成!")
    print(f"💡 你现在可以使用以下方式与系统交互:")
    print(f"   - EnhancedRAGSystem: 完整功能版本")
    print(f"   - UltraFastRAG(use_llm_ranking=True): LLM智能版本")
    print(f"   - UltraFastRAG(use_llm_ranking=False): 原始快速版本")


if __name__ == "__main__":
    main()