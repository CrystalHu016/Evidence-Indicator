#!/usr/bin/env python3
"""
Test script for article search and question answering functionality
测试文章搜索和问题回答功能的脚本
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_article_search():
    """测试文章搜索功能"""
    print("🔍 Testing Article Search Functionality")
    print("=" * 50)
    
    try:
        from enhanced_crawler import YahooNewsCrawler
        from dotenv import load_dotenv
        
        # 加载环境变量
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("❌ OpenAI API key not found")
            return
        
        # 创建爬虫实例
        crawler = YahooNewsCrawler(api_key)
        
        # 测试查询
        test_queries = [
            "AIで病院の業務効率化実例",
            "富士通のAIシステム",
            "医療AIの最新技術",
            "病院業務の自動化",
            "AIエージェントの医療応用"
        ]
        
        for query in test_queries:
            print(f"\n🎯 Testing query: {query}")
            print("-" * 40)
            
            # 搜索相关文章
            result = crawler.answer_question_with_articles(query)
            
            print(f"📝 Answer: {result['answer'][:200]}...")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"📊 Total articles found: {result['total_articles_found']}")
            
            if result['sources']:
                print("📚 Top sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"  {i}. {source['title']}")
                    print(f"     URL: {source['url']}")
                    print(f"     Score: {source['relevance_score']:.1f}")
                    print(f"     Category: {source['source']}")
            else:
                print("❌ No sources found")
            
            print()
        
        print("✅ Article search test completed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_specific_article():
    """测试特定文章的解析"""
    print("\n📰 Testing Specific Article Parsing")
    print("=" * 50)
    
    try:
        from enhanced_crawler import YahooNewsCrawler
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            print("❌ OpenAI API key not found")
            return
        
        crawler = YahooNewsCrawler(api_key)
        
        # 测试解析特定文章
        test_url = "https://news.yahoo.co.jp/articles/50e410b05cd15dd4acc318a5673ea3169189267c"
        
        print(f"🔍 Parsing article: {test_url}")
        
        article_content = crawler.extract_article_content(test_url)
        
        if article_content:
            print(f"✅ Article parsed successfully!")
            print(f"📝 Title: {article_content['title']}")
            print(f"📊 Word count: {article_content['word_count']}")
            print(f"⏰ Publish time: {article_content['publish_time']}")
            print(f"📄 Content preview: {article_content['content'][:300]}...")
        else:
            print("❌ Failed to parse article")
        
    except Exception as e:
        print(f"❌ Error during article parsing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 Yahoo!ニュース Article Search Test")
    print("=" * 60)
    
    # 测试文章搜索功能
    test_article_search()
    
    # 测试特定文章解析
    test_specific_article()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()
