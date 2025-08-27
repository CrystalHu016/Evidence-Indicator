#!/usr/bin/env python3
"""
Simple test script for basic article parsing
简单测试脚本 - 测试基本文章解析功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def test_specific_article():
    """测试特定文章的解析"""
    print("📰 Testing Specific Article Parsing")
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
            print(f"📄 Content preview: {article_content['content'][:500]}...")
            
            # 测试LLM回答生成
            print(f"\n🤖 Testing LLM answer generation...")
            user_query = "AIで病院の業務効率化実例について教えてください"
            
            answer = crawler.generate_llm_answer(user_query, str(article_content))
            
            if answer:
                print(f"✅ LLM answer generated:")
                print(f"📝 {answer}")
            else:
                print("❌ Failed to generate LLM answer")
                
        else:
            print("❌ Failed to parse article")
        
    except Exception as e:
        print(f"❌ Error during article parsing: {e}")
        import traceback
        traceback.print_exc()

def test_category_page():
    """测试分类页面解析"""
    print("\n📂 Testing Category Page Parsing")
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
        
        # 测试IT分类页面
        test_url = "https://news.yahoo.co.jp/categories/it"
        
        print(f"🔍 Parsing category page: {test_url}")
        
        html = crawler.get_page_content(test_url)
        if html:
            articles = crawler.parse_category_content(html, 'it')
            print(f"✅ Found {len(articles)} articles in IT category")
            
            if articles:
                print("📰 Sample articles:")
                for i, article in enumerate(articles[:3], 1):
                    print(f"  {i}. {article['title']}")
                    print(f"     URL: {article['url']}")
        else:
            print("❌ Failed to fetch category page")
        
    except Exception as e:
        print(f"❌ Error during category page parsing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主函数"""
    print("🚀 Yahoo!ニュース Simple Test")
    print("=" * 60)
    
    # 测试特定文章解析
    test_specific_article()
    
    # 测试分类页面解析
    test_category_page()
    
    print("\n🎉 Simple test completed!")

if __name__ == "__main__":
    main()
