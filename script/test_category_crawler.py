#!/usr/bin/env python3
"""
测试Yahoo!ニュース分类新闻爬虫的功能
"""

import os
import sys
import json
from category_news_crawler import CategoryNewsCrawler

def test_crawler_initialization():
    """测试爬虫初始化"""
    print("🧪 测试爬虫初始化...")
    
    try:
        crawler = CategoryNewsCrawler()
        print("✅ 爬虫初始化成功")
        print(f"   支持的类别数量: {len(crawler.category_urls)}")
        print(f"   输出目录: {crawler.output_dir}")
        return True
    except Exception as e:
        print(f"❌ 爬虫初始化失败: {e}")
        return False

def test_page_fetching():
    """测试页面获取功能"""
    print("\n🧪 测试页面获取功能...")
    
    try:
        crawler = CategoryNewsCrawler()
        
        # 测试获取主要页面
        test_url = "https://news.yahoo.co.jp"
        content = crawler.get_page_content(test_url, delay=1.0)
        
        if content:
            print("✅ 页面获取成功")
            print(f"   内容长度: {len(content)} 字符")
            return True
        else:
            print("❌ 页面获取失败")
            return False
            
    except Exception as e:
        print(f"❌ 页面获取测试失败: {e}")
        return False

def test_news_extraction():
    """测试新闻提取功能"""
    print("\n🧪 测试新闻提取功能...")
    
    try:
        crawler = CategoryNewsCrawler()
        
        # 测试从主要页面提取新闻
        test_url = "https://news.yahoo.co.jp"
        content = crawler.get_page_content(test_url, delay=1.0)
        
        if content:
            news_list = crawler.extract_news_from_category(content, "主要")
            
            if news_list:
                print("✅ 新闻提取成功")
                print(f"   提取的新闻数量: {len(news_list)}")
                print(f"   第一条新闻标题: {news_list[0]['title'][:50]}...")
                return True
            else:
                print("❌ 新闻提取失败")
                return False
        else:
            print("❌ 无法获取页面内容进行测试")
            return False
            
    except Exception as e:
        print(f"❌ 新闻提取测试失败: {e}")
        return False

def test_qa_generation():
    """测试查询和答案生成功能"""
    print("\n🧪 测试查询和答案生成功能...")
    
    try:
        crawler = CategoryNewsCrawler()
        
        # 测试简单的查询生成
        test_content = "これはテスト用のニュース記事の内容です。富士通が新しいAIシステムを開発しました。"
        test_category = "IT"
        
        query, answer = crawler.generate_simple_qa(test_content, test_category)
        
        if query and answer:
            print("✅ 查询和答案生成成功")
            print(f"   查询: {query}")
            print(f"   答案: {answer}")
            return True
        else:
            print("❌ 查询和答案生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 查询和答案生成测试失败: {e}")
        return False

def test_single_category_crawl():
    """测试单个类别的爬取"""
    print("\n🧪 测试单个类别爬取...")
    
    try:
        crawler = CategoryNewsCrawler()
        
        # 只测试主要类别
        test_category = "主要"
        test_url = crawler.category_urls[test_category]
        
        print(f"   测试类别: {test_category}")
        print(f"   测试URL: {test_url}")
        
        # 获取页面内容
        html_content = crawler.get_page_content(test_url, delay=2.0)
        if not html_content:
            print("❌ 无法获取页面内容")
            return False
        
        # 提取新闻
        news_list = crawler.extract_news_from_category(html_content, test_category)
        if not news_list:
            print("❌ 无法提取新闻")
            return False
        
        print(f"   找到新闻数量: {len(news_list)}")
        
        # 测试第一条新闻
        if len(news_list) > 0:
            first_news = news_list[0]
            print(f"   第一条新闻标题: {first_news['title']}")
            print(f"   第一条新闻URL: {first_news['url']}")
            
            # 获取新闻内容
            news_detail = crawler.get_news_content(first_news['url'])
            if news_detail:
                print("✅ 新闻内容获取成功")
                print(f"   内容长度: {len(news_detail['content'])} 字符")
                return True
            else:
                print("❌ 新闻内容获取失败")
                return False
        else:
            print("❌ 没有找到新闻")
            return False
            
    except Exception as e:
        print(f"❌ 单个类别爬取测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始运行分类新闻爬虫测试...\n")
    
    tests = [
        ("爬虫初始化", test_crawler_initialization),
        ("页面获取", test_page_fetching),
        ("新闻提取", test_news_extraction),
        ("查询生成", test_qa_generation),
        ("单个类别爬取", test_single_category_crawl)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ {test_name}测试出现异常: {e}\n")
    
    print("📊 测试结果汇总:")
    print(f"   通过: {passed}/{total}")
    print(f"   失败: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！爬虫功能正常")
        return True
    else:
        print("⚠️  部分测试失败，请检查问题")
        return False

def main():
    """主函数"""
    print("🧪 Yahoo!ニュース分类新闻爬虫测试套件")
    print("=" * 50)
    
    try:
        success = run_all_tests()
        
        if success:
            print("\n✅ 爬虫测试完成，可以开始正式爬取")
        else:
            print("\n❌ 爬虫测试失败，请修复问题后重试")
            
    except KeyboardInterrupt:
        print("\n⏹️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试套件运行出错: {e}")

if __name__ == "__main__":
    main()
