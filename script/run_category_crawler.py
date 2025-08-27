#!/usr/bin/env python3
"""
运行Yahoo!ニュース分类新闻爬虫的脚本
"""

import os
import sys
from category_news_crawler import CategoryNewsCrawler

def main():
    """主函数"""
    print("🚀 Yahoo!ニュース分类新闻爬虫启动中...")
    
    # 检查环境变量
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("⚠️  警告: 未找到OPENAI_API_KEY环境变量")
        print("   爬虫将使用简单的查询生成方法")
        print("   如需使用LLM生成查询，请设置OPENAI_API_KEY")
        print()
    
    try:
        # 创建爬虫实例
        crawler = CategoryNewsCrawler(openai_api_key)
        
        # 运行爬虫
        crawler.run()
        
        print("✅ 爬虫运行完成！")
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断了爬虫运行")
    except Exception as e:
        print(f"❌ 爬虫运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
