#!/usr/bin/env python3
"""
Configuration file for Yahoo!ニュース Enhanced Crawler
Yahoo!ニュース增强爬虫配置文件
"""

import os
from typing import List, Dict

class CrawlerConfig:
    """爬虫配置类"""
    
    # 基础URL配置 - Yahoo!ニュース
    BASE_URLS = {
        'main': 'https://news.yahoo.co.jp',
        'topics': 'https://news.yahoo.co.jp/topics',
        'domestic': 'https://news.yahoo.co.jp/domestic',
        'international': 'https://news.yahoo.co.jp/international',
        'economy': 'https://news.yahoo.co.jp/economy',
        'entertainment': 'https://news.yahoo.co.jp/entertainment',
        'sports': 'https://news.yahoo.co.jp/sports',
        'it': 'https://news.yahoo.co.jp/it',
        'science': 'https://news.yahoo.co.jp/science',
        'life': 'https://news.yahoo.co.jp/life',
        'ranking': 'https://news.yahoo.co.jp/ranking'
    }
    
    # 核心查询模板 - 基于Yahoo!ニュース内容
    CORE_QUERIES = [
        # 新闻分类和内容
        "What are the main news categories on Yahoo!ニュース?",
        "What are the current trending topics on Yahoo!ニュース?",
        "How does Yahoo!ニュース organize its news content?",
        "What types of news does Yahoo!ニュース cover?",
        
        # 特色功能
        "What is Yahoo!ニュース Live and how does it work?",
        "What are the ranking features on Yahoo!ニュース?",
        "How does Yahoo!ニュース recommend content to users?",
        "What are the comment and opinion features?",
        
        # 内容质量
        "How does Yahoo!ニュース ensure news quality?",
        "What are the sources of news on Yahoo!ニュース?",
        "How does Yahoo!ニュース handle breaking news?",
        "What makes Yahoo!ニュース different from other news platforms?",
        
        # 用户体验
        "What are the main features of Yahoo!ニュース?",
        "How does Yahoo!ニュース organize news by category?",
        "What are the trending topics and rankings?",
        "How does Yahoo!ニュース present news to users?"
    ]
    
    # 中文查询版本
    CORE_QUERIES_ZH = [
        "Yahoo!ニュース的主要新闻分类有哪些？",
        "Yahoo!ニュース上当前的热门话题是什么？",
        "Yahoo!ニュース如何组织其新闻内容？",
        "Yahoo!ニュース涵盖哪些类型的新闻？",
        "Yahoo!ニュース Live是什么，它是如何工作的？",
        "Yahoo!ニュース的排名功能有哪些？",
        "Yahoo!ニュース如何向用户推荐内容？",
        "评论和意见功能有哪些？",
        "Yahoo!ニュース如何确保新闻质量？",
        "Yahoo!ニュース的新闻来源是什么？",
        "Yahoo!ニュース如何处理突发新闻？",
        "Yahoo!ニュース与其他新闻平台有什么不同？",
        "Yahoo!ニュース的主要功能有哪些？",
        "Yahoo!ニュース如何按类别组织新闻？",
        "热门话题和排名是什么？",
        "Yahoo!ニュース如何向用户展示新闻？"
    ]
    
    # 日语查询版本
    CORE_QUERIES_JA = [
        "Yahoo!ニュースの主要なニュースカテゴリは何ですか？",
        "Yahoo!ニュースで現在話題になっているトピックは何ですか？",
        "Yahoo!ニュースはどのようにニュースコンテンツを整理していますか？",
        "Yahoo!ニュースはどのような種類のニュースをカバーしていますか？",
        "Yahoo!ニュースライブとは何で、どのように機能しますか？",
        "Yahoo!ニュースのランキング機能は何ですか？",
        "Yahoo!ニュースはどのようにユーザーにコンテンツを推薦していますか？",
        "コメントや意見の機能は何ですか？",
        "Yahoo!ニュースはどのようにニュースの品質を確保していますか？",
        "Yahoo!ニュースのニュースソースは何ですか？",
        "Yahoo!ニュースはどのように速報を処理していますか？",
        "Yahoo!ニュースを他のニュースプラットフォームと区別するものは何ですか？",
        "Yahoo!ニュースの主要な機能は何ですか？",
        "Yahoo!ニュースはどのようにカテゴリ別にニュースを整理していますか？",
        "話題のトピックとランキングは何ですか？",
        "Yahoo!ニュースはどのようにユーザーにニュースを提示していますか？",
        
        # 基于具体文章内容的问题
        "Yahoo!ニュースで三菱商事に関する現在の経済ニュースは何ですか？",
        "Yahoo!ニュースはビジネス・経済記事をどのように提示していますか？",
        "Yahoo!ニュースのニュース記事に対するコメント機能は何ですか？",
        "Yahoo!ニュースは記事の更新と修正をどのように処理していますか？",
        "Yahoo!ニュースの関連記事推薦機能は何ですか？",
        "Yahoo!ニュースは記事のメタデータ（公開時間、ソースなど）をどのように表示していますか？",
        "Yahoo!ニュースの異なるニュースカテゴリのランキングシステムは何ですか？",
        "Yahoo!ニュースはニュースソースと出版社別にコンテンツをどのように整理していますか？"
    ]
    
    # 请求配置
    REQUEST_CONFIG = {
        'timeout': 30,
        'delay_between_requests': 2.0,
        'max_retries': 3,
        'user_agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 内容解析配置
    PARSING_CONFIG = {
        'max_articles_per_section': 20,
        'min_title_length': 5,
        'max_content_length': 3000,  # LLM输入长度限制
        'content_tags': ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span', 'a']
    }
    
    # LLM配置
    LLM_CONFIG = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 500,
        'temperature': 0.3,
        'system_prompt': "你是一个专业的新闻分析师，擅长基于Yahoo!ニュース网站内容回答问题。请提供准确、有根据的回答。"
    }
    
    # 输出配置
    OUTPUT_CONFIG = {
        'output_dir': 'yahoo_news_dataset',
        'file_format': 'json',
        'encoding': 'utf-8',
        'include_timestamp': True,
        'include_metadata': True
    }
    
    # 数据集结构模板
    DATASET_TEMPLATE = {
        "core_query": "",
        "answer": "",
        "original_urls": [],
        "content_summary": "",
        "timestamp": "",
        "source": "Yahoo!ニュース",
        "crawler_version": "enhanced_v1.0",
        "metadata": {
            "word_count": 0,
            "content_sections": [],
            "publish_date": None,
            "article_type": "",
            "category": ""
        }
    }
    
    @classmethod
    def get_all_queries(cls, language: str = 'en') -> List[str]:
        """获取指定语言的所有查询"""
        if language.lower() == 'ja':
            return cls.CORE_QUERIES_JA
        return cls.CORE_QUERIES
    
    @classmethod
    def get_urls_for_section(cls, section: str) -> List[str]:
        """获取指定部分的URL列表"""
        urls = []
        if section in cls.BASE_URLS:
            urls.append(cls.BASE_URLS[section])
        
        # 添加相关子页面
        if section == 'topics':
            urls.append(cls.BASE_URLS['ranking'])
        elif section == 'domestic':
            urls.append(cls.BASE_URLS['international'])
        
        return urls
    
    @classmethod
    def validate_config(cls) -> Dict[str, bool]:
        """验证配置有效性"""
        validation_results = {
            'urls_valid': all(cls.BASE_URLS.values()),
            'queries_valid': len(cls.CORE_QUERIES) > 0,
            'output_config_valid': cls.OUTPUT_CONFIG['output_dir'] is not None,
            'llm_config_valid': cls.LLM_CONFIG['model'] is not None
        }
        
        return validation_results


# 环境变量配置
ENV_VARS = {
    'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    'YAHOO_CRAWLER_DELAY': os.getenv('YAHOO_CRAWLER_DELAY', '2.0'),
    'YAHOO_OUTPUT_DIR': os.getenv('YAHOO_OUTPUT_DIR', 'yahoo_news_dataset'),
    'YAHOO_MAX_ARTICLES': os.getenv('YAHOO_MAX_ARTICLES', '20')
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'yahoo_crawler.log',
    'max_size': '10MB',
    'backup_count': 5
}

if __name__ == "__main__":
    # 测试配置
    config = CrawlerConfig()
    print("🔧 Yahoo!ニュース Crawler Configuration Test")
    print("=" * 50)
    
    # 验证配置
    validation = config.validate_config()
    for key, value in validation.items():
        status = "✅" if value else "❌"
        print(f"{status} {key}: {value}")
    
    # 显示配置信息
    print(f"\n📊 Configuration Summary:")
    print(f"   Base URLs: {len(config.BASE_URLS)}")
    print(f"   Core Queries (EN): {len(config.CORE_QUERIES)}")
    print(f"   Core Queries (ZH): {len(config.CORE_QUERIES_ZH)}")
    print(f"   Output Directory: {config.OUTPUT_CONFIG['output_dir']}")
    print(f"   LLM Model: {config.LLM_CONFIG['model']}")
    
    # 检查环境变量
    print(f"\n🌍 Environment Variables:")
    for key, value in ENV_VARS.items():
        status = "✅" if value else "⚠️"
        print(f"{status} {key}: {value or 'Not Set'}")
