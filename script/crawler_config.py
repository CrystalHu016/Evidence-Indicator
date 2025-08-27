#!/usr/bin/env python3
"""
Configuration file for SmartNews Business Enhanced Crawler
SmartNews Business增强爬虫配置文件
"""

import os
from typing import List, Dict

class CrawlerConfig:
    """爬虫配置类"""
    
    # 基础URL配置
    BASE_URLS = {
        'main': 'https://business.smartnews.com',
        'newsroom': 'https://business.smartnews.com/newsroom',
        'blogs': 'https://business.smartnews.com/newsroom/blogs',
        'company': 'https://business.smartnews.com/company',
        'careers': 'https://business.smartnews.com/careers',
        'publishers': 'https://business.smartnews.com/publishers'
    }
    
    # 核心查询模板 - 基于SmartNews Business网站内容
    CORE_QUERIES = [
        # 公司基本信息
        "What is SmartNews and what is their mission?",
        "What are SmartNews' company values and principles?",
        "How does SmartNews contribute to society?",
        
        # 商业模式
        "How does SmartNews work with publishers and advertisers?",
        "What are the key features of SmartNews' business model?",
        "How does SmartNews balance editorial curation with algorithms?",
        
        # 产品特色
        "What is the SmartTake Newsletter and what does it offer?",
        "How does SmartNews ensure quality and trustworthy news?",
        "What makes SmartNews different from other news platforms?",
        
        # 职业发展
        "What career opportunities are available at SmartNews?",
        "What is it like to work at SmartNews?",
        "How does SmartNews support employee growth?",
        
        # 技术特色
        "How does SmartNews use AI and algorithms?",
        "What technology does SmartNews use for news aggregation?",
        "How does SmartNews handle content moderation?",
        
        # 合作伙伴
        "How can publishers partner with SmartNews?",
        "What are the benefits of advertising with SmartNews?",
        "How does SmartNews select its publishing partners?"
    ]
    
    # 中文查询版本
    CORE_QUERIES_ZH = [
        "SmartNews是什么公司，他们的使命是什么？",
        "SmartNews的公司价值观和原则是什么？",
        "SmartNews如何为社会做出贡献？",
        "SmartNews如何与出版商和广告商合作？",
        "SmartNews商业模式的关键特征是什么？",
        "SmartNews如何平衡编辑策划和算法？",
        "SmartTake Newsletter是什么，它提供什么服务？",
        "SmartNews如何确保新闻质量和可信度？",
        "SmartNews与其他新闻平台有什么不同？",
        "SmartNews提供哪些职业机会？",
        "在SmartNews工作是什么样的体验？",
        "SmartNews如何支持员工成长？",
        "SmartNews如何使用AI和算法？",
        "SmartNews使用什么技术进行新闻聚合？",
        "SmartNews如何处理内容审核？",
        "出版商如何与SmartNews合作？",
        "在SmartNews做广告有什么好处？",
        "SmartNews如何选择其出版合作伙伴？"
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
        'min_title_length': 10,
        'max_content_length': 3000,  # LLM输入长度限制
        'content_tags': ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'span']
    }
    
    # LLM配置
    LLM_CONFIG = {
        'model': 'gpt-3.5-turbo',
        'max_tokens': 500,
        'temperature': 0.3,
        'system_prompt': "你是一个专业的商业分析师，擅长基于SmartNews Business网站内容回答问题。请提供准确、有根据的回答。"
    }
    
    # 输出配置
    OUTPUT_CONFIG = {
        'output_dir': 'smartnews_dataset',
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
        "source": "SmartNews Business",
        "crawler_version": "enhanced_v1.0",
        "metadata": {
            "word_count": 0,
            "content_sections": [],
            "publish_date": None,
            "article_type": ""
        }
    }
    
    @classmethod
    def get_all_queries(cls, language: str = 'en') -> List[str]:
        """获取指定语言的所有查询"""
        if language.lower() == 'zh':
            return cls.CORE_QUERIES_ZH
        return cls.CORE_QUERIES
    
    @classmethod
    def get_urls_for_section(cls, section: str) -> List[str]:
        """获取指定部分的URL列表"""
        urls = []
        if section in cls.BASE_URLS:
            urls.append(cls.BASE_URLS[section])
        
        # 添加相关子页面
        if section == 'newsroom':
            urls.append(cls.BASE_URLS['blogs'])
        elif section == 'company':
            urls.append(cls.BASE_URLS['careers'])
        
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
    'SMARTNEWS_CRAWLER_DELAY': os.getenv('SMARTNEWS_CRAWLER_DELAY', '2.0'),
    'SMARTNEWS_OUTPUT_DIR': os.getenv('SMARTNEWS_OUTPUT_DIR', 'smartnews_dataset'),
    'SMARTNEWS_MAX_ARTICLES': os.getenv('SMARTNEWS_MAX_ARTICLES', '20')
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'crawler.log',
    'max_size': '10MB',
    'backup_count': 5
}

if __name__ == "__main__":
    # 测试配置
    config = CrawlerConfig()
    print("🔧 Crawler Configuration Test")
    print("=" * 40)
    
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
