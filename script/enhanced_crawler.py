#!/usr/bin/env python3
"""
Enhanced Yahoo!ニュース Crawler with LLM Integration
基于Yahoo!ニュース网站知识的智能回答系统
"""

import requests
from bs4 import BeautifulSoup
import os
import json
import time
import openai
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YahooNewsCrawler:
    """Yahoo!ニュース网站爬虫与LLM集成系统"""
    
    def __init__(self, openai_api_key: str = None):
        self.base_url = 'https://news.yahoo.co.jp'
        self.topics_url = 'https://news.yahoo.co.jp/topics'
        self.ranking_url = 'https://news.yahoo.co.jp/ranking'
        # 修复分类页面URL - 使用正确的格式
        self.domestic_url = 'https://news.yahoo.co.jp/categories/domestic'
        self.international_url = 'https://news.yahoo.co.jp/categories/international'
        self.economy_url = 'https://news.yahoo.co.jp/categories/economy'
        self.entertainment_url = 'https://news.yahoo.co.jp/categories/entertainment'
        self.sports_url = 'https://news.yahoo.co.jp/categories/sports'
        self.it_url = 'https://news.yahoo.co.jp/categories/it'
        self.science_url = 'https://news.yahoo.co.jp/categories/science'
        self.life_url = 'https://news.yahoo.co.jp/categories/life'
        
        # OpenAI配置
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("✅ OpenAI API key loaded")
        else:
            logger.warning("⚠️ OpenAI API key not found")
        
        # 创建输出目录
        self.output_dir = 'yahoo_news_dataset'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 请求头设置
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # 核心查询模板
        self.core_queries = [
            "What are the main news categories on Yahoo!ニュース?",
            "What are the current trending topics on Yahoo!ニュース?",
            "How does Yahoo!ニュース organize its news content?",
            "What types of news does Yahoo!ニュース cover?",
            "What is Yahoo!ニュース Live and how does it work?",
            "What are the ranking features on Yahoo!ニュース?",
            "How does Yahoo!ニュース recommend content to users?",
            "What are the comment and opinion features?",
            "How does Yahoo!ニュース ensure news quality?",
            "What are the sources of news on Yahoo!ニュース?",
            "How does Yahoo!ニュース handle breaking news?",
            "What makes Yahoo!ニュース different from other news platforms?",
            "What are the main features of Yahoo!ニュース?",
            "How does Yahoo!ニュース organize news by category?",
            "What are the trending topics and rankings?",
            "How does Yahoo!ニュース present news to users?"
        ]
    
    def get_page_content(self, url: str, delay: float = 2.0) -> Optional[str]:
        """获取网页内容，包含延迟和错误处理"""
        try:
            logger.info(f"🔄 Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # 添加延迟避免被反爬虫
            time.sleep(delay)
            return response.text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Error fetching {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def parse_main_page_content(self, html: str) -> Dict[str, str]:
        """解析Yahoo!ニュース主页内容"""
        soup = BeautifulSoup(html, 'html.parser')
        content = {}
        
        try:
            # 提取主要信息
            title_tag = soup.find('h1')
            content['title'] = title_tag.text.strip() if title_tag else "Yahoo!ニュース"
            
            # 提取新闻分类
            category_links = soup.find_all('a', href=True)
            categories = []
            for link in category_links:
                href = link.get('href')
                if href and any(cat in href for cat in ['/domestic', '/international', '/economy', '/sports', '/entertainment']):
                    categories.append(link.text.strip())
            content['categories'] = categories[:10]  # 限制数量
            
            # 提取热门话题
            topics = soup.find_all('a', href=True)
            trending_topics = []
            for topic in topics:
                if topic.text.strip() and len(topic.text.strip()) > 5:
                    trending_topics.append(topic.text.strip())
            content['trending_topics'] = trending_topics[:15]  # 限制数量
            
            # 提取其他关键信息
            content['description'] = "Yahoo!ニュース - 日本主要新闻门户网站"
            
            logger.info(f"✅ Parsed main page content: {len(content)} sections")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error parsing main page: {e}")
            return {}
    
    def parse_topics_content(self, html: str) -> List[Dict[str, str]]:
        """解析话题页面内容"""
        soup = BeautifulSoup(html, 'html.parser')
        topics = []
        
        try:
            # 查找话题链接
            topic_links = soup.find_all('a', href=True)
            
            for link in topic_links:
                href = link.get('href')
                if href and '/topics/' in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 5:  # 过滤掉太短的标题
                        topics.append({
                            'title': title,
                            'url': self.base_url + href if href.startswith('/') else href,
                            'type': 'topic'
                        })
            
            logger.info(f"✅ Found {len(topics)} topics")
            return topics[:20]  # 限制数量避免过多请求
            
        except Exception as e:
            logger.error(f"❌ Error parsing topics: {e}")
            return []
    
    def parse_ranking_content(self, html: str) -> List[Dict[str, str]]:
        """解析排名页面内容"""
        soup = BeautifulSoup(html, 'html.parser')
        rankings = []
        
        try:
            # 查找排名内容
            ranking_items = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            for item in ranking_items:
                text = item.get_text(strip=True)
                if text and any(keyword in text for keyword in ['ランキング', '1位', '2位', '3位', 'アクセス']):
                    rankings.append({
                        'title': text,
                        'type': 'ranking',
                        'content': text
                    })
            
            logger.info(f"✅ Found {len(rankings)} ranking items")
            return rankings[:15]
            
        except Exception as e:
            logger.error(f"❌ Error parsing rankings: {e}")
            return []
    
    def parse_category_content(self, html: str, category: str) -> List[Dict[str, str]]:
        """解析特定分类页面内容"""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        
        try:
            # 查找文章链接
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href')
                if href and '/articles/' in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 5:
                        articles.append({
                            'title': title,
                            'url': self.base_url + href if href.startswith('/') else href,
                            'type': 'article',
                            'category': category
                        })
            
            logger.info(f"✅ Found {len(articles)} articles in {category}")
            return articles[:15]
            
        except Exception as e:
            logger.error(f"❌ Error parsing {category}: {e}")
            return []
    
    def extract_article_content(self, url: str) -> Optional[Dict[str, str]]:
        """提取具体文章内容"""
        html = self.get_page_content(url, delay=1.5)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        try:
            # 提取标题
            title = soup.find('h1')
            title_text = title.text.strip() if title else "No Title"
            
            # 提取正文内容
            content_tags = soup.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6'])
            content_text = '\n'.join([tag.text.strip() for tag in content_tags if tag.text.strip()])
            
            # 提取发布时间
            time_tag = soup.find('time')
            publish_time = time_tag.get('datetime') if time_tag else None
            
            return {
                'title': title_text,
                'content': content_text,
                'url': url,
                'publish_time': publish_time,
                'word_count': len(content_text.split())
            }
            
        except Exception as e:
            logger.error(f"❌ Error extracting article content from {url}: {e}")
            return None
    
    def generate_llm_answer(self, query: str, context_content: str) -> Optional[str]:
        """使用LLM基于网站内容生成回答"""
        if not self.openai_api_key:
            logger.warning("⚠️ No OpenAI API key, skipping LLM generation")
            return None
        
        try:
            prompt = f"""
            基于以下Yahoo!ニュース网站的内容，回答用户问题。
            
            网站内容：
            {context_content[:3000]}  # 限制长度避免超出token限制
            
            用户问题：{query}
            
            要求：
            1. 只使用上述网站内容作为信息源
            2. 如果网站内容不足以回答问题，请说明
            3. 提供准确、有根据的回答
            4. 回答要简洁明了，突出重点
            5. 使用中文回答
            
            回答：
            """
            
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的新闻分析师，擅长基于Yahoo!ニュース网站内容回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info(f"✅ Generated LLM answer for: {query[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"❌ Error generating LLM answer: {e}")
            return None
    
    def create_dataset_entry(self, query: str, answer: str, urls: List[str], 
                           content_summary: str) -> Dict[str, any]:
        """创建数据集条目"""
        return {
            "core_query": query,
            "answer": answer,
            "original_urls": urls,
            "content_summary": content_summary,
            "timestamp": datetime.now().isoformat(),
            "source": "Yahoo!ニュース",
            "crawler_version": "enhanced_v1.0"
        }
    
    def crawl_and_generate_dataset(self) -> List[Dict[str, any]]:
        """主要爬取和数据集生成流程"""
        logger.info("🚀 Starting Yahoo!ニュース dataset generation...")
        
        dataset = []
        
        # 1. 爬取主页内容
        main_html = self.get_page_content(self.base_url)
        if main_html:
            main_content = self.parse_main_page_content(main_html)
            logger.info("✅ Main page content parsed")
        else:
            logger.error("❌ Failed to fetch main page")
            return dataset
        
        # 2. 爬取话题页面内容
        topics_html = self.get_page_content(self.topics_url)
        topics_content = []
        if topics_html:
            topics_content = self.parse_topics_content(topics_html)
        
        # 3. 爬取排名页面内容
        ranking_html = self.get_page_content(self.ranking_url)
        ranking_content = []
        if ranking_html:
            ranking_content = self.parse_ranking_content(ranking_html)
        
        # 4. 爬取分类页面内容
        category_urls = [
            (self.domestic_url, 'domestic'),
            (self.international_url, 'international'),
            (self.economy_url, 'economy'),
            (self.sports_url, 'sports'),
            (self.entertainment_url, 'entertainment')
        ]
        
        category_content = {}
        for url, category in category_urls:
            html = self.get_page_content(url)
            if html:
                category_content[category] = self.parse_category_content(html, category)
        
        # 5. 合并所有内容
        all_content = {
            'main_page': main_content,
            'topics': topics_content,
            'rankings': ranking_content,
            'categories': category_content
        }
        
        # 6. 为每个核心查询生成回答
        for query in self.core_queries:
            logger.info(f"🔍 Processing query: {query[:50]}...")
            
            # 生成LLM回答
            answer = self.generate_llm_answer(query, str(all_content))
            
            if answer:
                # 创建数据集条目
                entry = self.create_dataset_entry(
                    query=query,
                    answer=answer,
                    urls=[self.base_url, self.topics_url, self.ranking_url],
                    content_summary=f"Content from main page, topics, rankings, and category pages"
                )
                
                dataset.append(entry)
                logger.info(f"✅ Created dataset entry for: {query[:30]}...")
            else:
                logger.warning(f"⚠️ Failed to generate answer for: {query[:30]}...")
        
        logger.info(f"🎉 Dataset generation completed! Created {len(dataset)} entries")
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, any]], filename: str = None):
        """保存数据集到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yahoo_news_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Dataset saved to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
            return None
    
    def run_full_pipeline(self):
        """运行完整的爬取和数据集生成流程"""
        try:
            # 1. 爬取内容并生成数据集
            dataset = self.crawl_and_generate_dataset()
            
            if dataset:
                # 2. 保存数据集
                filepath = self.save_dataset(dataset)
                
                # 3. 输出统计信息
                logger.info("📊 Dataset Statistics:")
                logger.info(f"   Total entries: {len(dataset)}")
                logger.info(f"   Core queries: {len(self.core_queries)}")
                logger.info(f"   Output file: {filepath}")
                
                return filepath
            else:
                logger.error("❌ No dataset generated")
                return None
                
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return None


def main():
    """主函数"""
    print("🚀 Yahoo!ニュース Enhanced Crawler with LLM Integration")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   The crawler will work but LLM answer generation will be skipped")
        print("   Please set OPENAI_API_KEY in your .env file")
    
    # 创建爬虫实例
    crawler = YahooNewsCrawler(api_key)
    
    # 运行完整流程
    result = crawler.run_full_pipeline()
    
    if result:
        print(f"\n🎉 Success! Dataset saved to: {result}")
        print("📁 Check the 'yahoo_news_dataset' folder for your generated dataset")
    else:
        print("\n❌ Failed to generate dataset. Check the logs above for details.")


if __name__ == "__main__":
    main()
