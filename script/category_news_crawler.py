#!/usr/bin/env python3
"""
Yahoo!ニュース Category News Crawler with LLM Integration
爬取每个类别下的最新新闻并生成日语查询和答案
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
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CategoryNewsCrawler:
    """Yahoo!ニュース分类新闻爬虫与LLM集成系统"""
    
    def __init__(self, openai_api_key: str = None):
        self.base_url = 'https://news.yahoo.co.jp'
        
        # 各个分类的URL
        self.category_urls = {
            '主要': 'https://news.yahoo.co.jp',
            '国内': 'https://news.yahoo.co.jp/categories/domestic',
            '国際': 'https://news.yahoo.co.jp/categories/international',
            '経済': 'https://news.yahoo.co.jp/categories/economy',
            'エンタメ': 'https://news.yahoo.co.jp/categories/entertainment',
            'スポーツ': 'https://news.yahoo.co.jp/categories/sports',
            'IT': 'https://news.yahoo.co.jp/categories/it',
            '科学': 'https://news.yahoo.co.jp/categories/science',
            'ライフ': 'https://news.yahoo.co.jp/categories/life',
            '地域': 'https://news.yahoo.co.jp/categories/local'
        }
        
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
    
    def extract_news_from_category(self, html_content: str, category: str) -> List[Dict]:
        """从分类页面提取新闻信息"""
        news_list = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        try:
            # 尝试多种选择器来找到新闻链接
            news_selectors = [
                'a[href*="/articles/"]',
                'a[href*="/topics/"]',
                '.topicsList a',
                '.newsFeed a',
                '.newsList a'
            ]
            
            news_links = []
            for selector in news_selectors:
                links = soup.select(selector)
                if links:
                    news_links = links
                    break
            
            # 提取新闻信息
            for link in news_links[:10]:  # 获取前10个新闻
                href = link.get('href')
                if href and '/articles/' in href:
                    # 构建完整URL
                    if href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        full_url = href
                    
                    # 提取标题
                    title = link.get_text(strip=True)
                    if title and len(title) > 10:  # 过滤掉太短的标题
                        news_list.append({
                            'title': title,
                            'url': full_url,
                            'category': category
                        })
            
            logger.info(f"📰 Found {len(news_list)} news articles in {category}")
            
        except Exception as e:
            logger.error(f"❌ Error extracting news from {category}: {e}")
        
        return news_list
    
    def get_news_content(self, news_url: str) -> Optional[Dict]:
        """获取单个新闻的详细内容"""
        try:
            html_content = self.get_page_content(news_url, delay=1.0)
            if not html_content:
                return None
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 提取新闻内容
            content = ""
            
            # 尝试多种选择器
            content_selectors = [
                '.article_body',
                '.articleBody',
                '.newsContent',
                '.content',
                'article p',
                '.article p'
            ]
            
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content = ' '.join([elem.get_text(strip=True) for elem in elements])
                    break
            
            # 如果没有找到内容，尝试提取标题
            if not content:
                title_elem = soup.find('h1') or soup.find('title')
                if title_elem:
                    content = title_elem.get_text(strip=True)
            
            if content:
                return {
                    'content': content,
                    'url': news_url
                }
            
        except Exception as e:
            logger.error(f"❌ Error getting news content from {news_url}: {e}")
        
        return None
    
    def generate_query_and_answer(self, news_content: str, category: str) -> Tuple[str, str]:
        """使用LLM生成查询和答案"""
        try:
            if not self.openai_api_key:
                # 如果没有API key，生成简单的查询和答案
                return self.generate_simple_qa(news_content, category)
            
            # 使用OpenAI API生成查询和答案
            prompt = f"""
以下のニュース記事の内容に基づいて、日本語で質問と回答を生成してください。

記事内容: {news_content[:1000]}
カテゴリ: {category}

要求:
1. 質問は記事の内容に直接答えられるものにしてください
2. 質問と回答は両方とも日本語で書いてください
3. 質問は自然で自然な日本語表現を使用してください
4. 回答は記事の内容に基づいて正確で簡潔にしてください

出力形式:
質問: [質問内容]
回答: [回答内容]
"""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "あなたはニュース記事から質問と回答を生成する専門家です。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            result = response.choices[0].message.content.strip()
            
            # 解析结果
            lines = result.split('\n')
            query = ""
            answer = ""
            
            for line in lines:
                if line.startswith('質問:'):
                    query = line.replace('質問:', '').strip()
                elif line.startswith('回答:'):
                    answer = line.replace('回答:', '').strip()
            
            if query and answer:
                return query, answer
            else:
                return self.generate_simple_qa(news_content, category)
                
        except Exception as e:
            logger.error(f"❌ Error generating query and answer: {e}")
            return self.generate_simple_qa(news_content, category)
    
    def generate_simple_qa(self, news_content: str, category: str) -> Tuple[str, str]:
        """生成简单的查询和答案（备用方法）"""
        # 从内容中提取关键信息
        content_preview = news_content[:100] + "..." if len(news_content) > 100 else news_content
        
        # 根据类别生成不同类型的查询
        query_templates = {
            '主要': f"このニュース記事の主な内容は何ですか？",
            '国内': f"この国内ニュースの詳細はどのようなものですか？",
            '国際': f"この国際ニュースの要点は何ですか？",
            '経済': f"この経済ニュースの影響はどのようなものですか？",
            'エンタメ': f"このエンタメニュースの内容は何ですか？",
            'スポーツ': f"このスポーツニュースの結果はどうでしたか？",
            'IT': f"このITニュースの技術的な内容は何ですか？",
            '科学': f"この科学ニュースの発見や研究内容は何ですか？",
            'ライフ': f"このライフニュースの生活への影響は何ですか？",
            '地域': f"この地域ニュースの詳細はどのようなものですか？"
        }
        
        query = query_templates.get(category, "このニュース記事の内容は何ですか？")
        answer = f"この記事は{category}カテゴリのニュースで、{content_preview}について報じています。"
        
        return query, answer
    
    def crawl_category_news(self) -> List[Dict]:
        """爬取所有分类的最新新闻"""
        all_news_data = []
        
        for category, url in self.category_urls.items():
            logger.info(f"🔄 Crawling category: {category}")
            
            try:
                # 获取分类页面内容
                html_content = self.get_page_content(url, delay=2.0)
                if not html_content:
                    logger.warning(f"⚠️ Failed to get content for {category}")
                    continue
                
                # 提取新闻列表
                news_list = self.extract_news_from_category(html_content, category)
                
                # 获取前2个新闻的详细内容
                for i, news in enumerate(news_list[:2]):
                    logger.info(f"📰 Processing news {i+1} in {category}: {news['title'][:50]}...")
                    
                    # 获取新闻详细内容
                    news_detail = self.get_news_content(news['url'])
                    if news_detail:
                        # 生成查询和答案
                        query, answer = self.generate_query_and_answer(
                            news_detail['content'], 
                            category
                        )
                        
                        # 创建数据条目
                        news_data = {
                            "core_query": query,
                            "answer": answer,
                            "original_urls": [news['url']],
                            "content_summary": news['title'],
                            "timestamp": datetime.now().isoformat(),
                            "source": "Yahoo!ニュース",
                            "category": category,
                            "crawler_version": "category_news_v1.0"
                        }
                        
                        all_news_data.append(news_data)
                        logger.info(f"✅ Generated Q&A for {category} news {i+1}")
                    
                    # 添加延迟
                    time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"❌ Error crawling category {category}: {e}")
                continue
        
        return all_news_data
    
    def save_dataset(self, news_data: List[Dict], filename: str = None):
        """保存数据集到文件"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"category_news_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Dataset saved to: {filepath}")
            logger.info(f"📊 Total entries: {len(news_data)}")
            
        except Exception as e:
            logger.error(f"❌ Error saving dataset: {e}")
    
    def run(self):
        """运行爬虫"""
        logger.info("🚀 Starting Category News Crawler...")
        
        try:
            # 爬取所有分类的新闻
            news_data = self.crawl_category_news()
            
            if news_data:
                # 保存数据集
                self.save_dataset(news_data)
                logger.info("✅ Crawling completed successfully!")
            else:
                logger.warning("⚠️ No news data collected")
                
        except Exception as e:
            logger.error(f"❌ Crawler failed: {e}")

def main():
    """主函数"""
    # 创建爬虫实例
    crawler = CategoryNewsCrawler()
    
    # 运行爬虫
    crawler.run()

if __name__ == "__main__":
    main()
