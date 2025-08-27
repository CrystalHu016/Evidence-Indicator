#!/usr/bin/env python3
"""
Enhanced SmartNews Business Crawler with LLM Integration
基于SmartNews Business网站知识的智能回答系统
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

class SmartNewsBusinessCrawler:
    """SmartNews Business网站爬虫与LLM集成系统"""
    
    def __init__(self, openai_api_key: str = None):
        self.base_url = 'https://business.smartnews.com'
        self.newsroom_url = 'https://business.smartnews.com/newsroom'
        self.blogs_url = 'https://business.smartnews.com/newsroom/blogs'
        self.company_url = 'https://business.smartnews.com/company'
        
        # OpenAI配置
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            logger.info("✅ OpenAI API key loaded")
        else:
            logger.warning("⚠️ OpenAI API key not found")
        
        # 创建输出目录
        self.output_dir = 'smartnews_dataset'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 请求头设置
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # 核心查询模板
        self.core_queries = [
            "What is SmartNews and what is their mission?",
            "What are SmartNews' company values and principles?",
            "How does SmartNews work with publishers and advertisers?",
            "What career opportunities are available at SmartNews?",
            "What is the SmartTake Newsletter and what does it offer?",
            "How does SmartNews ensure quality and trustworthy news?",
            "What makes SmartNews different from other news platforms?",
            "How does SmartNews contribute to society?",
            "What are the key features of SmartNews' business model?",
            "How does SmartNews balance editorial curation with algorithms?"
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
        """解析SmartNews主页内容"""
        soup = BeautifulSoup(html, 'html.parser')
        content = {}
        
        try:
            # 提取主要信息
            content['title'] = soup.find('h1').text.strip() if soup.find('h1') else "SmartNews Business"
            
            # 提取使命宣言
            mission_section = soup.find('h3', string=lambda text: 'Mission' in text if text else False)
            if mission_section:
                mission_text = mission_section.find_next_sibling()
                if mission_text:
                    content['mission'] = mission_text.text.strip()
            
            # 提取公司价值观
            values_section = soup.find('h2', string=lambda text: 'Values' in text if text else False)
            if values_section:
                values_list = values_section.find_next_siblings()
                content['values'] = [v.text.strip() for v in values_list if v.text.strip()]
            
            # 提取其他关键信息
            content['description'] = soup.find('p').text.strip() if soup.find('p') else ""
            
            logger.info(f"✅ Parsed main page content: {len(content)} sections")
            return content
            
        except Exception as e:
            logger.error(f"❌ Error parsing main page: {e}")
            return {}
    
    def parse_newsroom_content(self, html: str) -> List[Dict[str, str]]:
        """解析新闻室页面内容"""
        soup = BeautifulSoup(html, 'html.parser')
        articles = []
        
        try:
            # 查找文章链接
            article_links = soup.find_all('a', href=True)
            
            for link in article_links:
                href = link.get('href')
                if href and '/newsroom/' in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 10:  # 过滤掉太短的标题
                        articles.append({
                            'title': title,
                            'url': self.base_url + href if href.startswith('/') else href,
                            'type': 'newsroom'
                        })
            
            logger.info(f"✅ Found {len(articles)} newsroom articles")
            return articles[:20]  # 限制数量避免过多请求
            
        except Exception as e:
            logger.error(f"❌ Error parsing newsroom: {e}")
            return []
    
    def parse_blog_content(self, html: str) -> List[Dict[str, str]]:
        """解析博客页面内容"""
        soup = BeautifulSoup(html, 'html.parser')
        blogs = []
        
        try:
            # 查找博客链接
            blog_links = soup.find_all('a', href=True)
            
            for link in blog_links:
                href = link.get('href')
                if href and '/blogs/' in href:
                    title = link.get_text(strip=True)
                    if title and len(title) > 10:
                        blogs.append({
                            'title': title,
                            'url': self.base_url + href if href.startswith('/') else href,
                            'type': 'blog'
                        })
            
            logger.info(f"✅ Found {len(blogs)} blog posts")
            return blogs[:20]
            
        except Exception as e:
            logger.error(f"❌ Error parsing blogs: {e}")
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
            基于以下SmartNews Business网站的内容，回答用户问题。
            
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
                    {"role": "system", "content": "你是一个专业的商业分析师，擅长基于SmartNews Business网站内容回答问题。"},
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
            "source": "SmartNews Business",
            "crawler_version": "enhanced_v1.0"
        }
    
    def crawl_and_generate_dataset(self) -> List[Dict[str, any]]:
        """主要爬取和数据集生成流程"""
        logger.info("🚀 Starting SmartNews Business dataset generation...")
        
        dataset = []
        
        # 1. 爬取主页内容
        main_html = self.get_page_content(self.base_url)
        if main_html:
            main_content = self.parse_main_page_content(main_html)
            logger.info("✅ Main page content parsed")
        else:
            logger.error("❌ Failed to fetch main page")
            return dataset
        
        # 2. 爬取新闻室内容
        newsroom_html = self.get_page_content(self.newsroom_url)
        newsroom_articles = []
        if newsroom_html:
            newsroom_articles = self.parse_newsroom_content(newsroom_html)
        
        # 3. 爬取博客内容
        blogs_html = self.get_page_content(self.blogs_url)
        blog_posts = []
        if blogs_html:
            blog_posts = self.blogs_url
        
        # 4. 合并所有内容
        all_content = {
            'main_page': main_content,
            'newsroom': newsroom_articles,
            'blogs': blog_posts
        }
        
        # 5. 为每个核心查询生成回答
        for query in self.core_queries:
            logger.info(f"🔍 Processing query: {query[:50]}...")
            
            # 生成LLM回答
            answer = self.generate_llm_answer(query, str(all_content))
            
            if answer:
                # 创建数据集条目
                entry = self.create_dataset_entry(
                    query=query,
                    answer=answer,
                    urls=[self.base_url, self.newsroom_url, self.blogs_url],
                    content_summary=f"Content from main page, newsroom, and blogs"
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
            filename = f"smartnews_dataset_{timestamp}.json"
        
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
    print("🚀 SmartNews Business Enhanced Crawler with LLM Integration")
    print("=" * 60)
    
    # 检查环境变量
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("   The crawler will work but LLM answer generation will be skipped")
        print("   Please set OPENAI_API_KEY in your .env file")
    
    # 创建爬虫实例
    crawler = SmartNewsBusinessCrawler(api_key)
    
    # 运行完整流程
    result = crawler.run_full_pipeline()
    
    if result:
        print(f"\n🎉 Success! Dataset saved to: {result}")
        print("📁 Check the 'smartnews_dataset' folder for your generated dataset")
    else:
        print("\n❌ Failed to generate dataset. Check the logs above for details.")


if __name__ == "__main__":
    main()
