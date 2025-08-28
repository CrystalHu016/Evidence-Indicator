#!/usr/bin/env python3
"""
Test LLM-only crawler with limited data for verification
LLM専用クローラーのテスト版
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
from typing import List, Dict
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test categories (limited)
category_urls = {
    '経済': 'https://news.yahoo.co.jp/categories/business',
    'スポーツ': 'https://news.yahoo.co.jp/categories/sports',
}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def generate_queries_and_answers(content: str, category: str) -> Dict:
    """Generate Q&A using LLM only"""
    if not os.getenv('OPENAI_API_KEY'):
        print("      ❌ OpenAI API key not found")
        return {'query1': '', 'answer1': '', 'query2': '', 'answer2': ''}
    
    try:
        print("      🤖 Using LLM to generate queries and answers...")
        
        prompt = f"""
以下のニュース記事の内容を分析して、2つの異なるタイプの質問と回答を生成してください。

記事内容:
{content[:1500]}...

カテゴリ: {category}

要求:
1. 詳細性の質問: 記事の具体的な事実や詳細について質問
2. 要約性の質問: 記事の要点や全体像について質問

両方の質問は記事の内容から答えを見つけることができるものにしてください。
回答は記事の内容に基づいて具体的に答えてください。

出力形式:
{{
    "query1": "詳細性の質問",
    "answer1": "詳細性の質問への回答",
    "query2": "要約性の質問", 
    "answer2": "要約性の質問への回答"
}}
"""
        
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたはニュース記事を分析して適切な質問と回答を生成する専門家です。記事の内容に基づいて具体的で有用な質問と回答を日本語で生成してください。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        
        llm_response = response.choices[0].message.content.strip()
        print(f"      📝 LLM Response: {llm_response[:100]}...")
        
        try:
            qa_data = json.loads(llm_response)
            return {
                'query1': qa_data.get('query1', ''),
                'answer1': qa_data.get('answer1', ''),
                'query2': qa_data.get('query2', ''),
                'answer2': qa_data.get('answer2', '')
            }
        except json.JSONDecodeError as e:
            print(f"      ⚠️  LLM response parsing failed: {e}")
            return {'query1': '', 'answer1': '', 'query2': '', 'answer2': ''}
            
    except Exception as e:
        print(f"      ❌ LLM error: {e}")
        return {'query1': '', 'answer1': '', 'query2': '', 'answer2': ''}

def extract_news_content(news_url: str) -> str:
    """Extract content from news URL"""
    try:
        print(f"      Extracting content from: {news_url}")
        response = requests.get(news_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        content_selectors = [
            'div.article_body',
            'div.article-body',
            'div[class*="article"]',
            'div[class*="content"]',
        ]
        
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                for script in content_elem(["script", "style"]):
                    script.decompose()
                
                content_text = content_elem.get_text(separator='\\n', strip=True)
                if content_text and len(content_text) > 50:
                    print(f"      Found content using selector: {selector}")
                    return content_text
        
        return "内容の取得に失敗しました。"
        
    except Exception as e:
        print(f"      Error extracting content: {e}")
        return "内容の取得に失敗しました。"

def fetch_category_news(category_name: str, url: str, top_n: int = 1) -> List[Dict]:
    """Fetch news from a category"""
    try:
        print(f"    Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        article_links = soup.find_all('a', href=True)
        news_links = [link for link in article_links if '/articles/' in link.get('href', '')]
        
        print(f"    Found {len(news_links)} article links")
        
        for i, link in enumerate(news_links[:top_n]):
            try:
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                if title and len(title) > 5 and not title.startswith('http'):
                    if href.startswith('/'):
                        full_url = 'https://news.yahoo.co.jp' + href
                    else:
                        full_url = href
                    
                    content = extract_news_content(full_url)
                    qa_data = generate_queries_and_answers(content, category_name)
                    
                    # Only create entries if we have valid Q&A
                    if qa_data['query1'] and qa_data['answer1']:
                        results.append({
                            'data_id': f"{category_name}_{i+1}_1",
                            'category': category_name,
                            'title': title,
                            'link': full_url,
                            'content': content,
                            'query': qa_data['query1'],
                            'answer': qa_data['answer1'],
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Yahoo!ニュース',
                            'crawler_version': 'llm_only_test_v1.0'
                        })
                    
                    if qa_data['query2'] and qa_data['answer2']:
                        results.append({
                            'data_id': f"{category_name}_{i+1}_2",
                            'category': category_name,
                            'title': title,
                            'link': full_url,
                            'content': content,
                            'query': qa_data['query2'],
                            'answer': qa_data['answer2'],
                            'timestamp': datetime.now().isoformat(),
                            'source': 'Yahoo!ニュース',
                            'crawler_version': 'llm_only_test_v1.0'
                        })
                    
                    time.sleep(2)  # Rate limiting
                    
            except Exception as e:
                print(f"    Error processing link {i+1}: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"    Error fetching {category_name}: {e}")
        return []

def main():
    """Main function"""
    print("🚀 Starting LLM-Only Yahoo!ニュース Crawler Test...")
    print("=" * 60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OpenAI API key not found. Exiting...")
        return
    
    all_news = []
    
    for category, url in category_urls.items():
        print(f"🔄 Crawling category: {category}")
        try:
            news = fetch_category_news(category, url, top_n=1)
            all_news.extend(news)
            print(f"   ✅ Found {len(news)} entries with LLM-generated Q&A")
            time.sleep(3)  # Delay between categories
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\\n📊 Total entries collected: {len(all_news)}")
    
    if all_news:
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"llm_only_test_results_{timestamp}.json"
        os.makedirs('yahoo_news_dataset', exist_ok=True)
        filepath = os.path.join('yahoo_news_dataset', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(all_news, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Results saved to: {filepath}")
        
        # Show sample results
        print("\\n📰 Sample Results:")
        for news in all_news[:2]:  # Show first 2 entries
            print(f"\\nCategory: {news['category']}")
            print(f"Query: {news['query']}")
            print(f"Answer: {news['answer'][:100]}...")
        
        print(f"\\n✅ Success! Generated {len(all_news)} LLM-based Q&A entries")
    else:
        print("⚠️  No valid entries generated")

if __name__ == "__main__":
    main()