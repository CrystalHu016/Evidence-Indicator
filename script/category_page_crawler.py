#!/usr/bin/env python3
"""
Category Page Yahoo!ニュース Crawler with Content Extraction and LLM Analysis
各カテゴリページからニュースを取得し、内容も抽出し、LLMで分析するYahoo!ニュースクローラー
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

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Yahoo!ニュース各分类页面URL
category_urls = {
    '国内': 'https://news.yahoo.co.jp/categories/domestic',
    '国際': 'https://news.yahoo.co.jp/categories/world',
    '経済': 'https://news.yahoo.co.jp/categories/business',
    'エンタメ': 'https://news.yahoo.co.jp/categories/entertainment',
    'スポーツ': 'https://news.yahoo.co.jp/categories/sports',
    'IT': 'https://news.yahoo.co.jp/categories/it',
    '科学': 'https://news.yahoo.co.jp/categories/science',
    'ライフ': 'https://news.yahoo.co.jp/categories/life',
    '地域': 'https://news.yahoo.co.jp/categories/local',
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
    """
    Generate two queries and answers using LLM based on content
    LLMを使用してコンテンツに基づいて2つのクエリと回答を生成
    """
    try:
        if not openai.api_key:
            print("      ⚠️  OpenAI API key not found, using fallback method")
            return generate_fallback_qa(content, category)
        
        print("      🤖 Using LLM to generate queries and answers...")
        
        # Generate two different types of queries
        # 2つの異なるタイプのクエリを生成
        prompt = f"""
以下のニュース記事の内容を分析して、2つの異なるタイプの質問と回答を生成してください。

記事内容:
{content[:1000]}...

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
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "あなたはニュース記事を分析して適切な質問と回答を生成する専門家です。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        
        # Parse LLM response
        llm_response = response.choices[0].message.content.strip()
        
        try:
            # Try to parse JSON response
            qa_data = json.loads(llm_response)
            return {
                'query1': qa_data.get('query1', ''),
                'answer1': qa_data.get('answer1', ''),
                'query2': qa_data.get('query2', ''),
                'answer2': qa_data.get('answer2', '')
            }
        except json.JSONDecodeError:
            print("      ⚠️  LLM response parsing failed, using fallback")
            return generate_fallback_qa(content, category)
            
    except Exception as e:
        print(f"      ⚠️  LLM error: {e}, using fallback method")
        return generate_fallback_qa(content, category)

def generate_fallback_qa(content: str, category: str) -> Dict:
    """
    Fallback method to generate queries and answers without LLM
    LLMなしでクエリと回答を生成するフォールバック方法
    """
    print("      📝 Using fallback method to generate queries and answers...")
    
    # Simple keyword-based query generation
    # シンプルなキーワードベースのクエリ生成
    content_lower = content.lower()
    
    # Generate first query (detail-oriented)
    # 最初のクエリを生成（詳細指向）
    if any(word in content_lower for word in ['死亡', '死亡者', '死亡した']):
        query1 = f"この{category}ニュースで死亡した人物の詳細は？"
        answer1 = "記事の内容から死亡に関する詳細情報を確認してください。"
    elif any(word in content_lower for word in ['金額', '円', '万円', '億円']):
        query1 = f"この{category}ニュースで言及されている金額は？"
        answer1 = "記事内で言及されている具体的な金額を確認してください。"
    elif any(word in content_lower for word in ['日時', '時間', '日付']):
        query1 = f"この{category}ニュースで言及されている日時は？"
        answer1 = "記事内で言及されている具体的な日時を確認してください。"
    else:
        query1 = f"この{category}ニュースの具体的な事実は？"
        answer1 = "記事の内容から具体的な事実や詳細を確認してください。"
    
    # Generate second query (summary-oriented)
    # 2番目のクエリを生成（要約指向）
    query2 = f"この{category}ニュースの要点は？"
    answer2 = "記事の内容を要約すると、主要なポイントや結論を確認してください。"
    
    return {
        'query1': query1,
        'answer1': answer1,
        'query2': query2,
        'answer2': answer2
    }

def extract_news_content(news_url: str) -> str:
    """
    Extract full content from a news article URL
    ニュース記事URLから完全な内容を抽出
    """
    try:
        print(f"      Extracting content from: {news_url}")
        response = requests.get(news_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple content selectors for Yahoo!ニュース
        # Yahoo!ニュース用の複数のコンテンツセレクターを試す
        content_selectors = [
            'div.article_body',
            'div.article-body',
            'div[class*="article"]',
            'div[class*="content"]',
            'div[class*="body"]',
            'article',
            'div.news-content',
            'div.news-body'
        ]
        
        content_text = ""
        for selector in content_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                # Remove script and style elements
                # スクリプトとスタイル要素を削除
                for script in content_elem(["script", "style"]):
                    script.decompose()
                
                # Get text content
                # テキストコンテンツを取得
                content_text = content_elem.get_text(separator='\n', strip=True)
                if content_text and len(content_text) > 50:
                    print(f"      Found content using selector: {selector}")
                    break
        
        if not content_text:
            # Fallback: try to get any text content
            # フォールバック: テキストコンテンツを取得
            content_text = soup.get_text(separator='\n', strip=True)
            # Limit content length to avoid too long text
            # コンテンツの長さを制限して長すぎるテキストを避ける
            if len(content_text) > 1000:
                content_text = content_text[:1000] + "..."
        
        return content_text
        
    except Exception as e:
        print(f"      Error extracting content: {e}")
        return "内容の取得に失敗しました。"

def fetch_category_news(category_name: str, url: str, top_n: int = 2) -> List[Dict]:
    """
    Fetch news from a specific category page with content extraction and LLM analysis
    特定のカテゴリページからニュースを取得し、内容も抽出し、LLMで分析
    """
    try:
        print(f"    Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        
        # Look for news links with /articles/ in href
        # hrefに/articles/を含むニュースリンクを探す
        article_links = soup.find_all('a', href=True)
        news_links = [link for link in article_links if '/articles/' in link.get('href', '')]
        
        print(f"    Found {len(news_links)} article links")
        
        # Process first top_n news links
        # 最初のtop_n個のニュースリンクを処理
        for i, link in enumerate(news_links[:top_n]):
            try:
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                # Clean title
                # タイトルをクリーンアップ
                if title and len(title) > 5 and not title.startswith('http'):
                    # Build full URL
                    # 完全なURLを構築
                    if href.startswith('/'):
                        full_url = 'https://news.yahoo.co.jp' + href
                    else:
                        full_url = href
                    
                    # Extract full content
                    # 完全な内容を抽出
                    content = extract_news_content(full_url)
                    
                    # Generate queries and answers using LLM
                    # LLMを使用してクエリと回答を生成
                    qa_data = generate_queries_and_answers(content, category_name)
                    
                    # Create first entry with query1
                    # query1で最初のエントリを作成
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
                        'crawler_version': 'category_page_v3.1'
                    })
                    
                    # Create second entry with query2
                    # query2で2番目のエントリを作成
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
                        'crawler_version': 'category_page_v3.1'
                    })
                    
                    # Add delay between content extraction and LLM analysis requests
                    # コンテンツ抽出とLLM分析リクエスト間に遅延を追加
                    time.sleep(2)
                    
            except Exception as e:
                print(f"    Error processing link {i+1}: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"    Error fetching {category_name}: {e}")
        return []

def save_to_json(data: List[Dict], filename: str = None) -> str:
    """
    Save news data to JSON file
    ニュースデータをJSONファイルに保存
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yahoo_news_categories_separate_entries_{timestamp}.json"
    
    # 确保输出目录存在
    # 出力ディレクトリが存在することを確認
    output_dir = 'yahoo_news_dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    return filepath

def main():
    """
    Main function to crawl all category pages with content extraction and LLM analysis
    すべてのカテゴリページをクロールし、内容も抽出し、LLMで分析するメイン関数
    """
    print("🚀 Starting Category Page Yahoo!ニュース Crawler with Separate Q&A Entries...")
    print("=" * 80)
    
    # Check OpenAI API key
    if not openai.api_key:
        print("⚠️  OpenAI API key not found. Will use fallback method for Q&A generation.")
        print("   Set OPENAI_API_KEY environment variable for LLM analysis.")
    
    all_news = []
    
    for category, url in category_urls.items():
        print(f"🔄 Crawling category: {category}")
        try:
            news = fetch_category_news(category, url, top_n=2)
            all_news.extend(news)
            print(f"   ✅ Found {len(news)} entries with content and Q&A")
            
            # Add delay between category requests
            # カテゴリリクエスト間に遅延を追加
            time.sleep(3)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Total entries collected: {len(all_news)}")
    
    # 显示收集到的新闻
    # 収集されたニュースを表示
    if all_news:
        print("\n📰 Collected News Entries:")
        print("-" * 80)
        
        # Group by category
        # カテゴリ別にグループ化
        categories = {}
        for news in all_news:
            cat = news['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(news)
        
        # Display by category
        # カテゴリ別に表示
        for category, items in sorted(categories.items()):
            print(f"\n🏷️  {category} ({len(items)} entries):")
            for i, news in enumerate(items, 1):
                print(f"   {i}. Data ID: {news['data_id']}")
                print(f"      Title: {news['title']}")
                if news['link']:
                    print(f"      Link: {news['link']}")
                if news.get('content'):
                    content_preview = news['content'][:100] + "..." if len(news['content']) > 100 else news['content']
                    print(f"      Content: {content_preview}")
                if news.get('query'):
                    print(f"      Query: {news['query']}")
                    print(f"      Answer: {news['answer'][:100]}...")
        
        # 保存到JSON文件
        # JSONファイルに保存
        filename = save_to_json(all_news)
        print(f"\n💾 News data with separate Q&A entries saved to: {filename}")
        
        # 显示统计信息
        # 統計情報を表示
        print(f"\n📈 Statistics:")
        print(f"   Total entries: {len(all_news)}")
        print(f"   Categories found: {len(categories)}")
        for cat, items in sorted(categories.items()):
            print(f"   - {cat}: {len(items)} entries")
        
        # 显示内容统计
        # コンテンツ統計を表示
        content_lengths = [len(news.get('content', '')) for news in all_news]
        if content_lengths:
            avg_length = sum(content_lengths) / len(content_lengths)
            print(f"   Average content length: {avg_length:.0f} characters")
            print(f"   Total content size: {sum(content_lengths)} characters")
        
        # 显示Q&A统计
        # Q&A統計を表示
        qa_count = sum(1 for news in all_news if news.get('query') and news.get('answer'))
        print(f"   Entries with Q&A generated: {qa_count}/{len(all_news)}")
        
    else:
        print("⚠️  No news data to save")
        print("💡 Tip: The page structure might have changed significantly.")

if __name__ == "__main__":
    main()
