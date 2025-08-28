#!/usr/bin/env python3
"""
Updated Yahoo!ニュース Crawler with correct selectors
正しいセレクターを使用した更新されたYahoo!ニュースクローラー
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
from typing import List, Dict

# Yahoo!ニュース主要分类URL
category_urls = {
    '国内': 'https://news.yahoo.co.jp/categories/domestic',
    '国際': 'https://news.yahoo.co.jp/categories/world',
    '経済': 'https://news.yahoo.co.jp/categories/business',
    'エンタメ': 'https://news.yahoo.co.jp/categories/entertainment',
    'スポーツ': 'https://news.yahoo.co.jp/categories/sports',
    'IT・科学': 'https://news.yahoo.co.jp/categories/it',
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

def fetch_top_news(category_name: str, url: str, top_n: int = 2) -> List[Dict]:
    """
    Fetch top news from a specific category with multiple selector strategies
    複数のセレクター戦略を使用して特定のカテゴリからトップニュースを取得
    """
    try:
        print(f"    Fetching: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        
        # Strategy 1: Try multiple possible selectors
        # 戦略1: 複数の可能なセレクターを試す
        selectors_to_try = [
            'div.NewsList_item',
            'li.NewsList_item',
            'div[class*="NewsList"]',
            'div[class*="news"]',
            'div[class*="item"]',
            'article',
            'div.news-item',
            'div.news-list-item'
        ]
        
        news_items = []
        for selector in selectors_to_try:
            items = soup.select(selector)
            if items:
                news_items = items[:top_n]
                print(f"    Found {len(items)} items using selector: {selector}")
                break
        
        if not news_items:
            # Strategy 2: Look for any links that might be news
            # 戦略2: ニュースである可能性のあるリンクを探す
            all_links = soup.find_all('a', href=True)
            news_links = [link for link in all_links if '/articles/' in link.get('href', '')]
            news_items = news_links[:top_n]
            print(f"    Found {len(news_items)} news links using href search")
        
        if not news_items:
            # Strategy 3: Look for any text that might be news titles
            # 戦略3: ニュースタイトルである可能性のあるテキストを探す
            potential_titles = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            news_items = potential_titles[:top_n]
            print(f"    Found {len(news_items)} potential titles using heading tags")
        
        # Process found items
        # 見つかった項目を処理
        for i, item in enumerate(news_items):
            try:
                # Extract title
                # タイトルを抽出
                if item.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    title = item.get_text(strip=True)
                    link = ''
                else:
                    title_tag = item.find('a') or item
                    title = title_tag.get_text(strip=True) if title_tag else f'ニュース{i+1}'
                    link = title_tag.get('href', '') if hasattr(title_tag, 'get') else ''
                
                # Clean title
                # タイトルをクリーンアップ
                if title and len(title) > 5:
                    # Extract timestamp if available
                    # 利用可能な場合はタイムスタンプを抽出
                    timestamp = datetime.now().isoformat()
                    
                    # Look for time elements
                    # 時間要素を探す
                    time_elem = item.find('time') or item.find('span', class_='time') or item.find('span', class_='date')
                    if time_elem:
                        timestamp = time_elem.get('datetime', time_elem.get_text(strip=True))
                    
                    # Build full link
                    # 完全なリンクを構築
                    if link and not link.startswith('http'):
                        link = 'https://news.yahoo.co.jp' + link
                    
                    results.append({
                        'category': category_name,
                        'title': title,
                        'timestamp': timestamp,
                        'link': link,
                        'source_url': url
                    })
                    
            except Exception as e:
                print(f"    Error processing item {i+1}: {e}")
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
        filename = f"yahoo_news_updated_{timestamp}.json"
    
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
    Main function to crawl all categories
    すべてのカテゴリをクロールするメイン関数
    """
    print("🚀 Starting Updated Yahoo!ニュース Crawler...")
    print("=" * 60)
    
    all_news = []
    
    for category, url in category_urls.items():
        print(f"🔄 Crawling category: {category}")
        try:
            news = fetch_top_news(category, url)
            all_news.extend(news)
            print(f"   ✅ Found {len(news)} news items")
            
            # Add delay between requests
            # リクエスト間に遅延を追加
            time.sleep(2)
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Total news collected: {len(all_news)}")
    
    # 显示收集到的新闻
    # 収集されたニュースを表示
    if all_news:
        print("\n📰 Collected News:")
        print("-" * 60)
        for news in all_news:
            print(f"[{news['category']}] {news['timestamp']} - {news['title']}")
            if news['link']:
                print(f"    Link: {news['link']}")
            print()
        
        # 保存到JSON文件
        # JSONファイルに保存
        filename = save_to_json(all_news)
        print(f"💾 News data saved to: {filename}")
        
        # 显示统计信息
        # 統計情報を表示
        print(f"\n📈 Statistics:")
        print(f"   Total categories processed: {len(category_urls)}")
        print(f"   Total news items: {len(all_news)}")
        print(f"   Average per category: {len(all_news)/len(category_urls):.1f}")
        
    else:
        print("⚠️  No news data to save")
        print("💡 Tip: The page structure might have changed. Check the selectors.")

if __name__ == "__main__":
    main()
