#!/usr/bin/env python3
"""
Category Page Yahoo!ニュース Crawler
各カテゴリページからニュースを取得するYahoo!ニュースクローラー
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
from typing import List, Dict

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

def fetch_category_news(category_name: str, url: str, top_n: int = 2) -> List[Dict]:
    """
    Fetch news from a specific category page
    特定のカテゴリページからニュースを取得
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
                    
                    results.append({
                        'category': category_name,
                        'title': title,
                        'link': full_url,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Yahoo!ニュース',
                        'crawler_version': 'category_page_v1.1'
                    })
                    
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
        filename = f"yahoo_news_categories_{timestamp}.json"
    
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
    Main function to crawl all category pages
    すべてのカテゴリページをクロールするメイン関数
    """
    print("🚀 Starting Category Page Yahoo!ニュース Crawler (IT & 科学 separated)...")
    print("=" * 70)
    
    all_news = []
    
    for category, url in category_urls.items():
        print(f"🔄 Crawling category: {category}")
        try:
            news = fetch_category_news(category, url, top_n=2)
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
        print("-" * 70)
        
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
            print(f"\n🏷️  {category} ({len(items)} items):")
            for i, news in enumerate(items, 1):
                print(f"   {i}. {news['title']}")
                if news['link']:
                    print(f"      Link: {news['link']}")
        
        # 保存到JSON文件
        # JSONファイルに保存
        filename = save_to_json(all_news)
        print(f"\n💾 News data saved to: {filename}")
        
        # 显示统计信息
        # 統計情報を表示
        print(f"\n📈 Statistics:")
        print(f"   Total news items: {len(all_news)}")
        print(f"   Categories found: {len(categories)}")
        for cat, items in sorted(categories.items()):
            print(f"   - {cat}: {len(items)} items")
        
    else:
        print("⚠️  No news data to save")
        print("💡 Tip: The page structure might have changed significantly.")

if __name__ == "__main__":
    main()
