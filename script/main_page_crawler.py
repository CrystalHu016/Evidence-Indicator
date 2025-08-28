#!/usr/bin/env python3
"""
Main Page Yahoo!ニュース Crawler with URL-based category detection
URLベースのカテゴリ検出を使用したメインページYahoo!ニュースクローラー
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
from typing import List, Dict

# Yahoo!ニュース主页
main_url = 'https://news.yahoo.co.jp'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def get_category_from_url(url: str) -> str:
    """
    Extract category from URL path
    URLパスからカテゴリを抽出
    """
    if '/categories/' in url:
        cat_part = url.split('/categories/')[-1].split('/')[0]
        category_map = {
            'domestic': '国内',
            'world': '国際',
            'business': '経済',
            'entertainment': 'エンタメ',
            'sports': 'スポーツ',
            'it': 'IT・科学',
            'life': 'ライフ',
            'local': '地域'
        }
        return category_map.get(cat_part, '主要')
    elif '/articles/' in url:
        return '主要'
    else:
        return '主要'

def fetch_main_page_news() -> List[Dict]:
    """
    Fetch news from the main page
    メインページからニュースを取得
    """
    try:
        print(f"🔄 Fetching main page: {main_url}")
        response = requests.get(main_url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        
        # Strategy 1: Look for news links with /articles/ in href
        # 戦略1: hrefに/articles/を含むニュースリンクを探す
        print("    Strategy 1: Looking for article links...")
        article_links = soup.find_all('a', href=True)
        news_links = [link for link in article_links if '/articles/' in link.get('href', '')]
        
        print(f"    Found {len(news_links)} article links")
        
        # Process first 20 news links
        # 最初の20個のニュースリンクを処理
        for i, link in enumerate(news_links[:20]):
            try:
                href = link.get('href', '')
                title = link.get_text(strip=True)
                
                # Clean title
                # タイトルをクリーンアップ
                if title and len(title) > 5 and not title.startswith('http'):
                    # Build full URL
                    # 完全なURLを構築
                    if href.startswith('/'):
                        full_url = main_url + href
                    else:
                        full_url = href
                    
                    # Get category from URL
                    # URLからカテゴリを取得
                    category = get_category_from_url(full_url)
                    
                    results.append({
                        'category': category,
                        'title': title,
                        'link': full_url,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'Yahoo!ニュース',
                        'crawler_version': 'main_page_v1.1'
                    })
                    
            except Exception as e:
                print(f"    Error processing link {i+1}: {e}")
                continue
        
        # Strategy 2: Look for trending topics
        # 戦略2: トレンドトピックを探す
        print("    Strategy 2: Looking for trending topics...")
        trending_selectors = [
            'div[class*="trending"]',
            'div[class*="topic"]',
            'div[class*="popular"]',
            'ul[class*="topic"]',
            'div[class*="ranking"]'
        ]
        
        for selector in trending_selectors:
            trending_items = soup.select(selector)
            if trending_items:
                print(f"    Found trending items using: {selector}")
                for item in trending_items[:5]:
                    try:
                        title_elem = item.find('a') or item
                        title = title_elem.get_text(strip=True) if title_elem else ''
                        
                        if title and len(title) > 5:
                            results.append({
                                'category': 'トレンド',
                                'title': title,
                                'link': '',
                                'timestamp': datetime.now().isoformat(),
                                'source': 'Yahoo!ニュース',
                                'crawler_version': 'main_page_v1.1'
                            })
                    except Exception as e:
                        continue
                break
        
        return results
        
    except Exception as e:
        print(f"    Error fetching main page: {e}")
        return []

def save_to_json(data: List[Dict], filename: str = None) -> str:
    """
    Save news data to JSON file
    ニュースデータをJSONファイルに保存
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yahoo_news_main_page_{timestamp}.json"
    
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
    Main function to crawl main page news
    メインページニュースをクロールするメイン関数
    """
    print("🚀 Starting Main Page Yahoo!ニュース Crawler (URL-based categories)...")
    print("=" * 70)
    
    # Fetch news from main page
    # メインページからニュースを取得
    news_items = fetch_main_page_news()
    
    print(f"\n📊 Total news collected: {len(news_items)}")
    
    # 显示收集到的新闻
    # 収集されたニュースを表示
    if news_items:
        print("\n📰 Collected News:")
        print("-" * 70)
        
        # Group by category
        # カテゴリ別にグループ化
        categories = {}
        for news in news_items:
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
        filename = save_to_json(news_items)
        print(f"\n💾 News data saved to: {filename}")
        
        # 显示统计信息
        # 統計情報を表示
        print(f"\n📈 Statistics:")
        print(f"   Total news items: {len(news_items)}")
        print(f"   Categories found: {len(categories)}")
        for cat, items in sorted(categories.items()):
            print(f"   - {cat}: {len(items)} items")
        
    else:
        print("⚠️  No news data to save")
        print("💡 Tip: The page structure might have changed significantly.")

if __name__ == "__main__":
    main()
