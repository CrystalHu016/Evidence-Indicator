#!/usr/bin/env python3
"""
New Yahoo!ニュース Crawler
新しいYahoo!ニュースクローラー
"""

import requests
from bs4 import BeautifulSoup
import json
import os
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
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36'
}

def fetch_top_news(category_name: str, url: str, top_n: int = 2) -> List[Dict]:
    """
    Fetch top news from a specific category
    特定のカテゴリからトップニュースを取得
    """
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # 根据页面结构选取新闻条目（Selector需根据真实页面调整）
        # ページ構造に基づいてニュース項目を選択（セレクターは実際のページに応じて調整が必要）
        news_items = soup.select('div.NewsList_item')[:top_n]

        results = []
        for item in news_items:
            title_tag = item.select_one('a.NewsList_link')
            title = title_tag.get_text(strip=True) if title_tag else '無題'

            time_tag = item.select_one('time')
            timestamp = time_tag['datetime'] if time_tag and time_tag.has_attr('datetime') else '無時間'

            # 获取新闻链接
            # ニュースリンクを取得
            link = title_tag['href'] if title_tag and title_tag.has_attr('href') else ''
            if link and not link.startswith('http'):
                link = 'https://news.yahoo.co.jp' + link

            results.append({
                'category': category_name,
                'title': title,
                'timestamp': timestamp,
                'link': link
            })

        return results
        
    except Exception as e:
        print(f"分類「{category_name}」の取得でエラーが発生しました: {e}")
        return []

def save_to_json(data: List[Dict], filename: str = None) -> str:
    """
    Save news data to JSON file
    ニュースデータをJSONファイルに保存
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"yahoo_news_{timestamp}.json"
    
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
    print("🚀 Starting Yahoo!ニュース Crawler...")
    print("=" * 50)
    
    all_news = []
    
    for category, url in category_urls.items():
        print(f"🔄 Crawling category: {category}")
        try:
            news = fetch_top_news(category, url)
            all_news.extend(news)
            print(f"   ✅ Found {len(news)} news items")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📊 Total news collected: {len(all_news)}")
    
    # 显示收集到的新闻
    # 収集されたニュースを表示
    print("\n📰 Collected News:")
    print("-" * 50)
    for news in all_news:
        print(f"[{news['category']}] {news['timestamp']} - {news['title']}")
        if news['link']:
            print(f"    Link: {news['link']}")
        print()
    
    # 保存到JSON文件
    # JSONファイルに保存
    if all_news:
        filename = save_to_json(all_news)
        print(f"💾 News data saved to: {filename}")
    else:
        print("⚠️  No news data to save")

if __name__ == "__main__":
    main()
