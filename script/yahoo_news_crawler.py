#!/usr/bin/env python3
"""
Yahoo!ニュース Universal Crawler with LLM Integration
Yahoo!ニュース統合クローラー - LLM質問応答生成機能付き

Features:
- Multi-category news extraction from Yahoo!ニュース
- LLM-based Q&A generation (OpenAI GPT-3.5-turbo)
- Diverse question patterns using varied interrogative words
- 100% LLM-generated responses (no template fallbacks)
- Structured JSON output for RAG systems
"""

import requests
from bs4 import BeautifulSoup
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Optional
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class YahooNewsCrawler:
    """Yahoo!ニュース統合クローラー"""
    
    def __init__(self):
        # OpenAI API key will be loaded from environment variable OPENAI_API_KEY
        # The new OpenAI client automatically reads from environment
        
        # Yahoo!ニュース各カテゴリページURL
        self.category_urls = {
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
        
        # HTTP request headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Output directory
        self.output_dir = '../data/yahoo_news_dataset'
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_queries_and_answers(self, content: str, category: str) -> Dict:
        """
        Generate two queries and answers using LLM based on content
        LLMを使用してコンテンツに基づいて2つのクエリと回答を生成
        """
        if not os.getenv('OPENAI_API_KEY'):
            print("      ❌ OpenAI API key not found. Cannot generate Q&A without LLM.")
            return {
                'query1': '',
                'answer1': '',
                'query2': '',
                'answer2': ''
            }
        
        try:
            print("      🤖 Using LLM to generate queries and answers...")
            
            # Generate two different types of queries
            # 2つの異なるタイプのクエリを生成
            prompt = f"""
以下のニュース記事の内容を分析して、2つの多様で具体的な質問と回答を生成してください。

記事内容:
{content[:1500]}...

カテゴリ: {category}

要求:
1. 記事の内容に基づいた具体的で実用的な質問を作成
2. 「なぜ」「どのように」「いつ」「どこで」「誰が」「何を」など異なる疑問詞を使用
3. 一般化可能で教育的価値のある質問を優先
4. 同じパターンの質問（要約、要点など）を避ける
5. 記事から学べる知識や仕組みに焦点を当てる

質問例のパターン:
- 「〜の理由は何ですか？」
- 「〜はどのような仕組みで動作しますか？」  
- 「〜する際に注意すべきポイントは何ですか？」
- 「〜の背景にはどのような要因がありますか？」
- 「〜に関する具体的な数値や詳細は？」

出力形式:
{{
    "query1": "具体的で泛化可能な質問1",
    "answer1": "質問1への詳細な回答",
    "query2": "異なる角度からの具体的な質問2", 
    "answer2": "質問2への詳細な回答"
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
            
            # Parse LLM response
            llm_response = response.choices[0].message.content.strip()
            print(f"      📝 LLM Response: {llm_response[:100]}...")
            
            try:
                # Try to parse JSON response
                qa_data = json.loads(llm_response)
                return {
                    'query1': qa_data.get('query1', ''),
                    'answer1': qa_data.get('answer1', ''),
                    'query2': qa_data.get('query2', ''),
                    'answer2': qa_data.get('answer2', '')
                }
            except json.JSONDecodeError as e:
                print(f"      ⚠️  LLM response parsing failed: {e}")
                print(f"      Raw response: {llm_response}")
                return {
                    'query1': '',
                    'answer1': '',
                    'query2': '',
                    'answer2': ''
                }
                
        except Exception as e:
            print(f"      ❌ LLM error: {e}")
            return {
                'query1': '',
                'answer1': '',
                'query2': '',
                'answer2': ''
            }

    def extract_news_content(self, news_url: str) -> str:
        """
        Extract full content from a news article URL
        ニュース記事URLから完全な内容を抽出
        """
        try:
            print(f"      Extracting content from: {news_url}")
            response = requests.get(news_url, headers=self.headers, timeout=30)
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

    def fetch_category_news(self, category_name: str, url: str, top_n: int = 2) -> List[Dict]:
        """
        Fetch news from a specific category page with content extraction and LLM analysis
        特定のカテゴリページからニュースを取得し、内容も抽出し、LLMで分析
        """
        try:
            print(f"    Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
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
                        content = self.extract_news_content(full_url)
                        
                        # Generate queries and answers using LLM
                        # LLMを使用してクエリと回答を生成
                        qa_data = self.generate_queries_and_answers(content, category_name)
                        
                        # Only create entries if we have valid Q&A from LLM
                        if qa_data['query1'] and qa_data['answer1']:
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
                                'crawler_version': 'unified_v1.0'
                            })
                        
                        if qa_data['query2'] and qa_data['answer2']:
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
                                'crawler_version': 'unified_v1.0'
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

    def save_to_json(self, data: List[Dict], filename: str = None) -> str:
        """
        Save news data to JSON file
        ニュースデータをJSONファイルに保存
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"yahoo_news_dataset_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return filepath

    def crawl_all_categories(self) -> List[Dict]:
        """
        Crawl all category pages and generate Q&A dataset
        全カテゴリページをクロールしてQ&Aデータセットを生成
        """
        print("🚀 Starting Yahoo!ニュース Universal Crawler...")
        print("=" * 80)
        
        # Check OpenAI API key
        if not os.getenv('OPENAI_API_KEY'):
            print("❌ OpenAI API key not found. Cannot generate Q&A without LLM.")
            print("   Set OPENAI_API_KEY environment variable to proceed.")
            print("   Exiting...")
            return []
        
        all_news = []
        
        for category, url in self.category_urls.items():
            print(f"🔄 Crawling category: {category}")
            try:
                news = self.fetch_category_news(category, url, top_n=2)
                all_news.extend(news)
                print(f"   ✅ Found {len(news)} entries with LLM-generated Q&A")
                
                # Add delay between category requests
                # カテゴリリクエスト間に遅延を追加
                time.sleep(3)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        return all_news

    def run(self, save_results: bool = True) -> Optional[str]:
        """
        Run the complete crawler pipeline
        完全なクローラーパイプラインを実行
        """
        # Crawl all categories
        all_news = self.crawl_all_categories()
        
        if not all_news:
            print("⚠️  No news data collected")
            return None
        
        print(f"\n📊 Total entries collected: {len(all_news)}")
        
        if save_results:
            # Save to JSON file
            filepath = self.save_to_json(all_news)
            print(f"💾 News data saved to: {filepath}")
            
            # Display statistics
            print(f"\n📈 Statistics:")
            print(f"   Total entries: {len(all_news)}")
            print(f"   Categories found: {len(set(n['category'] for n in all_news))}")
            
            # Show category breakdown
            categories = {}
            for news in all_news:
                cat = news['category']
                categories[cat] = categories.get(cat, 0) + 1
            
            for cat, count in sorted(categories.items()):
                print(f"   - {cat}: {count} entries")
            
            # Content statistics
            content_lengths = [len(news.get('content', '')) for news in all_news]
            if content_lengths:
                avg_length = sum(content_lengths) / len(content_lengths)
                print(f"   Average content length: {avg_length:.0f} characters")
            
            # Q&A statistics
            qa_count = sum(1 for news in all_news if news.get('query') and news.get('answer'))
            print(f"   Entries with Q&A generated: {qa_count}/{len(all_news)}")
            
            return filepath
        
        return all_news


def main():
    """
    Main function to run the crawler
    クローラーを実行するメイン関数
    """
    crawler = YahooNewsCrawler()
    result = crawler.run()
    
    if result:
        print(f"\n✅ Success! Dataset saved to: {result}")
    else:
        print("\n❌ Failed to generate dataset")


if __name__ == "__main__":
    main()