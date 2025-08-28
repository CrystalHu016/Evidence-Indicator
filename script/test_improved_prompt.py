#!/usr/bin/env python3
"""
Test improved LLM prompt for more diverse questions
改良されたLLMプロンプトのテスト
"""

import sys
sys.path.append('.')

from category_page_crawler import fetch_category_news
import time

# Test with 3 different categories
test_categories = {
    '経済': 'https://news.yahoo.co.jp/categories/business',
    'スポーツ': 'https://news.yahoo.co.jp/categories/sports', 
    'IT': 'https://news.yahoo.co.jp/categories/it',
}

print("🚀 Testing Improved LLM Prompt for Diverse Questions")
print("=" * 60)

all_queries = []

for category, url in test_categories.items():
    print(f"\n🔄 Testing category: {category}")
    try:
        news = fetch_category_news(category, url, top_n=1)
        print(f"   ✅ Found {len(news)} entries")
        
        for n in news:
            if n.get('query') and n.get('answer'):
                print(f"\n   📝 Query: {n['query']}")
                print(f"   📝 Answer: {n['answer'][:80]}...")
                all_queries.append(n['query'])
        
        time.sleep(3)  # Rate limiting
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n📊 Generated {len(all_queries)} diverse questions")
print("\n🎯 Question Diversity Analysis:")
question_patterns = {}
for query in all_queries:
    if "なぜ" in query:
        question_patterns["なぜ (Why)"] = question_patterns.get("なぜ (Why)", 0) + 1
    elif "どのように" in query or "どのような" in query:
        question_patterns["どのように/どのような (How/What kind)"] = question_patterns.get("どのように/どのような (How/What kind)", 0) + 1
    elif "いつ" in query:
        question_patterns["いつ (When)"] = question_patterns.get("いつ (When)", 0) + 1
    elif "どこで" in query:
        question_patterns["どこで (Where)"] = question_patterns.get("どこで (Where)", 0) + 1
    elif "誰が" in query:
        question_patterns["誰が (Who)"] = question_patterns.get("誰が (Who)", 0) + 1
    else:
        question_patterns["その他 (Other)"] = question_patterns.get("その他 (Other)", 0) + 1

for pattern, count in question_patterns.items():
    print(f"   - {pattern}: {count} questions")

print("\n✅ Diversity test completed!")