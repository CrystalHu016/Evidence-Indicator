#!/usr/bin/env python3
"""
Specific debug for crop harvest query showing complete process
"""

import os
import sys
from query_handler import setup_environment

def analyze_crop_harvest_query():
    """詳細分析：コンバインとは何作物を収穫できますか"""
    
    print("🌾 作物収穫クエリの詳細分析")
    print("="*70)
    
    # Setup environment
    original_cwd, parent_dir = setup_environment()
    
    try:
        from ultra_fast_rag_fixed import UltraFastRAG
        
        api_key = os.environ.get("OPENAI_API_KEY")
        rag = UltraFastRAG(api_key, "chroma")
        
        query = "コンバインとは何作物を収穫できますか"
        print(f"🎯 分析対象クエリ: {query}")
        print()
        
        # Step 1: Query analysis
        print("📝 Step 1: クエリ解析")
        print(f"  - 質問タイプ: 定義質問 + 具体的作物名")
        print(f"  - キーワード: ['コンバイン', '作物', '収穫']")
        print(f"  - 複雑度: 中程度（定義+詳細情報）")
        print()
        
        # Step 2: Document retrieval
        print("🔎 Step 2: 関連文書検索")
        search_results = rag.db.similarity_search_with_relevance_scores(query, k=5)
        
        for i, (doc, score) in enumerate(search_results):
            print(f"  📄 候補文書 {i+1} (関連度: {score:.3f})")
            content = doc.page_content
            print(f"     長さ: {len(content)}文字")
            print(f"     内容: {content[:80]}...")
            
            # Check for crop mentions
            crops = ['稲', '麦', '大豆', '小豆', '菜種', 'トウモロコシ', 'イネ', 'ムギ']
            found_crops = [crop for crop in crops if crop in content]
            if found_crops:
                print(f"     🌾 含まれる作物: {', '.join(found_crops)}")
            print()
        
        # Step 3: Best document selection
        print("🎯 Step 3: 最適文書選択プロセス")
        best_doc = rag._choose_best_doc(query, search_results)
        
        if isinstance(best_doc, tuple):
            source_text = best_doc[1] if len(best_doc) > 1 else ""
            print("  ➤ 合成回答が選択されました")
        else:
            source_text = best_doc.page_content
            print(f"  ➤ 単一文書が選択されました (長さ: {len(source_text)}文字)")
        
        print("📄 選択された文書の全文:")
        print(f'  "{source_text}"')
        print()
        
        # Step 4: Evidence generation
        print("💡 Step 4: 証拠生成プロセス")
        evidence = rag._generate_answer_fast(source_text, query)
        print(f"  生成方法: _generate_answer_fast()")
        print(f"  生成された証拠: \"{evidence}\"")
        print()
        
        # Step 5: Position calculation details
        print("📍 Step 5: 文字列位置特定の詳細アルゴリズム")
        print("  Algorithm: find() method for exact matching")
        
        start_pos = source_text.find(evidence)
        print(f"  source_text.find(evidence) = {start_pos}")
        
        if start_pos != -1:
            end_pos = start_pos + len(evidence)
            print(f"  end_pos = start_pos + len(evidence) = {start_pos} + {len(evidence)} = {end_pos}")
            
            print()
            print("🔤 文字位置の詳細マッピング:")
            print("  原文書での位置:")
            
            # Show character-by-character mapping for key positions
            for i in range(start_pos, min(start_pos + 20, end_pos)):
                char = source_text[i]
                print(f"    位置 {i:2d}: '{char}'")
            
            if end_pos - start_pos > 20:
                print(f"    ...  (省略: 位置{start_pos + 20}～{end_pos - 5})")
                for i in range(max(start_pos + 20, end_pos - 5), end_pos):
                    char = source_text[i]
                    print(f"    位置 {i:2d}: '{char}'")
        
        print()
        print("🎯 作物情報の抽出結果:")
        
        # Extract crop information specifically
        crops_mentioned = []
        crop_patterns = ['稲', '麦', '大豆', '小豆', '菜種', 'トウモロコシ']
        
        for crop in crop_patterns:
            if crop in evidence or crop in source_text:
                crops_mentioned.append(crop)
        
        print(f"  抽出された作物: {', '.join(crops_mentioned) if crops_mentioned else 'なし'}")
        
        # Final result structure
        result = {
            "query": query,
            "answer": evidence,
            "source_document": "農業機械データベース",
            "evidence_text": evidence,
            "start_char": start_pos,
            "end_char": end_pos if start_pos != -1 else 0,
            "character_length": len(evidence),
            "crops_mentioned": crops_mentioned,
            "source_text_full": source_text
        }
        
        print()
        print("📊 最終出力結果:")
        print(f"  query: {result['query']}")
        print(f"  answer: {result['answer']}")
        print(f"  evidence_text: {result['evidence_text']}")
        print(f"  start_char: {result['start_char']}")
        print(f"  end_char: {result['end_char']}")
        print(f"  character_length: {result['character_length']}")
        print(f"  crops_mentioned: {result['crops_mentioned']}")
        
        return result
        
    except Exception as e:
        import traceback
        print(f"❌ エラー: {str(e)}")
        print(f"トレースバック: {traceback.format_exc()}")
        return None
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    analyze_crop_harvest_query()