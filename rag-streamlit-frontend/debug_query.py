#!/usr/bin/env python3
"""
Debug version to show detailed evidence position calculation
"""

import os
import sys
import re
from query_handler import setup_environment, safe_print

def debug_evidence_position_calculation():
    """详细调试证拠情報の文字列範囲特定过程"""
    
    print("🔍 証拠情報の文字列範囲特定デバッグ")
    print("="*60)
    
    # Setup environment
    original_cwd, parent_dir = setup_environment()
    
    try:
        # Import RAG system
        from ultra_fast_rag_fixed import UltraFastRAG
        
        # Initialize RAG
        api_key = os.environ.get("OPENAI_API_KEY")
        rag = UltraFastRAG(api_key, "chroma")
        
        # Test query
        query = "コンバインとは何作物を収穫できますか"
        print(f"📝 テストクエリ: {query}")
        print()
        
        # Step 1: Get search results
        search_results = rag.db.similarity_search_with_relevance_scores(query, k=3)
        print("🔎 Step 1: ベクトル検索結果")
        for i, (doc, score) in enumerate(search_results):
            content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            print(f"  候補{i+1} (スコア:{score:.3f}): {content_preview}")
        print()
        
        # Step 2: Choose best document
        best_doc = rag._choose_best_doc(query, search_results)
        print("🎯 Step 2: 最適文書選択")
        if isinstance(best_doc, tuple):
            print("  合成回答が選択されました")
            source_text = best_doc[1] if len(best_doc) > 1 else ""
        else:
            source_text = best_doc.page_content
            print(f"  選択された文書長: {len(source_text)}文字")
        
        print(f"📄 原文書内容:")
        print(f"  \"{source_text}\"")
        print()
        
        # Step 3: Generate evidence
        evidence = rag._generate_answer_fast(source_text, query)
        print("💡 Step 3: 証拠生成")
        print(f"  生成された証拠: \"{evidence}\"")
        print()
        
        # Step 4: Find exact positions - DEBUG VERSION
        print("📍 Step 4: 文字列位置特定の詳細過程")
        
        # Method 1: Direct match
        print("  Method 1: 直接マッチング")
        start_pos = source_text.find(evidence)
        print(f"    検索パターン: \"{evidence[:30]}...\"")
        print(f"    find()結果: {start_pos}")
        
        if start_pos != -1:
            end_pos = start_pos + len(evidence)
            print(f"    ✅ 直接マッチング成功!")
            print(f"    開始位置: {start_pos}")
            print(f"    終了位置: {end_pos}")
            print(f"    証拠長: {len(evidence)}文字")
        else:
            print("    ❌ 直接マッチング失敗")
            
            # Method 2: Partial matching
            print("  Method 2: 部分マッチング")
            if len(evidence) >= 50:
                found = False
                for i in range(0, len(evidence) - 30, 10):
                    part = evidence[i:i+30]
                    pos = source_text.find(part)
                    print(f"    部分検索 [{i}:{i+30}]: \"{part[:20]}...\" -> 位置{pos}")
                    if pos != -1:
                        print(f"    ✅ 部分マッチング成功! 位置: {pos}")
                        # Expand range (simplified)
                        start_pos = max(0, pos - i)
                        end_pos = min(len(source_text), start_pos + len(evidence))
                        found = True
                        break
                
                if not found:
                    print("    ❌ 部分マッチング失敗")
                    start_pos, end_pos = 0, min(len(evidence), len(source_text))
        
        # Step 5: Character-by-character analysis
        print()
        print("🔤 Step 5: 文字単位解析")
        if start_pos != -1:
            print(f"  原文書[{start_pos}:{end_pos}]:")
            extracted = source_text[start_pos:end_pos]
            print(f"    \"{extracted}\"")
            print()
            
            # Character position mapping
            print("  文字位置マッピング:")
            for i in range(max(0, start_pos-5), min(len(source_text), end_pos+5)):
                char = source_text[i]
                marker = "→" if start_pos <= i < end_pos else " "
                print(f"    位置{i:3d}: '{char}' {marker}")
                if i - start_pos >= 10 and start_pos <= i < end_pos:  # Show first 10 chars of evidence
                    print("      ...")
                    break
        
        print()
        print("🎯 最終結果:")
        print(f"  証拠テキスト: \"{evidence}\"")
        print(f"  開始位置: {start_pos}")
        print(f"  終了位置: {end_pos}")
        print(f"  文字列長: {end_pos - start_pos}")
        
        return {
            "answer": evidence,
            "source_document": "debug_source",
            "evidence_text": evidence,
            "start_char": start_pos,
            "end_char": end_pos,
            "source_text": source_text
        }
        
    except Exception as e:
        import traceback
        print(f"❌ エラー: {str(e)}")
        print(f"トレースバック: {traceback.format_exc()}")
        return None
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    debug_evidence_position_calculation()