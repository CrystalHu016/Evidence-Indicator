#!/usr/bin/env python3
"""
Deep trace of evidence text generation process
详细追踪证据テキスト生成过程
"""

import os
import sys
from query_handler import setup_environment

def trace_evidence_generation():
    """详细追踪Step 2: 証拠テキスト生成的完整过程"""
    
    print("🔬 Step 2: 証拠テキスト生成の詳細追踪")
    print("="*70)
    
    original_cwd, parent_dir = setup_environment()
    
    try:
        from ultra_fast_rag_fixed import UltraFastRAG
        
        api_key = os.environ.get("OPENAI_API_KEY")
        rag = UltraFastRAG(api_key, "chroma")
        
        query = "コンバインとは何作物を収穫できますか"
        print(f"🎯 入力クエリ: {query}")
        print()
        
        # Step 1: Get the source document first
        print("📄 Step 1: ソース文書の取得")
        search_results = rag.db.similarity_search_with_relevance_scores(query, k=1)
        best_doc = search_results[0][0]
        source_text = best_doc.page_content
        
        print(f"完全なソース文書 ({len(source_text)}文字):")
        print(f'"{source_text}"')
        print()
        
        # Step 2A: Check if it goes through multi-chunk or regular processing
        print("🔄 Step 2A: 処理方式の判定")
        should_use_multi = rag._should_use_multi_chunk(query)
        print(f"マルチチャンク分析を使用: {should_use_multi}")
        print()
        
        # Step 2B: Trace the actual query processing
        print("🔍 Step 2B: クエリ処理の追跡")
        
        # Let's manually trace through the query method
        if should_use_multi and rag.multi_chunk_analyzer:
            print("➤ マルチチャンク分析ルート")
            
            # This would normally call _query_with_multi_chunk
            # Let's see what happens in regular processing instead for comparison
            print("  (比較のため通常処理も実行)")
            
        print("➤ 通常処理ルート (_query_regular)")
        
        # Step 2C: Document selection process
        print()
        print("🎯 Step 2C: 文書選択プロセス")
        chosen_doc = rag._choose_best_doc(query, search_results)
        
        if isinstance(chosen_doc, tuple):
            print("  ➤ 合成回答が選択されました")
            synthetic_answer = chosen_doc[0]
            source_for_evidence = chosen_doc[1] if len(chosen_doc) > 1 else chosen_doc[0]
            print(f"  合成回答: \"{synthetic_answer}\"")
            print(f"  証拠ソース: \"{source_for_evidence}\"")
        else:
            print("  ➤ 単一文書が選択されました")
            source_for_evidence = chosen_doc.page_content
        
        print()
        print("📝 Step 2D: 証拠生成の詳細プロセス")
        print(f"証拠生成に使用されるソース ({len(source_for_evidence)}文字):")
        print(f'"{source_for_evidence}"')
        print()
        
        # Step 2E: Evidence generation method analysis
        print("🧠 Step 2E: _generate_answer_fast()の動作解析")
        
        # Check query patterns
        definition_patterns = ['とは何', 'とは', '何ですか', '何でしょうか']
        pattern_match = None
        for pattern in definition_patterns:
            if pattern in query:
                pattern_match = pattern
                break
        
        print(f"定義質問パターン検出:")
        for pattern in definition_patterns:
            match_status = "✅" if pattern in query else "❌"
            print(f"  {match_status} '{pattern}' in query: {pattern in query}")
        
        if pattern_match:
            print(f"\n➤ 定義質問処理ルート (パターン: '{pattern_match}')")
            print("処理方法: 証拠の最初の文を抽出")
            
            # Extract first sentence
            import re
            first_sentence = re.split(r'[。！？.!?]', source_for_evidence)[0]
            print(f"抽出された最初の文: \"{first_sentence}\"")
            if first_sentence:
                evidence_result = first_sentence + '。'
                print(f"最終証拠 (定義処理): \"{evidence_result}\"")
        else:
            print(f"\n➤ LLM呼び出し処理ルート")
            print("処理方法: OpenAI GPT-3.5-turbo による証拠生成")
            
            # This would make the actual API call
            print("API呼び出しパラメータ:")
            print(f"  model: gpt-3.5-turbo")
            print(f"  system: 簡潔に日本語で答えてください。")
            print(f"  user: 証拠: {source_for_evidence[:50]}...\n        質問: {query}\n        回答:")
            print(f"  max_tokens: 100")
            print(f"  timeout: 30")
            
            # Actually call the method to see the result
            evidence_result = rag._generate_answer_fast(source_for_evidence, query)
            print(f"\n🎯 LLM生成結果: \"{evidence_result}\"")
        
        # Step 2F: Evidence text analysis
        print()
        print("📊 Step 2F: 生成された証拠テキストの分析")
        print(f"証拠テキスト長: {len(evidence_result)}文字")
        print(f"証拠テキスト: \"{evidence_result}\"")
        
        # Check if this matches our expected 75-character evidence
        expected_evidence = "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です"
        
        print()
        print("🔍 Step 2G: 期待される証拠との比較")
        print(f"期待される証拠 ({len(expected_evidence)}文字):")
        print(f'"{expected_evidence}"')
        print()
        print(f"マッチング結果:")
        print(f"  完全一致: {evidence_result == expected_evidence}")
        print(f"  部分一致: {expected_evidence in source_for_evidence}")
        print(f"  生成証拠がソースに含まれる: {evidence_result in source_for_evidence}")
        
        # Find where the expected evidence comes from in source
        if expected_evidence in source_for_evidence:
            pos = source_for_evidence.find(expected_evidence)
            print(f"  期待される証拠の位置: {pos}")
            print(f"  期待される証拠の終了位置: {pos + len(expected_evidence)}")
        
        # Step 2H: Explain the discrepancy if any
        print()
        print("🤔 Step 2H: 証拠生成メカニズムの解明")
        
        if evidence_result != expected_evidence:
            print("⚠️ 注意: 実際の生成結果と期待される結果が異なります")
            print()
            print("可能な原因:")
            print("1. _generate_answer_fast()は定義質問パターンを検出して最初の文のみ返した")
            print("2. 実際のマルチチャンク分析では異なる処理が行われる")
            print("3. LLM呼び出しでは異なる回答が生成される")
            print()
            print("実際の75文字証拠の生成源を確認するため、マルチチャンク処理を追跡します...")
            
            # Try with multi-chunk explicitly
            if rag.multi_chunk_analyzer:
                try:
                    multi_result = rag._query_with_multi_chunk(query)
                    print(f"\nマルチチャンク結果の証拠: \"{multi_result[2]}\"")
                    print(f"マルチチャンク証拠長: {len(multi_result[2])}文字")
                    
                    if multi_result[2] == expected_evidence:
                        print("✅ マルチチャンク分析が75文字証拠を生成しました！")
                except Exception as e:
                    print(f"❌ マルチチャンク分析エラー: {e}")
        else:
            print("✅ 証拠生成結果が期待値と一致しました")
        
        return {
            "query": query,
            "source_text": source_for_evidence,
            "generated_evidence": evidence_result,
            "expected_evidence": expected_evidence,
            "match": evidence_result == expected_evidence,
            "processing_route": "定義質問" if pattern_match else "LLM呼び出し"
        }
        
    except Exception as e:
        import traceback
        print(f"❌ エラー: {str(e)}")
        print(f"トレースバック: {traceback.format_exc()}")
        return None
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    trace_evidence_generation()