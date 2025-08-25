#!/usr/bin/env python3
"""
Step-by-step trace of the 75-character evidence generation
逐步追踪75文字证拠生成过程
"""

import os
import sys
import re
from query_handler import setup_environment

def trace_75char_evidence_generation():
    """详细追踪75文字証拠テキスト的生成过程"""
    
    print("🎯 75文字証拠テキスト生成の完全追跡")
    print("="*70)
    
    original_cwd, parent_dir = setup_environment()
    
    try:
        from ultra_fast_rag_fixed import UltraFastRAG
        
        api_key = os.environ.get("OPENAI_API_KEY")
        rag = UltraFastRAG(api_key, "chroma")
        
        query = "コンバインとは何作物を収穫できますか"
        print(f"🔍 分析クエリ: {query}")
        print()
        
        # 手动逐步执行 _query_regular 过程
        print("📝 Step-by-Step _query_regular 処理過程:")
        print()
        
        # Step 1: ベクトル検索
        print("🔎 Step 1: ベクトル類似検索")
        search_results = rag.db.similarity_search_with_relevance_scores(query, k=10)
        print(f"検索結果数: {len(search_results)}")
        print(f"最高スコア: {search_results[0][1]:.3f}")
        best_doc_content = search_results[0][0].page_content
        print(f"最高スコア文書 ({len(best_doc_content)}文字):")
        print(f'"{best_doc_content}"')
        print()
        
        # Step 2: 文書選択 (_choose_best_doc)
        print("🎯 Step 2: _choose_best_doc 処理")
        chosen_doc = rag._choose_best_doc(query, search_results)
        
        print("文書選択の詳細分析:")
        print(f"  返り値の型: {type(chosen_doc)}")
        
        if isinstance(chosen_doc, tuple):
            print("  ➤ 合成回答ルート")
            synthetic_answer = chosen_doc[0]
            source_text = chosen_doc[1] if len(chosen_doc) > 1 else chosen_doc[0]
            print(f"  合成回答: \"{synthetic_answer[:100]}...\"")
            print(f"  ソーステキスト: \"{source_text[:100]}...\"")
        else:
            print("  ➤ 単一文書ルート")
            source_text = chosen_doc.page_content
            print(f"  選択された文書: \"{source_text[:100]}...\"")
            synthetic_answer = None
        
        print()
        
        # Step 3: LLM証拠生成または合成回答の使用
        print("💡 Step 3: 証拠生成プロセス")
        
        if isinstance(chosen_doc, tuple):
            print("  ➤ 合成回答が存在する場合の処理")
            final_answer = synthetic_answer
            evidence_for_positioning = source_text if len(chosen_doc) > 1 else synthetic_answer
            
            print(f"  最終回答: \"{final_answer}\"")
            print(f"  位置特定用証拠: \"{evidence_for_positioning}\"")
        else:
            print("  ➤ LLM証拠生成の場合の処理")
            # 这里会调用 _generate_answer_fast
            evidence_text = rag._generate_answer_fast(source_text, query)
            print(f"  _generate_answer_fast結果: \"{evidence_text}\"")
            
            final_answer = evidence_text
            evidence_for_positioning = evidence_text
        
        print()
        
        # Step 4: 位置特定
        print("📍 Step 4: 文字列位置特定")
        print(f"位置特定対象の証拠 ({len(evidence_for_positioning)}文字):")
        print(f'"{evidence_for_positioning}"')
        
        # 在原文中查找位置
        start_pos = source_text.find(evidence_for_positioning)
        if start_pos != -1:
            end_pos = start_pos + len(evidence_for_positioning)
            print(f"✅ 直接マッチング成功:")
            print(f"  開始位置: {start_pos}")
            print(f"  終了位置: {end_pos}")
        else:
            print("❌ 直接マッチング失敗")
        
        # 关键分析：为什么会生成75字符的证据？
        print()
        print("🔍 Step 5: 75文字証拠の生成理由分析")
        
        expected_75char = "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です"
        
        # 检查这个75字符证据在原文中的位置
        pos_in_source = source_text.find(expected_75char)
        print(f"期待される75文字証拠の原文中位置: {pos_in_source}")
        
        if pos_in_source != -1:
            print(f"✅ 75文字証拠は原文中の位置 {pos_in_source}-{pos_in_source + len(expected_75char)} に存在")
            
            # 分析原文的结构
            print()
            print("📄 原文構造分析:")
            sentences = re.split(r'[。！？.!?]', source_text)
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    sentence_start = source_text.find(sentence.strip())
                    print(f"  文{i+1} (位置{sentence_start}): \"{sentence.strip()}\"")
                    
                    # 检查这个句子是否包含75字符证据
                    if expected_75char in sentence:
                        print(f"    ✅ この文に75文字証拠が含まれています！")
        
        # 深入分析 _choose_best_doc 的逻辑
        print()
        print("🤔 Step 6: _choose_best_doc の詳細ロジック分析")
        
        # 手动执行 _choose_best_doc 的关键步骤
        print("農業関連クエリの特別処理:")
        
        # 检查农业相关标记
        agri_markers = ['稲', '稲作', '田', '水田', '苗', '収穫', '脱穀', '選別', '籾', '精米', '農業']
        is_agri_query = any(m in query for m in agri_markers)
        print(f"  農業クエリ判定: {is_agri_query}")
        
        # 检查查询中的农业标记
        found_agri_markers = [m for m in agri_markers if m in query]
        print(f"  検出された農業マーカー: {found_agri_markers}")
        
        # 检查作物收获相关的特殊处理
        if '収穫' in query and '作物' in query:
            print("  ✅ 作物収穫クエリとして検出")
            print("  特別処理: 作物名を含む詳細な回答が必要")
            
            # 查找包含作物名的文档部分
            crops = ['稲', '麦', '大豆', '小豆', '菜種', 'トウモロコシ']
            print(f"  検索対象作物: {crops}")
            
            # 在原文中查找作物信息最丰富的部分
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    crop_count = sum(1 for crop in crops if crop in sentence)
                    if crop_count > 0:
                        print(f"    文{i+1}: {crop_count}種類の作物を含む")
                        if crop_count >= 4:  # 如果包含4种以上作物
                            print(f"      ✅ 最多作物情報を含む文: \"{sentence.strip()}\"")
                            
                            # 这可能就是75字符证据的来源！
                            if len(sentence.strip()) == 75:
                                print(f"      🎯 これが75文字証拠の正確な来源です！")
        
        return {
            "source_text": source_text,
            "chosen_doc_type": "tuple" if isinstance(chosen_doc, tuple) else "single",
            "final_evidence": evidence_for_positioning if 'evidence_for_positioning' in locals() else None,
            "evidence_position": (start_pos, start_pos + len(evidence_for_positioning)) if 'evidence_for_positioning' in locals() and start_pos != -1 else None
        }
        
    except Exception as e:
        import traceback
        print(f"❌ エラー: {str(e)}")
        print(f"トレースバック: {traceback.format_exc()}")
        return None
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    trace_75char_evidence_generation()