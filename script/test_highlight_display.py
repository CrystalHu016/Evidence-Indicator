#!/usr/bin/env python3
"""
测试高亮显示效果 - 模拟前端显示
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

def test_highlight_display():
    """测试高亮显示效果"""
    print('🔍 测试查询: 農業機械の種類について教えてください')
    print('=' * 80)

    try:
        from backend_integration import call_backend_query

        query = '農業機械の種類について教えてください'

        start_time = time.time()
        result, error = call_backend_query(query, 'enhanced')
        elapsed = time.time() - start_time

        if error:
            print(f'❌ 错误: {error}')
            return

        print(f'⏱️  处理时间: {elapsed:.2f}秒')
        print()
        
        # 1. 回答
        print('📝 【回答】')
        print('=' * 50)
        answer = result.get('answer', 'No answer')
        print(answer)
        print()
        
        # 2. 高亮部分
        print('🔍 【高亮部分/证据文本】')
        print('=' * 50)
        evidence = result.get('evidence_text', 'No evidence')
        print(f'"{evidence}"')
        print()
        
        # 3. 原文（带高亮标记）
        print('📄 【原文（带高亮标记）】')
        print('=' * 50)
        source = result.get('source_document', 'No source')
        
        # 模拟高亮效果
        if evidence and evidence in source:
            highlighted_source = source.replace(evidence, f'【{evidence}】')
            print(highlighted_source)
        else:
            print(source)
        print()
        
        # 4. 前端显示效果模拟
        print('🖥️  【前端显示效果模拟】')
        print('=' * 50)
        print('【回答】')
        print(answer)
        print()
        print('【検索ヒットのチャンクを含む文書】')
        print('💡 根拠部分のハイライト表示:')
        if evidence and evidence in source:
            highlighted_source = source.replace(evidence, f'**【根拠ハイライト】{evidence}**')
            print(highlighted_source)
        else:
            print(source)
        print()
        print('【根拠情報】')
        print(evidence)
        print()
        
        # 5. 详细分析
        print('📊 【详细分析】')
        print('=' * 50)
        print(f'回答长度: {len(answer)} 字符')
        print(f'证据长度: {len(evidence)} 字符')
        print(f'原文长度: {len(source)} 字符')
        print(f'信心度: {result.get("confidence", 0.0)}')
        print(f'模型: {result.get("model", "Unknown")}')
        print()
        
        # 检查回答质量
        print('🎯 【回答质量检查】')
        print('-' * 30)
        if "種類" in answer or "分類" in answer:
            print('✅ 包含"種類"关键词')
        if "普通型" in answer and "自立型" in answer:
            print('✅ 包含两种类型：普通型和自立型')
        if "コンバイン" in answer:
            print('✅ 提到コンバイン（联合收割机）')
        if len(answer) > len(evidence):
            print('✅ 使用生成式回答（回答比证据更长）')
        else:
            print('⚠️  可能使用抽取式回答')
            
        print()
        print('🎉 测试完成！')

    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_highlight_display()
