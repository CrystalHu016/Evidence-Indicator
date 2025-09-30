#!/usr/bin/env python3
"""
测试查询显示 - 显示回答、高亮部分和原文
"""

import sys
import os
import time

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

def test_query_display():
    """测试查询并显示完整结果"""
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
        print('-' * 50)
        answer = result.get('answer', 'No answer')
        print(answer)
        print()
        
        # 2. 高亮部分
        print('🔍 【高亮部分/证据文本】')
        print('-' * 50)
        evidence = result.get('evidence_text', 'No evidence')
        print(evidence)
        print()
        
        # 3. 原文
        print('📄 【原文】')
        print('-' * 50)
        source = result.get('source_document', 'No source')
        print(source)
        print()
        
        # 额外信息
        print('📊 【额外信息】')
        print('-' * 50)
        print(f'信心度: {result.get("confidence", 0.0)}')
        print(f'模型: {result.get("model", "Unknown")}')
        print(f'处理时间: {result.get("processing_time", "Unknown")}')
        
        # 分析对比
        print()
        print('🔍 【分析对比】')
        print('-' * 50)
        print(f'回答长度: {len(answer)} 字符')
        print(f'证据长度: {len(evidence)} 字符')
        print(f'原文长度: {len(source)} 字符')
        
        if answer == evidence:
            print('⚠️  回答和证据完全相同 - 可能仍在使用抽取式')
        elif len(answer) > len(evidence):
            print('✅ 回答比证据更长 - 使用了生成式方法')
        else:
            print('🤔 回答比证据更短 - 需要进一步检查')
            
        # 检查是否回答了"种类"问题
        if "種類" in answer or "分類" in answer or ("普通型" in answer and "自立型" in answer):
            print('✅ 正确回答了种类相关问题')
        else:
            print('⚠️  没有明确回答种类问题')

    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query_display()
