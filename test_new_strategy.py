#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from script.ultra_fast_rag_semantic_with_char_markers import ImprovedSemanticRAG

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

rag = ImprovedSemanticRAG(api_key)
rag.build_vector_store("")

query = '梅雨とは何季の一種か?'

print('\n' + '='*80)
print('【新策略测试】New Strategy Test')
print('='*80)
print(f'Query: {query}\n')

result = rag.query_with_answer(query, k=3)

print('='*80)
print('【回答】Answer')
print('='*80)
print(f'{result["answer"]}\n')

print('='*80)
print('【根拠情報】Evidence Information')
print('='*80)

for idx, evidence in enumerate(result.get('evidences', []), 1):
    print(f'\nChunk {idx}:')

    if evidence.get('is_empty', True):
        print('  状态: ❌ No evidence found')
        continue

    extracted = evidence.get('extracted_evidence', 'N/A')
    char_ranges = evidence.get('char_ranges', [])

    print(f'  【根拠情報】: "{extracted}"')

    if char_ranges:
        range_str = ', '.join([f'{s}文字目～{e}文字目' for s, e in char_ranges])
        print(f'  【根拠情報の文字列範囲】: {range_str}')
    else:
        print(f'  【根拠情報の文字列範囲】: N/A')

    print(f'  【コアターム】: {evidence.get("core_term", "N/A")}')
    print(f'  Semantic Relevance: {evidence.get("semantic_relevance", 0):.3f}')

    # 验证
    if char_ranges and not evidence.get('is_empty'):
        chunk_content = evidence.get('chunk_content', '')
        for start, end in char_ranges:
            if 1 <= start <= len(chunk_content) and start <= end <= len(chunk_content):
                actual = chunk_content[start-1:end]
                match = '✅' if actual == extracted else '❌'
                print(f'  【验证】: Position {start}～{end} = "{actual}" {match}')

print('\n' + '='*80)
print(f'⏱️  Processing Time: {result.get("processing_time", 0):.2f}s')
print('='*80)
