#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from script.ultra_fast_rag_semantic import PureSemanticRAG

load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')

rag = PureSemanticRAG(api_key, chroma_path='./chroma')
query = '梅雨とは何季の一種か?'

print('\n' + '='*80)
print('【查询】Query')
print('='*80)
print(f'{query}\n')

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

print('\n' + '='*80)
print(f'⏱️  Processing Time: {result.get("processing_time", 0):.2f}s')
print('='*80)
