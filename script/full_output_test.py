#!/usr/bin/env python3
"""
完整输出测试：生成式回答 + 高亮部分 + 原文
"""

import os
import sys
import openai
import json
from dotenv import load_dotenv

load_dotenv()

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rag-streamlit-frontend'))

def generate_full_output():
    """生成完整的三个输出"""

    query = "農業機械の種類について教えてください"

    # 模拟从向量数据库检索到的原文
    original_source = "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"

    print("=" * 80)
    print("🔍 用户查询: " + query)
    print("=" * 80)

    # 1. 生成LLM回答
    llm_answer = generate_llm_answer(query, original_source)

    # 2. 生成高亮证据（模拟原来的evidence_text）
    evidence_highlighted = extract_evidence_for_highlighting(original_source, query)

    # 3. 原文
    source_document = original_source

    print("\n📝 1. 生成式回答 (LLM生成):")
    print("-" * 50)
    print(llm_answer)

    print("\n🔍 2. 高亮部分/证据文本 (evidence_text):")
    print("-" * 50)
    print(evidence_highlighted)

    print("\n📄 3. 原文 (source_document):")
    print("-" * 50)
    print(source_document)

    # 4. 显示Streamlit前端会如何显示
    print("\n" + "=" * 80)
    print("🖥️  Streamlit前端显示效果:")
    print("=" * 80)

    print("【回答】")
    print(llm_answer)
    print()

    print("【検索ヒットのチャンクを含む文書】")
    print("💡 根拠部分のハイライト表示:")
    # 模拟高亮显示
    highlighted_source = create_highlighted_display(source_document, evidence_highlighted)
    print(highlighted_source)
    print()

    print("📄 元の文書:")
    print(source_document)
    print()

    print("【根拠情報】")
    print(evidence_highlighted)

def generate_llm_answer(query, context):
    """生成LLM回答"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "API密钥未设置，无法生成LLM回答"

    client = openai.OpenAI(api_key=api_key)

    prompt = f"""
查询: {query}
参考文本: {context}

请根据参考文本全面回答用户的查询。要求:
1. 直接回答用户的问题
2. 基于参考文本内容，但用你自己的话来表达
3. 如果是关于分类/种类的问题，要列出具体的分类
4. 回答要完整、准确、简洁
5. 用日语回答

返回JSON格式:
{{
    "generated_answer": "<根据参考文本生成的完整回答>"
}}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "你是一个智能问答助手，能够根据参考文本生成准确、完整的日语回答。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=400
        )

        result_text = response.choices[0].message.content.strip()
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            parsed = json.loads(json_str)
            return parsed.get('generated_answer', '生成失败')

    except Exception as e:
        return f"LLM生成失败: {e}"

    return "LLM生成失败"

def extract_evidence_for_highlighting(source_text, query):
    """提取用于高亮的证据文本"""
    import re

    # 对于"種類"查询，提取分类相关的句子
    if "種類" in query or "分類" in query:
        # 寻找包含分类信息的句子
        sentences = re.split(r'[。！？.!?]', source_text)
        for sentence in sentences:
            if "2種類に大別" in sentence or "普通型と自立型" in sentence:
                return sentence.strip()

    # 默认返回第一句作为证据
    sentences = re.split(r'[。！？.!?]', source_text)
    if sentences:
        return sentences[0].strip()

    return source_text[:100]

def create_highlighted_display(source_document, evidence_text):
    """创建带高亮显示的文档"""
    if evidence_text in source_document:
        highlighted = source_document.replace(evidence_text, f"**【根拠ハイライト】{evidence_text}**")
        return highlighted
    else:
        return f"**【根拠ハイライト】{evidence_text}**\n\n{source_document}"

if __name__ == "__main__":
    generate_full_output()