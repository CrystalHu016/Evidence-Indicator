#!/usr/bin/env python3
"""
直接测试LLM回答生成函数
"""

import os
import openai
import json
from dotenv import load_dotenv

load_dotenv()

def test_llm_answer_direct():
    """直接测试LLM回答生成"""
    query = "農業機械の種類について教えてください"
    context = "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"

    print(f"🔍 问题: {query}")
    print(f"📄 上下文: {context[:60]}...")

    # 获取API密钥
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY 未设置")
        return

    client = openai.OpenAI(api_key=api_key)

    # 生成回答的prompt
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
        print(f"\n📤 GPT响应: {result_text}")

        # 解析JSON
        json_start = result_text.find('{')
        json_end = result_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            parsed = json.loads(json_str)
            generated_answer = parsed.get('generated_answer', '')

            print(f"\n🎯 最终回答:")
            print(f"{generated_answer}")

            print(f"\n📊 分析:")
            print(f"- 回答长度: {len(generated_answer)} 字符")
            print(f"- 包含'種類': {'是' if '種類' in generated_answer else '否'}")
            print(f"- 包含'普通型': {'是' if '普通型' in generated_answer else '否'}")
            print(f"- 包含'自立型': {'是' if '自立型' in generated_answer else '否'}")

        else:
            print("❌ 无法解析JSON响应")

    except Exception as e:
        print(f"❌ 错误: {e}")

if __name__ == "__main__":
    test_llm_answer_direct()