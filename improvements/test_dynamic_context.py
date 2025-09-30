#!/usr/bin/env python3
"""
Quick test of dynamic context generation
"""

import os
from dynamic_context_generator import DynamicContextGenerator

def quick_test():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    generator = DynamicContextGenerator(api_key)

    # Simple test case
    chunks = [
        "コンバインは農業機械です。",
        "普通型と自立型があります。",
        "日本独自の技術です。"
    ]

    print("🧪 Testing dynamic context generation...")
    result = generator.generate_dynamic_context(
        query="コンバインとは何ですか",
        chunks=chunks,
        intent="definition"
    )

    print(f"✅ Generated context: {result.enhanced_context}")
    print(f"📊 Score: {result.coherence_score:.2f}")
    print(f"⏱️ Time: {result.processing_time:.2f}s")

if __name__ == "__main__":
    quick_test()