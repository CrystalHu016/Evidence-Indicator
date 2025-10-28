#!/usr/bin/env python3
"""
Normalize squad_test_100.json by extracting unique contexts and using references.
将 squad_test_100.json 标准化：提取唯一的 context，使用引用关联。

Output format:
{
  "contexts": [
    {
      "context_id": "ctx_0",
      "text": "梅雨 [SEP] 梅雨（つゆ、ばいう）は...",
      "title": "梅雨"
    },
    ...
  ],
  "qas": [
    {
      "id": "a10336p0q0",
      "context_id": "ctx_0",
      "question": "日本で梅雨がないのは北海道とどこか。",
      "answers": {
        "answer_start": [25, 25, 25],
        "text": ["小笠原諸島", "小笠原諸島を除く日本", "小笠原諸島"],
        "answer_end": [29, 34, 29]
      }
    },
    ...
  ]
}
"""

import json
import os

def normalize_squad_data(input_file: str, output_file: str):
    """Extract unique contexts and create normalized format"""

    print(f"📖 Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"   Total entries: {len(data)}")

    # Extract unique contexts
    context_map = {}  # text -> context_id
    contexts = []
    qas = []

    print(f"\n🔄 Processing entries...")

    for item in data:
        context_text = item.get('context', '')
        title = item.get('title', 'unknown')
        item_id = item.get('id', '')
        question = item.get('question', '')
        answers = item.get('answers', {})

        # Check if this context already exists
        if context_text not in context_map:
            # New context, add it
            context_id = f"ctx_{len(contexts)}"
            context_map[context_text] = context_id

            contexts.append({
                'context_id': context_id,
                'text': context_text,
                'title': title
            })

            print(f"   ➕ New context: {context_id} - {title} ({len(context_text)} chars)")
        else:
            # Context already exists, reuse it
            context_id = context_map[context_text]
            print(f"   🔗 Reusing context: {context_id} for question: {item_id}")

        # Add Q&A entry with reference to context
        qas.append({
            'id': item_id,
            'context_id': context_id,
            'question': question,
            'answers': answers,
            'is_impossible': item.get('is_impossible', False)
        })

    # Create normalized output
    normalized_data = {
        'contexts': contexts,
        'qas': qas
    }

    print(f"\n📊 Statistics:")
    print(f"   Original entries: {len(data)}")
    print(f"   Unique contexts: {len(contexts)}")
    print(f"   Total Q&A pairs: {len(qas)}")
    print(f"   Storage reduction: {(1 - len(contexts) / len(data)) * 100:.1f}% (for contexts)")

    # Calculate storage size estimation
    original_context_size = sum(len(item['context']) for item in data)
    normalized_context_size = sum(len(ctx['text']) for ctx in contexts)
    print(f"   Context storage: {original_context_size:,} chars → {normalized_context_size:,} chars")
    print(f"   Reduction: {(1 - normalized_context_size / original_context_size) * 100:.1f}%")

    # Save normalized data
    print(f"\n💾 Saving normalized data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Normalized format saved")

    # Show example usage
    print(f"\n📋 Example usage:")
    print(f"   # Load normalized data")
    print(f"   with open('{output_file}', 'r') as f:")
    print(f"       data = json.load(f)")
    print(f"   ")
    print(f"   # Get Q&A with context")
    print(f"   qa = data['qas'][0]")
    print(f"   context_id = qa['context_id']")
    print(f"   context = next(c for c in data['contexts'] if c['context_id'] == context_id)")
    print(f"   print(f\"Question: {{qa['question']}}\")")
    print(f"   print(f\"Context: {{context['text'][:50]}}...\")")

    return normalized_data

def convert_back_to_original_format(normalized_file: str, output_file: str):
    """Convert normalized format back to original SQuAD format for compatibility"""

    print(f"\n🔄 Converting normalized format back to original SQuAD format...")
    print(f"   Input: {normalized_file}")
    print(f"   Output: {output_file}")

    with open(normalized_file, 'r', encoding='utf-8') as f:
        normalized_data = json.load(f)

    # Create context lookup
    context_lookup = {ctx['context_id']: ctx for ctx in normalized_data['contexts']}

    # Convert back
    original_format = []
    for qa in normalized_data['qas']:
        context_id = qa['context_id']
        context_obj = context_lookup[context_id]

        original_format.append({
            'id': qa['id'],
            'title': context_obj['title'],
            'context': context_obj['text'],
            'question': qa['question'],
            'answers': qa['answers'],
            'is_impossible': qa.get('is_impossible', False)
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(original_format, f, ensure_ascii=False, indent=2)

    print(f"✅ Converted {len(original_format)} entries back to original format")

def main():
    # File paths
    input_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100.json")
    normalized_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100_normalized.json")
    converted_back_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100_converted_back.json")

    # Normalize
    normalized_data = normalize_squad_data(input_file, normalized_file)

    # Convert back to original format (for verification)
    convert_back_to_original_format(normalized_file, converted_back_file)

    print(f"\n{'='*80}")
    print(f"✅ Summary:")
    print(f"   Original: {input_file}")
    print(f"   Normalized: {normalized_file} (distinct contexts + Q&A references)")
    print(f"   Converted back: {converted_back_file} (for verification)")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
