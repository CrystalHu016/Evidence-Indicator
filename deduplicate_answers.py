#!/usr/bin/env python3
"""
Deduplicate answers within each Q&A entry.
If answer text array has duplicates, keep only unique ones and adjust answer_start/answer_end.

Example:
Before:
  "answers": {
    "answer_start": [39, 39, 39],
    "text": ["シベリア気団", "シベリア気団", "シベリア気団"],
    "answer_end": [44, 44, 44]
  }

After:
  "answers": {
    "answer_start": [39],
    "text": ["シベリア気団"],
    "answer_end": [44]
  }
"""

import json
import os

def deduplicate_answers_in_entry(answers: dict) -> dict:
    """
    Deduplicate answer text and corresponding positions
    保留唯一的答案文本和对应的位置
    """
    text_list = answers.get('text', [])
    start_list = answers.get('answer_start', [])
    end_list = answers.get('answer_end', [])

    if not text_list:
        return answers

    # Track unique combinations of (text, start, end)
    seen = set()
    unique_texts = []
    unique_starts = []
    unique_ends = []

    for i, text in enumerate(text_list):
        start = start_list[i] if i < len(start_list) else None
        end = end_list[i] if i < len(end_list) else None

        # Create a unique key
        key = (text, start, end)

        if key not in seen:
            seen.add(key)
            unique_texts.append(text)
            if start is not None:
                unique_starts.append(start)
            if end is not None:
                unique_ends.append(end)

    return {
        'answer_start': unique_starts,
        'text': unique_texts,
        'answer_end': unique_ends
    }

def deduplicate_squad_answers(input_file: str, output_file: str):
    """Deduplicate answers in squad dataset"""

    print(f"📖 Loading data from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"   Total entries: {len(data)}")

    total_original_answers = 0
    total_unique_answers = 0
    entries_with_duplicates = 0

    print(f"\n🔄 Processing entries...")

    for idx, item in enumerate(data):
        answers = item.get('answers', {})
        original_count = len(answers.get('text', []))
        total_original_answers += original_count

        # Deduplicate
        deduplicated_answers = deduplicate_answers_in_entry(answers)
        unique_count = len(deduplicated_answers.get('text', []))
        total_unique_answers += unique_count

        # Update the entry
        item['answers'] = deduplicated_answers

        # Log if duplicates were found
        if original_count > unique_count:
            entries_with_duplicates += 1
            item_id = item.get('id', f'entry_{idx}')
            question = item.get('question', '')[:50]
            print(f"   🔄 {item_id}: {original_count} → {unique_count} answers")
            print(f"      Question: {question}...")
            print(f"      Original texts: {answers.get('text', [])}")
            print(f"      Unique texts: {deduplicated_answers.get('text', [])}")

    print(f"\n📊 Statistics:")
    print(f"   Total entries: {len(data)}")
    print(f"   Entries with duplicate answers: {entries_with_duplicates}")
    print(f"   Total original answers: {total_original_answers}")
    print(f"   Total unique answers: {total_unique_answers}")
    print(f"   Duplicate answers removed: {total_original_answers - total_unique_answers}")
    if total_original_answers > 0:
        print(f"   Reduction: {(total_original_answers - total_unique_answers) / total_original_answers * 100:.1f}%")

    # Save deduplicated data
    print(f"\n💾 Saving deduplicated data to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Done!")

    return data

def main():
    # File paths
    input_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100.json")
    output_file = os.path.join(os.path.dirname(__file__), "data", "squad_test_100_dedup_answers.json")

    # Create backup
    backup_file = input_file + ".backup2"
    if not os.path.exists(backup_file):
        import shutil
        shutil.copy2(input_file, backup_file)
        print(f"📦 Created backup: {backup_file}\n")

    # Deduplicate answers
    deduplicated_data = deduplicate_squad_answers(input_file, output_file)

    # Show example
    print(f"\n📋 Example - Before and After:")
    print(f"   Before (from backup):")
    with open(backup_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
        if len(original_data) > 0:
            example = original_data[10]  # Show entry with duplicates
            print(f"      ID: {example.get('id')}")
            print(f"      Question: {example.get('question')[:50]}...")
            print(f"      Answers: {json.dumps(example.get('answers'), ensure_ascii=False, indent=8)}")

    print(f"\n   After:")
    if len(deduplicated_data) > 0:
        example = deduplicated_data[10]
        print(f"      ID: {example.get('id')}")
        print(f"      Question: {example.get('question')[:50]}...")
        print(f"      Answers: {json.dumps(example.get('answers'), ensure_ascii=False, indent=8)}")

    print(f"\n{'='*80}")
    print(f"💡 To replace original file, run:")
    print(f"   mv {output_file} {input_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
