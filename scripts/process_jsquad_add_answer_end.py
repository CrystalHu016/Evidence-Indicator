#!/usr/bin/env python3
"""
Process JSQuAD datasets to add answer_end field
Based on answer_start and answer text
"""
import json
from pathlib import Path

def add_answer_end(data):
    """Add answer_end to each entry based on answer_start and answer text"""
    processed_count = 0

    for item in data:
        if 'answers' in item and item['answers']:
            # Get answer text and start position
            answer_texts = item['answers']['text']
            answer_starts = item['answers']['answer_start']

            # Calculate answer_end for each answer
            answer_ends = []
            for text, start in zip(answer_texts, answer_starts):
                # answer_end is the index of the last character
                # answer_end = answer_start + len(answer_text) - 1
                end = start + len(text) - 1
                answer_ends.append(end)

            # Add answer_end field
            item['answers']['answer_end'] = answer_ends
            processed_count += 1

    return processed_count

def process_dataset(input_file, output_file):
    """Process a dataset file"""
    print(f"\n{'='*60}")
    print(f"Processing: {input_file}")
    print(f"{'='*60}")

    # Load data
    print(f"🔄 Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📊 Total entries: {len(data):,}")

    # Add answer_end
    print(f"🔄 Adding answer_end fields...")
    processed_count = add_answer_end(data)

    # Save processed data
    print(f"💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Processed {processed_count:,} entries")

    # Show sample
    if data:
        sample = data[0]
        print(f"\n📝 Sample entry:")
        print(f"   Question: {sample['question'][:50]}...")
        if 'answers' in sample and sample['answers']:
            ans_text = sample['answers']['text'][0]
            ans_start = sample['answers']['answer_start'][0]
            ans_end = sample['answers']['answer_end'][0]
            print(f"   Answer text: {ans_text}")
            print(f"   Answer start: {ans_start}")
            print(f"   Answer end: {ans_end}")
            print(f"   Length: {len(ans_text)} chars")

            # Verify by extracting from context
            # Note: answer_end is the index of last char, so we need context[start:end+1]
            context = sample['context']
            extracted = context[ans_start:ans_end+1]
            match = "✅" if extracted == ans_text else "❌"
            print(f"   Verification: {match} (extracted: '{extracted}')")

    # File size
    import os
    file_size = os.path.getsize(output_file)
    print(f"\n💾 Output file size: {file_size / 1024 / 1024:.1f} MB")

def main():
    """Process both JSQuAD datasets"""
    print("🚀 Processing JSQuAD datasets to add answer_end field")

    datasets = [
        ("data/jsquad_train_10k.json", "data/jsquad_train_10k.json"),
        ("data/jsquad_train_full.json", "data/jsquad_train_full.json")
    ]

    for input_file, output_file in datasets:
        if Path(input_file).exists():
            process_dataset(input_file, output_file)
        else:
            print(f"⚠️  File not found: {input_file}")

    print(f"\n{'='*60}")
    print("✅ All datasets processed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
