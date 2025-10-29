#!/usr/bin/env python3
"""
Extract random 10k samples from JSQuAD train dataset
"""
import json
import random
from pathlib import Path

def extract_random_samples(input_file, output_file, sample_size=10000):
    """Extract random samples from dataset"""
    print(f"🔄 Loading data from {input_file}...")

    # Load the full dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        full_data = json.load(f)

    print(f"📊 Total examples in dataset: {len(full_data):,}")

    # Set random seed for reproducibility
    random.seed(42)

    # Randomly sample
    if len(full_data) <= sample_size:
        sampled_data = full_data
        print(f"⚠️  Dataset has only {len(full_data)} examples, using all")
    else:
        sampled_data = random.sample(full_data, sample_size)
        print(f"✅ Randomly sampled {sample_size:,} examples")

    # Save to output file
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {output_file}")

    # Show statistics
    print(f"\n📈 Statistics:")
    print(f"   Input:  {len(full_data):>7,} examples")
    print(f"   Output: {len(sampled_data):>7,} examples")

    # Show sample
    print(f"\n📝 Sample entry:")
    sample = sampled_data[0]
    print(f"   ID: {sample['id']}")
    print(f"   Title: {sample['title']}")
    print(f"   Question: {sample['question'][:50]}...")
    print(f"   Answer: {sample['answers']['text'][0][:30]}...")
    print(f"   Context length: {len(sample['context'])} characters")

    # File size
    import os
    file_size = os.path.getsize(output_file)
    print(f"\n💾 File size: {file_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    input_file = "data/jsquad_train_full.json"
    output_file = "data/jsquad_train_10k.json"

    extract_random_samples(input_file, output_file, sample_size=10000)
