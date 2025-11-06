#!/usr/bin/env python3
"""
Merge squad_test_100.json and jsquad_train_10k.json into a single dataset
"""
import json
from pathlib import Path

def merge_datasets():
    """Merge two datasets"""
    print("🔄 Merging datasets...")
    print("=" * 70)

    # Load squad_test_100.json
    print("\n📂 Loading squad_test_100.json...")
    with open('data/squad_test_100.json', 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    print(f"   ✅ Loaded {len(squad_data)} entries")

    # Load jsquad_train_10k.json
    print("\n📂 Loading jsquad_train_10k.json...")
    with open('data/jsquad_train_10k.json', 'r', encoding='utf-8') as f:
        jsquad_data = json.load(f)
    print(f"   ✅ Loaded {len(jsquad_data)} entries")

    # Merge datasets
    print("\n🔀 Merging datasets...")
    merged_data = squad_data + jsquad_data
    print(f"   ✅ Total entries: {len(merged_data)}")

    # Save merged dataset
    output_file = 'data/merged_qa_dataset.json'
    print(f"\n💾 Saving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # File size
    import os
    file_size = os.path.getsize(output_file)
    print(f"   ✅ Saved {len(merged_data)} entries")
    print(f"   💾 File size: {file_size / 1024 / 1024:.1f} MB")

    # Show statistics
    print("\n📊 Dataset Statistics:")
    print(f"   squad_test_100.json:     {len(squad_data):>6,} entries")
    print(f"   jsquad_train_10k.json:   {len(jsquad_data):>6,} entries")
    print(f"   " + "-" * 40)
    print(f"   merged_qa_dataset.json:  {len(merged_data):>6,} entries")

    # Show sample from each source
    print("\n📝 Sample from squad_test_100.json:")
    sample1 = squad_data[0]
    print(f"   Question: {sample1['question']}")
    print(f"   Answer: {sample1['answers']['text'][0]}")

    print("\n📝 Sample from jsquad_train_10k.json:")
    sample2 = jsquad_data[0]
    print(f"   Question: {sample2['question']}")
    print(f"   Answer: {sample2['answers']['text'][0]}")

    print("\n" + "=" * 70)
    print("✅ Dataset merge completed successfully!")
    print(f"📄 Output file: {output_file}")

    return output_file

if __name__ == "__main__":
    merge_datasets()
