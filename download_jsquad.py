#!/usr/bin/env python3
"""
Download complete JSQuAD dataset from HuggingFace
"""
import json
from datasets import load_dataset
from pathlib import Path

def download_jsquad():
    """Download JSQuAD train and validation splits"""
    print("🔄 Downloading JSQuAD dataset from HuggingFace...")

    # Load the dataset
    dataset = load_dataset("sbintuitions/JSQuAD")

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Process train split
    train_data = []
    print(f"\n📊 Processing train split ({len(dataset['train'])} examples)...")
    for item in dataset['train']:
        train_data.append({
            'id': item['id'],
            'title': item['title'],
            'context': item['context'],
            'question': item['question'],
            'answers': item['answers']
        })

    # Save train data
    train_file = data_dir / "jsquad_train_full.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(train_data)} training examples to {train_file}")

    # Process validation split
    val_data = []
    print(f"\n📊 Processing validation split ({len(dataset['validation'])} examples)...")
    for item in dataset['validation']:
        val_data.append({
            'id': item['id'],
            'title': item['title'],
            'context': item['context'],
            'question': item['question'],
            'answers': item['answers']
        })

    # Save validation data
    val_file = data_dir / "jsquad_validation_full.json"
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(val_data)} validation examples to {val_file}")

    # Print statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Train examples: {len(train_data)}")
    print(f"   Validation examples: {len(val_data)}")
    print(f"   Total examples: {len(train_data) + len(val_data)}")

    # Show a sample
    print(f"\n📝 Sample from train split:")
    sample = train_data[0]
    print(f"   ID: {sample['id']}")
    print(f"   Title: {sample['title']}")
    print(f"   Question: {sample['question']}")
    print(f"   Answer: {sample['answers']['text'][0]}")
    print(f"   Context length: {len(sample['context'])} characters")

if __name__ == "__main__":
    download_jsquad()
