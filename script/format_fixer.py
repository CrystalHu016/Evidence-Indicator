#!/usr/bin/env python3
"""
Format Fixer for Ichikara Dataset
Fix format issues for ChromaDB compatibility while preserving ALL original data
"""

import json
import os
from typing import Dict, List, Any

def flatten_metadata(meta: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Flatten nested metadata while preserving all data"""
    flattened = {}
    
    for key, value in meta.items():
        # Create safe key name (replace problematic characters)
        safe_key = key.replace("-", "_").replace(" ", "_")
        full_key = f"{prefix}{safe_key}" if prefix else safe_key
        
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            nested = flatten_metadata(value, f"{full_key}_")
            flattened.update(nested)
        elif isinstance(value, list):
            # Convert lists to JSON strings to preserve data
            if value:  # Only if list is not empty
                if all(isinstance(item, str) for item in value):
                    # Simple string list - join with separator
                    flattened[full_key] = " | ".join(str(item) for item in value)
                else:
                    # Complex list - convert to JSON string
                    flattened[full_key] = json.dumps(value, ensure_ascii=False)
        else:
            # Simple values - convert to string
            flattened[full_key] = str(value) if value is not None else ""
    
    return flattened

def fix_dataset_format(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Fix format while preserving ALL original data"""
    
    # Create the main content field (combination of text and output for RAG)
    content = entry.get("text", "") + "\n\n" + entry.get("output", "")
    
    # Start with basic structure
    fixed = {
        "ID": str(entry.get("ID", "")),
        "text": str(entry.get("text", "")),
        "output": str(entry.get("output", "")),
        "content": content,  # Combined content for RAG search
    }
    
    # Flatten and add metadata while preserving everything
    if "meta" in entry and isinstance(entry["meta"], dict):
        flattened_meta = flatten_metadata(entry["meta"])
        fixed.update(flattened_meta)
    
    return fixed

def fix_ichikara_format(input_file: str, output_file: str) -> bool:
    """Fix format issues while preserving all original data"""
    try:
        print(f"🔧 Fixing format: {input_file}")
        
        # Load original dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        print(f"📊 Original dataset: {len(original_data)} entries")
        
        # Fix format for each entry
        fixed_data = []
        for i, entry in enumerate(original_data):
            try:
                fixed_entry = fix_dataset_format(entry)
                fixed_data.append(fixed_entry)
                
                # Show what fields were created
                field_count = len(fixed_entry)
                print(f"✅ Fixed entry {i+1}/{len(original_data)}: {fixed_entry['ID'][:50]}... ({field_count} fields)")
                
            except Exception as e:
                print(f"❌ Error fixing entry {i+1}: {e}")
                continue
        
        # Save fixed dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Fixed dataset saved: {output_file}")
        print(f"📊 Final dataset: {len(fixed_data)} entries")
        
        # Show sample of fields created
        if fixed_data:
            print(f"\n📋 Fields in fixed dataset:")
            sample_entry = fixed_data[0]
            for key in sample_entry.keys():
                value_preview = str(sample_entry[key])[:50]
                if len(str(sample_entry[key])) > 50:
                    value_preview += "..."
                print(f"  - {key}: {value_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix format: {e}")
        return False

def validate_chromadb_compatibility(data: List[Dict[str, Any]]) -> bool:
    """Validate ChromaDB compatibility"""
    print("\n🔍 Validating ChromaDB compatibility:")
    
    compatible = True
    
    for i, entry in enumerate(data):
        for key, value in entry.items():
            if not isinstance(value, (str, int, float, bool, type(None))):
                print(f"❌ Entry {i+1}, field '{key}': {type(value)} not compatible")
                compatible = False
                break
    
    if compatible:
        print("✅ All entries are ChromaDB compatible")
    else:
        print("❌ Dataset has compatibility issues")
    
    return compatible

def main():
    """Main function"""
    print("🔧 Ichikara Dataset Format Fixer")
    print("=" * 50)
    print("📝 Preserving ALL original data, only fixing format issues")
    print("")
    
    # File paths
    input_file = "./data/ichikara-rag-sampleToMF-rebuilt.json"
    output_file = "./data/ichikara-rag-formatted.json"
    
    # Check input file exists
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        return
    
    # Fix format
    if not fix_ichikara_format(input_file, output_file):
        print("❌ Format fixing failed")
        return
    
    # Validate compatibility
    with open(output_file, 'r', encoding='utf-8') as f:
        fixed_data = json.load(f)
    
    validate_chromadb_compatibility(fixed_data)
    
    print(f"\n🎉 Format fixing completed successfully!")
    print(f"📁 Output file: {output_file}")
    print(f"🔐 All original data preserved, only format changed for ChromaDB compatibility")

if __name__ == "__main__":
    main()