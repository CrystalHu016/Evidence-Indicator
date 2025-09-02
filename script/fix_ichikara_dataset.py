#!/usr/bin/env python3
"""
Fix Ichikara Dataset Format Issues
Convert RTF-formatted JSON to clean JSON format
"""

import re
import json
import os
from typing import List, Dict, Any

def clean_rtf_content(content: str) -> str:
    """Remove RTF formatting and clean the content"""
    
    print("🔧 Cleaning RTF content...")
    
    # Remove RTF header and formatting
    content = re.sub(r'^.*?\\cf0\s*', '', content, flags=re.DOTALL)
    
    # Remove RTF escape sequences but preserve important ones
    content = re.sub(r'\\[a-zA-Z0-9]+(?=\s|$)', '', content)
    
    # Remove RTF control characters
    content = re.sub(r'\\[{}]', '', content)
    
    # Clean up newlines and formatting
    content = re.sub(r'\\n', '\n', content)
    content = re.sub(r'\\t', '\t', content)
    
    # Remove extra whitespace but preserve structure
    content = re.sub(r'\s+', ' ', content)
    
    # Clean up quotes and escaping
    content = content.replace('\\"', '"')
    
    # Remove RTF-specific formatting
    content = re.sub(r'\\f0\\fs24', '', content)
    content = re.sub(r'\\pard', '', content)
    
    # Fix common RTF artifacts
    content = re.sub(r'\\tx\d+', '', content)
    content = re.sub(r'\\pardirnatural', '', content)
    content = re.sub(r'\\partightenfactor0', '', content)
    
    return content.strip()

def extract_json_from_rtf(file_path: str) -> List[Dict[str, Any]]:
    """Extract JSON data from RTF-formatted file"""
    
    print(f"📖 Reading RTF file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📊 File size: {len(content)} characters")
    
    # Find the JSON array start
    json_start = content.find('[')
    if json_start == -1:
        raise ValueError("Could not find JSON array start '['")
    
    # Find the JSON array end
    json_end = content.rfind(']')
    if json_end == -1:
        raise ValueError("Could not find JSON array end ']'")
    
    # Extract the JSON content
    json_content = content[json_start:json_end + 1]
    
    print(f"🔍 Extracted JSON content: {len(json_content)} characters")
    
    # Clean the JSON content
    cleaned_content = clean_rtf_content(json_content)
    
    print(f"🧹 Cleaned content: {len(cleaned_content)} characters")
    
    # Show first 500 characters of cleaned content for debugging
    print(f"🔍 First 500 chars of cleaned content:")
    print(cleaned_content[:500])
    print("...")
    
    # Try to fix the content step by step
    fixed_content = cleaned_content
    
    # Step 1: Fix basic structure issues
    print("🔧 Step 1: Fixing basic structure...")
    
    # Remove any remaining RTF artifacts at the beginning
    fixed_content = re.sub(r'^[^[]*', '', fixed_content)
    
    # Ensure proper array structure
    if not fixed_content.startswith('['):
        fixed_content = '[' + fixed_content
    if not fixed_content.endswith(']'):
        fixed_content = fixed_content + ']'
    
    # Step 2: Fix JSON syntax issues
    print("🔧 Step 2: Fixing JSON syntax...")
    
    # Fix trailing commas
    fixed_content = re.sub(r',(\s*[}\]])', r'\1', fixed_content)
    
    # Fix missing quotes around keys
    fixed_content = re.sub(r'(\w+):', r'"\1":', fixed_content)
    
    # Fix escaped quotes
    fixed_content = fixed_content.replace('\\"', '"')
    
    # Step 3: Try parsing with different approaches
    print("🔧 Step 3: Attempting to parse...")
    
    # Try the fixed content first
    try:
        data = json.loads(fixed_content)
        print(f"✅ Successfully parsed JSON with {len(data)} entries")
        return data
    except json.JSONDecodeError as e:
        print(f"❌ First attempt failed: {e}")
        
        # Try to extract individual entries
        print("🔧 Attempting to extract individual entries...")
        
        # Find all entry objects
        entry_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        entries = re.findall(entry_pattern, fixed_content)
        
        print(f"🔍 Found {len(entries)} potential entries")
        
        if entries:
            # Try to parse each entry individually
            valid_entries = []
            for i, entry in enumerate(entries):
                try:
                    # Clean up the entry
                    clean_entry = entry.strip()
                    if clean_entry.startswith('{') and clean_entry.endswith('}'):
                        # Try to parse as JSON
                        parsed_entry = json.loads(clean_entry)
                        valid_entries.append(parsed_entry)
                        print(f"✅ Entry {i+1} parsed successfully")
                    else:
                        print(f"⚠️ Entry {i+1} has invalid structure")
                except json.JSONDecodeError as e:
                    print(f"❌ Entry {i+1} failed to parse: {e}")
                    continue
            
            if valid_entries:
                print(f"✅ Successfully parsed {len(valid_entries)} entries")
                return valid_entries
        
        # If all else fails, try manual reconstruction
        print("🔧 Attempting manual reconstruction...")
        
        # This is a fallback approach - we'll need to manually fix the content
        raise ValueError("Automatic parsing failed. Manual intervention required.")

def validate_dataset_structure(data: List[Dict[str, Any]]) -> bool:
    """Validate the structure of the cleaned dataset"""
    
    print("\n🔍 Validating dataset structure...")
    
    required_fields = ['ID', 'text', 'output', 'meta']
    valid_entries = 0
    
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            print(f"❌ Entry {i} is not a dictionary")
            continue
            
        missing_fields = [field for field in required_fields if field not in entry]
        if missing_fields:
            print(f"❌ Entry {i} missing fields: {missing_fields}")
            continue
            
        # Check meta structure
        if 'meta' in entry and isinstance(entry['meta'], dict):
            meta_fields = ['misc', 'output-reference', 'output-lines']
            missing_meta = [field for field in meta_fields if field not in entry['meta']]
            if missing_meta:
                print(f"⚠️ Entry {i} missing meta fields: {missing_meta}")
        
        valid_entries += 1
    
    print(f"✅ Valid entries: {valid_entries}/{len(data)}")
    return valid_entries == len(data)

def save_clean_json(data: List[Dict[str, Any]], output_path: str):
    """Save the cleaned data as proper JSON"""
    
    print(f"\n💾 Saving cleaned JSON to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Cleaned JSON saved successfully")
    
    # Verify the saved file
    file_size = os.path.getsize(output_path)
    print(f"📁 Output file size: {file_size} bytes")

def main():
    """Main function to fix the dataset"""
    
    input_file = "./data/ichikara-rag-sampleToMF.json"
    output_file = "./data/ichikara-rag-sampleToMF-clean.json"
    
    print("🚀 Starting Ichikara Dataset Format Fix...")
    print("=" * 50)
    
    try:
        # Extract and clean the data
        data = extract_json_from_rtf(input_file)
        
        # Validate the structure
        if not validate_dataset_structure(data):
            print("⚠️ Dataset structure validation failed")
            return False
        
        # Save the cleaned data
        save_clean_json(data, output_file)
        
        print("\n🎉 Dataset format fix completed successfully!")
        print(f"📁 Clean file saved as: {output_file}")
        
        # Show sample of cleaned data
        if data:
            print(f"\n📊 Sample entry structure:")
            sample = data[0]
            print(f"  ID: {sample.get('ID', 'N/A')}")
            print(f"  Text length: {len(sample.get('text', ''))}")
            print(f"  Output length: {len(sample.get('output', ''))}")
            print(f"  Meta fields: {list(sample.get('meta', {}).keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during dataset fix: {e}")
        print("\n🔧 Manual intervention required.")
        print("The dataset appears to have complex RTF formatting that cannot be automatically cleaned.")
        print("Consider:")
        print("1. Re-exporting the dataset from the original source as pure JSON")
        print("2. Using a different export format")
        print("3. Manual cleaning of the specific formatting issues")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
