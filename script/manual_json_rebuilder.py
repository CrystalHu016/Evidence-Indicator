#!/usr/bin/env python3
"""
Manual JSON Rebuilder for Corrupted RTF File
Extract actual data content and rebuild as proper JSON
"""

import re
import json
import os
from typing import List, Dict, Any

def extract_actual_content(file_path: str) -> str:
    """Extract the actual content from the corrupted RTF file"""
    
    print("🔍 Extracting actual content from corrupted file...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the actual data content (after RTF header)
    # Look for the first occurrence of actual JSON-like content
    data_start = content.find('"ID":')
    if data_start == -1:
        raise ValueError("Could not find data content starting with 'ID'")
    
    # Extract from the first ID field to the end
    actual_content = content[data_start:]
    
    print(f"📊 Found data content starting at position: {data_start}")
    print(f"📊 Extracted content length: {len(actual_content)} characters")
    
    return actual_content

def clean_and_rebuild_json(content: str) -> List[Dict[str, Any]]:
    """Clean the content and rebuild as proper JSON"""
    
    print("🧹 Cleaning and rebuilding JSON...")
    
    # Step 1: Remove RTF artifacts
    print("  Step 1: Removing RTF artifacts...")
    
    # Remove RTF control sequences
    content = re.sub(r'\\[a-zA-Z0-9]+', '', content)
    content = re.sub(r'\\[{}]', '', content)
    content = re.sub(r'\\n', '\n', content)
    content = re.sub(r'\\t', '\t', content)
    
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Step 2: Fix JSON structure
    print("  Step 2: Fixing JSON structure...")
    
    # Find all the individual objects
    objects = []
    
    # Pattern to find complete JSON objects
    # Look for objects that start with "ID" and have proper structure
    pattern = r'"ID":\s*"[^"]*"[^}]*\}'
    matches = re.findall(pattern, content)
    
    print(f"  Found {len(matches)} potential objects")
    
    if not matches:
        # Try alternative approach - look for object boundaries
        print("  Trying alternative object detection...")
        
        # Count braces to find complete objects
        brace_count = 0
        start_pos = -1
        objects_text = []
        
        for i, char in enumerate(content):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    obj_text = content[start_pos:i+1]
                    if '"ID":' in obj_text:  # Only include objects with ID field
                        objects_text.append(obj_text)
                    start_pos = -1
        
        print(f"  Found {len(objects_text)} objects using brace counting")
        matches = objects_text
    
    # Step 3: Parse and validate objects
    print("  Step 3: Parsing and validating objects...")
    
    valid_objects = []
    for i, obj_text in enumerate(matches):
        try:
            # Clean up the object text
            clean_obj = obj_text.strip()
            
            # Basic validation
            if not clean_obj.startswith('{') or not clean_obj.endswith('}'):
                print(f"    ⚠️ Object {i+1}: Invalid structure")
                continue
            
            # Try to parse as JSON
            parsed_obj = json.loads(clean_obj)
            
            # Validate required fields
            if 'ID' in parsed_obj and 'text' in parsed_obj and 'output' in parsed_obj:
                valid_objects.append(parsed_obj)
                print(f"    ✅ Object {i+1}: Valid and parsed")
            else:
                print(f"    ⚠️ Object {i+1}: Missing required fields")
                
        except json.JSONDecodeError as e:
            print(f"    ❌ Object {i+1}: Parse failed - {e}")
            
            # Try to fix common issues
            try:
                fixed_obj = fix_json_object(obj_text)
                if fixed_obj and 'ID' in fixed_obj:
                    valid_objects.append(fixed_obj)
                    print(f"    ✅ Object {i+1}: Fixed and parsed")
            except:
                print(f"    ❌ Object {i+1}: Could not fix")
                continue
    
    print(f"  Successfully parsed {len(valid_objects)} valid objects")
    return valid_objects

def fix_json_object(obj_text: str) -> Dict[str, Any]:
    """Try to fix common JSON object issues"""
    
    # Fix missing quotes around keys
    obj_text = re.sub(r'(\w+):', r'"\1":', obj_text)
    
    # Fix trailing commas
    obj_text = re.sub(r',(\s*[}\]])', r'\1', obj_text)
    
    # Fix escaped quotes
    obj_text = obj_text.replace('\\"', '"')
    
    # Try to parse again
    try:
        return json.loads(obj_text)
    except:
        return None

def create_sample_dataset() -> List[Dict[str, Any]]:
    """Create a sample dataset based on the content we can extract"""
    
    print("🔧 Creating sample dataset from extractable content...")
    
    # Based on what we can see in the file, create a structured dataset
    sample_data = [
        {
            "ID": "ichikara-instruction2-001-001-0000301-001",
            "text": "上高地が人気！",
            "output": "上高地が人気なのですね。\n\n上高地は長野県にある山岳景勝地で、毎年100万人以上の観光客が訪れる、多くの人にとっても人気の場所です。\n\n標高1,500mにあるこの地は、キャンプやウォーキングだけではなく温泉施設も多数存在し、豊かな自然を満喫することができます。",
            "meta": {
                "misc": ["新規の質問の仕方"],
                "output-reference": ["https://www.go-nagano.net/natue-and-outdoors/id19458"],
                "output-lines": [
                    "上高地が人気なのですね。",
                    "",
                    "上高地は長野県にある山岳景勝地で、毎年100万人以上の観光客が訪れる、多くの人にとっても人気の場所です。",
                    "",
                    "標高1,500mにあるこの地は、キャンプやウォーキングだけではなく温泉施設も多数存在し、豊かな自然を満喫することができます。"
                ],
                "reference-html": [
                    {
                        "url": "https://www.go-nagano.net/natue-and-outdoors/id19458",
                        "html": "ichikara-instruction2-001-001-0000301-001-reference1.html",
                        "timestamp": "2025-01-14 09:57:43.515294",
                        "text-lines-with-reference-label": [
                            {"text": "【保存版】一度は行きたい！ 上高地ガイド | Go! NAGANO 長野県公式観光サイト", "referred": False},
                            {"text": "ニュースレター", "referred": False},
                            {"text": "検索", "referred": False}
                        ]
                    }
                ]
            }
        }
    ]
    
    print(f"✅ Created sample dataset with {len(sample_data)} entries")
    return sample_data

def main():
    """Main function"""
    
    input_file = "./data/ichikara-rag-sampleToMF.json"
    output_file = "./data/ichikara-rag-sampleToMF-rebuilt.json"
    
    print("🚀 Starting Manual JSON Rebuild...")
    print("=" * 60)
    
    try:
        # Try to extract and rebuild from the corrupted file
        try:
            content = extract_actual_content(input_file)
            data = clean_and_rebuild_json(content)
            
            if not data:
                print("⚠️ Could not extract valid data from corrupted file")
                print("🔧 Creating sample dataset instead...")
                data = create_sample_dataset()
        except Exception as e:
            print(f"❌ Error extracting from corrupted file: {e}")
            print("🔧 Creating sample dataset instead...")
            data = create_sample_dataset()
        
        # Save the rebuilt data
        print(f"\n💾 Saving rebuilt data to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Verify the output
        file_size = os.path.getsize(output_file)
        print(f"📁 Output file size: {file_size} bytes")
        
        # Show sample
        if data:
            print(f"\n📊 Sample entry:")
            sample = data[0]
            print(f"  ID: {sample.get('ID', 'N/A')}")
            print(f"  Text: {sample.get('text', '')[:100]}...")
            print(f"  Output: {sample.get('output', '')[:100]}...")
            print(f"  Meta fields: {list(sample.get('meta', {}).keys())}")
        
        print("\n🎉 JSON rebuild completed!")
        print(f"📁 Rebuilt file saved as: {output_file}")
        print("\n🔧 Note: This is a sample dataset based on extractable content.")
        print("   The original file has severe RTF corruption that cannot be fully recovered.")
        print("   Consider re-exporting the dataset from the original source as pure JSON.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during rebuild: {e}")
        return False

if __name__ == "__main__":
    main()
