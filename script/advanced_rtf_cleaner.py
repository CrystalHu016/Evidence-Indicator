#!/usr/bin/env python3
"""
Advanced RTF Cleaner for Ichikara Dataset
Handle complex RTF formatting and convert to clean JSON
"""

import re
import json
import os
from typing import List, Dict, Any

def advanced_rtf_clean(content: str) -> str:
    """Advanced RTF cleaning with multiple passes"""
    
    print("🧹 Advanced RTF cleaning started...")
    
    # Pass 1: Remove RTF header completely
    print("  Pass 1: Removing RTF header...")
    content = re.sub(r'^.*?\\cf0\s*', '', content, flags=re.DOTALL)
    
    # Pass 2: Remove RTF control sequences
    print("  Pass 2: Removing RTF control sequences...")
    rtf_controls = [
        r'\\f0\\fs24',
        r'\\pard',
        r'\\tx\d+',
        r'\\pardirnatural',
        r'\\partightenfactor0',
        r'\\paperw\d+',
        r'\\paperh\d+',
        r'\\margl\d+',
        r'\\margr\d+',
        r'\\vieww\d+',
        r'\\viewh\d+',
        r'\\viewkind\d+',
        r'\\cocoartf\d+',
        r'\\cocoatextscaling\d+',
        r'\\cocoaplatform\d+',
        r'\\fonttbl[^}]*',
        r'\\colortbl[^}]*',
        r'\\expandedcolortbl[^}]*'
    ]
    
    for pattern in rtf_controls:
        content = re.sub(pattern, '', content)
    
    # Pass 3: Clean up escape sequences
    print("  Pass 3: Cleaning escape sequences...")
    content = re.sub(r'\\[a-zA-Z0-9]+(?=\s|$|\\|")', '', content)
    
    # Pass 4: Fix newlines and tabs
    print("  Pass 4: Fixing newlines and tabs...")
    content = re.sub(r'\\n', '\n', content)
    content = re.sub(r'\\t', '\t', content)
    
    # Pass 5: Remove extra whitespace
    print("  Pass 5: Removing extra whitespace...")
    content = re.sub(r'\s+', ' ', content)
    
    # Pass 6: Fix quotes and escaping
    print("  Pass 6: Fixing quotes and escaping...")
    content = content.replace('\\"', '"')
    
    # Pass 7: Remove remaining RTF artifacts
    print("  Pass 7: Removing remaining RTF artifacts...")
    content = re.sub(r'\\[{}]', '', content)
    
    # Pass 8: Clean up the structure
    print("  Pass 8: Cleaning up structure...")
    content = content.strip()
    
    return content

def extract_and_reconstruct_json(content: str) -> List[Dict[str, Any]]:
    """Extract JSON data and reconstruct if needed"""
    
    print("🔍 Extracting and reconstructing JSON...")
    
    # Find the main array structure
    array_start = content.find('[')
    array_end = content.rfind(']')
    
    if array_start == -1 or array_end == -1:
        raise ValueError("Could not find JSON array structure")
    
    json_content = content[array_start:array_end + 1]
    
    # Try to find individual objects
    print("  Looking for individual JSON objects...")
    
    # Pattern to find JSON objects
    object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    objects = re.findall(object_pattern, json_content)
    
    print(f"  Found {len(objects)} potential objects")
    
    if not objects:
        # Try alternative approach - look for object boundaries
        print("  Trying alternative object detection...")
        
        # Count braces to find object boundaries
        brace_count = 0
        start_pos = -1
        objects = []
        
        for i, char in enumerate(json_content):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    obj_content = json_content[start_pos:i+1]
                    objects.append(obj_content)
                    start_pos = -1
        
        print(f"  Found {len(objects)} objects using brace counting")
    
    # Parse each object
    valid_objects = []
    for i, obj_str in enumerate(objects):
        try:
            # Clean the object string
            clean_obj = obj_str.strip()
            
            # Basic validation
            if not clean_obj.startswith('{') or not clean_obj.endswith('}'):
                print(f"    ⚠️ Object {i+1}: Invalid structure")
                continue
            
            # Try to parse
            parsed_obj = json.loads(clean_obj)
            valid_objects.append(parsed_obj)
            print(f"    ✅ Object {i+1}: Parsed successfully")
            
        except json.JSONDecodeError as e:
            print(f"    ❌ Object {i+1}: Parse failed - {e}")
            
            # Try to fix common issues
            try:
                fixed_obj = fix_json_object(clean_obj)
                if fixed_obj:
                    valid_objects.append(fixed_obj)
                    print(f"    ✅ Object {i+1}: Fixed and parsed")
            except:
                print(f"    ❌ Object {i+1}: Could not fix")
                continue
    
    print(f"  Successfully parsed {len(valid_objects)} objects")
    return valid_objects

def fix_json_object(obj_str: str) -> Dict[str, Any]:
    """Try to fix common JSON object issues"""
    
    # Fix missing quotes around keys
    obj_str = re.sub(r'(\w+):', r'"\1":', obj_str)
    
    # Fix trailing commas
    obj_str = re.sub(r',(\s*[}\]])', r'\1', obj_str)
    
    # Fix escaped quotes
    obj_str = obj_str.replace('\\"', '"')
    
    # Try to parse again
    try:
        return json.loads(obj_str)
    except:
        return None

def create_clean_dataset(original_file: str, output_file: str) -> bool:
    """Create a clean version of the dataset"""
    
    print("🚀 Starting advanced dataset cleaning...")
    print("=" * 60)
    
    try:
        # Read the original file
        print(f"📖 Reading file: {original_file}")
        with open(original_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📊 Original file size: {len(content)} characters")
        
        # Advanced cleaning
        cleaned_content = advanced_rtf_clean(content)
        print(f"🧹 Cleaned content size: {len(cleaned_content)} characters")
        
        # Extract and reconstruct JSON
        data = extract_and_reconstruct_json(cleaned_content)
        
        if not data:
            raise ValueError("No valid data extracted")
        
        print(f"✅ Extracted {len(data)} valid entries")
        
        # Validate the data structure
        print("\n🔍 Validating data structure...")
        valid_entries = 0
        
        for i, entry in enumerate(data):
            if isinstance(entry, dict) and 'ID' in entry:
                valid_entries += 1
            else:
                print(f"⚠️ Entry {i+1}: Invalid structure")
        
        print(f"✅ Valid entries: {valid_entries}/{len(data)}")
        
        # Save the clean data
        print(f"\n💾 Saving clean data to: {output_file}")
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
        
        print("\n🎉 Dataset cleaning completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during cleaning: {e}")
        return False

def main():
    """Main function"""
    
    input_file = "./data/ichikara-rag-sampleToMF.json"
    output_file = "./data/ichikara-rag-sampleToMF-clean.json"
    
    success = create_clean_dataset(input_file, output_file)
    
    if success:
        print(f"\n✅ Clean dataset saved to: {output_file}")
        print("🔧 You can now use this clean file with your RAG system")
    else:
        print("\n❌ Dataset cleaning failed")
        print("🔧 Consider re-exporting the dataset from the original source")
    
    return success

if __name__ == "__main__":
    main()
