#!/usr/bin/env python3
"""
Debug RTF Content
Examine the RTF content to understand formatting issues
"""

import re

def examine_rtf_content(file_path: str):
    """Examine the RTF content structure"""
    
    print("🔍 Examining RTF content structure...")
    print("=" * 50)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📊 Total file size: {len(content)} characters")
    
    # Check file header
    print(f"\n📋 File header (first 200 chars):")
    print(content[:200])
    
    # Find JSON array markers
    array_start = content.find('[')
    array_end = content.rfind(']')
    
    if array_start != -1 and array_end != -1:
        print(f"\n🔍 JSON array found:")
        print(f"  Start position: {array_start}")
        print(f"  End position: {array_end}")
        print(f"  Array length: {array_end - array_start}")
        
        # Extract array content
        array_content = content[array_start:array_end + 1]
        print(f"\n📋 Array content (first 500 chars):")
        print(array_content[:500])
        print("...")
        
        # Look for object patterns
        print(f"\n🔍 Looking for object patterns...")
        
        # Count braces
        open_braces = array_content.count('{')
        close_braces = array_content.count('}')
        print(f"  Open braces: {open_braces}")
        print(f"  Close braces: {close_braces}")
        
        # Look for specific patterns
        id_pattern = r'"ID":\s*"[^"]*"'
        ids = re.findall(id_pattern, array_content)
        print(f"  Found {len(ids)} ID fields")
        
        if ids:
            print(f"  Sample IDs:")
            for i, id_field in enumerate(ids[:3]):
                print(f"    {i+1}: {id_field}")
        
        # Look for text fields
        text_pattern = r'"text":\s*"[^"]*"'
        texts = re.findall(text_pattern, array_content)
        print(f"  Found {len(texts)} text fields")
        
        # Look for output fields
        output_pattern = r'"output":\s*"[^"]*"'
        outputs = re.findall(output_pattern, array_content)
        print(f"  Found {len(outputs)} output fields")
        
        # Check for RTF artifacts
        print(f"\n🔍 RTF artifacts found:")
        rtf_patterns = [
            (r'\\[a-zA-Z0-9]+', 'RTF control sequences'),
            (r'\\[{}]', 'RTF braces'),
            (r'\\n', 'RTF newlines'),
            (r'\\t', 'RTF tabs'),
            (r'\\cf0', 'RTF color formatting'),
            (r'\\f0', 'RTF font formatting'),
            (r'\\fs24', 'RTF font size'),
            (r'\\pard', 'RTF paragraph formatting')
        ]
        
        for pattern, description in rtf_patterns:
            matches = re.findall(pattern, array_content)
            if matches:
                print(f"  {description}: {len(matches)} occurrences")
                if len(matches) <= 5:
                    print(f"    Examples: {matches[:3]}")
        
        # Try to find object boundaries manually
        print(f"\n🔍 Manual object boundary detection...")
        
        # Look for the first complete object
        first_obj_start = array_content.find('{')
        if first_obj_start != -1:
            print(f"  First object starts at: {first_obj_start}")
            
            # Try to find the end of the first object
            brace_count = 0
            obj_end = -1
            
            for i in range(first_obj_start, len(array_content)):
                char = array_content[i]
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        obj_end = i
                        break
            
            if obj_end != -1:
                first_obj = array_content[first_obj_start:obj_end + 1]
                print(f"  First object length: {len(first_obj)}")
                print(f"  First object content:")
                print(first_obj[:300])
                print("...")
                
                # Try to parse this object
                try:
                    import json
                    parsed_obj = json.loads(first_obj)
                    print(f"  ✅ First object parsed successfully!")
                    print(f"  Keys: {list(parsed_obj.keys())}")
                except json.JSONDecodeError as e:
                    print(f"  ❌ First object parse failed: {e}")
                    
                    # Show the problematic parts
                    print(f"  🔍 Problematic content:")
                    print(first_obj)
            else:
                print(f"  ❌ Could not find end of first object")
    
    else:
        print(f"❌ No JSON array markers found")
        print(f"  '[' found at: {array_start}")
        print(f"  ']' found at: {array_end}")

def main():
    """Main function"""
    
    file_path = "./data/ichikara-rag-sampleToMF.json"
    examine_rtf_content(file_path)

if __name__ == "__main__":
    main()
