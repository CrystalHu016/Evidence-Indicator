#!/usr/bin/env python3
"""
Manual JSON Extractor for RTF Ichikara Dataset
Manually extract and rebuild JSON from RTF content
"""

import json
import os
import subprocess
import re
from typing import List, Dict, Any

def extract_entries_from_text(text: str) -> List[Dict[str, Any]]:
    """Manually extract JSON entries from text using regex patterns"""
    entries = []
    
    print("🔍 Manually extracting entries using pattern matching...")
    
    # Pattern to match complete JSON objects
    # Look for patterns like "ID": "ichikara-..." to "}" 
    entry_pattern = r'"ID":\s*"([^"]+)"[^{]*?(?="ID":|$)'
    
    # More specific pattern for each field
    id_pattern = r'"ID":\s*"([^"]+)"'
    text_pattern = r'"text":\s*"([^"]*)"'
    output_pattern = r'"output":\s*"([^"]*?)"(?=,\s*"meta")'
    
    # Split text into potential entry sections
    # Look for ID markers to split entries
    id_matches = list(re.finditer(id_pattern, text))
    
    print(f"📊 Found {len(id_matches)} potential entries")
    
    for i, id_match in enumerate(id_matches):
        try:
            # Extract section for this entry
            start_pos = id_match.start()
            if i + 1 < len(id_matches):
                end_pos = id_matches[i + 1].start()
                entry_text = text[start_pos:end_pos]
            else:
                entry_text = text[start_pos:]
            
            # Extract fields from this section
            entry = {}
            
            # Extract ID
            id_val = id_match.group(1)
            entry["ID"] = id_val
            
            # Extract text
            text_match = re.search(text_pattern, entry_text)
            if text_match:
                entry["text"] = text_match.group(1)
            
            # Extract output (more complex due to multiline)
            output_match = re.search(r'"output":\s*"(.*?)"(?=,\s*"meta")', entry_text, re.DOTALL)
            if output_match:
                output_text = output_match.group(1)
                # Clean up escaped characters
                output_text = output_text.replace('\\n', '\n').replace('\\"', '"')
                entry["output"] = output_text
            
            # Extract meta section (simplified)
            meta_match = re.search(r'"meta":\s*\{(.*?)\}(?:\s*\}|\s*,|\s*$)', entry_text, re.DOTALL)
            if meta_match:
                try:
                    # Try to parse the meta section
                    meta_content = "{" + meta_match.group(1) + "}"
                    # This is complex, so we'll create a simplified meta
                    meta = {"extracted": True}
                    
                    # Try to extract some simple meta fields
                    misc_match = re.search(r'"misc":\s*\[\s*"([^"]+)"', meta_content)
                    if misc_match:
                        meta["misc"] = misc_match.group(1)
                    
                    ref_match = re.search(r'"output-reference":\s*\[\s*"([^"]+)"', meta_content)
                    if ref_match:
                        meta["output_reference"] = ref_match.group(1)
                    
                    entry["meta"] = meta
                except:
                    entry["meta"] = {"extraction_error": True}
            
            # Only add entry if it has essential fields
            if "ID" in entry and "text" in entry and "output" in entry:
                entries.append(entry)
                print(f"✅ Extracted entry {len(entries)}: {entry['ID'][:50]}...")
            else:
                print(f"⚠️ Incomplete entry {i+1}, skipping")
                
        except Exception as e:
            print(f"❌ Error extracting entry {i+1}: {e}")
            continue
    
    return entries

def create_complete_json_manually(rtf_path: str, output_path: str) -> bool:
    """Manually create JSON from RTF file"""
    try:
        print("🔄 Manual JSON Creation Started")
        print("=" * 50)
        
        # Convert RTF to text
        txt_path = "/tmp/manual_convert.txt"
        result = subprocess.run([
            'textutil', '-convert', 'txt', rtf_path, '-output', txt_path
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ RTF conversion failed: {result.stderr}")
            return False
        
        print("✅ RTF converted to text")
        
        # Read the converted text
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        print(f"📄 Text file size: {len(text)} characters")
        
        # Extract entries manually
        entries = extract_entries_from_text(text)
        
        if not entries:
            print("❌ No entries extracted")
            return False
        
        print(f"✅ Successfully extracted {len(entries)} entries")
        
        # Save JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        
        print(f"💾 JSON file saved: {output_path}")
        
        # Show summary
        print("\n📊 Extraction Summary:")
        for i, entry in enumerate(entries, 1):
            text_preview = entry.get("text", "")[:50]
            print(f"  {i}. ID: {entry.get('ID', 'Unknown')}")
            print(f"     Text: {text_preview}...")
            print(f"     Output: {len(entry.get('output', ''))} characters")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Manual extraction failed: {e}")
        return False
    
    finally:
        # Clean up
        if os.path.exists("/tmp/manual_convert.txt"):
            os.remove("/tmp/manual_convert.txt")

def main():
    """Main function"""
    print("✋ Manual JSON Extractor for Ichikara Dataset")
    print("=" * 60)
    
    rtf_path = "/Users/hu.crystal/Downloads/ichikara-rag-sampleToMF (1).json"
    output_path = "./data/ichikara-rag-manual.json"
    
    if not os.path.exists(rtf_path):
        print(f"❌ RTF file not found: {rtf_path}")
        return
    
    if create_complete_json_manually(rtf_path, output_path):
        print(f"\n🎉 Manual extraction completed!")
        print(f"📁 Output file: {output_path}")
    else:
        print(f"\n❌ Manual extraction failed!")

if __name__ == "__main__":
    main()