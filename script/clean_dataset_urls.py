#!/usr/bin/env python3
"""
清理Yahoo!ニュース数据集中的无用URL
删除通用的页面URL，每个条目只保留一个最相关的URL
"""

import json
import os
from typing import List, Dict

def clean_dataset_urls(input_file: str, output_file: str = None):
    """清理数据集中的无用URL，每个条目只保留一个URL"""
    
    # 读取原始数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📖 读取数据集: {len(data)} 个条目")
    
    # 无用的URL列表
    useless_urls = [
        "https://news.yahoo.co.jp",
        "https://news.yahoo.co.jp/topics", 
        "https://news.yahoo.co.jp/ranking"
    ]
    
    cleaned_count = 0
    
    # 清理每个条目
    for i, item in enumerate(data):
        if "original_urls" in item:
            original_urls = item["original_urls"]
            
            # 过滤掉无用的URL
            cleaned_urls = [url for url in original_urls if url not in useless_urls]
            
            # 如果清理后没有URL了，保留一个主要的URL
            if not cleaned_urls:
                cleaned_urls = ["https://news.yahoo.co.jp"]
            
            # 确保每个条目只保留一个URL（取第一个）
            if len(cleaned_urls) > 1:
                cleaned_urls = [cleaned_urls[0]]
                print(f"  条目 {i+1}: 多个URL -> 只保留第一个: {cleaned_urls[0]}")
            
            # 更新条目
            if len(cleaned_urls) != len(original_urls):
                data[i]["original_urls"] = cleaned_urls
                cleaned_count += 1
                print(f"  条目 {i+1}: 清理了 {len(original_urls)} -> {len(cleaned_urls)} 个URL")
    
    print(f"🧹 总共清理了 {cleaned_count} 个条目")
    
    # 确定输出文件名
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_single_url.json"
    
    # 保存清理后的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 清理后的数据集已保存到: {output_file}")
    
    return output_file

def main():
    """主函数"""
    print("🧹 Yahoo!ニュース数据集URL清理工具")
    print("每个条目只保留一个最相关的URL")
    print("=" * 50)
    
    # 输入文件路径
    input_file = "yahoo_news_dataset/yahoo_news_dataset_20250827_155647.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    try:
        # 清理数据集
        output_file = clean_dataset_urls(input_file)
        
        print(f"\n✅ 清理完成！")
        print(f"   输入文件: {input_file}")
        print(f"   输出文件: {output_file}")
        print(f"   现在每个条目只包含一个URL")
        
    except Exception as e:
        print(f"❌ 清理过程中出现错误: {e}")

if __name__ == "__main__":
    main()
