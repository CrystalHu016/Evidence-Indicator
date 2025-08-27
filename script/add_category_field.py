#!/usr/bin/env python3
"""
为Yahoo!ニュース数据集添加缺失的category字段
"""

import json
import os
from typing import List, Dict

def determine_category(entry: Dict) -> str:
    """根据条目内容确定类别"""
    
    # 获取查询和答案内容
    query = entry.get("core_query", "").lower()
    answer = entry.get("answer", "").lower()
    
    # 富士通医疗AI相关的条目
    if any(keyword in query or keyword in answer for keyword in ["富士通", "医療", "ai", "システム"]):
        return "IT"
    
    # 根据查询内容判断类别
    if any(keyword in query for keyword in ["国内", "政治", "社会", "事件"]):
        return "国内"
    elif any(keyword in query for keyword in ["国際", "外交", "海外", "外国"]):
        return "国際"
    elif any(keyword in query for keyword in ["経済", "金融", "株価", "企業"]):
        return "経済"
    elif any(keyword in query for keyword in ["エンタメ", "芸能", "映画", "音楽"]):
        return "エンタメ"
    elif any(keyword in query for keyword in ["スポーツ", "野球", "サッカー", "オリンピック"]):
        return "スポーツ"
    elif any(keyword in query for keyword in ["it", "技術", "ai", "システム", "インターネット"]):
        return "IT"
    elif any(keyword in query for keyword in ["科学", "研究", "発見", "宇宙"]):
        return "科学"
    elif any(keyword in query for keyword in ["ライフ", "生活", "健康", "料理"]):
        return "ライフ"
    elif any(keyword in query for keyword in ["地域", "地方", "市", "県"]):
        return "地域"
    else:
        # 默认类别
        return "主要"

def add_category_field(input_file: str, output_file: str = None):
    """为数据集添加category字段"""
    
    # 读取数据集
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📖 读取数据集: {len(data)} 个条目")
    
    # 为每个条目添加category字段
    categories_added = 0
    category_counts = {}
    
    for i, entry in enumerate(data):
        # 如果已经有category字段，跳过
        if "category" in entry:
            continue
        
        # 确定类别
        category = determine_category(entry)
        entry["category"] = category
        
        # 统计类别数量
        category_counts[category] = category_counts.get(category, 0) + 1
        categories_added += 1
        
        print(f"  条目 {i+1}: 添加类别 '{category}'")
    
    print(f"✅ 总共添加了 {categories_added} 个category字段")
    
    # 显示类别统计
    print(f"\n📊 类别统计:")
    for category, count in sorted(category_counts.items()):
        print(f"   {category}: {count} 个条目")
    
    # 确定输出文件名
    if not output_file:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_with_category.json"
    
    # 保存添加了category字段的数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 添加了category字段的数据集已保存到: {output_file}")
    
    return output_file

def main():
    """主函数"""
    print("🏷️  Yahoo!ニュース数据集 - 添加Category字段")
    print("=" * 50)
    
    # 输入文件路径（使用修复后的文件）
    input_file = "yahoo_news_dataset/yahoo_news_dataset_20250827_155647_fixed.json"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    try:
        # 添加category字段
        output_file = add_category_field(input_file)
        
        if output_file:
            print(f"\n✅ 处理完成！")
            print(f"   输入文件: {input_file}")
            print(f"   输出文件: {output_file}")
            print(f"   现在所有条目都包含category字段")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")

if __name__ == "__main__":
    main()
