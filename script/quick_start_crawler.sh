#!/bin/bash

# Yahoo!ニュース分类新闻爬虫快速启动脚本

echo "🚀 Yahoo!ニュース分类新闻爬虫快速启动"
echo "=================================="

# 检查Python版本
python_version=$(python3 --version 2>&1)
if [[ $? -eq 0 ]]; then
    echo "✅ Python版本: $python_version"
else
    echo "❌ 未找到Python3，请先安装Python3"
    exit 1
fi

# 检查pip
if command -v pip3 &> /dev/null; then
    echo "✅ pip3已安装"
else
    echo "❌ 未找到pip3，请先安装pip3"
    exit 1
fi

# 安装依赖
echo "📦 安装依赖包..."
pip3 install -r requirements_category_crawler.txt

if [[ $? -eq 0 ]]; then
    echo "✅ 依赖安装完成"
else
    echo "❌ 依赖安装失败"
    exit 1
fi

# 检查环境变量
if [[ -z "$OPENAI_API_KEY" ]]; then
    echo "⚠️  警告: 未设置OPENAI_API_KEY环境变量"
    echo "   爬虫将使用简单的查询生成方法"
    echo "   如需使用LLM生成查询，请设置OPENAI_API_KEY"
    echo ""
fi

# 运行测试
echo "🧪 运行功能测试..."
python3 test_category_crawler.py

if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 测试通过！开始运行爬虫..."
    echo ""
    
    # 运行爬虫
    python3 run_category_crawler.py
else
    echo ""
    echo "❌ 测试失败，请检查问题后重试"
    exit 1
fi
