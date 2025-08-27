#!/usr/bin/env python3
"""
Run script for Yahoo!ニュース Enhanced Crawler
Yahoo!ニュース增强爬虫运行脚本
"""

import os
import sys
import time
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = [
        ('requests', 'requests'),
        ('bs4', 'beautifulsoup4'),  # 修复导入名称
        ('openai', 'openai'),
        ('dotenv', 'python-dotenv')
    ]
    
    missing_packages = []
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All required packages are installed")
    return True

def check_environment():
    """检查环境配置"""
    print("\n🔍 Checking environment configuration...")
    
    # 检查.env文件
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        print("✅ .env file found")
        
        # 检查OpenAI API key
        from dotenv import load_dotenv
        load_dotenv(env_file)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            print("✅ OpenAI API key is set")
            return True
        else:
            print("⚠️  OpenAI API key not properly configured")
            print("   Please set OPENAI_API_KEY in your .env file")
            return False
    else:
        print("❌ .env file not found")
        print("   Please create a .env file with your OpenAI API key")
        return False

def test_config():
    """测试配置文件"""
    print("\n🔧 Testing crawler configuration...")
    
    try:
        from crawler_config import CrawlerConfig
        
        config = CrawlerConfig()
        validation = config.validate_config()
        
        all_valid = True
        for key, value in validation.items():
            status = "✅" if value else "❌"
            print(f"{status} {key}: {value}")
            if not value:
                all_valid = False
        
        if all_valid:
            print("✅ Configuration validation passed")
            return True
        else:
            print("❌ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False

def run_crawler():
    """运行爬虫"""
    print("\n🚀 Starting enhanced crawler...")
    
    try:
        from enhanced_crawler import YahooNewsCrawler
        
        # 获取API key
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        
        # 创建爬虫实例
        crawler = YahooNewsCrawler(api_key)
        
        # 运行完整流程
        start_time = time.time()
        result = crawler.run_full_pipeline()
        end_time = time.time()
        
        if result:
            print(f"\n🎉 Success! Dataset generation completed in {end_time - start_time:.2f} seconds")
            print(f"📁 Dataset saved to: {result}")
            
            # 显示数据集统计
            import json
            with open(result, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            print(f"\n📊 Dataset Statistics:")
            print(f"   Total entries: {len(dataset)}")
            print(f"   File size: {os.path.getsize(result) / 1024:.2f} KB")
            print(f"   Output directory: {crawler.output_dir}")
            
            return True
        else:
            print("\n❌ Failed to generate dataset")
            return False
            
    except Exception as e:
        print(f"❌ Error running crawler: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 Yahoo!ニュース Enhanced Crawler - Test Runner")
    print("=" * 60)
    
    # 1. 检查依赖
    if not check_dependencies():
        return
    
    # 2. 检查环境
    if not check_environment():
        return
    
    # 3. 测试配置
    if not test_config():
        return
    
    # 4. 运行爬虫
    success = run_crawler()
    
    if success:
        print("\n🎊 All tests passed! Enhanced crawler is working correctly.")
        print("\n📝 Next steps:")
        print("   1. Check the generated dataset in 'yahoo_news_dataset' folder")
        print("   2. Review the LLM-generated answers")
        print("   3. Customize queries in 'crawler_config.py' if needed")
        print("   4. Integrate with your RAG system")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
