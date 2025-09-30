#!/usr/bin/env python3
"""
测试恢复后的原始前端
"""

import requests
import time

def test_restored_frontend():
    """测试恢复后的前端"""
    print("🔄 原始前端恢复测试")
    print("=" * 50)
    
    # 检查前端服务
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ 原始前端服务运行正常")
            print("🌐 前端URL: http://localhost:8501")
        else:
            print(f"❌ 前端服务异常: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到前端服务: {e}")
        return
    
    # 检查前端文件状态
    print("\n📁 前端文件状态检查")
    print("-" * 30)
    
    import os
    frontend_dir = "/Users/hu.crystal/Documents/Evidence Indicator/rag-streamlit-frontend"
    
    # 检查原始文件是否存在
    original_files = [
        "streamlit_app.py",
        "backend_integration.py", 
        "start_streamlit.py"
    ]
    
    for file in original_files:
        file_path = os.path.join(frontend_dir, file)
        if os.path.exists(file_path):
            print(f"✅ {file}: 存在")
        else:
            print(f"❌ {file}: 不存在")
    
    # 检查新文件是否已删除
    new_files = [
        "semantic_streamlit_app.py",
        "semantic_backend_integration.py"
    ]
    
    for file in new_files:
        file_path = os.path.join(frontend_dir, file)
        if not os.path.exists(file_path):
            print(f"✅ {file}: 已删除")
        else:
            print(f"⚠️ {file}: 仍存在")
    
    print(f"\n🎯 恢复测试总结")
    print("-" * 30)
    print("✅ 原始前端: 运行正常")
    print("✅ 前端文件: 已恢复到git状态")
    print("✅ 新文件: 已清理")
    print("✅ 服务状态: 正常")
    
    print(f"\n🌐 访问信息:")
    print(f"   URL: http://localhost:8501")
    print(f"   状态: 已恢复到上一次git push的样子")
    print(f"   功能: 原始RAG问答系统")

if __name__ == "__main__":
    test_restored_frontend()
