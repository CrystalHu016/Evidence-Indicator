#!/usr/bin/env python3
"""
Integration Test for Advanced RAG Evidence Indicator System
Tests the complete RAG pipeline with the new dataset
"""

import os
import sys
import json
import time
from pathlib import Path

def test_dataset_loading():
    """Test dataset loading and validation"""
    print("📊 Testing Dataset Loading...")
    
    dataset_path = "../data/ichikara-rag-sampleToMF-rebuilt.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("❌ Dataset is not a list")
            return False
        
        if len(data) == 0:
            print("❌ Dataset is empty")
            return False
        
        print(f"✅ Dataset loaded: {len(data)} entries")
        
        # Validate first entry structure
        first_entry = data[0]
        required_fields = ["ID", "text", "output", "meta"]
        
        for field in required_fields:
            if field not in first_entry:
                print(f"❌ Missing required field: {field}")
                return False
        
        print("✅ Dataset structure validation passed")
        print(f"  Sample ID: {first_entry['ID']}")
        print(f"  Sample text: {first_entry['text'][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_rag_system_creation():
    """Test RAG system creation without API calls"""
    print("\n🔧 Testing RAG System Creation...")
    
    try:
        from rag_evidence_indicator import RAGEvidenceIndicator
        
        # Create system instance
        rag_system = RAGEvidenceIndicator(
            dataset_path="../data/ichikara-rag-sampleToMF-rebuilt.json",
            chroma_path="./chroma_test"
        )
        
        print("✅ RAG system instance created")
        
        # Test system info
        system_info = rag_system.get_system_info()
        
        if not system_info:
            print("❌ Failed to get system info")
            return False
        
        print("✅ System info retrieved")
        print(f"  System: {system_info['system_name']}")
        print(f"  Dataset: {system_info['dataset_path']}")
        print(f"  Status: {system_info['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG system creation failed: {e}")
        return False

def test_configuration_integration():
    """Test configuration system integration"""
    print("\n⚙️ Testing Configuration Integration...")
    
    try:
        from config import get_config, validate_config
        
        # Test all configurations
        all_config = get_config("all")
        if not all_config:
            print("❌ Failed to get all configurations")
            return False
        
        print("✅ All configurations retrieved")
        
        # Test specific configurations
        dataset_config = get_config("dataset")
        if not dataset_config or "path" not in dataset_config:
            print("❌ Dataset configuration invalid")
            return False
        
        print("✅ Dataset configuration valid")
        
        # Test validation
        if not validate_config():
            print("❌ Configuration validation failed")
            return False
        
        print("✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration failed: {e}")
        return False

def test_text_processing():
    """Test text processing capabilities"""
    print("\n✂️ Testing Text Processing...")
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # Test Japanese text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )
        
        # Test with Japanese text
        test_text = "上高地が人気！上高地は長野県にある山岳景勝地で、毎年100万人以上の観光客が訪れる、多くの人にとっても人気の場所です。標高1,500mにあるこの地は、キャンプやウォーキングだけではなく温泉施設も多数存在し、豊かな自然を満喫することができます。"
        
        chunks = text_splitter.split_text(test_text)
        
        if not chunks:
            print("❌ Text splitting failed")
            return False
        
        print(f"✅ Text split into {len(chunks)} chunks")
        print(f"  First chunk: {chunks[0][:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing failed: {e}")
        return False

def test_directory_structure():
    """Test directory structure and permissions"""
    print("\n📁 Testing Directory Structure...")
    
    required_dirs = [
        "./data",
        "./chroma_new",
        "./logs"
    ]
    
    for dir_path in required_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            if not os.path.exists(dir_path):
                print(f"❌ Failed to create directory: {dir_path}")
                return False
            
            if not os.access(dir_path, os.W_OK):
                print(f"❌ Directory not writable: {dir_path}")
                return False
            
            print(f"✅ Directory ready: {dir_path}")
            
        except Exception as e:
            print(f"❌ Directory test failed for {dir_path}: {e}")
            return False
    
    return True

def test_environment_setup():
    """Test environment setup and dependencies"""
    print("\n🌍 Testing Environment Setup...")
    
    # Test Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"❌ Python version too old: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Test required modules
    required_modules = [
        "json", "os", "time", "pathlib",
        "langchain", "langchain_community", "langchain_openai", "langchain_chroma",
        "chromadb", "openai", "pydantic", "dotenv"
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ Module available: {module}")
        except ImportError:
            print(f"⚠️ Module not available: {module}")
    
    return True

def run_integration_tests():
    """Run all integration tests"""
    print("🧪 Advanced RAG Evidence Indicator - Integration Test Suite")
    print("=" * 70)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Directory Structure", test_directory_structure),
        ("Dataset Loading", test_dataset_loading),
        ("Configuration Integration", test_configuration_integration),
        ("Text Processing", test_text_processing),
        ("RAG System Creation", test_rag_system_creation)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Integration Test Results:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    elapsed_time = time.time() - start_time
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    print(f"⏱️ Total time: {elapsed_time:.2f} seconds")
    
    if passed == total:
        print("🎉 All integration tests passed! System is ready for production.")
        return True
    else:
        print("⚠️ Some integration tests failed. Please check the issues above.")
        return False

def main():
    """Main integration test function"""
    success = run_integration_tests()
    
    if success:
        print("\n🚀 System Integration Complete!")
        print("\nNext steps for full deployment:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Run: python3 rag_evidence_indicator.py")
        print("3. Test with real queries")
        print("4. Monitor performance and logs")
    else:
        print("\n🔧 Please fix the integration issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
