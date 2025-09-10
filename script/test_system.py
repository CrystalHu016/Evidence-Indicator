#!/usr/bin/env python3
"""
Test script for Advanced RAG Evidence Indicator System
Tests all major components and functionality
"""

import os
import sys
import json
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        from rag_evidence_indicator import RAGEvidenceIndicator
        print("✅ RAGEvidenceIndicator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RAGEvidenceIndicator: {e}")
        return False
    
    try:
        from config import get_config, validate_config
        print("✅ Configuration modules imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import configuration modules: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration system"""
    print("\n🔧 Testing configuration system...")
    
    try:
        from config import get_config, validate_config
        
        # Test configuration retrieval
        all_config = get_config("all")
        if all_config:
            print("✅ Configuration retrieval successful")
        else:
            print("❌ Configuration retrieval failed")
            return False
        
        # Test specific config sections
        dataset_config = get_config("dataset")
        if dataset_config and "path" in dataset_config:
            print("✅ Dataset configuration accessible")
        else:
            print("❌ Dataset configuration not accessible")
            return False
        
        # Test configuration validation
        if validate_config():
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dataset_access():
    """Test dataset file access"""
    print("\n📁 Testing dataset access...")
    
    dataset_path = "../data/ichikara-rag-sampleToMF-rebuilt.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"✅ Dataset loaded successfully: {len(data)} entries")
            
            # Check first entry structure
            first_entry = data[0]
            required_fields = ["ID", "text", "output", "meta"]
            missing_fields = [field for field in required_fields if field not in first_entry]
            
            if not missing_fields:
                print("✅ Dataset structure validation passed")
                return True
            else:
                print(f"❌ Missing required fields: {missing_fields}")
                return False
        else:
            print("❌ Dataset is empty or invalid format")
            return False
            
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def test_system_initialization():
    """Test system initialization (without OpenAI API)"""
    print("\n🚀 Testing system initialization...")
    
    try:
        from rag_evidence_indicator import RAGEvidenceIndicator
        
        # Create system instance
        rag_system = RAGEvidenceIndicator(
            dataset_path="../data/ichikara-rag-sampleToMF-rebuilt.json",
            chroma_path="./chroma_test"
        )
        
        print("✅ System instance created successfully")
        
        # Test system info
        system_info = rag_system.get_system_info()
        if system_info and "system_name" in system_info:
            print("✅ System info retrieval successful")
            print(f"  System: {system_info['system_name']}")
            print(f"  Dataset: {system_info['dataset_path']}")
            print(f"  Status: {system_info['status']}")
        else:
            print("❌ System info retrieval failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ System initialization test failed: {e}")
        return False

def test_directory_structure():
    """Test if required directories exist or can be created"""
    print("\n📂 Testing directory structure...")
    
    required_dirs = [
        "./data",
        "./chroma_new",
        "./logs"
    ]
    
    for dir_path in required_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory ready: {dir_path}")
        except Exception as e:
            print(f"❌ Failed to create directory {dir_path}: {e}")
            return False
    
    return True

def run_all_tests():
    """Run all tests"""
    print("🧪 Advanced RAG Evidence Indicator System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration System", test_configuration),
        ("Dataset Access", test_dataset_access),
        ("System Initialization", test_system_initialization),
        ("Directory Structure", test_directory_structure)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the issues above.")
        return False

def main():
    """Main test function"""
    success = run_all_tests()
    
    if success:
        print("\n🚀 System is ready for deployment!")
        print("\nNext steps:")
        print("1. Set your OPENAI_API_KEY in environment")
        print("2. Run: python rag_evidence_indicator.py")
        print("3. Test with sample queries")
    else:
        print("\n🔧 Please fix the issues before proceeding.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
