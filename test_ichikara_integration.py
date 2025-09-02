#!/usr/bin/env python3
"""
Quick test script for Ichikara dataset integration
Run this to verify the integration is working properly
"""

import sys
import os

# Add the script directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'script'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

def test_configuration():
    """Test the configuration system"""
    print("🔧 Testing Configuration System...")
    
    try:
        from config.ichikara_config import validate_config, get_dataset_path, get_chunk_settings
        
        # Test configuration validation
        if validate_config():
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            return False
        
        # Test configuration access
        dataset_path = get_dataset_path()
        chunk_settings = get_chunk_settings()
        
        print(f"📁 Dataset Path: {dataset_path}")
        print(f"✂️ Chunk Settings: {chunk_settings}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading capabilities"""
    print("\n📚 Testing Dataset Loading...")
    
    try:
        from script.ichikara_dataset_integration import IchikaraDatasetIntegrator
        
        # Initialize integrator
        integrator = IchikaraDatasetIntegrator()
        print("✅ Integrator initialized successfully")
        
        # Test dataset loading
        documents = integrator.load_ichikara_dataset("./data/ichikara-rag-sampleToMF.json")
        print(f"✅ Loaded {len(documents)} documents")
        
        # Test chunk creation
        chunks = integrator.create_enhanced_chunks(documents)
        print(f"✅ Created {len(chunks)} enhanced chunks")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        return False

def test_query_capabilities():
    """Test basic query capabilities"""
    print("\n🔍 Testing Query Capabilities...")
    
    try:
        from script.ichikara_dataset_integration import IchikaraDatasetIntegrator
        
        integrator = IchikaraDatasetIntegrator()
        
        # Test queries
        test_queries = [
            "上高地について教えて",
            "観光地の情報",
            "伝統的な治療法"
        ]
        
        for query in test_queries:
            print(f"  Testing query: {query}")
            try:
                answer, source, evidence, start, end = integrator.query_ichikara_dataset(query)
                if answer and answer != "情報が見つかりませんでした。":
                    print(f"    ✅ Query successful (answer length: {len(answer)})")
                else:
                    print(f"    ⚠️ Query returned no results")
            except Exception as e:
                print(f"    ❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query capabilities test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Ichikara Dataset Integration Tests...\n")
    
    tests = [
        ("Configuration System", test_configuration),
        ("Dataset Loading", test_dataset_loading),
        ("Query Capabilities", test_query_capabilities)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🧪 Running {test_name} Test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print("=" * 50)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ichikara integration is working correctly.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the configuration and setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
