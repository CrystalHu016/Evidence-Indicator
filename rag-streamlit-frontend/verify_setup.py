#!/usr/bin/env python3
"""
Setup Verification Script for Evidence Indicator RAG System Streamlit Frontend
"""

import sys
import importlib
import os

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.9+")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'requests', 
        'dotenv',
        'pandas',
        'plotly',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_files():
    """Check if all required files exist"""
    print("\n📁 Checking files...")
    
    required_files = [
        'app.py',
        'app_enhanced.py', 
        'test_ui.py',
        'config.py',
        'requirements.txt',
        'run_streamlit.sh',
        'README.md'
    ]
    
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - OK")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def check_virtual_environment():
    """Check if virtual environment is activated"""
    print("\n🔧 Checking virtual environment...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment is activated")
        return True
    else:
        print("⚠️  Virtual environment not detected")
        print("Recommendation: Activate virtual environment with 'source streamlit-env/bin/activate'")
        return False

def test_imports():
    """Test if the main modules can be imported"""
    print("\n🧪 Testing imports...")
    
    try:
        import app
        print("✅ app.py - Import OK")
    except Exception as e:
        print(f"❌ app.py - Import failed: {e}")
        return False
    
    try:
        import app_enhanced
        print("✅ app_enhanced.py - Import OK")
    except Exception as e:
        print(f"❌ app_enhanced.py - Import failed: {e}")
        return False
    
    try:
        import test_ui
        print("✅ test_ui.py - Import OK")
    except Exception as e:
        print(f"❌ test_ui.py - Import failed: {e}")
        return False
    
    try:
        import config
        print("✅ config.py - Import OK")
    except Exception as e:
        print(f"❌ config.py - Import failed: {e}")
        return False
    
    return True

def main():
    """Main verification function"""
    print("🔍 Evidence Indicator RAG System - Streamlit Frontend Setup Verification")
    print("=" * 70)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_files(),
        check_virtual_environment(),
        test_imports()
    ]
    
    print("\n" + "=" * 70)
    print("📊 Verification Summary")
    print("=" * 70)
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! Your setup is ready.")
        print("\n🚀 To start the application:")
        print("   streamlit run app_enhanced.py")
        print("   or")
        print("   ./run_streamlit.sh")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\n💡 Common solutions:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Activate virtual environment: source streamlit-env/bin/activate")
        print("   3. Check file permissions: chmod +x run_streamlit.sh")

if __name__ == "__main__":
    main() 