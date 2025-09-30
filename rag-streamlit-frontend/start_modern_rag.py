#!/usr/bin/env python3
"""
Modern RAG System - Streamlit Launcher
Launch the zero-hardcoded, multilingual, configuration-driven RAG system
"""

import subprocess
import sys
import os
import signal
import time
from pathlib import Path

def signal_handler(sig, frame):
    print('\n🛑 Shutting down Modern RAG System...')
    sys.exit(0)

def check_prerequisites():
    """Check if all required components are available"""
    print("🔍 Checking prerequisites...")

    # Check API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment variables")
        print("   Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return False
    else:
        print(f"✅ OPENAI_API_KEY is set (length: {len(api_key)})")

    # Check improvements directory
    current_dir = Path(__file__).parent
    improvements_dir = current_dir.parent / "improvements"

    required_files = [
        "config_driven_rag.py",
        "semantic_keyword_extractor.py",
        "llm_intent_classifier.py",
        "dynamic_context_generator.py"
    ]

    missing_files = []
    for file in required_files:
        file_path = improvements_dir / file
        if not file_path.exists():
            missing_files.append(file)
        else:
            print(f"✅ Found {file}")

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    # Check if Streamlit is installed
    try:
        import streamlit
        print(f"✅ Streamlit version {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not installed")
        print("   Install with: pip install streamlit")
        return False

    return True

def main():
    print("🚀 Modern RAG System - Web Interface")
    print("=" * 60)
    print("🎯 Zero Hardcoded Architecture")
    print("🌍 Multilingual Support")
    print("⚙️  Configuration-Driven")
    print("=" * 60)
    print()

    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return

    print("\n📱 Application will be available at: http://localhost:8502")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)

        # Start Streamlit with the modern RAG app
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "modern_rag_app.py",
            "--server.port", "8502",  # Different port to avoid conflicts
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.address", "localhost",
            "--theme.base", "light",
            "--theme.primaryColor", "#1f77b4",
            "--theme.backgroundColor", "#ffffff"
        ]

        print("🚀 Starting Modern RAG System with command:")
        print(" ".join(cmd))
        print()
        print("⏳ Initializing modern RAG components...")
        print("   - Semantic keyword extractor")
        print("   - LLM intent classifier")
        print("   - Dynamic context generator")
        print("   - Configuration-driven system")
        print()

        # Run Streamlit in foreground
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
            if "You can now view your Streamlit app in your browser." in line:
                print("\n" + "="*60)
                print("✅ SUCCESS! Modern RAG System is now running!")
                print("📱 Open your browser and go to: http://localhost:8502")
                print("🎯 Features available:")
                print("   ✅ Multilingual query processing")
                print("   ✅ Semantic keyword extraction")
                print("   ✅ LLM intent understanding")
                print("   ✅ Dynamic context generation")
                print("   ✅ Zero hardcoded values")
                print("="*60)
                print()

        # Wait for the process
        process.wait()

    except KeyboardInterrupt:
        print('\n🛑 Received interrupt signal')
    except Exception as e:
        print(f'❌ Error: {e}')
    finally:
        print('👋 Modern RAG System stopped')

if __name__ == "__main__":
    main()