#!/usr/bin/env python3
"""
Reliable startup script for the Evidence Indicator RAG System Streamlit Frontend
"""

import subprocess
import sys
import os
import signal
import time

def signal_handler(sig, frame):
    print('\n🛑 Shutting down Streamlit...')
    sys.exit(0)

def main():
    print("🔍 Starting Evidence Indicator RAG System - Streamlit Frontend")
    print("=" * 70)
    print("📱 Application will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 70)
    print()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Change to the correct directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # Start Streamlit with the comprehensive app
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "streamlit_app.py",
            "--server.port", "8501",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--server.address", "localhost"
        ]

        print("🚀 Starting Streamlit with command:")
        print(" ".join(cmd))
        print()
        print("⏳ Please wait for Streamlit to initialize...")
        print()

        # Run Streamlit in foreground
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
            if "You can now view your Streamlit app in your browser." in line:
                print("\n" + "="*70)
                print("✅ SUCCESS! Streamlit is now running!")
                print("📱 Open your browser and go to: http://localhost:8501")
                print("="*70)
                print()

        # Wait for the process
        process.wait()

    except KeyboardInterrupt:
        print('\n🛑 Received interrupt signal')
    except Exception as e:
        print(f'❌ Error: {e}')
    finally:
        print('👋 Streamlit stopped')

if __name__ == "__main__":
    main()