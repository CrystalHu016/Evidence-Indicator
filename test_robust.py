#!/usr/bin/env python3
"""
Robust test script to verify the system works without pipe errors
"""

import os
import sys
import signal

# Change to the correct directory
os.chdir("/Users/hu.crystal/Documents/Evidence Indicator")
sys.path.insert(0, os.getcwd())

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

def test_with_timeout(test_func, timeout_seconds=60):
    """Run a test function with timeout protection"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = test_func()
        signal.alarm(0)  # Cancel the alarm
        return result, None
    except TimeoutError:
        return None, "Operation timed out"
    except Exception as e:
        signal.alarm(0)
        return None, str(e)

def simple_query_test():
    """Test simple query without multi-chunk"""
    from dotenv import load_dotenv
    load_dotenv()
    
    from ultra_fast_rag_fixed import UltraFastRAG
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    rag = UltraFastRAG(api_key, "chroma")
    answer, source, evidence, start, end = rag.query("コンバインとは何ですか", use_multi_chunk=False)
    
    return {
        "answer": answer,
        "evidence": evidence,
        "source_length": len(source),
        "evidence_range": f"{start}-{end}"
    }

def backend_integration_test():
    """Test backend integration"""
    sys.path.insert(0, os.path.join(os.getcwd(), "rag-streamlit-frontend"))
    from backend_integration import call_backend_query
    
    result, error = call_backend_query("コンバインとは何ですか", use_multi_chunk=False)
    
    if error:
        raise Exception(f"Backend error: {error}")
    
    return {
        "answer": result.get("answer", ""),
        "processing_time": result.get("processing_time", 0),
        "confidence": result.get("confidence", 0)
    }

def main():
    print("🧪 Robust System Testing")
    print("=" * 40)
    
    # Test 1: Simple query
    print("\n1️⃣ Testing simple query...")
    result, error = test_with_timeout(simple_query_test, 30)
    
    if error:
        print(f"❌ Simple query failed: {error}")
        return False
    else:
        print(f"✅ Simple query successful!")
        print(f"   Answer: {result['answer'][:50]}...")
        print(f"   Evidence: {result['evidence'][:30]}...")
    
    # Test 2: Backend integration
    print("\n2️⃣ Testing backend integration...")
    result, error = test_with_timeout(backend_integration_test, 45)
    
    if error:
        print(f"❌ Backend integration failed: {error}")
        return False
    else:
        print(f"✅ Backend integration successful!")
        print(f"   Answer: {result['answer'][:50]}...")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Confidence: {result['confidence']:.2f}")
    
    print("\n" + "=" * 40)
    print("🎉 All robust tests passed!")
    print("✅ System is working correctly without pipe errors")
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)