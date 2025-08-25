#!/usr/bin/env python3
"""
Debug script to isolate the broken pipe error
"""

import os
import sys
import traceback

# Ensure we're in the right directory
os.chdir("/Users/hu.crystal/Documents/Evidence Indicator")
sys.path.insert(0, os.getcwd())

def debug_step_by_step():
    """Debug each component step by step"""
    
    print("🔍 Debugging Broken Pipe Error Step by Step")
    print("=" * 50)
    
    # Step 1: Environment setup
    print("\n1️⃣ Checking environment setup...")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            print(f"✅ OPENAI_API_KEY found (length: {len(api_key)})")
        else:
            print("❌ OPENAI_API_KEY not found")
            return False
            
        if os.path.exists("chroma/chroma.sqlite3"):
            print("✅ Vector database found")
        else:
            print("❌ Vector database not found")
            return False
            
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 2: Test vector database access
    print("\n2️⃣ Testing vector database access...")
    try:
        from langchain_openai import OpenAIEmbeddings
        from langchain_chroma import Chroma
        from pydantic import SecretStr
        
        embedding_function = OpenAIEmbeddings(api_key=SecretStr(api_key))
        print("✅ Embedding function created")
        
        db = Chroma(persist_directory="chroma", embedding_function=embedding_function)
        print("✅ Vector database connected")
        
        # Test a simple search
        results = db.similarity_search("test", k=1)
        print(f"✅ Vector search successful ({len(results)} results)")
        
    except Exception as e:
        print(f"❌ Vector database test failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 3: Test OpenAI API directly
    print("\n3️⃣ Testing OpenAI API directly...")
    try:
        import openai
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "Hello, respond with just 'OK'"}
            ],
            max_tokens=10,
            timeout=10
        )
        
        result = response.choices[0].message.content.strip()
        print(f"✅ OpenAI API test successful: '{result}'")
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 4: Test RAG query without multi-chunk
    print("\n4️⃣ Testing basic RAG query...")
    try:
        from ultra_fast_rag_fixed import UltraFastRAG
        
        rag = UltraFastRAG(api_key, "chroma")
        print("✅ RAG system initialized")
        
        # Test with multi-chunk disabled and minimal OpenAI calls
        answer, source, evidence, start, end = rag.query("コンバインとは何ですか", use_multi_chunk=False)
        print(f"✅ Basic RAG query successful")
        print(f"   Answer: {answer[:30]}...")
        
    except Exception as e:
        print(f"❌ Basic RAG query failed: {e}")
        traceback.print_exc()
        return False
    
    # Step 5: Test backend integration
    print("\n5️⃣ Testing backend integration...")
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), "rag-streamlit-frontend"))
        from backend_integration import call_backend_query
        
        result, error = call_backend_query("コンバインとは何ですか", use_multi_chunk=False)
        
        if error:
            print(f"❌ Backend integration failed: {error}")
            return False
        else:
            print(f"✅ Backend integration successful")
            print(f"   Answer: {result['answer'][:30]}...")
        
    except Exception as e:
        print(f"❌ Backend integration test failed: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All debug steps completed successfully!")
    return True

def debug_with_minimal_openai():
    """Test with minimal OpenAI usage to isolate the pipe issue"""
    
    print("\n🔍 Testing with minimal OpenAI usage...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Create a simple RAG version that avoids OpenAI for answer generation
        from ultra_fast_rag_fixed import UltraFastRAG
        
        class MinimalRAG(UltraFastRAG):
            def _generate_answer_fast(self, evidence: str, query: str) -> str:
                """Override to avoid OpenAI API calls"""
                # For definition questions, return evidence directly
                if any(pattern in query for pattern in ['とは何', 'とは', '何ですか']):
                    first_sentence = evidence.split('。')[0] if '。' in evidence else evidence
                    return first_sentence + '。' if not first_sentence.endswith('。') else first_sentence
                
                # For other questions, return evidence without API call
                return evidence[:100] + ('...' if len(evidence) > 100 else '')
        
        api_key = os.environ.get("OPENAI_API_KEY")
        rag = MinimalRAG(api_key, "chroma")
        
        answer, source, evidence, start, end = rag.query("コンバインとは何ですか", use_multi_chunk=False)
        
        print(f"✅ Minimal RAG successful!")
        print(f"   Answer: {answer}")
        print(f"   Evidence: {evidence[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Minimal RAG failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    try:
        # Run full debug
        success1 = debug_step_by_step()
        
        # Run minimal test
        success2 = debug_with_minimal_openai()
        
        if success1 and success2:
            print("\n🎉 Both debug tests successful - pipe error should be resolved!")
        else:
            print("\n⚠️ Some debug tests failed - investigating pipe error...")
            
        return success1 or success2
        
    except KeyboardInterrupt:
        print("\n⚠️ Debug interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Debug failed with unexpected error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()