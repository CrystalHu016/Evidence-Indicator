#!/usr/bin/env python3
"""
Query handler for Streamlit frontend - handles path and environment issues
"""

import os
import sys
import logging
from typing import Optional, Dict, Tuple

# Configure logging to avoid print statement issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def safe_print(message: str, level: str = "INFO"):
    """Safely print messages without causing BrokenPipeError"""
    try:
        if level == "DEBUG":
            logger.debug(message)
        elif level == "INFO":
            logger.info(message)
        elif level == "WARNING":
            logger.warning(message)
        elif level == "ERROR":
            logger.error(message)
        else:
            logger.info(message)
    except Exception as e:
        # Silently fail if logging fails
        pass

def setup_environment():
    """Setup environment and paths for RAG system"""
    
    # Get the correct directory paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Change to parent directory to ensure relative paths work
    original_cwd = os.getcwd()
    os.chdir(parent_dir)
    
    # Add parent directory to Python path
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        env_path = os.path.join(parent_dir, '.env')
        load_dotenv(env_path)
        safe_print(f"✅ Environment loaded from: {env_path}", "INFO")
    except Exception as e:
        safe_print(f"⚠️ Warning: Could not load environment: {e}", "WARNING")
    
    return original_cwd, parent_dir

def query_rag_system(query: str, use_multi_chunk: bool = True) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Query the RAG system with proper path and environment setup
    """
    
    original_cwd, parent_dir = setup_environment()
    
    try:
        # Check if necessary files exist
        chroma_path = os.path.join(parent_dir, 'chroma', 'chroma.sqlite3')
        if not os.path.exists(chroma_path):
            return None, f"Vector database not found at: {chroma_path}"
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None, "OPENAI_API_KEY not found in environment"
        
        # Import and use the RAG system
        from ultra_fast_rag_fixed import UltraFastRAG
        
        rag = UltraFastRAG(api_key, "chroma")
        answer, source_document, evidence_text, start_char, end_char = rag.query(query, use_multi_chunk)
        
        # Calculate processing time (simplified)
        import time
        processing_time = 1.0  # Placeholder since we don't track actual time here
        
        result = {
            "answer": answer,
            "source_document": source_document,
            "evidence_text": evidence_text,
            "start_char": start_char,
            "end_char": end_char,
            "processing_time": processing_time,
            "confidence": 0.95,
            "model": "UltraFastRAG (Fixed Path)",
            "timestamp": time.time()
        }
        
        return result, None
        
    except Exception as e:
        import traceback
        error_msg = f"RAG system error: {str(e)}\nTraceback: {traceback.format_exc()}"
        return None, error_msg
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)

def test_query_handler():
    """Test the query handler"""
    safe_print("🧪 Testing Query Handler", "INFO")
    safe_print("=" * 30, "INFO")
    
    result, error = query_rag_system("コンバインとは何ですか", use_multi_chunk=False)
    
    if error:
        safe_print(f"❌ Error: {error}", "ERROR")
        return False
    else:
        safe_print(f"✅ Success!", "INFO")
        safe_print(f"Answer: {result['answer'][:50]}...", "INFO")
        safe_print(f"Evidence: {result['evidence_text'][:30]}...", "INFO")
        return True

if __name__ == "__main__":
    test_query_handler()