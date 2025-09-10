#!/usr/bin/env python3
"""
Fast RAG Options - Multiple performance configurations
提供多种性能配置选项的快速RAG系统
"""

import os
import time
from typing import Dict, Any, Tuple, List
from dotenv import load_dotenv

# Import the different RAG systems
try:
    from enhanced_rag_system import EnhancedRAGSystem
    from ultra_fast_rag import UltraFastRAG
    from llm_evidence_ranker import LLMEvidenceRanker
    SYSTEMS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Some systems not available: {e}")
    SYSTEMS_AVAILABLE = False

class FastRAGOptions:
    """Multiple RAG system configurations for different performance needs"""
    
    def __init__(self, openai_api_key: str, chroma_path: str):
        self.openai_api_key = openai_api_key
        self.chroma_path = chroma_path
        self.systems = {}
        self._initialize_systems()
    
    def _initialize_systems(self):
        """Initialize all available RAG systems"""
        if not SYSTEMS_AVAILABLE:
            return
        
        try:
            # 1. Ultra Fast RAG (Original) - Fastest
            self.systems['ultra_fast_original'] = UltraFastRAG(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path,
                use_llm_ranking=False  # Disable LLM ranking for maximum speed
            )
            print("✅ Ultra Fast RAG (Original) initialized")
            
            # 2. Ultra Fast RAG (LLM Mode) - Balanced
            self.systems['ultra_fast_llm'] = UltraFastRAG(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path,
                use_llm_ranking=True
            )
            print("✅ Ultra Fast RAG (LLM Mode) initialized")
            
            # 3. Enhanced RAG (Optimized) - Smart but faster
            self.systems['enhanced_optimized'] = EnhancedRAGSystem(
                openai_api_key=self.openai_api_key,
                chroma_path=self.chroma_path,
                model="gpt-4o-mini"
            )
            print("✅ Enhanced RAG (Optimized) initialized")
            
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
    
    def query_with_system(self, system_name: str, query: str) -> Dict[str, Any]:
        """Query using a specific system"""
        if system_name not in self.systems:
            return {"error": f"System {system_name} not available"}
        
        start_time = time.time()
        
        try:
            if system_name == 'ultra_fast_original':
                # Ultra Fast RAG (Original) - ~1-2 seconds
                answer, source_doc, evidence, start_pos, end_pos = self.systems[system_name].query(query)
                result = {
                    "answer": answer,
                    "source_document": source_doc,
                    "evidence_text": evidence,
                    "start_char": start_pos,
                    "end_char": end_pos,
                    "processing_time": time.time() - start_time,
                    "system": "Ultra Fast RAG (Original)",
                    "features": ["Vector search", "Regex evidence extraction"]
                }
                
            elif system_name == 'ultra_fast_llm':
                # Ultra Fast RAG (LLM Mode) - ~5-10 seconds
                answer, source_doc, evidence, start_pos, end_pos = self.systems[system_name].query(query, k=3)
                result = {
                    "answer": answer,
                    "source_document": source_doc,
                    "evidence_text": evidence,
                    "start_char": start_pos,
                    "end_char": end_pos,
                    "processing_time": time.time() - start_time,
                    "system": "Ultra Fast RAG (LLM Mode)",
                    "features": ["Vector search", "LLM ranking", "Smart evidence selection"]
                }
                
            elif system_name == 'enhanced_optimized':
                # Enhanced RAG (Optimized) - ~15-20 seconds (reduced from 25-35)
                result = self.systems[system_name].query(
                    query_text=query,
                    initial_k=5,  # Reduced from 8
                    final_k=2,    # Reduced from 3
                    use_llm_ranking=True
                )
                result["system"] = "Enhanced RAG (Optimized)"
                result["features"] = ["Vector search", "LLM ranking", "LLM answer generation", "Evidence highlighting"]
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "processing_time": time.time() - start_time,
                "system": system_name
            }
    
    def benchmark_all_systems(self, query: str) -> Dict[str, Any]:
        """Benchmark all systems with the same query"""
        print(f"🏁 Benchmarking all systems with query: '{query}'")
        print("=" * 60)
        
        results = {}
        
        for system_name in self.systems.keys():
            print(f"\n🚀 Testing {system_name}...")
            result = self.query_with_system(system_name, query)
            results[system_name] = result
            
            if "error" in result:
                print(f"❌ {system_name}: {result['error']}")
            else:
                print(f"✅ {system_name}: {result['processing_time']:.2f}s")
                print(f"   Answer: {result['answer'][:50]}...")
        
        return results
    
    def get_system_recommendations(self) -> Dict[str, str]:
        """Get recommendations for different use cases"""
        return {
            "ultra_fast_original": "Use for: Real-time applications, high-volume queries, when speed is critical",
            "ultra_fast_llm": "Use for: Balanced performance, good accuracy with reasonable speed",
            "enhanced_optimized": "Use for: High accuracy requirements, detailed analysis, when quality matters most"
        }

def main():
    """Demo the different RAG systems"""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return
    
    # Initialize the fast RAG options
    fast_rag = FastRAGOptions(api_key, "./chroma")
    
    # Test query
    test_query = "コンバインとは何ですか"
    
    print("🎯 Fast RAG Options Demo")
    print("=" * 60)
    
    # Show recommendations
    print("\n📋 System Recommendations:")
    recommendations = fast_rag.get_system_recommendations()
    for system, recommendation in recommendations.items():
        print(f"  {system}: {recommendation}")
    
    # Benchmark all systems
    results = fast_rag.benchmark_all_systems(test_query)
    
    # Summary
    print(f"\n📊 Performance Summary:")
    print("-" * 40)
    for system_name, result in results.items():
        if "error" not in result:
            print(f"{system_name}: {result['processing_time']:.2f}s")
    
    # Find fastest system
    fastest_system = min(
        [(name, result) for name, result in results.items() if "error" not in result],
        key=lambda x: x[1]["processing_time"]
    )
    print(f"\n🏆 Fastest system: {fastest_system[0]} ({fastest_system[1]['processing_time']:.2f}s)")

if __name__ == "__main__":
    main()
