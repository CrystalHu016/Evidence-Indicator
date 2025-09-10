"""
Configuration for Advanced RAG Evidence Indicator System
New dataset version with enhanced capabilities
"""

import os
from typing import Dict, Any

# System Configuration
SYSTEM_CONFIG = {
    "name": "RAG Evidence Indicator - Advanced",
    "version": "2.0.0",
    "description": "Enhanced RAG system using cleaned Ichikara dataset",
    "author": "AI Assistant",
    "created_date": "2025-01-15"
}

# Dataset Configuration
DATASET_CONFIG = {
    "path": "./data/ichikara-rag-sampleToMF-rebuilt.json",
    "type": "json",
    "format": "ichikara_rebuilt",
    "encoding": "utf-8",
    "validation": {
        "required_fields": ["ID", "text", "output", "meta"],
        "content_validation": True,
        "structure_validation": True
    }
}

# Vector Store Configuration
VECTORSTORE_CONFIG = {
    "type": "chromadb",
    "path": "./chroma_new",
    "collection_name": "evidence_indicator_collection",
    "embedding_model": "text-embedding-ada-002",
    "persistence": True,
    "metadata": True
}

# Text Processing Configuration
TEXT_PROCESSING_CONFIG = {
    "chunk_size": 300,
    "chunk_overlap": 100,
    "max_chunk_size": 500,
    "separators": ["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
    "japanese_support": True,
    "min_chunk_length": 50
}

# Search Configuration
SEARCH_CONFIG = {
    "default_k": 3,
    "max_k": 10,
    "similarity_threshold": 0.7,
    "reranking": False,
    "metadata_filtering": True,
    "source_filtering": True
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "batch_size": 100,
    "max_workers": 4,
    "caching": True,
    "cache_ttl": 3600,
    "optimization_level": "balanced"
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "log_file": "./logs/rag_system.log"
}

# API Configuration
API_CONFIG = {
    "openai_api_key_env": "OPENAI_API_KEY",
    "rate_limiting": True,
    "max_requests_per_minute": 60,
    "timeout": 30,
    "retry_attempts": 3
}

# Quality Control Configuration
QUALITY_CONFIG = {
    "content_filtering": True,
    "duplicate_detection": True,
    "relevance_scoring": True,
    "confidence_threshold": 0.6,
    "metadata_validation": True
}

def get_config(config_type: str = "all") -> Dict[str, Any]:
    """Get configuration by type"""
    configs = {
        "system": SYSTEM_CONFIG,
        "dataset": DATASET_CONFIG,
        "vectorstore": VECTORSTORE_CONFIG,
        "text_processing": TEXT_PROCESSING_CONFIG,
        "search": SEARCH_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "logging": LOGGING_CONFIG,
        "api": API_CONFIG,
        "quality": QUALITY_CONFIG
    }
    
    if config_type == "all":
        return configs
    elif config_type in configs:
        return configs[config_type]
    else:
        return {}

def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check dataset file exists
        if not os.path.exists(DATASET_CONFIG["path"]):
            print(f"❌ Dataset file not found: {DATASET_CONFIG['path']}")
            return False
        
        # Check if chroma directory can be created
        chroma_path = VECTORSTORE_CONFIG["path"]
        if not os.path.exists(chroma_path):
            try:
                os.makedirs(chroma_path, exist_ok=True)
                print(f"✅ Created ChromaDB directory: {chroma_path}")
            except Exception as e:
                print(f"❌ Cannot create ChromaDB directory: {e}")
                return False
        
        # Check environment variables
        if not os.getenv(API_CONFIG["openai_api_key_env"]):
            print(f"⚠️ Warning: {API_CONFIG['openai_api_key_env']} not set")
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def print_config_summary():
    """Print configuration summary"""
    print("🔧 Advanced RAG Evidence Indicator Configuration")
    print("=" * 60)
    
    print(f"📁 Dataset: {DATASET_CONFIG['path']}")
    print(f"🗄️ Vector Store: {VECTORSTORE_CONFIG['path']}")
    print(f"✂️ Chunk Size: {TEXT_PROCESSING_CONFIG['chunk_size']}")
    print(f"🔍 Default Search K: {SEARCH_CONFIG['default_k']}")
    print(f"🚀 Embedding Model: {VECTORSTORE_CONFIG['embedding_model']}")
    print(f"⚙️ Performance: {PERFORMANCE_CONFIG['optimization_level']}")

if __name__ == "__main__":
    print_config_summary()
    print("\n" + "=" * 60)
    
    if validate_config():
        print("✅ Configuration is valid and ready to use")
    else:
        print("❌ Configuration has issues that need to be resolved")