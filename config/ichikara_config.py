"""
Configuration file for Ichikara Dataset Integration
Centralized settings for the enhanced Japanese RAG system
"""

import os
from typing import Dict, List

# Dataset Configuration
ICHIKARA_CONFIG = {
    # File paths
    "dataset_path": "./data/ichikara-rag-sampleToMF.json",
    "chroma_path": "./chroma",
    "collection_name": "ichikara_collection",
    
    # Text processing
    "chunk_size": 300,
    "chunk_overlap": 100,
    "max_chunk_size": 500,
    
    # Search configuration
    "search_k": 3,
    "similarity_threshold": 0.7,
    "max_results": 5,
    
    # Metadata fields to extract
    "metadata_fields": [
        "id",
        "type", 
        "instruction",
        "response",
        "references",
        "timestamp",
        "misc_tags",
        "source_type",
        "chunk_id",
        "chunk_size",
        "dataset"
    ],
    
    # Content types
    "content_types": {
        "instruction": "instruction",
        "response": "response",
        "mixed": "mixed"
    },
    
    # Japanese language settings
    "japanese_settings": {
        "sentence_delimiters": ["。", "！", "？", ".", "!", "?"],
        "word_delimiters": [" ", "　", "、", "，"],
        "min_word_length": 2,
        "max_sentence_length": 200
    },
    
    # Reference handling
    "reference_settings": {
        "extract_urls": True,
        "extract_timestamps": True,
        "validate_references": False,  # Set to True in production
        "max_references": 10
    },
    
    # Quality settings
    "quality_settings": {
        "min_content_length": 50,
        "max_content_length": 2000,
        "filter_empty_content": True,
        "validate_json_structure": True
    }
}

# Enhanced RAG Configuration
ENHANCED_RAG_CONFIG = {
    # Query processing
    "query_enhancement": {
        "enable_japanese_processing": True,
        "enable_instruction_matching": True,
        "enable_metadata_search": True,
        "enable_reference_validation": True
    },
    
    # Answer generation
    "answer_generation": {
        "include_metadata": True,
        "include_references": True,
        "include_timestamps": True,
        "include_confidence_scores": True,
        "format_output": "structured"  # "structured", "simple", "detailed"
    },
    
    # Performance optimization
    "performance": {
        "enable_caching": True,
        "cache_ttl": 3600,  # 1 hour
        "batch_processing": True,
        "batch_size": 100
    }
}

# Integration settings
INTEGRATION_CONFIG = {
    # Existing system compatibility
    "compatibility": {
        "maintain_existing_format": True,
        "extend_metadata": True,
        "backward_compatible": True
    },
    
    # Data migration
    "migration": {
        "enable_auto_migration": False,
        "backup_existing_data": True,
        "validate_migration": True
    },
    
    # Monitoring and logging
    "monitoring": {
        "enable_performance_tracking": True,
        "log_queries": True,
        "log_metadata": True,
        "enable_metrics": True
    }
}

def get_config(config_type: str = "ichikara") -> Dict:
    """Get configuration by type"""
    configs = {
        "ichikara": ICHIKARA_CONFIG,
        "enhanced_rag": ENHANCED_RAG_CONFIG,
        "integration": INTEGRATION_CONFIG
    }
    return configs.get(config_type, {})

def get_dataset_path() -> str:
    """Get the dataset file path"""
    return ICHIKARA_CONFIG["dataset_path"]

def get_chroma_path() -> str:
    """Get the ChromaDB path"""
    return ICHIKARA_CONFIG["chroma_path"]

def get_collection_name() -> str:
    """Get the collection name"""
    return ICHIKARA_CONFIG["collection_name"]

def get_chunk_settings() -> Dict:
    """Get text chunking settings"""
    return {
        "chunk_size": ICHIKARA_CONFIG["chunk_size"],
        "chunk_overlap": ICHIKARA_CONFIG["chunk_overlap"],
        "max_chunk_size": ICHIKARA_CONFIG["max_chunk_size"]
    }

def get_japanese_settings() -> Dict:
    """Get Japanese language processing settings"""
    return ICHIKARA_CONFIG["japanese_settings"]

def get_reference_settings() -> Dict:
    """Get reference handling settings"""
    return ICHIKARA_CONFIG["reference_settings"]

def get_quality_settings() -> Dict:
    """Get quality control settings"""
    return ICHIKARA_CONFIG["quality_settings"]

def validate_config() -> bool:
    """Validate the configuration settings"""
    try:
        # Check if dataset file exists
        if not os.path.exists(get_dataset_path()):
            print(f"❌ Dataset file not found: {get_dataset_path()}")
            return False
        
        # Check if chroma directory is writable
        chroma_path = get_chroma_path()
        if not os.path.exists(chroma_path):
            try:
                os.makedirs(chroma_path, exist_ok=True)
            except Exception as e:
                print(f"❌ Cannot create ChromaDB directory: {e}")
                return False
        
        # Validate chunk settings
        chunk_settings = get_chunk_settings()
        if chunk_settings["chunk_size"] <= chunk_settings["chunk_overlap"]:
            print("❌ Chunk size must be greater than chunk overlap")
            return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    print("🔧 Testing Ichikara Configuration...")
    validate_config()
    
    print(f"\n📁 Dataset Path: {get_dataset_path()}")
    print(f"🗄️ ChromaDB Path: {get_chroma_path()}")
    print(f"📚 Collection Name: {get_collection_name()}")
    print(f"✂️ Chunk Settings: {get_chunk_settings()}")
