"""
Configuration file for Ichikara Dataset Integration
Centralized settings for the enhanced Japanese RAG system
"""

import os
from typing import Dict, List

# Dataset Configuration
ICHIKARA_CONFIG = {
    # File paths - Updated to use the rebuilt dataset
    "dataset_path": "./data/ichikara-rag-sampleToMF-rebuilt.json",  # Fixed dataset
    "original_dataset_path": "./data/ichikara-rag-sampleToMF.json",  # Corrupted original (for reference)
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
        "meta"
    ],
    
    # Processing options
    "enable_metadata_extraction": True,
    "enable_reference_validation": True,
    "enable_quality_metrics": True,
    
    # Performance settings
    "batch_size": 100,
    "max_workers": 4,
    "chunking_strategy": "recursive"
}

def get_dataset_path() -> str:
    """Get the path to the usable dataset file"""
    return ICHIKARA_CONFIG["dataset_path"]

def get_original_dataset_path() -> str:
    """Get the path to the original corrupted dataset file (for reference)"""
    return ICHIKARA_CONFIG["original_dataset_path"]

def get_chroma_path() -> str:
    """Get the ChromaDB storage path"""
    return ICHIKARA_CONFIG["chroma_path"]

def get_collection_name() -> str:
    """Get the ChromaDB collection name"""
    return ICHIKARA_CONFIG["collection_name"]

def get_chunk_settings() -> Dict[str, int]:
    """Get text chunking settings"""
    return {
        "chunk_size": ICHIKARA_CONFIG["chunk_size"],
        "chunk_overlap": ICHIKARA_CONFIG["chunk_overlap"],
        "max_chunk_size": ICHIKARA_CONFIG["max_chunk_size"]
    }

def get_search_settings() -> Dict[str, any]:
    """Get search configuration settings"""
    return {
        "search_k": ICHIKARA_CONFIG["search_k"],
        "similarity_threshold": ICHIKARA_CONFIG["similarity_threshold"],
        "max_results": ICHIKARA_CONFIG["max_results"]
    }

def get_metadata_fields() -> List[str]:
    """Get the list of metadata fields to extract"""
    return ICHIKARA_CONFIG["metadata_fields"]

def get_processing_options() -> Dict[str, bool]:
    """Get processing option flags"""
    return {
        "enable_metadata_extraction": ICHIKARA_CONFIG["enable_metadata_extraction"],
        "enable_reference_validation": ICHIKARA_CONFIG["enable_reference_validation"],
        "enable_quality_metrics": ICHIKARA_CONFIG["enable_quality_metrics"]
    }

def get_performance_settings() -> Dict[str, any]:
    """Get performance-related settings"""
    return {
        "batch_size": ICHIKARA_CONFIG["batch_size"],
        "max_workers": ICHIKARA_CONFIG["max_workers"],
        "chunking_strategy": ICHIKARA_CONFIG["chunking_strategy"]
    }

def validate_config() -> bool:
    """Validate the configuration settings"""
    try:
        # Check if the rebuilt dataset exists
        if not os.path.exists(get_dataset_path()):
            print(f"❌ Rebuilt dataset not found: {get_dataset_path()}")
            return False
        
        # Check if ChromaDB directory exists or can be created
        chroma_path = get_chroma_path()
        if not os.path.exists(chroma_path):
            try:
                os.makedirs(chroma_path, exist_ok=True)
                print(f"✅ Created ChromaDB directory: {chroma_path}")
            except Exception as e:
                print(f"❌ Cannot create ChromaDB directory: {e}")
                return False
        
        # Validate chunk settings
        chunk_settings = get_chunk_settings()
        if chunk_settings["chunk_size"] <= 0 or chunk_settings["chunk_overlap"] < 0:
            print("❌ Invalid chunk settings")
            return False
        
        # Validate search settings
        search_settings = get_search_settings()
        if search_settings["search_k"] <= 0 or search_settings["max_results"] <= 0:
            print("❌ Invalid search settings")
            return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def print_config_summary():
    """Print a summary of the current configuration"""
    print("🔧 Ichikara Dataset Configuration Summary")
    print("=" * 50)
    print(f"📁 Dataset Path: {get_dataset_path()}")
    print(f"📁 Original Dataset: {get_original_dataset_path()}")
    print(f"🗄️ ChromaDB Path: {get_chroma_path()}")
    print(f"📚 Collection Name: {get_collection_name()}")
    print(f"✂️ Chunk Settings: {get_chunk_settings()}")
    print(f"🔍 Search Settings: {get_search_settings()}")
    print(f"🏷️ Metadata Fields: {get_metadata_fields()}")
    print(f"⚙️ Processing Options: {get_processing_options()}")
    print(f"🚀 Performance Settings: {get_performance_settings()}")

if __name__ == "__main__":
    print_config_summary()
    print("\n" + "=" * 50)
    
    if validate_config():
        print("✅ Configuration is valid and ready to use")
    else:
        print("❌ Configuration has issues that need to be resolved")
