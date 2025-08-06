#!/usr/bin/env python3
"""
Flask Application for Evidence Indicator RAG System
Deployable on AWS ECS/Fargate
"""

import os
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultra_fast_rag import UltraFastRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global RAG instance (singleton)
rag_instance = None

def get_rag_instance():
    """Get or create RAG instance (singleton pattern)"""
    global rag_instance
    if rag_instance is None:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        chroma_path = os.environ.get('CHROMA_PATH', './chroma')
        rag_instance = UltraFastRAG(api_key, chroma_path)
    
    return rag_instance

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Evidence Indicator RAG',
        'version': '1.0.0',
        'timestamp': time.time(),
        'model': 'UltraFastRAG'
    })

@app.route('/query', methods=['POST'])
def query():
    """Main query endpoint"""
    start_time = time.time()
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        query_text = data.get('query', '')
        use_advanced_rag = data.get('use_advanced_rag', True)
        
        if not query_text:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        # Get RAG instance
        rag = get_rag_instance()
        
        # Process query
        answer, source_document, evidence_text, start_char, end_char = rag.query(query_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            'answer': answer,
            'source_document': source_document,
            'evidence_text': evidence_text,
            'start_char': start_char + 1,  # Convert to 1-indexed
            'end_char': end_char,
            'processing_time': round(processing_time, 2),
            'confidence': 0.85,  # Default confidence for UltraFastRAG
            'model': 'UltraFastRAG',
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_response = {
            'error': str(e),
            'processing_time': round(processing_time, 2),
            'timestamp': time.time()
        }
        return jsonify(error_response), 500

@app.route('/batch_query', methods=['POST'])
def batch_query():
    """Batch query endpoint for multiple queries"""
    start_time = time.time()
    
    try:
        # Parse request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        queries = data.get('queries', [])
        if not queries:
            return jsonify({'error': 'Queries array is required'}), 400
        
        # Get RAG instance
        rag = get_rag_instance()
        
        results = []
        total_processing_time = 0
        
        for query_text in queries:
            query_start = time.time()
            
            # Process query
            answer, source_document, evidence_text, start_char, end_char = rag.query(query_text)
            
            query_time = time.time() - query_start
            total_processing_time += query_time
            
            results.append({
                'query': query_text,
                'answer': answer,
                'source_document': source_document,
                'evidence_text': evidence_text,
                'start_char': start_char + 1,
                'end_char': end_char,
                'processing_time': round(query_time, 2),
                'confidence': 0.85
            })
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        response = {
            'results': results,
            'total_queries': len(queries),
            'total_processing_time': round(total_time, 2),
            'average_processing_time': round(total_processing_time / len(queries), 2),
            'model': 'UltraFastRAG',
            'timestamp': time.time()
        }
        
        return jsonify(response)
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_response = {
            'error': str(e),
            'processing_time': round(processing_time, 2),
            'timestamp': time.time()
        }
        return jsonify(error_response), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Metrics endpoint for monitoring"""
    try:
        rag = get_rag_instance()
        
        # Get basic metrics
        metrics_data = {
            'service': 'Evidence Indicator RAG',
            'version': '1.0.0',
            'model': 'UltraFastRAG',
            'status': 'operational',
            'timestamp': time.time(),
            'environment': {
                'chroma_path': os.environ.get('CHROMA_PATH', './chroma'),
                'data_path': os.environ.get('DATA_PATH', './data'),
                'openai_api_key_set': bool(os.environ.get('OPENAI_API_KEY'))
            }
        }
        
        return jsonify(metrics_data)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error',
            'timestamp': time.time()
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Evidence Indicator RAG API',
        'version': '1.0.0',
        'endpoints': {
            'POST /query': 'Process a single query',
            'POST /batch_query': 'Process multiple queries',
            'GET /health': 'Health check',
            'GET /metrics': 'Service metrics'
        },
        'example_request': {
            'query': 'コンバインとは何ですか',
            'use_advanced_rag': True
        },
        'example_response': {
            'answer': 'コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。',
            'source_document': '検索ヒットのチャンクを含む文書...',
            'evidence_text': '根拠情報',
            'start_char': 1,
            'end_char': 35,
            'processing_time': 1.25,
            'confidence': 0.85
        }
    })

if __name__ == '__main__':
    # Get port from environment or default to 8000
    port = int(os.environ.get('PORT', 8000))
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Bind to all interfaces
        port=port,
        debug=os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    ) 