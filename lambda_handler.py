#!/usr/bin/env python3
"""
AWS Lambda Handler for Evidence Indicator RAG System
"""

import json
import os
import time
import boto3
from typing import Dict, Any
from ultra_fast_rag import UltraFastRAG

# Initialize AWS clients
secrets_client = boto3.client('secrets-manager')

def get_secret(secret_name: str) -> str:
    """Retrieve secret from AWS Secrets Manager"""
    try:
        response = secrets_client.get_secret_value(SecretId=secret_name)
        secret = json.loads(response['SecretString'])
        return secret.get('OPENAI_API_KEY', '')
    except Exception as e:
        print(f"Error retrieving secret: {e}")
        return os.environ.get('OPENAI_API_KEY', '')

def initialize_rag():
    """Initialize RAG system (singleton pattern)"""
    # Get OpenAI API key from Secrets Manager or environment
    api_key = get_secret('evidence-indicator/openai-key')
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    # Initialize UltraFastRAG
    chroma_path = "/tmp/chroma"  # Lambda temp directory
    return UltraFastRAG(api_key, chroma_path)

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function
    
    Expected event format:
    {
        "query": "コンバインとは何ですか",
        "use_advanced_rag": true
    }
    
    Returns:
    {
        "statusCode": 200,
        "body": {
            "answer": "回答内容",
            "source_document": "検索ヒットのチャンクを含む文書",
            "evidence_text": "根拠情報",
            "start_char": 1,
            "end_char": 35,
            "processing_time": 1.25,
            "confidence": 0.85
        }
    }
    """
    start_time = time.time()
    
    try:
        # Parse request
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else:
            body = event.get('body', {})
        
        query = body.get('query', '')
        use_advanced_rag = body.get('use_advanced_rag', True)
        
        if not query:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Query parameter is required'
                })
            }
        
        # Initialize RAG system
        rag = initialize_rag()
        
        # Process query
        answer, source_document, evidence_text, start_char, end_char = rag.query(query)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response_body = {
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
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS'
            },
            'body': json.dumps(response_body, ensure_ascii=False)
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_response = {
            'error': str(e),
            'processing_time': round(processing_time, 2),
            'timestamp': time.time()
        }
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(error_response, ensure_ascii=False)
        }

def health_check(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check endpoint"""
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'status': 'healthy',
            'service': 'Evidence Indicator RAG',
            'version': '1.0.0',
            'timestamp': time.time()
        })
    } 