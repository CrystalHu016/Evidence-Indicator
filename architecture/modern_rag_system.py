#!/usr/bin/env python3
"""
Modern RAG System Architecture - Configuration-Driven Approach
Replaces hardcoded patterns with learned representations and dynamic processing
"""

import yaml
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

@dataclass
class QueryContext:
    """Dynamic query context - no hardcoded patterns"""
    question: str
    domain: Optional[str] = None
    language: Optional[str] = None
    embedding: Optional[List[float]] = None
    intent_scores: Optional[Dict[str, float]] = None

@dataclass
class RetrievalResult:
    """Clean result structure"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str

class TextProcessor(ABC):
    """Configurable text processing - no hardcoded rules"""

    @abstractmethod
    def process(self, text: str, config: Dict) -> str:
        pass

    @abstractmethod
    def detect_language(self, text: str) -> str:
        pass

class SemanticTextProcessor(TextProcessor):
    """LLM-powered text processing"""

    def __init__(self, llm_client):
        self.llm = llm_client

    def process(self, text: str, config: Dict) -> str:
        """Dynamic text processing based on config"""
        if config.get("normalize_whitespace", True):
            text = " ".join(text.split())

        # Use LLM for complex processing instead of regex rules
        if config.get("semantic_cleaning", False):
            text = self._llm_clean_text(text)

        return text

    def detect_language(self, text: str) -> str:
        """LLM-based language detection"""
        prompt = f"Detect the language of this text and return only the language code: {text[:100]}"
        response = self.llm.invoke(prompt)
        return response.content.strip().lower()

    def _llm_clean_text(self, text: str) -> str:
        """Use LLM for semantic text cleaning"""
        prompt = f"""Clean and normalize this text for better semantic understanding:
        Text: {text}
        Return only the cleaned text."""

        response = self.llm.invoke(prompt)
        return response.content.strip()

class ConfigurableRAGSystem:
    """Modern RAG system with minimal hardcoding"""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self._initialize_components()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _initialize_components(self):
        """Initialize components from config"""
        # Embedding model (configurable)
        embedding_config = self.config.get("embedding", {})
        self.embedding_model = self._create_embedding_model(embedding_config)

        # LLM (configurable)
        llm_config = self.config.get("llm", {})
        self.llm = self._create_llm(llm_config)

        # Text processor (minimal, configurable)
        processor_config = self.config.get("text_processing", {})
        self.text_processor = SemanticTextProcessor(self.llm)

        # Vector database
        vector_config = self.config.get("vector_db", {})
        self.vector_db = self._create_vector_db(vector_config)

    def query(self, question: str, domain: Optional[str] = None) -> Dict[str, Any]:
        """Main query interface - no hardcoded patterns"""

        # Step 1: Create query context
        context = self._create_query_context(question, domain)

        # Step 2: Semantic retrieval (no keyword matching)
        relevant_chunks = self._semantic_retrieval(context)

        # Step 3: Dynamic context enhancement
        enhanced_context = self._enhance_context(context, relevant_chunks)

        # Step 4: LLM-based answer generation
        answer = self._generate_answer(context, enhanced_context)

        return {
            "answer": answer,
            "context": enhanced_context,
            "retrieval_results": relevant_chunks,
            "metadata": {
                "language": context.language,
                "domain": context.domain,
                "processing_method": "semantic_llm"
            }
        }

    def _create_query_context(self, question: str, domain: Optional[str]) -> QueryContext:
        """Create dynamic query context using LLM"""

        # Auto-detect language
        language = self.text_processor.detect_language(question)

        # Generate embedding
        embedding = self.embedding_model.encode(question)

        # LLM-based intent analysis (replaces hardcoded patterns)
        intent_prompt = f"""Analyze this question and return a JSON with intent scores (0-1):
        Question: {question}

        Return format:
        {{
            "definition_seeking": <score>,
            "classification_seeking": <score>,
            "procedural": <score>,
            "comparative": <score>,
            "factual": <score>
        }}"""

        try:
            intent_response = self.llm.invoke(intent_prompt)
            intent_scores = json.loads(intent_response.content)
        except:
            intent_scores = {"general": 1.0}  # Fallback

        return QueryContext(
            question=question,
            domain=domain,
            language=language,
            embedding=embedding,
            intent_scores=intent_scores
        )

    def _semantic_retrieval(self, context: QueryContext) -> List[RetrievalResult]:
        """Pure semantic retrieval - no hardcoded keywords"""

        # Get domain-specific boosting from config (not hardcoded)
        domain_config = self.config.get("domains", {}).get(context.domain, {})

        # Similarity search
        raw_results = self.vector_db.similarity_search_with_score(
            context.embedding,
            k=self.config.get("retrieval", {}).get("top_k", 10)
        )

        # Convert to structured results
        results = []
        for doc, score in raw_results:
            result = RetrievalResult(
                content=doc.page_content,
                score=score,
                metadata=doc.metadata,
                source=doc.metadata.get("source", "unknown")
            )

            # Apply domain boosting if configured (not hardcoded)
            if domain_config.get("boost_relevance", False):
                result.score *= domain_config.get("boost_factor", 1.0)

            results.append(result)

        return sorted(results, key=lambda x: x.score, reverse=True)

    def _enhance_context(self, query_context: QueryContext, results: List[RetrievalResult]) -> str:
        """Dynamic context enhancement using LLM"""

        # Combine relevant chunks
        combined_content = "\n\n".join([r.content for r in results[:5]])

        # LLM-based context enhancement (replaces hardcoded templates)
        enhancement_prompt = f"""
        Enhance this context for answering the question. Make it coherent and comprehensive:

        Question: {query_context.question}
        Raw Context: {combined_content}

        Return enhanced context that directly supports answering the question.
        """

        enhanced = self.llm.invoke(enhancement_prompt)
        return enhanced.content.strip()

    def _generate_answer(self, context: QueryContext, enhanced_context: str) -> str:
        """LLM-based answer generation - handles any question type"""

        # Dynamic prompt based on intent (not hardcoded patterns)
        intent_context = ""
        if context.intent_scores:
            top_intent = max(context.intent_scores.items(), key=lambda x: x[1])
            intent_context = f"This appears to be a {top_intent[0]} question. "

        generation_prompt = f"""
        {intent_context}Answer this question based on the provided context:

        Question: {context.question}
        Context: {enhanced_context}

        Provide a clear, accurate, and comprehensive answer.
        """

        answer = self.llm.invoke(generation_prompt)
        return answer.content.strip()

    def _create_embedding_model(self, config: Dict):
        """Factory method for embedding model"""
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.get("model", "text-embedding-3-small"),
            api_key=config.get("api_key")
        )

    def _create_llm(self, config: Dict):
        """Factory method for LLM"""
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.get("model", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.1),
            api_key=config.get("api_key")
        )

    def _create_vector_db(self, config: Dict):
        """Factory method for vector database"""
        from langchain_chroma import Chroma
        # Implementation depends on your vector DB choice
        pass

# Example usage
def main():
    """Example of modern RAG system usage"""

    # Initialize with configuration
    rag = ConfigurableRAGSystem("config/rag_config.yaml")

    # Query examples (no hardcoded limitations)
    queries = [
        "What is a combine harvester?",           # English definition
        "コンバインとは何ですか",                    # Japanese definition
        "农业机械有哪些类型",                        # Chinese classification
        "Explain the differences between types", # Comparison
        "How does crop harvesting work?"         # Procedural
    ]

    for query in queries:
        result = rag.query(query, domain="agriculture")
        print(f"Q: {query}")
        print(f"A: {result['answer']}\n")

if __name__ == "__main__":
    main()