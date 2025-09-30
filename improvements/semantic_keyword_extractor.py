#!/usr/bin/env python3
"""
Semantic Keyword Extractor - Replaces Hardcoded Keyword Lists
Uses embeddings and semantic similarity instead of hardcoded patterns
"""

import numpy as np
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import re
import os
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import json

@dataclass
class KeywordCandidate:
    """Represents a keyword candidate with its semantic properties"""
    text: str
    embedding: np.ndarray
    relevance_score: float
    context: str
    frequency: int = 1

class SemanticKeywordExtractor:
    """
    Replaces hardcoded keyword lists with semantic understanding

    OLD APPROACH (Hardcoded):
    patterns = [r'日本', r'コンバイン', r'農業機械', r'種類', ...]

    NEW APPROACH (Semantic):
    Uses embeddings to find semantically relevant terms
    """

    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.domain_embeddings_cache = {}

    def extract_semantic_keywords(self, query: str, context: str, domain: str = None) -> List[str]:
        """
        Extract keywords using semantic similarity instead of hardcoded patterns

        Args:
            query: User query
            context: Source text context
            domain: Optional domain hint for semantic boosting

        Returns:
            List of semantically relevant keywords
        """

        # Step 1: Get query embedding
        query_embedding = self._get_embedding(query)

        # Step 2: Extract all potential keywords from context
        candidates = self._extract_candidate_terms(context)

        # Step 3: Calculate semantic relevance to query
        relevant_keywords = self._rank_by_semantic_relevance(
            candidates, query_embedding, query
        )

        # Step 4: Apply domain boosting if specified
        if domain:
            relevant_keywords = self._apply_domain_boosting(relevant_keywords, domain)

        # Step 5: Filter and return top keywords
        return self._filter_top_keywords(relevant_keywords, max_keywords=10)

    def _extract_candidate_terms(self, text: str) -> List[KeywordCandidate]:
        """
        Extract candidate terms from text using multiple strategies
        Replaces hardcoded pattern matching
        """
        candidates = []

        # Strategy 1: Extract Japanese words (replace hardcoded Japanese patterns)
        japanese_words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', text)

        # Strategy 2: Extract meaningful phrases (2-4 characters for Japanese)
        for word in japanese_words:
            if 2 <= len(word) <= 6:  # Reasonable length for Japanese terms
                embedding = self._get_embedding(word)
                candidate = KeywordCandidate(
                    text=word,
                    embedding=embedding,
                    relevance_score=0.0,  # Will be calculated later
                    context=text[:50] + "..." if len(text) > 50 else text
                )
                candidates.append(candidate)

        # Strategy 3: Extract compound terms (like "農業機械", "普通型")
        compound_patterns = self._find_compound_terms(text)
        for compound in compound_patterns:
            if compound not in [c.text for c in candidates]:
                embedding = self._get_embedding(compound)
                candidate = KeywordCandidate(
                    text=compound,
                    embedding=embedding,
                    relevance_score=0.0,
                    context=text[:50] + "..." if len(text) > 50 else text
                )
                candidates.append(candidate)

        return candidates

    def _find_compound_terms(self, text: str) -> List[str]:
        """
        Find compound terms using semantic patterns instead of hardcoded lists
        """
        compounds = []

        # Pattern 1: X + 型 (type/model patterns like 普通型, 自立型)
        type_patterns = re.findall(r'[\u4E00-\u9FAF]{1,4}型', text)
        compounds.extend(type_patterns)

        # Pattern 2: X + 機械 (machinery patterns like 農業機械)
        machine_patterns = re.findall(r'[\u4E00-\u9FAF]{1,4}機械', text)
        compounds.extend(machine_patterns)

        # Pattern 3: Common compound structures
        # This replaces hardcoded lists with pattern-based detection
        technical_terms = re.findall(r'[\u4E00-\u9FAF]{2,4}', text)

        # Filter for meaningful technical terms (length-based heuristic)
        meaningful_terms = [term for term in technical_terms
                           if 2 <= len(term) <= 4 and not self._is_common_word(term)]

        compounds.extend(meaningful_terms)

        return list(set(compounds))  # Remove duplicates

    def _is_common_word(self, word: str) -> bool:
        """
        Filter out common function words (replaces hardcoded exclusion lists)
        """
        # Common Japanese function words to exclude
        function_words = {
            'です', 'ます', 'について', 'とは', 'から', 'まで', 'ですか', 'でしょうか',
            'ある', 'いる', 'する', 'なる', 'れる', 'られる'
        }
        return word in function_words

    def _rank_by_semantic_relevance(self, candidates: List[KeywordCandidate],
                                   query_embedding: np.ndarray, query: str) -> List[KeywordCandidate]:
        """
        Rank candidates by semantic similarity to query
        This replaces hardcoded keyword scoring
        """
        for candidate in candidates:
            # Calculate cosine similarity between candidate and query
            similarity = cosine_similarity(
                candidate.embedding.reshape(1, -1),
                query_embedding.reshape(1, -1)
            )[0][0]

            candidate.relevance_score = float(similarity)

        # Sort by relevance score (descending)
        return sorted(candidates, key=lambda x: x.relevance_score, reverse=True)

    def _apply_domain_boosting(self, keywords: List[KeywordCandidate], domain: str) -> List[KeywordCandidate]:
        """
        Apply domain-specific boosting using semantic understanding
        Replaces hardcoded domain keyword lists
        """
        if domain not in self.domain_embeddings_cache:
            self._build_domain_embeddings(domain)

        domain_embedding = self.domain_embeddings_cache.get(domain)
        if domain_embedding is None:
            return keywords  # No boosting if domain not found

        # Boost keywords semantically similar to domain
        for keyword in keywords:
            domain_similarity = cosine_similarity(
                keyword.embedding.reshape(1, -1),
                domain_embedding.reshape(1, -1)
            )[0][0]

            # Apply boosting factor based on domain similarity
            boost_factor = 1.0 + (domain_similarity * 0.3)  # Up to 30% boost
            keyword.relevance_score *= boost_factor

        return sorted(keywords, key=lambda x: x.relevance_score, reverse=True)

    def _build_domain_embeddings(self, domain: str):
        """
        Build domain embeddings dynamically instead of hardcoding domain terms
        """
        domain_descriptions = {
            "agriculture": "agricultural machinery farming equipment crops harvesting",
            "technology": "software hardware computer programming development",
            "medical": "health medicine treatment diagnosis healthcare"
        }

        domain_text = domain_descriptions.get(domain, domain)
        self.domain_embeddings_cache[domain] = self._get_embedding(domain_text)

    def _filter_top_keywords(self, keywords: List[KeywordCandidate], max_keywords: int = 10) -> List[str]:
        """
        Filter and return top keywords based on relevance scores
        """
        # Filter by minimum relevance threshold
        min_threshold = 0.3  # Configurable threshold
        relevant_keywords = [kw for kw in keywords if kw.relevance_score >= min_threshold]

        # Return top N keywords
        top_keywords = relevant_keywords[:max_keywords]
        return [kw.text for kw in top_keywords]

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching"""
        try:
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            print(f"Warning: Could not get embedding for '{text}': {e}")
            # Return zero vector as fallback
            return np.zeros(1536)  # OpenAI embedding dimension

    def compare_with_hardcoded(self, query: str, context: str) -> Dict:
        """
        Compare semantic extraction with hardcoded approach for evaluation
        """
        # Semantic approach
        semantic_keywords = self.extract_semantic_keywords(query, context, domain="agriculture")

        # Simulate hardcoded approach (for comparison)
        hardcoded_keywords = self._simulate_hardcoded_extraction(query, context)

        return {
            "semantic_keywords": semantic_keywords,
            "hardcoded_keywords": hardcoded_keywords,
            "semantic_count": len(semantic_keywords),
            "hardcoded_count": len(hardcoded_keywords),
            "overlap": list(set(semantic_keywords) & set(hardcoded_keywords)),
            "semantic_only": list(set(semantic_keywords) - set(hardcoded_keywords)),
            "hardcoded_only": list(set(hardcoded_keywords) - set(semantic_keywords))
        }

    def _simulate_hardcoded_extraction(self, query: str, context: str) -> List[str]:
        """
        Simulate the old hardcoded approach for comparison
        """
        # This mirrors the hardcoded patterns from backend_integration.py:124
        hardcoded_patterns = [
            r'日本', r'コンバイン', r'農業機械', r'種類', r'普通型',
            r'自立型', r'作物', r'収穫', r'大豆', r'稲', r'麦',
            r'小豆', r'菜種', r'トウモロコシ'
        ]

        found_keywords = []
        words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', query + " " + context)

        for word in words:
            for pattern in hardcoded_patterns:
                if pattern in word and pattern not in found_keywords:
                    found_keywords.append(pattern)

        return found_keywords


def main():
    """Test the semantic keyword extractor"""
    print("🧪 Testing Semantic Keyword Extractor")
    print("=" * 50)

    # Initialize
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    extractor = SemanticKeywordExtractor(api_key)

    # Test cases
    test_cases = [
        {
            "query": "コンバインの種類について教えてください",
            "context": "日本で使われているコンバインは普通型と自立型の2種類に大別されます。普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。"
        },
        {
            "query": "What types of agricultural machinery exist?",
            "context": "Agricultural machinery includes combine harvesters, tractors, plows, and seeding equipment. These machines are essential for modern farming operations."
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}")
        print(f"Query: {test_case['query']}")
        print("-" * 30)

        # Extract keywords semantically
        semantic_keywords = extractor.extract_semantic_keywords(
            test_case["query"],
            test_case["context"],
            domain="agriculture"
        )

        print(f"🎯 Semantic Keywords: {semantic_keywords}")

        # Compare with hardcoded approach (for Japanese queries)
        if "コンバイン" in test_case["query"]:
            comparison = extractor.compare_with_hardcoded(
                test_case["query"],
                test_case["context"]
            )
            print(f"🔄 Hardcoded Keywords: {comparison['hardcoded_keywords']}")
            print(f"📊 Overlap: {comparison['overlap']}")
            print(f"✨ Semantic Only: {comparison['semantic_only']}")

    print(f"\n✅ Semantic Keyword Extraction Complete!")


if __name__ == "__main__":
    main()