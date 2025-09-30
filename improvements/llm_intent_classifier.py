#!/usr/bin/env python3
"""
LLM Intent Classifier - Replaces Hardcoded Question Pattern Matching
Uses LLM to understand query intent instead of hardcoded patterns
"""

import json
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import os
from langchain_openai import ChatOpenAI

class QueryIntent(Enum):
    """Intent categories to replace hardcoded question patterns"""
    DEFINITION = "definition"          # Replaces: ['とは何', 'とは', '何ですか', '定義']
    CLASSIFICATION = "classification"  # Replaces: ['いくつ', '何種類', '何個', '種類', '分類']
    ENUMERATION = "enumeration"       # Replaces: ['どのよう', '何があり', '作物', '対応']
    PROCEDURE = "procedure"           # Replaces: ['手順', '方法', 'ステップ']
    ATTRIBUTE = "attribute"           # Replaces: ['ですか', 'でしょうか'] + ['日本独自', '独自']
    COMPARISON = "comparison"         # New: handles comparison questions
    FACTUAL = "factual"              # New: general factual questions

@dataclass
class IntentAnalysis:
    """Intent analysis result"""
    primary_intent: QueryIntent
    confidence: float
    intent_scores: Dict[str, float]
    reasoning: str
    language: str

class LLMIntentClassifier:
    """
    Replaces hardcoded question pattern matching with LLM-based intent understanding

    OLD APPROACH (Hardcoded in backend_integration.py:98-111):
    def qtype(q: str) -> str:
        if any(p in q for p in ['とは何', 'とは', '何ですか', '定義']):
            return 'definition'
        # ... more hardcoded patterns

    NEW APPROACH (LLM-based):
    Uses LLM to understand query intent dynamically
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.1  # Low temperature for consistent classification
        )

    def classify_intent(self, query: str) -> IntentAnalysis:
        """
        Classify query intent using LLM instead of hardcoded patterns

        This completely replaces the hardcoded qtype() function
        """

        # Create LLM prompt for intent classification
        classification_prompt = self._create_classification_prompt(query)

        try:
            # Get LLM response
            response = self.llm.invoke(classification_prompt)
            result_text = response.content.strip()

            # Parse JSON response
            result = self._parse_llm_response(result_text)

            # Create intent analysis
            return self._create_intent_analysis(result, query)

        except Exception as e:
            print(f"⚠️ Intent classification failed: {e}")
            # Fallback to basic classification
            return self._fallback_classification(query)

    def _create_classification_prompt(self, query: str) -> str:
        """Create prompt for LLM intent classification"""

        prompt = f"""
Analyze this query and classify its intent. Return a JSON response.

Query: "{query}"

Classify the query intent with confidence scores (0.0 to 1.0) for each category:

Intent Categories:
- definition: Asking "what is X" or requesting definitions
- classification: Asking about types, categories, or "how many kinds"
- enumeration: Asking for lists of items or examples
- procedure: Asking about steps, methods, or how to do something
- attribute: Asking about specific properties or characteristics
- comparison: Asking about differences or similarities
- factual: General factual questions not fitting other categories

Return JSON format:
{{
    "primary_intent": "<most likely intent>",
    "confidence": <0.0-1.0>,
    "intent_scores": {{
        "definition": <score>,
        "classification": <score>,
        "enumeration": <score>,
        "procedure": <score>,
        "attribute": <score>,
        "comparison": <score>,
        "factual": <score>
    }},
    "reasoning": "<brief explanation of why this intent was chosen>",
    "language": "<detected language>"
}}

Analyze the query semantically, not just by keyword matching.
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM JSON response"""

        # Extract JSON from response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If JSON parsing fails, try to extract key information
        return self._extract_basic_info(response_text)

    def _extract_basic_info(self, text: str) -> Dict:
        """Extract basic intent information if JSON parsing fails"""

        # Try to find intent keywords in response
        intent_mapping = {
            'definition': ['definition', 'define', 'what is', 'とは'],
            'classification': ['classification', 'types', 'categories', '種類', '分類'],
            'enumeration': ['enumeration', 'list', 'examples', 'what are'],
            'procedure': ['procedure', 'steps', 'how to', '方法', '手順'],
            'attribute': ['attribute', 'property', 'characteristic', '特徴'],
            'comparison': ['comparison', 'difference', 'versus', '違い'],
            'factual': ['factual', 'fact', 'information']
        }

        for intent, keywords in intent_mapping.items():
            if any(keyword in text.lower() for keyword in keywords):
                return {
                    'primary_intent': intent,
                    'confidence': 0.7,
                    'intent_scores': {intent: 0.7},
                    'reasoning': f'Detected {intent} keywords in response',
                    'language': 'auto-detected'
                }

        # Default fallback
        return {
            'primary_intent': 'factual',
            'confidence': 0.5,
            'intent_scores': {'factual': 0.5},
            'reasoning': 'Fallback classification',
            'language': 'unknown'
        }

    def _create_intent_analysis(self, result: Dict, query: str) -> IntentAnalysis:
        """Create IntentAnalysis from parsed result"""

        primary_intent_str = result.get('primary_intent', 'factual')

        try:
            primary_intent = QueryIntent(primary_intent_str)
        except ValueError:
            primary_intent = QueryIntent.FACTUAL

        return IntentAnalysis(
            primary_intent=primary_intent,
            confidence=float(result.get('confidence', 0.5)),
            intent_scores=result.get('intent_scores', {}),
            reasoning=result.get('reasoning', ''),
            language=result.get('language', 'unknown')
        )

    def _fallback_classification(self, query: str) -> IntentAnalysis:
        """Fallback classification when LLM fails"""

        # Basic pattern matching as fallback (minimal hardcoding)
        query_lower = query.lower()

        if any(pattern in query_lower for pattern in ['what is', 'とは', '何ですか']):
            intent = QueryIntent.DEFINITION
            confidence = 0.8
        elif any(pattern in query_lower for pattern in ['types', 'kinds', '種類', '分類']):
            intent = QueryIntent.CLASSIFICATION
            confidence = 0.8
        elif any(pattern in query_lower for pattern in ['how to', 'steps', '方法', '手順']):
            intent = QueryIntent.PROCEDURE
            confidence = 0.8
        else:
            intent = QueryIntent.FACTUAL
            confidence = 0.6

        return IntentAnalysis(
            primary_intent=intent,
            confidence=confidence,
            intent_scores={intent.value: confidence},
            reasoning="Fallback pattern matching",
            language="auto-detected"
        )

    def compare_with_hardcoded(self, query: str) -> Dict:
        """
        Compare LLM classification with the original hardcoded approach
        """

        # LLM classification
        llm_result = self.classify_intent(query)

        # Simulate hardcoded classification (from original backend_integration.py)
        hardcoded_result = self._simulate_hardcoded_classification(query)

        return {
            "llm_classification": {
                "intent": llm_result.primary_intent.value,
                "confidence": llm_result.confidence,
                "reasoning": llm_result.reasoning
            },
            "hardcoded_classification": hardcoded_result,
            "agreement": llm_result.primary_intent.value == hardcoded_result,
            "llm_more_confident": llm_result.confidence > 0.8
        }

    def _simulate_hardcoded_classification(self, query: str) -> str:
        """
        Simulate the original hardcoded classification for comparison
        This mirrors the exact logic from backend_integration.py:98-111
        """
        q = query.lower()

        # Original hardcoded patterns
        if any(p in q for p in ['とは何', 'とは', '何ですか', '定義']):
            return 'definition'

        # Counting/classification questions
        if any(p in q for p in ['いくつ', '何種類', '何個', '何つ', '分類']) or ('種類' in q and any(num in q for num in ['いくつ', '何', '数'])):
            return 'classification'

        # Enumeration questions
        if any(p in q for p in ['どのよう', '何があり', '作物', '対応']) and '種類' not in q:
            return 'enumeration'

        # Procedure questions
        if any(p in q for p in ['手順', '方法', 'ステップ']):
            return 'procedure'

        # Attribute questions
        if any(p in q for p in ['ですか', 'でしょうか']) and any(attr in q for attr in ['日本独自', '独自', '日本', '特徴']):
            return 'attribute'

        return 'generic'


def main():
    """Test the LLM intent classifier"""
    print("🧠 Testing LLM Intent Classifier")
    print("🔄 Replacing Hardcoded Pattern Matching with LLM Understanding")
    print("=" * 70)

    # Initialize
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    classifier = LLMIntentClassifier(api_key)

    # Test queries that match original hardcoded patterns
    test_queries = [
        # Definition queries (original: ['とは何', 'とは', '何ですか', '定義'])
        "コンバインとは何ですか",
        "What is a combine harvester?",

        # Classification queries (original: ['いくつ', '何種類', '何個', '種類', '分類'])
        "コンバインの種類はいくつありますか",
        "農業機械は何種類ありますか",

        # Enumeration queries (original: ['どのよう', '何があり', '作物', '対応'])
        "どのような作物に対応していますか",

        # Procedure queries (original: ['手順', '方法', 'ステップ'])
        "コンバインの使用方法を教えてください",

        # Attribute queries (original: ['ですか', 'でしょうか'] + ['日本独自', '独自'])
        "自立型は日本独自の機械ですか",

        # New types not handled by hardcoded patterns
        "普通型と自立型の違いは何ですか",  # Comparison
        "農業機械の歴史について教えて",    # Factual
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}: {query}")
        print("-" * 50)

        # LLM classification
        result = classifier.classify_intent(query)
        print(f"🎯 LLM Intent: {result.primary_intent.value} (confidence: {result.confidence:.2f})")
        print(f"💭 Reasoning: {result.reasoning}")

        # Compare with hardcoded approach
        comparison = classifier.compare_with_hardcoded(query)
        print(f"🔄 Hardcoded would classify as: {comparison['hardcoded_classification']}")
        print(f"🤝 Agreement: {comparison['agreement']}")

        if result.intent_scores:
            print(f"📊 All scores: {dict(sorted(result.intent_scores.items(), key=lambda x: x[1], reverse=True)[:3])}")

    print(f"\n✅ LLM Intent Classification Complete!")
    print("📈 Benefits over Hardcoded Patterns:")
    print("  ✓ Handles multiple languages automatically")
    print("  ✓ Understands semantic intent, not just keywords")
    print("  ✓ Can handle new question types without code changes")
    print("  ✓ Provides confidence scores and reasoning")
    print("  ✓ No maintenance of hardcoded pattern lists")


if __name__ == "__main__":
    main()