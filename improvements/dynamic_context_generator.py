#!/usr/bin/env python3
"""
Dynamic Context Generator - Replaces Hardcoded Context Templates
Uses LLM to generate context dynamically instead of fixed templates
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os
from langchain_openai import ChatOpenAI

@dataclass
class ContextEnhancement:
    """Context enhancement result"""
    enhanced_context: str
    original_chunks: List[str]
    enhancement_method: str
    coherence_score: float
    processing_time: float

class DynamicContextGenerator:
    """
    Replaces hardcoded context templates with dynamic LLM-based generation

    OLD APPROACH (Hardcoded in backend_integration.py:342-346):
    context_templates = {
        "コンバイン": "コンバインは、一台で穀物の収穫・脱穀・選別をする...",
        "普通型": "普通型は主にアメリカやヨーロッパ等大規模農業で...",
        "自立型": "自立型は収穫時に水分含有率が高い稲の収穫に..."
    }

    NEW APPROACH (Dynamic LLM):
    Generates contextually relevant text based on query and retrieved chunks
    """

    def __init__(self, openai_api_key: str, model: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0.2  # Low temperature for consistent context generation
        )

    def generate_dynamic_context(self, query: str, chunks: List[str],
                                intent: str = None) -> ContextEnhancement:
        """
        Generate dynamic context from multiple chunks

        This completely replaces the hardcoded template approach in
        _build_extended_context function
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Analyze chunks for coherent context assembly
            chunk_analysis = self._analyze_chunks(chunks, query)

            # Step 2: Generate coherent context using LLM
            enhanced_context = self._llm_context_synthesis(
                query=query,
                chunks=chunks,
                analysis=chunk_analysis,
                intent=intent
            )

            # Step 3: Validate and score the enhanced context
            coherence_score = self._calculate_coherence_score(enhanced_context, chunks)

            processing_time = time.time() - start_time

            return ContextEnhancement(
                enhanced_context=enhanced_context,
                original_chunks=chunks,
                enhancement_method="llm_dynamic",
                coherence_score=coherence_score,
                processing_time=processing_time
            )

        except Exception as e:
            print(f"⚠️ Dynamic context generation failed: {e}")
            # Fallback to simple concatenation
            fallback_context = self._fallback_context_generation(chunks)

            return ContextEnhancement(
                enhanced_context=fallback_context,
                original_chunks=chunks,
                enhancement_method="fallback_concatenation",
                coherence_score=0.5,
                processing_time=time.time() - start_time
            )

    def _analyze_chunks(self, chunks: List[str], query: str) -> Dict:
        """
        Analyze chunks for optimal context synthesis
        """
        analysis_prompt = f"""
Analyze these text chunks for context synthesis:

Query: {query}

Chunks:
{chr(10).join([f"{i+1}. {chunk}" for i, chunk in enumerate(chunks)])}

Analyze and return JSON:
{{
    "main_topics": ["topic1", "topic2", ...],
    "relationships": ["relationship between chunks"],
    "information_gaps": ["what's missing"],
    "synthesis_strategy": "how to best combine these chunks",
    "key_concepts": ["important concepts to highlight"]
}}
"""

        try:
            response = self.llm.invoke(analysis_prompt)
            return self._parse_analysis_response(response.content)
        except Exception as e:
            print(f"⚠️ Chunk analysis failed: {e}")
            return self._default_analysis(chunks)

    def _llm_context_synthesis(self, query: str, chunks: List[str],
                              analysis: Dict, intent: str = None) -> str:
        """
        Use LLM to synthesize coherent context from chunks
        """

        synthesis_prompt = self._create_synthesis_prompt(query, chunks, analysis, intent)

        try:
            response = self.llm.invoke(synthesis_prompt)
            synthesized_context = response.content.strip()

            # Clean and validate the synthesized context
            return self._clean_synthesized_context(synthesized_context)

        except Exception as e:
            print(f"⚠️ Context synthesis failed: {e}")
            return self._fallback_context_generation(chunks)

    def _create_synthesis_prompt(self, query: str, chunks: List[str],
                                analysis: Dict, intent: str = None) -> str:
        """
        Create optimized prompt for context synthesis
        """

        intent_guidance = ""
        if intent:
            intent_prompts = {
                "definition": "Focus on providing a clear, comprehensive definition",
                "classification": "Organize information into clear categories and types",
                "comparison": "Highlight differences and similarities between concepts",
                "procedure": "Structure information as clear steps or methods",
                "enumeration": "Present information as organized lists or examples"
            }
            intent_guidance = intent_prompts.get(intent, "")

        prompt = f"""
Create a coherent, comprehensive context by synthesizing these text chunks.

Query: {query}
Intent: {intent} - {intent_guidance}

Source Chunks:
{chr(10).join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks)])}

Analysis:
- Main topics: {analysis.get('main_topics', [])}
- Key concepts: {analysis.get('key_concepts', [])}
- Synthesis strategy: {analysis.get('synthesis_strategy', 'Combine logically')}

Requirements:
1. Create a single, coherent paragraph or section
2. Maintain factual accuracy from source chunks
3. Eliminate redundancy and improve flow
4. Ensure information directly supports answering the query
5. Preserve important details and relationships
6. Use natural, readable language

Return only the synthesized context text, no explanations.
"""

        return prompt

    def _clean_synthesized_context(self, context: str) -> str:
        """
        Clean and validate synthesized context
        """
        # Remove any meta-commentary
        lines = context.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            # Skip explanatory lines that aren't part of the content
            if line and not any(skip_pattern in line.lower() for skip_pattern in
                              ['here is', 'this synthesized', 'the context', 'based on']):
                cleaned_lines.append(line)

        cleaned_context = ' '.join(cleaned_lines)

        # Basic length validation
        if len(cleaned_context) < 50:
            return "情報が不足しています。" if self._is_japanese(cleaned_context) else "Insufficient information available."

        return cleaned_context

    def _is_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters"""
        return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]', text))

    def _calculate_coherence_score(self, enhanced_context: str, original_chunks: List[str]) -> float:
        """
        Calculate coherence score for the enhanced context
        """
        try:
            scoring_prompt = f"""
Rate the coherence and quality of this synthesized context:

Enhanced Context: {enhanced_context}

Original Chunks:
{chr(10).join(original_chunks)}

Rate on these criteria (0.0-1.0 each):
1. Coherence: Does the text flow logically?
2. Completeness: Does it preserve important information?
3. Readability: Is it easy to understand?
4. Accuracy: Does it maintain factual correctness?

Return only a single number between 0.0 and 1.0 representing overall quality.
"""

            response = self.llm.invoke(scoring_prompt)
            score_text = response.content.strip()

            # Extract numeric score
            score_match = re.search(r'(\d+\.?\d*)', score_text)
            if score_match:
                score = float(score_match.group(1))
                return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1

        except Exception as e:
            print(f"⚠️ Coherence scoring failed: {e}")

        # Fallback scoring based on length and content diversity
        return self._fallback_coherence_scoring(enhanced_context, original_chunks)

    def _fallback_coherence_scoring(self, context: str, chunks: List[str]) -> float:
        """
        Simple fallback coherence scoring
        """
        score = 0.5  # Base score

        # Length appropriateness
        if 100 <= len(context) <= 500:
            score += 0.2

        # Information preservation (simple keyword overlap check)
        all_chunk_text = ' '.join(chunks).lower()
        context_lower = context.lower()

        # Extract key terms and check preservation
        key_terms = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]+', all_chunk_text)
        preserved_terms = [term for term in key_terms if term in context_lower]

        if key_terms:
            preservation_ratio = len(preserved_terms) / len(key_terms)
            score += preservation_ratio * 0.3

        return min(score, 1.0)

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse LLM analysis response"""
        try:
            # Try to extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
        except:
            pass

        # Fallback parsing
        return {
            "main_topics": ["general information"],
            "key_concepts": ["key concepts"],
            "synthesis_strategy": "combine chunks coherently"
        }

    def _default_analysis(self, chunks: List[str]) -> Dict:
        """Default analysis when LLM analysis fails"""
        return {
            "main_topics": ["information synthesis"],
            "key_concepts": ["extracted from chunks"],
            "synthesis_strategy": "combine and organize logically",
            "relationships": ["sequential information flow"]
        }

    def _fallback_context_generation(self, chunks: List[str]) -> str:
        """
        Fallback context generation when LLM fails
        Simple intelligent concatenation
        """
        if not chunks:
            return "No context available."

        # Remove duplicates while preserving order
        unique_chunks = []
        seen = set()
        for chunk in chunks:
            if chunk not in seen:
                unique_chunks.append(chunk)
                seen.add(chunk)

        # Join with appropriate separators
        return ' '.join(unique_chunks)

    def compare_with_hardcoded_templates(self, query: str, chunks: List[str]) -> Dict:
        """
        Compare dynamic context generation with hardcoded template approach
        """

        # Dynamic generation
        dynamic_result = self.generate_dynamic_context(query, chunks)

        # Simulate hardcoded template approach
        hardcoded_context = self._simulate_hardcoded_context(chunks)

        return {
            "dynamic": {
                "context": dynamic_result.enhanced_context,
                "method": dynamic_result.enhancement_method,
                "coherence_score": dynamic_result.coherence_score,
                "processing_time": dynamic_result.processing_time,
                "length": len(dynamic_result.enhanced_context)
            },
            "hardcoded": {
                "context": hardcoded_context,
                "method": "template_matching",
                "coherence_score": 0.7,  # Assumed for hardcoded
                "processing_time": 0.001,  # Very fast
                "length": len(hardcoded_context)
            },
            "comparison": {
                "dynamic_more_coherent": dynamic_result.coherence_score > 0.7,
                "dynamic_more_comprehensive": len(dynamic_result.enhanced_context) > len(hardcoded_context),
                "flexibility": "dynamic handles any domain, hardcoded only agriculture"
            }
        }

    def _simulate_hardcoded_context(self, chunks: List[str]) -> str:
        """
        Simulate the hardcoded template approach from backend_integration.py
        """
        # Hardcoded templates (from original code)
        templates = {
            "コンバイン": "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
            "普通型": "普通型は主にアメリカやヨーロッパ等大規模農業で使われていて、稲・麦・大豆の他にも小豆・菜種・トウモロコシなどの幅広い作物に対応した汎用性の農業機械です。",
            "自立型": "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。"
        }

        # Find matching template (same logic as original)
        combined_chunks = ' '.join(chunks)
        for key, template in templates.items():
            if key in combined_chunks:
                return template

        # Fallback if no template matches
        return combined_chunks if combined_chunks else "No relevant information found."


def main():
    """Test the dynamic context generator"""
    print("🔄 Testing Dynamic Context Generator")
    print("🎯 Replacing Hardcoded Templates with LLM-Generated Context")
    print("=" * 70)

    # Initialize
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    generator = DynamicContextGenerator(api_key)

    # Test cases with multiple chunks
    test_cases = [
        {
            "query": "コンバインとは何ですか",
            "chunks": [
                "コンバインは、一台で穀物の収穫・脱穀・選別をする自走機能を有した農業機械です。",
                "日本で使われているコンバインは普通型と自立型の2種類に大別されます。",
                "普通型は主にアメリカやヨーロッパ等大規模農業で使われています。"
            ],
            "intent": "definition"
        },
        {
            "query": "What types of agricultural machinery exist?",
            "chunks": [
                "Agricultural machinery includes combine harvesters for grain processing.",
                "Tractors are essential for field preparation and cultivation.",
                "Plowing equipment helps prepare soil for planting."
            ],
            "intent": "classification"
        },
        {
            "query": "普通型と自立型の違いは何ですか",
            "chunks": [
                "普通型は主にアメリカやヨーロッパ等大規模農業で使われています。",
                "自立型は収穫時に水分含有率が高い稲の収穫に対応するために開発された日本独自の農業機械です。",
                "両者は用途と設計思想が異なります。"
            ],
            "intent": "comparison"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['query']}")
        print(f"Intent: {test_case['intent']}")
        print("-" * 50)

        # Generate dynamic context
        result = generator.generate_dynamic_context(
            query=test_case["query"],
            chunks=test_case["chunks"],
            intent=test_case["intent"]
        )

        print(f"🎯 Dynamic Context:\n{result.enhanced_context}\n")
        print(f"📊 Coherence Score: {result.coherence_score:.2f}")
        print(f"⏱️  Processing Time: {result.processing_time:.2f}s")

        # Compare with hardcoded approach
        comparison = generator.compare_with_hardcoded_templates(
            test_case["query"],
            test_case["chunks"]
        )

        print(f"\n🔄 Hardcoded Template Would Produce:")
        print(f"{comparison['hardcoded']['context'][:100]}...")

        print(f"\n📈 Comparison:")
        print(f"  Dynamic more coherent: {comparison['comparison']['dynamic_more_coherent']}")
        print(f"  Dynamic more comprehensive: {comparison['comparison']['dynamic_more_comprehensive']}")
        print(f"  Flexibility: {comparison['comparison']['flexibility']}")

    print(f"\n✅ Dynamic Context Generation Complete!")
    print("📈 Benefits over Hardcoded Templates:")
    print("  ✓ No hardcoded templates to maintain")
    print("  ✓ Works with any domain or language")
    print("  ✓ Adapts to query intent automatically")
    print("  ✓ Synthesizes multiple chunks coherently")
    print("  ✓ Improves information organization")


if __name__ == "__main__":
    main()