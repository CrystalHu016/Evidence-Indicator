#!/usr/bin/env python3
"""
Test script to verify the new prompt with persona is working correctly
"""
import os
import sys
from dotenv import load_dotenv

# Add script directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'script'))

# Load environment variables
load_dotenv()

from ultra_fast_rag_semantic import PureSemanticRAG
from query_history_manager import QueryHistoryManager

def test_new_prompt():
    """Test the new prompt with persona"""
    print("=" * 80)
    print("🧪 Testing New Prompt with Persona")
    print("=" * 80)

    # Get API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        return

    # Initialize RAG system
    print("\n📦 Initializing RAG system...")
    chroma_path = os.path.join(os.path.dirname(__file__), "script", "chroma_squad_dedup")
    rag = PureSemanticRAG(api_key, chroma_path=chroma_path)

    # Initialize history manager
    print("📦 Initializing history manager...")
    db_path = os.path.join(os.path.dirname(__file__), "query_history.db")
    history_manager = QueryHistoryManager(db_path)

    # Test query
    test_query = "梅雨とは何季の一種か?"
    print(f"\n🔍 Test Query: {test_query}")
    print("-" * 80)

    # Execute query
    print("\n⏳ Executing query...")
    result = rag.query_with_answer(test_query, k=3)

    # Display results
    print("\n" + "=" * 80)
    print("📊 QUERY RESULTS")
    print("=" * 80)
    print(f"\n💬 Answer: {result['answer']}")
    print(f"\n⏱️  Processing Time: {result['processing_time']:.2f}s")
    print(f"💯 Confidence: {result['confidence']:.2f}")
    print(f"📄 Chunks Used: {result['chunks_used']}")

    # Display evidences with prompt details
    evidences = result.get('evidences', [])
    print(f"\n📌 Total Evidences Found: {len(evidences)}")

    for idx, evidence in enumerate(evidences, 1):
        print("\n" + "=" * 80)
        print(f"Evidence #{idx}")
        print("=" * 80)

        # Core term identified
        core_term = evidence.get('core_term', '')
        if core_term:
            print(f"🎯 Core Term: {core_term}")

        # Character ranges
        char_ranges = evidence.get('char_ranges', [])
        if char_ranges:
            ranges_str = ', '.join([f"{s}～{e}" for s, e in char_ranges])
            print(f"📍 Character Ranges: {ranges_str}")

        # Extracted evidence
        extracted = evidence.get('extracted_evidence', '')
        if extracted:
            print(f"📝 Extracted Text: {extracted}")

        # Similarity scores
        print(f"🎯 Similarity Score: {evidence.get('similarity_score', 0):.3f}")
        print(f"🧠 Semantic Relevance: {evidence.get('semantic_relevance', 0):.3f}")

        # Show the NEW PROMPT with persona
        prompt = evidence.get('evidence_range_prompt', '')
        if prompt:
            print("\n" + "-" * 80)
            print("🔍 EXTRACTION PROMPT (First 800 chars):")
            print("-" * 80)
            # Show first 800 characters of the prompt to verify the persona
            print(prompt[:800])
            if len(prompt) > 800:
                print("\n... (truncated)")
            print("-" * 80)

    # Save to history
    print("\n💾 Saving to history database...")
    query_id = history_manager.add_query(
        query=test_query,
        generated_answer=result['answer'],
        processing_time=result['processing_time'],
        model=result.get('model', 'PureSemanticRAG'),
        confidence=result['confidence'],
        num_chunks=result['chunks_used']
    )

    # Save evidences
    for evidence in evidences:
        history_manager.add_evidence_extraction(
            query_id=query_id,
            chunk_id=evidence.get('chunk_id', 0),
            chunk_content=evidence.get('chunk_content', ''),
            extraction_prompt=evidence.get('evidence_range_prompt', ''),
            llm_raw_response=evidence.get('llm_response', ''),
            extracted_ranges=evidence.get('char_ranges', []),
            extracted_texts=[evidence.get('extracted_evidence', '')] if evidence.get('extracted_evidence') else [],
            similarity_score=evidence.get('similarity_score', 0),
            semantic_relevance=evidence.get('semantic_relevance', 0),
            core_term=evidence.get('core_term', '')
        )

    print(f"\n✅ Query saved to database with ID: {query_id}")

    # Verify saved data
    print("\n📖 Verifying saved data from database...")
    saved_query = history_manager.get_query_details(query_id)

    if saved_query:
        print(f"✅ Query retrieved from database")
        print(f"   - Query: {saved_query['query']}")
        print(f"   - Answer: {saved_query['generated_answer'][:100]}...")
        print(f"   - Evidences saved: {len(saved_query['evidences'])}")

        # Check if the new prompt with persona is saved
        if saved_query['evidences']:
            first_evidence = saved_query['evidences'][0]
            saved_prompt = first_evidence.get('extraction_prompt', '')

            # Check if the persona is present in the saved prompt
            if "precise text analysis expert" in saved_prompt:
                print("\n✅ NEW PROMPT WITH PERSONA VERIFIED IN DATABASE!")
                print("   The persona 'precise text analysis expert' is present in saved prompt")
            else:
                print("\n⚠️  Warning: Persona not found in saved prompt")
                print(f"   Saved prompt preview: {saved_prompt[:200]}...")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)
    print("\nYou can now:")
    print("1. Start Streamlit app: cd rag-streamlit-frontend && streamlit run streamlit_app.py")
    print("2. View the query history to see the new prompt with persona")
    print("3. Check the '🔍 Extraction Prompt Instructions' section in the history")

if __name__ == "__main__":
    test_new_prompt()
