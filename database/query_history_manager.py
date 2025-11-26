#!/usr/bin/env python3
"""
Query History Manager - Persistent storage for all RAG queries and results
Stores query history in SQLite database for analysis and debugging
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import os


class QueryHistoryManager:
    """Manages persistent storage of query history"""

    def __init__(self, db_path: str = "./query_history.db"):
        """Initialize the query history manager

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()

    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Main query history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                generated_answer TEXT,
                processing_time REAL,
                model TEXT,
                confidence REAL,
                num_chunks INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Evidence extraction details table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS evidence_extraction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                chunk_id INTEGER,
                chunk_content TEXT,
                extraction_prompt TEXT,
                llm_raw_response TEXT,
                extracted_ranges TEXT,
                extracted_texts TEXT,
                similarity_score REAL,
                semantic_relevance REAL,
                core_term TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id)
            )
        """)

        # Dataset ground truth comparison table (for evaluation)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_comparison (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id INTEGER NOT NULL,
                dataset_answer TEXT,
                dataset_answer_start INTEGER,
                dataset_answer_end INTEGER,
                rag_extracted_text TEXT,
                rag_start INTEGER,
                rag_end INTEGER,
                match_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (query_id) REFERENCES query_history(id)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_timestamp
            ON query_history(timestamp)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_evidence_query_id
            ON evidence_extraction(query_id)
        """)

        # Migration: Add core_term column if it doesn't exist
        try:
            cursor.execute("SELECT core_term FROM evidence_extraction LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            print("🔧 Migrating database: Adding core_term column to evidence_extraction table...")
            cursor.execute("ALTER TABLE evidence_extraction ADD COLUMN core_term TEXT")
            print("✅ Migration completed: core_term column added")

        # Migration: Add answer_judgment column if it doesn't exist
        try:
            cursor.execute("SELECT answer_judgment FROM query_history LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            print("🔧 Migrating database: Adding answer_judgment column to query_history table...")
            cursor.execute("ALTER TABLE query_history ADD COLUMN answer_judgment TEXT")
            print("✅ Migration completed: answer_judgment column added")

        conn.commit()
        conn.close()

        print(f"✅ Query history database initialized: {self.db_path}")

    def add_query(
        self,
        query: str,
        generated_answer: str,
        processing_time: float,
        model: str = "PureSemanticRAG",
        confidence: float = 0.0,
        num_chunks: int = 0,
        dataset_answer: str = "",
        evidences: str = "",
        answer_judgment: str = ""
    ) -> int:
        """Add a new query to history

        Args:
            query: User's question
            generated_answer: System's generated answer
            processing_time: Time taken to process (seconds)
            model: Model name used
            confidence: Confidence score
            num_chunks: Number of chunks used
            dataset_answer: Ground truth answer from dataset (optional)
            evidences: JSON string of evidence chunks (optional)
            answer_judgment: Gemini yes/no judgment on answer relevance (optional)

        Returns:
            query_id: ID of the inserted query record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        timestamp = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO query_history
            (timestamp, query, generated_answer, processing_time, model, confidence, num_chunks, dataset_answer, evidences, answer_judgment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, query, generated_answer, processing_time, model, confidence, num_chunks, dataset_answer, evidences, answer_judgment))

        query_id = cursor.lastrowid
        conn.commit()
        conn.close()

        print(f"📝 Query saved to history: ID={query_id}")
        return query_id

    def add_evidence_extraction(
        self,
        query_id: int,
        chunk_id: int,
        chunk_content: str,
        extraction_prompt: str,
        llm_raw_response: str,
        extracted_ranges: List[Dict[str, int]],
        extracted_texts: List[str],
        similarity_score: float = 0.0,
        semantic_relevance: float = 0.0,
        core_term: str = ""
    ):
        """Add evidence extraction details

        Args:
            query_id: ID of the parent query
            chunk_id: Chunk number
            chunk_content: Original chunk text
            extraction_prompt: Full prompt used for extraction
            llm_raw_response: Raw LLM response
            extracted_ranges: List of {"start": int, "end": int}
            extracted_texts: List of extracted text strings
            similarity_score: Vector similarity score
            semantic_relevance: LLM semantic relevance score
            core_term: Identified core term from the answer
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert lists to JSON strings for storage
        ranges_json = json.dumps(extracted_ranges, ensure_ascii=False)
        texts_json = json.dumps(extracted_texts, ensure_ascii=False)

        cursor.execute("""
            INSERT INTO evidence_extraction
            (query_id, chunk_id, chunk_content, extraction_prompt, llm_raw_response,
             extracted_ranges, extracted_texts, similarity_score, semantic_relevance, core_term)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (query_id, chunk_id, chunk_content, extraction_prompt, llm_raw_response,
              ranges_json, texts_json, similarity_score, semantic_relevance, core_term))

        conn.commit()
        conn.close()

        print(f"📌 Evidence extraction saved for query_id={query_id}, chunk={chunk_id}, core_term='{core_term}'")

    def add_dataset_comparison(
        self,
        query_id: int,
        dataset_answer: str,
        dataset_answer_start: int,
        dataset_answer_end: int,
        rag_extracted_text: str,
        rag_start: int,
        rag_end: int,
        match_score: float = 0.0
    ):
        """Add dataset ground truth comparison

        Args:
            query_id: ID of the parent query
            dataset_answer: Original dataset answer
            dataset_answer_start: Start position in dataset
            dataset_answer_end: End position in dataset
            rag_extracted_text: RAG extracted text
            rag_start: RAG extraction start position
            rag_end: RAG extraction end position
            match_score: Similarity score between dataset and RAG extraction
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO dataset_comparison
            (query_id, dataset_answer, dataset_answer_start, dataset_answer_end,
             rag_extracted_text, rag_start, rag_end, match_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (query_id, dataset_answer, dataset_answer_start, dataset_answer_end,
              rag_extracted_text, rag_start, rag_end, match_score))

        conn.commit()
        conn.close()

        print(f"📊 Dataset comparison saved for query_id={query_id}")

    def get_recent_queries(self, limit: int = 10, version: str = 'v1_v2_hybrid') -> List[Dict[str, Any]]:
        """Get recent queries

        Args:
            limit: Number of recent queries to retrieve
            version: Version to filter by (v1, v2, v3, v1_v2_hybrid, v2_requery). Default is v1_v2_hybrid (has evidence chunks)

        Returns:
            List of query dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM query_history
            WHERE version = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (version, limit))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def get_evidence_for_query(self, query_id: int) -> List[tuple]:
        """Get all evidence extractions for a specific query

        Args:
            query_id: Query ID

        Returns:
            List of evidence extraction tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM evidence_extraction
            WHERE query_id = ?
            ORDER BY chunk_id
        """, (query_id,))

        rows = cursor.fetchall()
        conn.close()

        return rows

    def get_query_details(self, query_id: int) -> Optional[Dict[str, Any]]:
        """Get complete details for a specific query

        Args:
            query_id: Query ID

        Returns:
            Dictionary with query, evidence extractions, and comparisons
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get main query
        cursor.execute("SELECT * FROM query_history WHERE id = ?", (query_id,))
        query_row = cursor.fetchone()

        if not query_row:
            conn.close()
            return None

        result = dict(query_row)

        # Get evidence extractions
        cursor.execute("""
            SELECT * FROM evidence_extraction
            WHERE query_id = ?
            ORDER BY chunk_id
        """, (query_id,))

        evidences = []
        for row in cursor.fetchall():
            evidence = dict(row)
            # Parse JSON fields
            evidence['extracted_ranges'] = json.loads(evidence['extracted_ranges'])
            evidence['extracted_texts'] = json.loads(evidence['extracted_texts'])
            evidences.append(evidence)

        result['evidences'] = evidences

        # Get dataset comparisons
        cursor.execute("""
            SELECT * FROM dataset_comparison
            WHERE query_id = ?
        """, (query_id,))

        comparisons = [dict(row) for row in cursor.fetchall()]
        result['dataset_comparisons'] = comparisons

        conn.close()
        return result

    def search_queries(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search queries by text

        Args:
            search_term: Search term
            limit: Max results

        Returns:
            List of matching queries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM query_history
            WHERE query LIKE ? OR generated_answer LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (f"%{search_term}%", f"%{search_term}%", limit))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def delete_query(self, query_id: int) -> bool:
        """Delete a query and all its related data (evidences, comparisons)

        Args:
            query_id: Query ID to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Delete related evidence extractions (foreign key constraint will handle this if enabled)
            cursor.execute("DELETE FROM evidence_extraction WHERE query_id = ?", (query_id,))

            # Delete related dataset comparisons
            cursor.execute("DELETE FROM dataset_comparison WHERE query_id = ?", (query_id,))

            # Delete the main query
            cursor.execute("DELETE FROM query_history WHERE id = ?", (query_id,))

            conn.commit()
            conn.close()

            print(f"✅ Query {query_id} and related data deleted successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to delete query {query_id}: {e}")
            return False

    def export_to_json(self, output_file: str):
        """Export all history to JSON file

        Args:
            output_file: Output JSON file path
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get all queries with details
        cursor.execute("SELECT id FROM query_history ORDER BY created_at")
        query_ids = [row[0] for row in cursor.fetchall()]

        conn.close()

        all_data = []
        for query_id in query_ids:
            details = self.get_query_details(query_id)
            if details:
                all_data.append(details)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)

        print(f"📦 Exported {len(all_data)} queries to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about query history

        Returns:
            Dictionary with statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total queries
        cursor.execute("SELECT COUNT(*) FROM query_history")
        stats['total_queries'] = cursor.fetchone()[0]

        # Average processing time
        cursor.execute("SELECT AVG(processing_time) FROM query_history")
        stats['avg_processing_time'] = cursor.fetchone()[0] or 0.0

        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM query_history")
        stats['avg_confidence'] = cursor.fetchone()[0] or 0.0

        # Total evidence extractions
        cursor.execute("SELECT COUNT(*) FROM evidence_extraction")
        stats['total_evidence_extractions'] = cursor.fetchone()[0]

        # Average similarity score
        cursor.execute("SELECT AVG(similarity_score) FROM evidence_extraction")
        stats['avg_similarity_score'] = cursor.fetchone()[0] or 0.0

        # Average semantic relevance
        cursor.execute("SELECT AVG(semantic_relevance) FROM evidence_extraction")
        stats['avg_semantic_relevance'] = cursor.fetchone()[0] or 0.0

        conn.close()
        return stats


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = QueryHistoryManager("./query_history.db")

    # Test add query
    query_id = manager.add_query(
        query="初夏に入った5月ごろ、北上する気流は何か？",
        generated_answer="初夏に入った5月ごろ、亜熱帯ジェット気流が北上します。",
        processing_time=2.5,
        confidence=0.85,
        num_chunks=3
    )

    # Test add evidence extraction
    manager.add_evidence_extraction(
        query_id=query_id,
        chunk_id=1,
        chunk_content="梅雨 [SEP] 一方、初夏に入った5月ごろ、亜熱帯ジェット気流も北上し...",
        extraction_prompt="Task: Extract core keyword...",
        llm_raw_response="character 28～character 38",
        extracted_ranges=[{"start": 28, "end": 38}],
        extracted_texts=["亜熱帯ジェット気流"],
        similarity_score=0.213,
        semantic_relevance=0.800
    )

    # Get statistics
    stats = manager.get_statistics()
    print("\n📊 Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Get recent queries
    recent = manager.get_recent_queries(limit=5)
    print(f"\n📝 Recent {len(recent)} queries:")
    for q in recent:
        print(f"  - {q['query'][:50]}...")
