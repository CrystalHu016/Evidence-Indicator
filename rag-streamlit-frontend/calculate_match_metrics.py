#!/usr/bin/env python3
"""
Character-level Match Rate Metrics for Evidence Evaluation
Compares extracted evidence with dataset ground truth answers
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def calculate_llm_semantic_score(extracted_evidence: str, dataset_answer: str) -> float:
    """
    Use Gemini LLM to evaluate semantic match between extracted evidence and dataset answer

    Args:
        extracted_evidence: Evidence text extracted by the system
        dataset_answer: Ground truth answer from dataset

    Returns:
        float: Semantic match score from 0.0 to 1.0 (0% to 100%)
    """
    if not extracted_evidence or not dataset_answer:
        return 0.0

    try:
        import google.generativeai as genai

        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("⚠️ GEMINI_API_KEY not found in environment")
            return 0.0

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')

        # Construct prompt for semantic evaluation
        prompt = f"""あなたは質問応答システムの評価者です。抽出された根拠情報が正解答案の意味を正確に含んでいるかを評価してください。

正解答案（元のデータセット回答）：
{dataset_answer}

抽出された根拠情報：
{extracted_evidence}

評価基準：
- 100%: 根拠情報が正解答案の意味を完全に含んでいる、または同じ意味である
- 80-99%: 根拠情報が正解答案の主要な意味をほぼ全て含んでいるが、一部表現が異なる
- 60-79%: 根拠情報が正解答案の意味の大部分を含んでいるが、重要な詳細が欠けている
- 40-59%: 根拠情報が正解答案の意味の一部を含んでいるが、不完全または不正確
- 20-39%: 根拠情報が正解答案とわずかに関連しているが、主要な意味が異なる
- 0-19%: 根拠情報が正解答案と全く関係ない、または完全に不正確

**数値のみを返してください（0-100の整数）。説明は不要です。**

スコア:"""

        # Call Gemini API
        response = model.generate_content(prompt)
        score_text = response.text.strip()

        # Extract numeric score
        match = re.search(r'\d+', score_text)
        if match:
            score = int(match.group())
            # Normalize to 0.0-1.0 range
            return min(max(score / 100.0, 0.0), 1.0)
        else:
            print(f"⚠️ Could not parse LLM score from response: {score_text}")
            return 0.0

    except Exception as e:
        print(f"⚠️ Error calculating LLM semantic score: {e}")
        return 0.0


def calculate_char_match_rate(extracted_evidence: str, dataset_answer: str) -> dict:
    """
    Calculate character-level match rate between extracted evidence and dataset answer
    
    Args:
        extracted_evidence: Evidence text extracted by the system
        dataset_answer: Ground truth answer from dataset
    
    Returns:
        dict with metrics:
        - exact_match: bool, whether it's 100% match
        - match_rate: float, percentage of matching characters (0.0-1.0)
        - overlap_chars: int, number of overlapping characters
        - precision: float, what percentage of extracted evidence is correct
        - recall: float, what percentage of dataset answer is found
        - f1_score: float, harmonic mean of precision and recall
    """
    if not extracted_evidence or not dataset_answer:
        return {
            'exact_match': False,
            'match_rate': 0.0,
            'overlap_chars': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Normalize strings (remove whitespace for comparison)
    evidence_clean = extracted_evidence.strip()
    answer_clean = dataset_answer.strip()
    
    # Check exact match
    exact_match = (evidence_clean == answer_clean)
    
    # Calculate character overlap using longest common subsequence
    overlap_chars = longest_common_subsequence_length(evidence_clean, answer_clean)
    
    # Calculate metrics
    evidence_len = len(evidence_clean)
    answer_len = len(answer_clean)
    
    # Precision: how much of extracted evidence is correct
    precision = overlap_chars / evidence_len if evidence_len > 0 else 0.0
    
    # Recall: how much of dataset answer is found
    recall = overlap_chars / answer_len if answer_len > 0 else 0.0
    
    # F1 score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Match rate (using F1 as overall match rate)
    match_rate = f1_score
    
    return {
        'exact_match': exact_match,
        'match_rate': match_rate,
        'overlap_chars': overlap_chars,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'evidence_length': evidence_len,
        'answer_length': answer_len
    }


def longest_common_subsequence_length(s1: str, s2: str) -> int:
    """
    Calculate the length of longest common subsequence between two strings
    This gives us the number of matching characters in order
    """
    m, n = len(s1), len(s2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def calculate_substring_match(extracted_evidence: str, dataset_answer: str) -> dict:
    """
    Alternative: Calculate substring match (simpler, faster)
    Checks if dataset answer is contained in extracted evidence
    """
    if not extracted_evidence or not dataset_answer:
        return {
            'contains_answer': False,
            'match_rate': 0.0
        }
    
    evidence_clean = extracted_evidence.strip()
    answer_clean = dataset_answer.strip()
    
    # Check if answer is substring of evidence
    contains_answer = answer_clean in evidence_clean
    
    if contains_answer:
        match_rate = 1.0  # Perfect match if contained
    else:
        # Calculate partial overlap
        max_overlap = 0
        answer_len = len(answer_clean)
        
        for i in range(len(evidence_clean)):
            for j in range(i+1, len(evidence_clean)+1):
                substring = evidence_clean[i:j]
                if substring in answer_clean:
                    max_overlap = max(max_overlap, len(substring))
        
        match_rate = max_overlap / answer_len if answer_len > 0 else 0.0
    
    return {
        'contains_answer': contains_answer,
        'match_rate': match_rate
    }


# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        {
            'evidence': '古今集',
            'answer': '古今集',
            'description': 'Exact match'
        },
        {
            'evidence': '『古今集』の成立は『万葉集』よりも時代が下る',
            'answer': '古今集',
            'description': 'Answer contained in evidence'
        },
        {
            'evidence': '万葉集',
            'answer': '古今集',
            'description': 'No match'
        },
        {
            'evidence': '環太平洋パートナーシップ協定',
            'answer': 'TPP',
            'description': 'Partial match'
        }
    ]
    
    print("=" * 70)
    print("Character-level Match Rate Metrics Test")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['description']}")
        print(f"Evidence: {test['evidence']}")
        print(f"Answer: {test['answer']}")
        
        metrics = calculate_char_match_rate(test['evidence'], test['answer'])
        
        print(f"\nMetrics:")
        print(f"  Exact Match: {metrics['exact_match']}")
        print(f"  Match Rate: {metrics['match_rate']:.2%}")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1_score']:.2%}")
        print(f"  Overlap Chars: {metrics['overlap_chars']}")
        print("-" * 70)

