import os
import time
import shutil
from dotenv import load_dotenv
from typing import Union, List, cast, Any
from pydantic import SecretStr
import chromadb
import argparse
import openai

from langchain_community.document_loaders import DirectoryLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from edge_case_handler import EdgeCaseHandler, EvidenceResult
from ultra_fast_rag import UltraFastRAG

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables.")
    print("Please set your OpenAI API key in the .env file or as an environment variable.")

CHROMA_PATH = "chroma"
DATA_PATH = "./data/single_20240229.json"
TEST_DATA_PATH = "./data/test_sample.json"

def clean_evidence_text(evidence_text: str, query_text: str) -> str:
    """Clean evidence text to remove query repetition and improve readability"""
    import re
    
    # Remove the query text if it appears at the beginning of evidence
    query_clean = query_text.replace('\n', ' ').strip()
    if evidence_text.startswith(query_clean):
        evidence_text = evidence_text[len(query_clean):].strip()
    
    # Remove common question patterns that might be repeated
    question_patterns = [
        r'次の文章の慣用句の間違いを指摘し修正しなさい。',
        r'次の文章の.*?を指摘し修正しなさい。',
        r'次の文章の.*?を教えてください。',
        r'次の文章の.*?について説明してください。',
    ]
    
    for pattern in question_patterns:
        evidence_text = re.sub(pattern, '', evidence_text)
    
    # Clean up extra whitespace and newlines
    evidence_text = re.sub(r'\n+', '\n', evidence_text)
    evidence_text = evidence_text.strip()
    
    return evidence_text

def main_setup():
    # This main function is primarily for command-line execution.
    # In Colab, you can call generate_data_store() or query_data() directly.
    pass # Keep main but remove the call to generate_data_store here

def generate_data_store(use_test_data=False):
    start = time.time()
    documents = load_documents(use_test_data)
    print(f"Loaded documents in {time.time() - start:.2f} seconds")
    start = time.time()
    chunks = split_text(documents)
    print(f"Split text in {time.time() - start:.2f} seconds")
    start = time.time()
    save_to_chroma(chunks)
    print(f"Saved to Chroma in {time.time() - start:.2f} seconds")

def load_documents(use_test_data=False):
    """
    Loads documents from the specified JSON file using JSONLoader.
    The jq_schema '.[ ]' indicates that the JSON file is a top-level array,
    and each object within that array should be treated as a document.
    The content_key 'output' specifies that the 'output' field of each object
    contains the main text content for the LangChain Document.
    """
    file_path = TEST_DATA_PATH if use_test_data else DATA_PATH
    loader = JSONLoader(
        file_path=file_path, # Use the appropriate data path
        jq_schema='.[ ]',  # Selects each object in the top-level array as a document
        content_key='output' # Extracts content from the 'output' field for each document
    )
    documents = loader.load()
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")



    return chunks

def save_to_chroma(chunks: list[Document]):
    print(f"Starting Chroma save for {len(chunks)} chunks...")
    
    # Remove the directory for a fresh start
    if os.path.exists(CHROMA_PATH) and os.path.isdir(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared existing directory: {CHROMA_PATH}")

    os.makedirs(CHROMA_PATH, exist_ok=True)
    print(f"Created directory: {CHROMA_PATH}")

    # Prepare data for Chroma
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    
    print("Initializing embedding function...")
    embedding_function = OpenAIEmbeddings(
        api_key=SecretStr(openai_api_key),
        model="text-embedding-ada-002",  # Explicitly specify model
        chunk_size=1000  # Optimize chunk size for embeddings
    )
    
    print("Creating Chroma database...")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # Optimize batch size based on chunk count
    if len(chunks) > 15000:
        batch_size = 500   # Very small batches for very large datasets
    elif len(chunks) > 10000:
        batch_size = 1000  # Smaller batches for large datasets
    else:
        batch_size = 2000  # Larger batches for smaller datasets
    
    print(f"Using batch size: {batch_size}")
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for i in range(0, len(texts), batch_size):
        batch_num = i // batch_size + 1
        batch_end = min(i + batch_size, len(texts))
        print(f"Processing batch {batch_num}/{total_batches} (chunks {i} to {batch_end-1})")
        
        start_time = time.time()
        db.add_texts(
            texts=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
            ids=ids[i:batch_end]
        )
        batch_time = time.time() - start_time
        print(f"Batch {batch_num} completed in {batch_time:.2f} seconds")
    
    print(f"Chroma save completed. Total chunks: {len(chunks)}")

# Remove problematic module-level execution
# documents = load_documents()
# split_text(documents)

def query_data(query_text: str, use_advanced_rag: bool = True):
    """
    Queries the Chroma vector store and returns the response and source documents.
    Enhanced with Advanced RAG system for multi-chunk information aggregation.
    """
    total_start = time.time()
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    
    if use_advanced_rag:
        # 使用超高速RAG系统
        ultra_fast_rag = UltraFastRAG(api_key, CHROMA_PATH)
        answer, source_document, evidence_text, start_char, end_char = ultra_fast_rag.query(query_text)
        
        return (
            answer,
            source_document,  # 完整的検索ヒット文書
            evidence_text,
            start_char,
            end_char
        )
    
    # 原始系统（备用）
    start = time.time()
    print("Initializing Chroma DB...")
    embedding_function = OpenAIEmbeddings(api_key=SecretStr(api_key))
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    print(f"Initialized Chroma DB in {time.time() - start:.2f} seconds")
    
    # 原始系统已被高级RAG系统替代

        # Extract key terms from the query for better similarity search
    # Remove common question words and focus on the main topic
    import re
    
    # More sophisticated query processing
    # First, try to extract the main topic by looking for patterns
    search_query = query_text
    
    # Common Japanese question patterns
    patterns = [
        r'(.+?)とは何か.*',  # "Xとは何か..."
        r'(.+?)について.*',  # "Xについて..."
        r'(.+?)を説明してください.*',  # "Xを説明してください..."
        r'(.+?)を教えてください.*',  # "Xを教えてください..."
        r'(.+?)とは.*',  # "Xとは..."
        r'(.+?)何ですか.*',  # "X何ですか..."
        r'(.+?)何でしょうか.*',  # "X何でしょうか..."
    ]
    
    # Try to extract the main topic using regex patterns
    for pattern in patterns:
        match = re.match(pattern, query_text)
        if match:
            search_query = match.group(1).strip()
            break
    
    # If no pattern matched, try simple word removal
    if search_query == query_text:
        question_words = [
            "とは何か", "について", "を説明してください", "を教えてください", 
            "とは", "について教えて", "について説明して", "について詳しく",
            "何ですか", "何でしょうか", "何か", "説明してください", "教えてください"
        ]
        
        for word in question_words:
            search_query = search_query.replace(word, "")
    
    # Clean up the search query
    search_query = search_query.strip()
    if search_query.endswith('。'):
        search_query = search_query[:-1]
    
    # If the cleaned query is too short, use the original
    if len(search_query.strip()) < 2:
        search_query = query_text
    
    start = time.time()
    results = db.similarity_search_with_relevance_scores(search_query, k=5)
    print(f"Similarity search completed in {time.time() - start:.2f} seconds")
    if not results or results[0][1] < 0.3:
        # Try edge case handling for multi-chunk scenarios
        print("Low similarity scores detected. Trying edge case handling...")
        try:
            edge_handler = EdgeCaseHandler(api_key, CHROMA_PATH)
            edge_result = edge_handler.handle_edge_case(query_text, strategy="auto")
            
            if edge_result.is_complete and edge_result.confidence_score > 0.3:
                print(f"Edge case handling successful using {edge_result.strategy_used} (confidence: {edge_result.confidence_score:.3f})")
                # Generate response using edge case evidence
                prompt_template = """
                You are an intelligent assistant. Use the following evidence to answer the question.
                
                CRITICAL INSTRUCTIONS:
                1. You MUST provide a comprehensive answer based on the evidence provided.
                2. Only say "I don't know" if the evidence contains ZERO relevant information for the question.
                3. Even if the evidence only partially answers the question, provide what you can from the available information.
                4. If the question is asked in Japanese, respond entirely in Japanese. If the question is asked in English, respond entirely in English. Do not mix languages in your response.

                Evidence: {context}

                Question: {question}
                Answer:
                """
                prompt = prompt_template.format(context=edge_result.evidence_text, question=query_text)
                
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ]
                )
                
                return response.choices[0].message.content, results, edge_result.evidence_text, 0, len(edge_result.evidence_text)
            else:
                print(f"Edge case handling found insufficient evidence (confidence: {edge_result.confidence_score:.3f})")
        except Exception as e:
            print(f"Edge case handling failed: {e}")
        
        return "Unable to find matching results.", None, None, None, None

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = """
    You are an intelligent assistant. Use the following pieces of context to answer the question at the end.
    
    CRITICAL INSTRUCTIONS:
    1. If the context contains ANY relevant information that can help answer the question, you MUST provide a comprehensive answer based on that information.
    2. Only say "I don't know" if the context contains ZERO relevant information for the question.
    3. Even if the context only partially answers the question, provide what you can from the available information.
    4. Focus on providing the answer without repeating the question text.
    5. LANGUAGE MATCHING: If the question is asked in Japanese, respond ENTIRELY in Japanese. If the question is asked in English, respond ENTIRELY in English. Never mix languages in your response.
    6. For Japanese questions, provide natural, fluent Japanese responses that directly answer the question.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    prompt = prompt_template.format(context=context_text, question=query_text)

    # 检测问题语言并设置系统提示
    def is_japanese(text):
        japanese_chars = sum(1 for char in text if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FAF')
        return japanese_chars > len(text) * 0.1
    
    if is_japanese(query_text):
        system_prompt = "あなたは日本語で回答する知的なアシスタントです。質問には必ず日本語で答えてください。"
    else:
        system_prompt = "You are a helpful assistant that answers in English."
    
    start = time.time()
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    if results:
        original_evidence = results[0][0].page_content
        # 使用TextRank提取关键句
        start = time.time()
        key_result = textrank_extractor.extract_concise_answer(original_evidence, query_text)
        print(f"TextRank extraction completed in {time.time() - start:.2f} seconds")
        
        if key_result:
            # 使用TextRank提取的关键句作为证据
            evidence_text = key_result.key_sentence
            start_index = key_result.start_position
            end_index = key_result.end_position
            print(f"Key sentence extracted: {evidence_text}")
            print(f"Position: {start_index}〜{end_index} (relevance: {key_result.relevance_score:.3f})")
        else:
            # 备用方案：使用清理后的原始文本
            evidence_text = clean_evidence_text(original_evidence, query_text)
            start_index = 0
            end_index = len(evidence_text)
    else:
        evidence_text = None
        start_index = None
        end_index = None

    return response.choices[0].message.content, results, evidence_text, start_index, end_index

def main():
    print("Script loaded. In a Colab notebook, call generate_data_store() or query_data('your query') directly in a cell.")
    print("For local execution, use command-line arguments: 'generate' or 'query --query \"your text\"'.")

    parser = argparse.ArgumentParser(description="RAG Query System with Evidence Indicator")
    parser.add_argument("action", nargs='?', default=None, choices=["generate", "query"], help="Action to perform.")
    parser.add_argument("--query", type=str, help="Query text to search for.")
    args = parser.parse_args()

    if args.action == "generate":
        generate_data_store()
    elif args.action == "query":
        if not args.query:
            print("Error: --query argument is required for the 'query' action.")
            return
        response, sources, evidence, start, end = query_data(args.query)
        print("--- RAGの通常回答 ---")
        print(response)
        if sources and evidence:
            print("\n--- Evidence Indicator による付帯情報の出力 ---")
            print(f"【検索ヒットのチャンクを含む文書】")
            print(f"{sources}")
            print(f"\n【根拠情報の文字列範囲】{start + 1}文字目～{end}文字目")
            print(f"\n【根拠情報】")
            print(f"{evidence.strip()}")

# Interactive prompt for real-time queries
def interactive_query_loop():
    print("\n--- Entering Interactive Query Mode ---")
    print("Type your query and press Enter. Type 'exit' to quit.\n")
    while True:
        query = input("Your query: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting interactive query mode.")
            break
        if not query:
            continue
        response, sources, evidence, start, end = query_data(query)
        print("\n--- RAGの通常回答 ---")
        print(response)
        if sources and evidence:
            print("\n--- Evidence Indicator による付帯情報の出力 ---")
            print(f"【検索ヒットのチャンクを含む文書】")
            print(f"{sources}")
            print(f"\n【根拠情報の文字列範囲】{start + 1}文字目～{end}文字目")
            print(f"\n【根拠情報】")
            print(f"{evidence.strip()}")
        else:
            print("No relevant sources found to generate evidence.\n")

if __name__ == "__main__":
    print("--- Generating Data Store ---")
    # Use test data for faster development
    use_test_data = os.environ.get('USE_TEST_DATA', 'false').lower() == 'true'
    if use_test_data:
        print("Using test data for faster development...")
        # Skip the full dataset loading when using test data
        generate_data_store(use_test_data=True)
    else:
        generate_data_store(use_test_data=False)
    print("\n--- Data Store Generation Complete ---\n")
    
    interactive_query_loop()





# Commented out IPython magic to ensure Python compatibility.
# %pip install unstructured

# Commented out IPython magic to ensure Python compatibility.
# %pip install jq

