from setuptools import setup, find_packages

setup(
    name="rag-system",
    version="0.1.0",
    description="A RAG (Retrieval-Augmented Generation) system with evidence indicators",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "langchain==0.1.0",
        "langchain-community==0.0.20",
        "langchain-openai==0.0.5",
        "langchain-core==0.1.10",
        "openai==1.12.0",
        "chromadb==0.4.22",
        "python-dotenv==1.0.0",
        "unstructured==0.11.8",
        "jq==1.4.1",
    ],
    python_requires=">=3.8",
) 