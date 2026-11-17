# Lab 5 – RAG Architecture

**Student:** [Hingurala Perera - 10965291]
**Module:** [PUSL3205]

## Overview
For this lab, I built a Retrieval-Augmented Generation (RAG) application using LangChain, ChromaDB, Ollama, and Streamlit. The application was initially designed to answer questions based on a static set of documents. I then modified it to allow users to dynamically upload multiple PDF files via a sidebar, which are then processed, embedded, and stored for chat-based querying.

## Contents
* `app.py` – The main Streamlit application code.
* `requirements.txt` – List of Python dependencies.
* `screenshots/` – Evidence of the working app, including the chat interface and document upload sidebar.

## Setup Summary
| Item | Value |
|---|---|
| **LLM** | Llama 3.2 3B (via Ollama) |
| **Embedding Model** | Qwen3 Embedding 0.6B (via Ollama) |
| **Vector Database** | ChromaDB |
| **Frameworks** | LangChain, Streamlit |
| **Document Loader** | PyPDFLoader |

## Key Takeaways
* Learned how to build a complete RAG pipeline from document ingestion to answer generation.
* Implemented dynamic file uploading in Streamlit to make the app interactive and user-friendly.
* Used local models (Ollama) for privacy and to avoid API costs.
* Understood the importance of text chunking and embedding quality for retrieval performance.