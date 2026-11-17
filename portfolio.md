# Portfolio – Lab 5: RAG Architecture

## Step 1 & 2: Environment Setup and Model Installation
I set up a Python virtual environment and installed all necessary libraries, including `langchain`, `langchain-community`, `langchain-ollama`, `langchain-chroma`, `chromadb`, `streamlit`, and `pypdf`. I then installed Ollama and pulled the required models: `llama3.2:3b` for text generation and `qwen3-embedding:0.6b` for embeddings.

**Reflection:** Using a local environment with Ollama is great because it doesn't require API keys and keeps all data private. The main challenge was ensuring the correct versions of the libraries were installed.

## Step 3 & 4: Building the Basic RAG Pipeline
I created an `app.py` script that loads a PDF, splits it into chunks using `RecursiveCharacterTextSplitter`, creates embeddings with Ollama, and stores them in ChromaDB. A retriever was then used to fetch relevant chunks for a user query, which were passed to the LLM along with a prompt template to generate an answer.

**Reflection:** Understanding how the retriever interacts with the vector store was a key learning point. The quality of the answer heavily depends on retrieving the correct context chunks.

## Step 5: Modifying the App for Multiple PDFs and User Uploads
I modified the application to use `st.file_uploader` in the sidebar, allowing users to upload multiple PDFs at once. I implemented a temporary file saving mechanism so `PyPDFLoader` could process them. The `vectorstore` and `retriever` are stored in `st.session_state` so the chat can persist across interactions without reprocessing documents.

**Struggle:** I initially tried to process files directly from memory, which caused issues with `PyPDFLoader`. 
**Fix:** I saved the uploaded files to temporary paths using Python's `tempfile` module, processed them, and then deleted the temporary files to keep the workspace clean.

## Step 6: Testing and Results
I tested the app with several university-related PDFs. The app successfully processed the documents and answered questions based on their content. The chat interface provided a smooth user experience.

**(Evidence: `screenshots/app_interface.png`, `screenshots/streamlit_sidebar.png`)**

## Step 7 & 8: Reflection and Future Work
This lab provided a solid foundation in building RAG applications. The ability to dynamically upload documents makes the tool much more practical.

**Future Enhancements (Group Activity):** I explored the advanced RAG architectures for the group activity.
*   **Hybrid Retrieval:** Combining BM25 with vector search could improve recall for exact academic terms.
*   **Context-Aware Chunking:** Moving away from fixed-size chunks to section-based chunking could improve answers for long handbooks.
*   **Corrective RAG:** Adding a retrieval quality checker would reduce hallucinations and improve abstention when the answer is not in the documents.