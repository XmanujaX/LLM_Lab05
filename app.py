import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CONFIGURATION
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "llama3.2:3b"
CHROMA_DIR = "./chroma_db_user_uploads"
TOP_K = 3

# RAG PROMPT TEMPLATE
RAG_PROMPT = """You are a helpful assistant for a University Knowledge Assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""

# STREAMLIT UI
st.set_page_config(page_title="University AI Assistant", page_icon="🎓", layout="wide")
st.title("🎓 University AI Assistant")
st.caption("Upload your university PDF documents and ask questions about them.")

# SIDEBAR: FILE UPLOADER
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="You can upload one or more PDF files at a time."
    )
    
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # 1. Save uploaded files to temporary paths
                temp_file_paths = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        temp_file_paths.append(tmp_file.name)
                
                try:
                    # 2. Load and split documents
                    all_docs = []
                    for pdf_path in temp_file_paths:
                        loader = PyPDFLoader(pdf_path)
                        docs = loader.load()
                        all_docs.extend(docs)
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(all_docs)
                    
                    # 3. Create embeddings and store in ChromaDB
                    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
                    vectorstore = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory=CHROMA_DIR
                    )
                    
                    st.success(f"✅ Processed {len(uploaded_files)} document(s) into {len(splits)} chunks.")
                    
                    # Store the vectorstore in session state for the chat
                    st.session_state.vectorstore = vectorstore
                    st.session_state.retriever = vectorstore.as_retriever(
                        search_kwargs={"k": TOP_K}
                    )
                
                finally:
                    # Clean up temporary files
                    for path in temp_file_paths:
                        os.unlink(path)
        else:
            st.warning("Please upload at least one PDF file.")

#  MAIN CHAT INTERFACE
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents have been processed
    if "retriever" not in st.session_state:
        st.error("Please upload and process at least one PDF document first.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build the RAG chain
                    retriever = st.session_state.retriever
                    llm = OllamaLLM(model=LLM_MODEL)
                    prompt_template = PromptTemplate.from_template(RAG_PROMPT)
                    
                    def format_docs(docs):
                        return "\n\n".join(doc.page_content for doc in docs)
                    
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Invoke the chain
                    response = rag_chain.invoke(prompt)
                    
                    # Display and store the response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                except Exception as e:
                    st.error(f"An error occurred: {e}")