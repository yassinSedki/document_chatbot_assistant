import os
from dotenv import load_dotenv
import tempfile
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- Constants ---
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CUSTOM_PROMPT_TEMPLATE = """
Answer using ONLY the provided context. If unsure, say "I don't know".

Context: {context}
Question: {question}

Answer concisely:
"""

# --- Core Functions ---
def process_uploaded_file(uploaded_file):
    """Process PDF into text chunks with error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        return []
    finally:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)

def initialize_vectorstore(chunks):
    """Create FAISS vectorstore with progress tracking"""
    if not chunks:
        return None
    with st.spinner("Creating search index..."):
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return FAISS.from_documents(chunks, embeddings)

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Document Chatbot", page_icon="📄")
    st.title("📄 Smart Document Assistant")
    
    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs to begin chatting"}]
    
    # File upload with auto-processing
    with st.sidebar:
        st.subheader("Document Management")
        uploaded_files = st.file_uploader(
            "Add PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more PDFs to analyze"
        )
        
        if uploaded_files:
            with st.status("Processing documents...", expanded=True) as status:
                all_chunks = []
                for file in uploaded_files:
                    st.write(f"• {file.name}")
                    chunks = process_uploaded_file(file)
                    all_chunks.extend(chunks)
                
                if all_chunks:
                    st.session_state.vectorstore = initialize_vectorstore(all_chunks)
                    status.update(
                        label=f"Ready! Processed {len(all_chunks)} text chunks",
                        state="complete",
                        expanded=False
                    )
                else:
                    status.update(label="Processing failed", state="error")
    
    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    # Chat interaction
    if prompt := st.chat_input("Ask about your documents"):
        if not HF_TOKEN:
            st.error("Missing HuggingFace token in .env file")
            return
        if st.session_state.vectorstore is None:
            st.error("Please upload documents first")
            return
        
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Analyzing documents..."):
            try:
                llm = HuggingFaceHub(
                    repo_id=HUGGINGFACE_REPO_ID,
                    huggingfacehub_api_token=HF_TOKEN,
                    model_kwargs={"temperature": 0.3, "max_length": 512}
                )
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": PromptTemplate(
                            template=CUSTOM_PROMPT_TEMPLATE,
                            input_variables=["context", "question"]
                        )
                    }
                )
                
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"].split("Answer concisely:")[-1].strip()
                
                # Format sources
                sources = [
                    f"📌 **{doc.metadata.get('source', 'Document')}** (page {doc.metadata.get('page', '?')})\n"
                    f"{doc.page_content[:150]}..."
                    for doc in response["source_documents"]
                ]
                
                # Display response
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("View sources"):
                        st.markdown("\n\n".join(sources))
                
                st.session_state.messages.extend([
                    {"role": "assistant", "content": answer},
                    {"role": "sources", "content": "\n".join(sources)}
                ])
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    main()