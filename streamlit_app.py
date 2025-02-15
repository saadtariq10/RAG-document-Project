import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from pdf2image import convert_from_path
import pytesseract

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Gemma Model Document Q&A (With OCR Support)")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions in a detailed information based on the provided context only if do not found in the document search online.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

def extract_text_with_ocr(pdf_path):
    """Convert scanned PDF pages to text using OCR"""

    # Set local paths for Poppler and Tesseract
    poppler_path = os.path.join(os.path.dirname(__file__), "poppler", "Library", "bin")  # Adjust the path to your directory
    pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(__file__), "Tesseract-OCR", "tesseract.exe")  # Path to tesseract.exe
    
    # Convert PDF to images using Poppler
    images = convert_from_path(pdf_path, poppler_path=poppler_path)

    # Extract text from images using Tesseract
    text = "\n".join(pytesseract.image_to_string(img) for img in images)
    return text



def process_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary directory"""
    temp_dir = "./uploaded_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

from langchain.schema import Document  # Import this at the top

def vector_embedding(file_path):
    if "vectors" not in st.session_state:
        # Initialize Hugging Face embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load the PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Check if extracted text is empty (indicating a scanned PDF)
        if not any(doc.page_content.strip() for doc in docs):
            st.warning("Scanned PDF detected. Running OCR...")  # Run OCR if the document is scanned
            extracted_text = extract_text_with_ocr(file_path)

            # ✅ Create a Document object manually
            docs = [Document(page_content=extracted_text)]

        # Split documents into chunks
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)

        # Create vector embeddings using FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# File uploader widget
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_path = process_uploaded_file(uploaded_file)
    st.success(f"Uploaded file: {uploaded_file.name}")
    
    if st.button("Embed Document"):
        vector_embedding(file_path)
        st.write("Vector Store DB is ready")

prompt1 = st.text_input("Enter your question about the uploaded document")

if prompt1:
    if "vectors" in st.session_state:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response Time: {time.process_time() - start} seconds")
        st.write(response['answer'])

        # Display relevant document chunks
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.warning("Please embed a document first!")




