# import streamlit as st
# import os
# import time
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv
# from pdf2image import convert_from_path
# import pytesseract
# from langchain.schema import Document  

# # Load environment variables
# load_dotenv()
# groq_api_key = os.getenv('GROQ_API_KEY')

# st.title("Gemma Model Document Q&A (With OCR Support)")

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# prompt = ChatPromptTemplate.from_template(
# """
# You are an advanced AI assistant specializing in document analysis. You will provide detailed, context-aware, and elaborative responses based on the retrieved document content. Follow these principles while answering:

# 1. General Q&A:
# - Answer questions strictly based on the provided context.
# - Provide in-depth responses that include all relevant details, explanations, and examples.
# - If the information is not found in the documents, state: "The answer is not available in the provided documents."
# - Avoid assumptions and hallucinations.

# 2. Document Comparison:
# - If the user asks to compare documents, retrieve relevant sections from each document.
# - Provide a thorough comparison, highlighting all key similarities, differences, and implications.
# - Present comparisons in structured formats, using tables, bullet points, or detailed paragraphs.

# 3. Data Extraction & Summarization:
# - If the user asks for specific data (e.g., dates, names, figures), extract and return all relevant details.
# - If summarization is requested, provide a concise yet comprehensive summary, elaborating on key points and underlying themes.

# 4. Context Management:
# - Always use the most relevant sections of the documents for answering.
# - Avoid including redundant or unrelated information.

# 5. Response Format:
# - Maintain clarity, coherence, and proper structuring.
# - Use bullet points, tables, or numbered lists where necessary for readability.
# - For comparisons, highlight pros/cons, factual differences, and unique aspects of each document.

# Context:
# <context>
# {context}
# <context>

# User Question:
# {input}

# Provide a detailed, well-explained, and long-form response based on the provided context.
# """
# )



# def extract_text_with_ocr(pdf_path):
#     """Convert scanned PDF pages to text using OCR"""
#     poppler_path = os.path.join(os.path.dirname(__file__), "poppler", "Library", "bin")  # Adjust path
#     pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(__file__), "Tesseract-OCR", "tesseract.exe")  

#     images = convert_from_path(pdf_path, poppler_path=poppler_path)
#     text = "\n".join(pytesseract.image_to_string(img) for img in images)
#     return text

# def vector_embedding_from_directory(directory):
#     """Processes all PDFs in a directory and creates a vector store."""
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#         all_docs = []

#         for filename in os.listdir(directory):
#             if filename.endswith(".pdf"):
#                 file_path = os.path.join(directory, filename)
#                 loader = PyPDFLoader(file_path)
#                 docs = loader.load()

#                 # Check if extracted text is empty (indicating scanned PDF)
#                 if not any(doc.page_content.strip() for doc in docs):
#                     st.warning(f"Scanned PDF detected: {filename}. Running OCR...")
#                     extracted_text = extract_text_with_ocr(file_path)
#                     docs = [Document(page_content=extracted_text)]

#                 all_docs.extend(docs)

#         # Split documents into chunks
#         st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
#         st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)

#         # Create vector embeddings using FAISS
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#         st.success("All PDFs have been embedded into the vector database.")

# # Load PDFs from directory
# pdf_directory = "./input_sample"  # Adjust folder name as needed
# if st.button("Embed PDFs from Directory"):
#     if os.path.exists(pdf_directory) and os.listdir(pdf_directory):
#         vector_embedding_from_directory(pdf_directory)
#     else:
#         st.error("No PDFs found in the directory!")

# # Ask questions
# prompt1 = st.text_input("Enter your question about the documents")

# if prompt1:
#     if "vectors" in st.session_state:
#         document_chain = create_stuff_documents_chain(llm, prompt)
#         retriever = st.session_state.vectors.as_retriever()
#         retrieval_chain = create_retrieval_chain(retriever, document_chain)

#         start = time.process_time()
#         response = retrieval_chain.invoke({'input': prompt1})
#         st.write(f"Response Time: {time.process_time() - start} seconds")
#         st.write(response['answer'])

#         # Display relevant document chunks
#         with st.expander("Document Similarity Search"):
#             for i, doc in enumerate(response["context"]):
#                 st.write(doc.page_content)
#                 st.write("--------------------------------")
#     else:
#         st.warning("Please embed the documents first!")












































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
from langchain.schema import Document  

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Gemma Model Document Q&A (With OCR Support)")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Debugging message for prompt setup
st.text("Setting up the AI assistant with the provided prompt template...")

prompt = ChatPromptTemplate.from_template(
"""
You are an advanced AI assistant specializing in document analysis. You will provide detailed, context-aware, and elaborative responses based on the retrieved document content. Follow these principles while answering:

1. General Q&A:
- Answer questions strictly based on the provided context.
- Provide in-depth responses that include all relevant details, explanations, and examples.
- If the information is not found in the documents, state: "The answer is not available in the provided documents."
- Avoid assumptions and hallucinations.

2. Document Comparison:
- If the user asks to compare documents, retrieve relevant sections from each document.
- Provide a thorough comparison, highlighting all key similarities, differences, and implications.
- Present comparisons in structured formats, using tables, bullet points, or detailed paragraphs.

3. Data Extraction & Summarization:
- If the user asks for specific data (e.g., dates, names, figures), extract and return all relevant details.
- If summarization is requested, provide a concise yet comprehensive summary, elaborating on key points and underlying themes.

4. Context Management:
- Always use the most relevant sections of the documents for answering.
- Avoid including redundant or unrelated information.

5. Response Format:
- Maintain clarity, coherence, and proper structuring.
- Use bullet points, tables, or numbered lists where necessary for readability.
- For comparisons, highlight pros/cons, factual differences, and unique aspects of each document.

Context:
<context>
{context}
<context>

User Question:
{input}

Provide a detailed, well-explained, and long-form response based on the provided context.
"""
)


def extract_text_with_ocr(pdf_path):
    """Convert scanned PDF pages to text using OCR"""
    poppler_path = os.path.join(os.path.dirname(__file__), "poppler", "Library", "bin")  # Adjust path
    pytesseract.pytesseract.tesseract_cmd = os.path.join(os.path.dirname(__file__), "Tesseract-OCR", "tesseract.exe")  

    images = convert_from_path(pdf_path, poppler_path=poppler_path)
    text = "\n".join(pytesseract.image_to_string(img) for img in images)
    return text

def vector_embedding_from_directory(directory):
    """Processes all PDFs in a directory and creates a vector store."""
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        all_docs = []

        st.text("Processing PDFs and embedding them into the vector store...")  # Debugging message

        for filename in os.listdir(directory):
            if filename.endswith(".pdf"):
                file_path = os.path.join(directory, filename)
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                # Debugging message for file processing
                st.text(f"Processing file: {filename}")

                # Check if extracted text is empty (indicating scanned PDF)
                if not any(doc.page_content.strip() for doc in docs):
                    st.warning(f"Scanned PDF detected: {filename}. Running OCR...")
                    extracted_text = extract_text_with_ocr(file_path)
                    docs = [Document(page_content=extracted_text)]

                all_docs.extend(docs)

        # Split documents into chunks
        st.text("Splitting documents into chunks...")  # Debugging message
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(all_docs)

        # Create vector embeddings using FAISS
        st.text("Creating vector embeddings using FAISS...")  # Debugging message
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("All PDFs have been embedded into the vector database.")

# Load PDFs from directory
pdf_directory = "./input_sample"  # Adjust folder name as needed
if st.button("Embed PDFs from Directory"):
    if os.path.exists(pdf_directory) and os.listdir(pdf_directory):
        vector_embedding_from_directory(pdf_directory)
    else:
        st.error("No PDFs found in the directory!")

# Ask questions
prompt1 = st.text_input("Enter your question about the documents")

if prompt1:
    if "vectors" in st.session_state:
        st.text("Retrieving relevant documents and generating response...")  # Debugging message
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})

        # Debugging message for response time
        st.text(f"Response generated in: {time.process_time() - start} seconds")

        st.write(response['answer'])

        # Display relevant document chunks
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
    else:
        st.warning("Please embed the documents first!")
