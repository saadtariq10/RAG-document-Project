# 📄 Gemma Model Document Q&A (With OCR Support)  

An **AI-powered PDF Q&A tool** that enables users to ask questions about uploaded PDF documents. The application supports **scanned PDFs** using **OCR (Optical Character Recognition)** and leverages **LLM-based document retrieval** for accurate responses.  

## 🚀 Features  
✅ Upload PDF documents (both text-based and scanned)  
✅ Extract text using OCR for scanned PDFs  
✅ Generate vector embeddings using FAISS  
✅ Retrieve and answer questions based on document content  
✅ Uses **Llama3-8b-8192** model via **Groq API**  

## 🖥️ Demo  
![Demo](https://github.com/saadtariq10/RAG-document-Project/blob/main/rag.gif)  

## 📥 Installation  

1. **Clone the repository:**  
   ```sh
   git clone https://github.com/saadtariq10/YourRepoName.git
   cd YourRepoName
   ```
2. **Install dependencies:**  
   ```sh
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**  
   Create a `.env` file and add your **Groq API Key**:  
   ```
   GROQ_API_KEY=your_api_key_here
   ```
4. **Run the application:**  
   ```sh
   streamlit run app.py
   ```

## 📌 Requirements  
- **Python 3.x**  
- **Streamlit**  
- **LangChain**  
- **FAISS**  
- **PyPDFLoader**  
- **HuggingFace Embeddings**  
- **pytesseract** (For OCR)  
- **pdf2image** (For scanned PDFs)  

## 🛠️ Usage  
1. **Upload a PDF document.**  
2. **The app extracts text** (uses OCR if the document is scanned).  
3. **Click "Embed Document"** to generate vector embeddings.  
4. **Ask any question** related to the document.  
5. **Get AI-generated answers** in real-time!  

## 🏆 License  
This project is **open-source** and free to use.  

---
Made with ❤️ using **Python, Streamlit, and LLMs**.
```
