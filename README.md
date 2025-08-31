# 📚 Multi-PDF Q&A Bot with LangChain & Hugging Face

An interactive Streamlit application that lets you **upload multiple PDF documents** and ask natural language questions based on their content.  
It uses **LangChain**, **HuggingFace models**, and **FAISS** for document embedding, retrieval, and conversational Q&A.

---

## 🚀 Features
- Upload one or more PDFs and extract text automatically
- Chunk documents for efficient semantic search
- Embed text with `sentence-transformers/all-MiniLM-L12-v2`
- Store & retrieve vectors using **FAISS**
- Ask natural language questions and get accurate answers
- Conversation history for context-aware responses
- Caches processed PDFs for faster reloads

---

## 🛠️ Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)  
- **LLM**: HuggingFace `google/flan-t5-large`  
- **Embeddings**: `sentence-transformers/all-MiniLM-L12-v2`  
- **Vector Store**: FAISS  
- **PDF Parsing**: pdfplumber  
- **LangChain** for chaining LLM + retrieval

---
## Project Srructure
├── app.py # Main Streamlit app
├── requirements.txt # Dependencies
├── faiss_cache/ # Cached vector stores
├── .env # HuggingFace API key (optional)
└── README.md # Documentation

## ⚙️ Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/mohdsami01/multi-pdf-Q-A-App.git
   cd multi-pdf-Q-A-App

2. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate    # Windows (PowerShell)

3. Install dependencies
pip install -r requirements.tx

4.Set HuggingFace API token 
HUGGINGFACEHUB_API_TOKEN=your_api_token_here

5.Run the Streamlit app:

streamlit run app.py



## 📂 Project Structure
