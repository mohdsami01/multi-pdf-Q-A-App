import streamlit as st
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import tempfile
import pdfplumber
from pathlib import Path
from transformers.pipelines import pipeline
import hashlib


load_dotenv()
hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_api_token:
    st.warning(
        "Hugging Face API token not found. Local inference will proceed without it."
    )


st.set_page_config(
    page_title="PDF Q&A Bot with LangChain & Hugging Face", layout="wide"
)
st.title("📚 PDF Q&A APP")
st.write("Upload one or more PDFs and ask questions based on their content!")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "pdf_hashes" not in st.session_state:
    st.session_state.pdf_hashes = None


def get_files_hash(uploaded_files):
    hasher = hashlib.md5()
    for uploaded_file in sorted(uploaded_files, key=lambda x: x.name):
        hasher.update(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
    return hasher.hexdigest()


def process_pdfs(uploaded_files):
    try:
        all_docs = []
        for uploaded_file in uploaded_files:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            with pdfplumber.open(tmp_file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        all_docs.append(
                            {
                                "page_content": text,
                                "metadata": {
                                    "page": page_num + 1,
                                    "source": uploaded_file.name,
                                },
                            }
                        )

            os.unlink(tmp_file_path)

        from langchain_core.documents import Document

        docs = [
            Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for doc in all_docs
        ]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L12-v2",
            encode_kwargs={"batch_size": 32},
        )
        vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)

        cache_dir = Path("faiss_cache")
        cache_dir.mkdir(exist_ok=True)
        files_hash = get_files_hash(uploaded_files)
        vectorstore.save_local(cache_dir / files_hash)

        return vectorstore, files_hash
    except Exception as e:
        st.error(f"Error processing PDFs: {str(e)}")
        return None, None


def load_cached_vectorstore(files_hash):
    try:
        cache_dir = Path("faiss_cache")
        if (cache_dir / files_hash).exists():
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L12-v2",
                encode_kwargs={"batch_size": 32},
            )
            return FAISS.load_local(
                cache_dir / files_hash, embeddings, allow_dangerous_deserialization=True
            )
        return None
    except Exception as e:
        st.warning(f"Error loading cached vector store: {str(e)}")
        return None


def initialize_qa_chain(vectorstore):
    try:

        llm = HuggingFacePipeline.from_model_id(
            model_id="google/flan-t5-large",
            task="text2text-generation",
            pipeline_kwargs={"max_length": 512, "temperature": 0.7},
            device=-1,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=retriever, return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error initializing QA chain: {str(e)}")
        return None


uploaded_files = st.file_uploader(
    "Upload one or more PDF files", type=["pdf"], accept_multiple_files=True
)
if uploaded_files:
    with st.spinner("Processing PDFs..."):

        files_hash = get_files_hash(uploaded_files)
        if files_hash != st.session_state.pdf_hashes:

            cached_vectorstore = load_cached_vectorstore(files_hash)
            if cached_vectorstore:
                st.session_state.vectorstore = cached_vectorstore
                st.session_state.pdf_hashes = files_hash
            else:

                st.session_state.vectorstore, st.session_state.pdf_hashes = (
                    process_pdfs(uploaded_files)
                )
        else:
            st.session_state.vectorstore = (
                load_cached_vectorstore(files_hash) or st.session_state.vectorstore
            )

        if st.session_state.vectorstore:
            st.session_state.qa_chain = initialize_qa_chain(
                st.session_state.vectorstore
            )
            st.success(
                f"{len(uploaded_files)} PDF(s) processed successfully! You can now ask questions."
            )


if st.session_state.qa_chain:
    with st.form("question_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="e.g., What is the main topic of the documents?",
        )
        submitted = st.form_submit_button("Ask")

        if submitted and question:
            with st.spinner("Generating answer..."):

                result = st.session_state.qa_chain(
                    {
                        "question": question,
                        "chat_history": st.session_state.chat_history,
                    }
                )
                answer = result["answer"]
                source_documents = result["source_documents"]

                st.session_state.chat_history.append((question, answer))

                st.markdown("**Answer:**")
                st.write(answer)

                with st.expander("Source Documents", expanded=False):
                    for i, doc in enumerate(source_documents, 1):
                        st.markdown(
                            f"**Document {i}** (Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'N/A')}):"
                        )
                        st.write(f"{doc.page_content[:200]}...")
else:
    st.info("Please upload one or more PDF files to start asking questions.")


st.markdown("### Chat History")
if st.session_state.chat_history:
    for i, (q, a) in enumerate(st.session_state.chat_history, 1):
        st.markdown(f"**Q{i}**: {q}")
        st.markdown(f"**A{i}**: {a}")
else:
    st.write("No questions asked yet.")
