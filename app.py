import os
import io
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
import pytesseract

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from docx import Document as DocxDocument

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="ResearchVault AI",
    page_icon="📚",
    layout="wide"
)

# Custom CSS: Title Styling & Hiding 200MB Limit Text
st.markdown("""
    <style>
    .main-title { text-align: center; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .sub-title { text-align: center; color: #4B5563; font-size: 1.1rem; margin-bottom: 25px; }
    
    /* Hide the default 200MB file size & extension instruction text below file uploader */
    [data-testid="stFileUploaderInstructions"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INITIALIZE SESSION STATE
# ==========================================
if "api_authenticated" not in st.session_state:
    st.session_state["api_authenticated"] = False
if "api_key" not in st.session_state:
    st.session_state["api_key"] = ""
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "file_sig" not in st.session_state:
    st.session_state["file_sig"] = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "total_chunks_read" not in st.session_state:
    st.session_state["total_chunks_read"] = 0

# ==========================================
# 3. SECURE ACCESS PORTAL (LOCK SCREEN)
# ==========================================
if not st.session_state["api_authenticated"]:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h2 style='text-align: center;'>📚 ResearchVault AI</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: gray;'>Stack: OpenAI | FAISS | LangChain | Streamlit</p>", unsafe_allow_html=True)
        
        api_key_input = st.text_input("Enter OpenAI API Key", type="password")

        if st.button("Unlock Dashboard", use_container_width=True):
            if api_key_input.strip() != "":
                try:
                    test_embeddings = OpenAIEmbeddings(openai_api_key=api_key_input.strip())
                    st.session_state["api_key"] = api_key_input.strip()
                    st.session_state["api_authenticated"] = True
                    st.rerun()
                except Exception as e:
                    st.error("Authentication failed! Please check your OpenAI API key.")
            else:
                st.error("API key cannot be empty.")
    st.stop()

# ==========================================
# 4. SIDEBAR CONFIGURATION (Parameters Only)
# ==========================================
with st.sidebar:
    st.title("⚙️ RAG Control Panel")
    st.caption("**App:** ResearchVault AI")
    
    st.markdown("---")
    st.subheader("🛠️ FAISS Chunking Parameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", min_value=0, max_value=400, value=200, step=25)
    
    st.markdown("---")
    st.subheader("🔍 Retrieval Settings")
    top_k = st.slider("Retrieved Chunks (top_k)", min_value=1, max_value=10, value=4, step=1)
    model_name = st.selectbox("LLM Model", ["gpt-4o-mini", "gpt-4o"])

    # Display Active Stack & Metrics
    st.markdown("---")
    st.subheader("📊 Active RAG Metrics")
    st.caption("**Tech Stack:** OpenAI Key, FAISS, LangChain, Streamlit")
    st.write(f"• **Chunk Size:** {chunk_size}")
    st.write(f"• **Chunk Overlap:** {chunk_overlap}")
    st.write(f"• **Total Chunks Read:** {st.session_state['total_chunks_read']}")
    st.write(f"• **Retrieved Chunks (k):** {top_k}")

    st.markdown("---")
    if st.button("🔒 Lock / Clear Session", use_container_width=True):
        st.session_state["api_authenticated"] = False
        st.session_state["vectorstore"] = None
        st.session_state["messages"] = []
        st.session_state["total_chunks_read"] = 0
        st.rerun()

# ==========================================
# 5. MULTI-FORMAT DOCUMENT PROCESSING & FAISS
# ==========================================
@st.cache_resource
def build_vectorstore(file_obj, c_size, c_overlap, api_key):
    docs = []
    file_name = file_obj.name.lower()
    data = file_obj.getvalue()

    try:
        # PDF Format
        if file_name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                tmp_pdf.write(data)
                tmp_pdf_path = tmp_pdf.name
            loader = PyPDFLoader(tmp_pdf_path)
            docs.extend(loader.load())
            os.unlink(tmp_pdf_path)

        # DOCX Format
        elif file_name.endswith(".docx"):
            doc = DocxDocument(io.BytesIO(data))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            docs.append(Document(page_content=text, metadata={"source": file_obj.name, "page": "DOCX Content"}))

        # EXCEL Format (.xlsx, .xls)
        elif file_name.endswith((".xlsx", ".xls")):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_excel:
                tmp_excel.write(data)
                tmp_excel_path = tmp_excel.name
            excel_dfs = pd.read_excel(tmp_excel_path, sheet_name=None)
            for sheet_name, df in excel_dfs.items():
                csv_data = df.to_string(index=False)
                doc_text = f"Sheet Name: {sheet_name}\n\nDataset Content:\n{csv_data}"
                docs.append(Document(page_content=doc_text, metadata={"source": file_obj.name, "page": f"Sheet: {sheet_name}"}))
            os.unlink(tmp_excel_path)

        # IMAGE Format (.png, .jpg, .jpeg) with Safe OCR Error Handling
        elif file_name.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(io.BytesIO(data))
            try:
                ocr_text = pytesseract.image_to_string(img)
                if not ocr_text.strip():
                    ocr_text = "[Image processed, but no readable text was detected via OCR.]"
            except Exception as e:
                st.error("Tesseract OCR system path par detect nahi hua. SystemDependencies check karein.")
                ocr_text = "[Image uploaded, but OCR extraction failed due to missing system dependencies.]"
            
            docs.append(Document(page_content=ocr_text, metadata={"source": file_obj.name, "page": "Image OCR"}))

        if not docs:
            return None, 0

        # FAISS Chunking & Embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
        split_docs = text_splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        
        return vectorstore, len(split_docs)

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, 0

# ==========================================
# 6. RAG CHAIN BUILDER
# ==========================================
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert academic research assistant.\n"
     "Answer questions strictly using the provided document context.\n"
     "If the answer cannot be found, state: \"I couldn't find that in the uploaded document.\"\n"
     "Cite sources and page/sheet labels clearly.\n\n"
     "Context:\n{context}"
    ),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join([f"[Source: {d.metadata.get('source', 'Unknown')} | Location: {d.metadata.get('page', 'N/A')}]\n{d.page_content}" for d in docs])

def get_chain(vectorstore, model, k):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model_name=model, temperature=0.1, openai_api_key=st.session_state["api_key"])
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# ==========================================
# 7. MAIN UI & CHAT INTERFACE
# ==========================================
st.markdown("<h1 class='main-title'>📚 ResearchVault AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Multi-Format RAG Engine (PDF, DOCX, Excel, Image OCR)</p>", unsafe_allow_html=True)

# Front Dashboard Document Uploader (Main Section)
uploaded_file = st.file_uploader(
    "Attach Document (PDF, DOCX, EXCEL, IMAGE)", 
    type=["pdf", "docx", "xlsx", "xls", "png", "jpg", "jpeg"]
)

# Trigger Vector Indexing on Upload
if uploaded_file is not None:
    file_sig = (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)

    if st.session_state.get("file_sig") != file_sig:
        with st.spinner("🔄 Extracting text & indexing chunks into FAISS vector space..."):
            vstore, chunk_count = build_vectorstore(
                uploaded_file, chunk_size, chunk_overlap, st.session_state["api_key"]
            )
            if vstore:
                st.session_state["vectorstore"] = vstore
                st.session_state["file_sig"] = file_sig
                st.session_state["total_chunks_read"] = chunk_count
                st.session_state["messages"] = []
                st.success(f"✅ Indexed **{uploaded_file.name}** into **{chunk_count} FAISS chunks**!")
                st.rerun()

st.markdown("---")

# Chat Interface Section
if st.session_state["vectorstore"] is not None:
    # Replay History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if question := st.chat_input("Ask any question about your document..."):
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            chain, retriever = get_chain(st.session_state["vectorstore"], model_name, top_k)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving FAISS chunks & generating response..."):
                    answer = chain.invoke(question)
                    st.markdown(answer)

                # Transparency Drawer: Retrieved Chunks
                with st.expander(f"🔍 Retrieved Chunks Transparency (top_k = {top_k})"):
                    retrieved_docs = retriever.invoke(question)
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Retrieved Chunk {i+1} | Location: {doc.metadata.get('page', 'N/A')}**")
                        st.text(doc.page_content[:400] + "...")
                        st.divider()

            st.session_state["messages"].append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("👆 Upload a document using the box above to start asking questions.")