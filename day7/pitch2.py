import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 🔑 Set Gemini API key
genai.configure(api_key="AIzaSyCdnYK_FQY_5-qkOBDRYMaI-nQDW1fFVA0")

# --- Call Gemini ---
def call_gemini(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text.strip()

# --- Agent 1: Extract Startup Context ---
def extract_context(startup_desc):
    prompt = f"""
Extract the following details in clear sentences:
- Product
- Domain
- Value Proposition
- Traction

Description:
{startup_desc}
"""
    return call_gemini(prompt)

# --- Agent 2: Load PDF and RAG ---
def load_pdf_and_create_vectorstore(pdf_path):
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

def run_rag_query(vectorstore, query):
    docs = vectorstore.similarity_search(query, k=3)
    combined = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
Use the following context to answer the question in plain English.

Context:
{combined}

Q: {query}
A:"""
    return call_gemini(prompt)

# --- Agent 3: Slide Generator ---
def generate_slides(context, market_info):
    prompt = f"""
Based on the context and market data, create the following pitch deck slides with titles and content:

Startup Context:
{context}

Market Info:
{market_info}

Slides:
1. Problem
2. Solution
3. Market Size
4. Business Model
5. Team
6. Roadmap
"""
    return call_gemini(prompt)

# --- Streamlit UI ---
st.set_page_config(page_title="AI Pitch Deck Builder", layout="centered")
st.title("🚀 AI Pitch Deck Builder")
st.caption("Generate a pitch deck using multi-agent AI (Context Extraction + RAG + Slide Builder)")

# --- Sidebar ---
st.sidebar.header("📋 About")
st.sidebar.info("""
This tool uses Google Gemini and LangChain agents to extract startup insights and build a pitch deck automatically.
- No file upload required
- PDF is read from code
- Powered by Gemini (`google.generativeai`)
""")

# --- Input Section ---
st.subheader("1️⃣ Describe Your Startup")
startup_input = st.text_area("What does your startup do?", height=200, placeholder="E.g., We built an AI health assistant...")

# --- Buttons Section ---
generate_all = st.button("✨ Generate Full Pitch Deck")

# Hardcoded PDF file path (no UI upload)
pdf_path = "/Users/user/demo1/ecostart_pitch_deck.pdf"

# --- Output Sections ---
if generate_all:
    if not startup_input.strip():
        st.error("Please enter a startup description.")
    else:
        # Agent 1
        with st.spinner("🧠 Extracting startup context..."):
            context_output = extract_context(startup_input)
            st.success("Startup context extracted!")

        # Agent 2
        with st.spinner("🔎 Running market research (RAG)..."):
            vectorstore = load_pdf_and_create_vectorstore(pdf_path)
            market_output = run_rag_query(vectorstore, "What is the market size and trend for this domain?")
            st.success("Market data retrieved!")

        # Agent 3
        with st.spinner("🛠️ Generating pitch slides..."):
            slide_output = generate_slides(context_output, market_output)
            st.success("Pitch deck generated!")

        # --- Display Output ---
        st.subheader("🧠 Startup Context")
        st.markdown(f"<div style='background-color:#f0f0f5; padding:10px; border-radius:10px'>{context_output}</div>", unsafe_allow_html=True)

        st.subheader("📊 Market Research (RAG)")
        st.markdown(f"<div style='background-color:#f0f0f5; padding:10px; border-radius:10px'>{market_output}</div>", unsafe_allow_html=True)

        st.subheader("📋 Pitch Deck Slides")
        for line in slide_output.split("\n"):
            if line.strip().endswith(":"):
                st.markdown(f"### {line.strip()}")
            else:
                st.markdown(f"{line.strip()}")

