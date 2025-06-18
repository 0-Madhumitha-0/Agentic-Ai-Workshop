import os
import streamlit as st
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# === GOOGLE GENAI SETUP ===
GOOGLE_API_KEY = "AIzaSyCdnYK_FQY_5-qkOBDRYMaI-nQDW1fFVA0"
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)

# === LOAD INTERNAL PDF ===
pdf_path = "/Users/user/demo1/AgriTech_Market_Research.pdf"
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(split_docs, embedding)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# === AGENT 1: CONTEXT EXTRACTION ===
def extract_context_fn(user_input: str) -> str:
    prompt = f"""
Extract startup context (Product, Domain, Value Proposition, and Traction) from the following input:

{user_input}
"""
    return llm.invoke(prompt).content

context_tool = Tool(
    name="ContextExtractor",
    func=extract_context_fn,
    description="Extracts startup product, domain, value proposition, and traction."
)

# === AGENT 2: MARKET RESEARCH (RAG) ===
def market_research_fn(user_input: str) -> str:
    return qa_chain.run(f"Provide market size, competitors, and key trends for this startup: {user_input}")

market_tool = Tool(
    name="MarketResearcher",
    func=market_research_fn,
    description="Retrieves market size, competitor analysis, and trends using the RAG-embedded PDF."
)

# === AGENT 3: SLIDE GENERATION ===
slide_prompt = ChatPromptTemplate.from_template("""
Create pitch deck slides for the following startup.

Startup Context:
{context}

Market Info:
{market}

Include:
- Problem
- Solution
- Market Size
- Business Model
- Team
- Roadmap
""")

def slide_generation_fn(user_input: str) -> str:
    context = extract_context_fn(user_input)
    market = market_research_fn(user_input)
    prompt = slide_prompt.format(context=context, market=market)
    return llm.invoke(prompt).content

slide_tool = Tool(
    name="SlideGenerator",
    func=slide_generation_fn,
    description="Generates pitch deck slide content from startup context and market information."
)

# === BUILD AGENT EXECUTOR ===
tools = [context_tool, market_tool, slide_tool]

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert startup advisor AI. Use the available tools to build a pitch deck."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template
)

executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# === STREAMLIT UI ===
st.set_page_config(page_title="AI Pitch Deck Builder", layout="wide")
st.title("🚀 AI Pitch Deck Builder (3-Agent System)")

st.markdown("Enter your startup description including **problem, product, traction, and differentiation**.")

user_input = st.text_area("💡 Describe Your Startup", height=250)

if st.button("Generate Pitch Deck"):
    if not user_input.strip():
        st.warning("Please enter a valid startup description.")
    else:
        with st.spinner("Running AI agents..."):
            result = executor.invoke({"input": user_input})
            st.success("✅ Pitch deck created!")
            st.markdown(result["output"])
