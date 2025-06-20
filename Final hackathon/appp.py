import os
import streamlit as st
from dotenv import load_dotenv
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, LLMChain
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import PromptTemplate

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# RAG Setup
PDF_PATH = "/Users/user/idea_generator_ai/Untitled document - Google Docs.pdf"
loader = PyPDFLoader(PDF_PATH)
pages = loader.load_and_split()

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(pages, embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

custom_prompt = PromptTemplate.from_template("""
Use only the information in the following context to generate 5–10 innovative startup ideas based on real-world trends.

Context:
{context}

Question:
{question}

Answer:
""")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True,
    verbose=True
)

# Agent functions
def opportunity_context_agent(input_text: str) -> str:
    prompt = PromptTemplate.from_template(
        "Extract 3-5 user pain points and summarize the opportunity scope.\n\nInput:\n{input_text}\n\nFormat:\nPain Points:\n1. ...\nOpportunity Scope: ..."
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    result = chain.invoke({"input_text": input_text})
    return result["text"]

def trend_idea_generator_agent(opportunity_summary: str) -> dict:
    query = f"What are 5–10 innovative startup ideas aligned with this opportunity context?\n\n{opportunity_summary}"
    result = rag_chain.invoke({"query": query})
    docs = result["source_documents"]
    if not docs:
        return {"result": "No relevant trend data found in RAG. Cannot generate ideas.", "source_documents": []}
    return result

def problem_solution_mapper_agent(ideas_and_context: str) -> str:
    prompt = PromptTemplate.from_template(
        "Map each idea to a user pain point and list 2 innovative features.\n\nInput:\n{ideas_and_context}\n\nFormat:\n1. Idea:\nProblem Solved:\nFeatures:\n- ...\n- ..."
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    result = chain.invoke({"ideas_and_context": ideas_and_context})
    return result["text"]

def idea_evaluator_selector_agent(mapped_ideas: str) -> str:
    prompt = PromptTemplate.from_template(
        "Score each idea on Relevance, Feasibility, and Innovation (1-10), rank them, and select the top idea.\n\nInput:\n{mapped_ideas}\n\nOutput Format:\nRanked List:\nBest Idea:\nJustification:"
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    result = chain.invoke({"mapped_ideas": mapped_ideas})
    return result["text"]

def idea_description_composer_agent(top_idea_summary: str) -> str:
    prompt = PromptTemplate.from_template(
        "Write a final description of the idea.\n\nInput:\n{top_idea_summary}\n\nOutput Format:\nName:\nDescription:\nTarget User:\nValue Proposition:"
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    result = chain.invoke({"top_idea_summary": top_idea_summary})
    return result["text"]

# Tools and Agent Executor
tools = [
    Tool(name="Opportunity Context Agent", func=opportunity_context_agent, description="Extracts user pain points and opportunity scope."),
    Tool(name="Trend-Aware Idea Generator", func=lambda x: trend_idea_generator_agent(x)["result"], description="Generates trend-aligned ideas using RAG."),
    Tool(name="Problem-Solution Mapper", func=problem_solution_mapper_agent, description="Maps each idea to a pain point and features."),
    Tool(name="Idea Evaluator", func=idea_evaluator_selector_agent, description="Scores and ranks ideas with justification."),
    Tool(name="Description Composer", func=idea_description_composer_agent, description="Composes final product description.")
]

agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

# Streamlit UI
st.set_page_config(page_title="AI Idea Generator (RAG)", layout="wide")
st.title("AI Idea Generator using RAG + Gemini")

with st.form("input_form"):
    metadata = st.text_area("Enter Project Metadata", height=100)
    feedback = st.text_area("Enter User Feedback or Research Insights", height=150)
    submitted = st.form_submit_button("Run Full Pipeline")

if submitted:
    with st.spinner("Running AI agents..."):
        context = opportunity_context_agent(metadata + "\n" + feedback)
        st.subheader("1. Opportunity Context")
        st.code(context)

        result = trend_idea_generator_agent(context)
        ideas = result["result"]
        docs = result["source_documents"]

        st.subheader("2. Trend-Aware Idea Generator (RAG)")
        st.code(ideas)

        if "No relevant trend data" not in ideas:
            mappings = problem_solution_mapper_agent(ideas + "\n" + context)
            st.subheader("3. Problem-Solution Mapper")
            st.code(mappings)

            evaluation = idea_evaluator_selector_agent(mappings)
            st.subheader("4. Idea Evaluator & Selector")
            st.code(evaluation)

            final = idea_description_composer_agent(evaluation)
            st.subheader("5. Final Idea Description")
            st.success("Final Output Ready")
            st.code(final)
        else:
            st.warning("No matching trend found in RAG. Please revise your input.")
            st.stop()