Name:Madhumitha
TITLE:Agentic AI System for User-Centered Idea Generation and Selection
# 💡 AI Idea Generator using LangChain + Gemini + RAG

This Streamlit application uses a multi-agent AI workflow to generate, evaluate, and refine user-centered startup ideas, using trend data retrieved from a PDF via RAG (Retrieval-Augmented Generation).

## 🔧 Features

- ✅ 5-step agentic pipeline
- 🔍 RAG with FAISS + HuggingFace for trend-based idea generation
- 🤖 Google Gemini Pro (via LangChain) as the LLM
- 📄 Source PDF-based idea validation only (prevents hallucination)
- 📊 Streamlit UI with controlled LLM invocation

## 🧠 Agent Workflow

1. **Opportunity Context Agent** — Extracts pain points and project scope.
2. **Trend-Aware Idea Generator (RAG)** — Uses vector search to retrieve trend-aligned ideas.
3. **Problem-Solution Mapper** — Maps each idea to a problem and adds core features.
4. **Idea Evaluator & Selector** — Ranks ideas based on relevance, feasibility, and innovation.
5. **Idea Description Composer** — Outputs final product description (name, user, value prop).

## 📁 Project Structure

```
📦 idea_generator_ai
 ┣ 📜 app.py
 ┃ ┗ 📄 untitled document-google docs.pdf
 ┣ 📜 .env
 ┗ 📜 requirements.txt
```

## 📚 Setup Instructions

```bash
# Install required packages
pip install -r requirements.txt
```

Ensure your `.env` contains your Google API key:

```env
GOOGLE_API_KEY=your_gemini_key
```

## 🚀 Run the App

```bash
streamlit run app.py
```

## 📝 Notes

- The system uses a **HuggingFace MiniLM** model for vector embedding.
- All startup ideas are **strictly derived** from your trend PDF (RAG-enabled).
- LLM will refuse to proceed if no relevant content is found.

---

Built with ❤️ using LangChain, Streamlit, Gemini Pro, and FAISS.

code video link: https://drive.google.com/file/d/1IVgXjXvbdyGEKcJbGIpGy0BfBeF6qS4I/view?usp=drive_link