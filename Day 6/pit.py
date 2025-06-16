import streamlit as st
import google.generativeai as genai
import PyPDF2

# 🔑 Set API Key for Gemini
GOOGLE_API_KEY = "AIzaSyA1ViY5PHERmKH1bTCZY5DFxmaRsElSXAs"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# 🎯 Streamlit App Setup
st.set_page_config(page_title="Pitch Deck AI", layout="centered")
st.title("🚀 AI-Driven Pitch Deck Generator")
st.markdown("Generate slides using Google Gemini + a local Market PDF file.")

# 📝 Input fields
startup_name = st.text_input("Startup Name", "EcoCart")
description = st.text_area("Describe your startup idea")

# 🧠 Generate pitch
if st.button("Generate Pitch Deck"):
    if not description.strip():
        st.warning("⚠️ Please enter a startup description.")
    else:
        # Step 1: Load and extract text from local PDF file
        pdf_path = "/Users/user/pitch.py/ecostart_pitch_deck.pdf"  # 🔁 Change to your actual path
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                pdf_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pdf_text += text
        except Exception as e:
            st.error(f"❌ Error reading PDF: {e}")
            st.stop()

        if not pdf_text.strip():
            st.error("❌ No text extracted from the PDF.")
        else:
            # Step 2: Prompt Gemini with extracted PDF text
            with st.spinner("🧠 Generating pitch deck with Gemini..."):
                prompt = f"""
You are a startup pitch expert.

Startup Name: {startup_name}
Startup Description: {description}

Market Research (from local PDF):
{pdf_text[:3000]}

Generate concise content for these slides:
1. Problem
2. Solution
3. Market Size
4. Business Model
5. Team
6. Roadmap
                """

                response = model.generate_content(prompt)
                st.success("✅ Pitch deck generated!")

                # Step 3: Display slides
                st.subheader("📊 Slide Content")
                slides = response.text.split("\n\n")
                for slide in slides:
                    st.markdown(slide)
