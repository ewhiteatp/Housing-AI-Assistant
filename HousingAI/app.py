import streamlit as st
import openai
import pinecone

# Access API keys securely from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENVIRONMENT"]
)

# Streamlit app interface
st.title("Housing AI Assistant")
st.write("Upload a PDF document and ask questions about its content.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    st.success(f"File uploaded successfully: {uploaded_file.name}")
    # Placeholder for processing and question answering
    st.write("This is where PDF processing and AI answering will happen.")
else:
    st.info("Please upload a file to proceed.")