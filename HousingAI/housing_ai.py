import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from pinecone import Pinecone, ServerlessSpec
import tempfile

# Set API keys
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_7GLZih_9N5zJucrPbJdidh6iE2JV7qkNfXS2Pg9F9bBUx7JvLrwwVUTfZmBfF5p86fGxd4"
PINECONE_ENVIRONMENT = "us-west2-gcp"

# Initialize Pinecone
pinecone_instance = Pinecone(
    api_key=PINECONE_API_KEY, 
    environment=PINECONE_ENVIRONMENT
)

# Check if index exists, and create if it doesn't
INDEX_NAME = "housing-ai-index"
if INDEX_NAME not in pinecone_instance.list_indexes().names():
    pinecone_instance.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# Initialize the vector store
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
    index_name=INDEX_NAME,
    embedding_function=embeddings.embed_query
)

# Streamlit app setup
st.title("Housing AI Assistant")
st.write("Upload your documents or ask a question below.")

# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# Process the uploaded file
if uploaded_file:
    st.success(f"PDF uploaded successfully! File name: {uploaded_file.name}")
    
    # Save the file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load PDF and split text
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # Add data to Pinecone
    vector_store.from_documents(chunks, embedding_function=embeddings.embed_query)
    st.success("Document processed and added to the vector database!")

# Question and Answer Section
st.write("### Ask a question:")
question = st.text_input("Your question:")
if question:
    try:
        # Create a retrieval-based QA chain
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4"),
            retriever=retriever,
        )
        
        # Get the answer
        answer = qa_chain.run(question)
        st.write("**Answer:**", answer)
    except Exception as e:
        st.error(f"Error processing your question: {e}")

# Closing note
st.info("This is a simplified test. Once it works, we will add more features.")