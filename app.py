import streamlit as st
import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from io import BytesIO
import PyPDF2
import pinecone

# Streamlit App Configuration
st.title("Housing AI Assistant")
st.write("Upload a PDF document and ask questions about its content.")

# Access API keys securely from Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize Pinecone using the Pinecone class
pinecone.init(
    api_key=st.secrets["PINECONE_API_KEY"],
    environment=st.secrets["PINECONE_ENVIRONMENT"]
)

# Define the Pinecone index name
INDEX_NAME = "housing-ai-index"

# Check if the index exists and create it if it doesn't
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Dimension of OpenAI embeddings
        metric="cosine"  # Use "cosine" for similarity
    )

# Initialize Pinecone Vector Store
index = pinecone.Index(INDEX_NAME)
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
vector_store = PineconeVectorStore(index, embedding_function=embeddings.embed_query)

# File Upload Section
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    try:
        # Read the PDF file
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

        # Extract text from PDF
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Prepare documents to add to Pinecone
        documents = [
            {
                "id": f"chunk-{i}",
                "page_content": chunk,
                "metadata": {"source": uploaded_file.name}
            }
            for i, chunk in enumerate(chunks)
        ]

        # Add documents to Pinecone Vector Store
        vector_store.add_documents(documents)
        st.success("PDF uploaded and indexed successfully!")
    except Exception as e:
        st.error(f"Error processing the PDF: {e}")

# Question and Answer Section
st.write("### Ask a question:")
question = st.text_input("Type your question below:")
if question:
    try:
        # Create a retriever and QA chain
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(openai_api_key=openai.api_key, model="gpt-4"),
            retriever=retriever
        )

        # Get the answer
        answer = qa_chain.run(question)
        st.write("**Answer:**", answer)
    except Exception as e:
        st.error(f"Error answering your question: {e}")
