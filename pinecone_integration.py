import os
from pinecone import Pinecone
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid

# Replace with your actual API keys
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment="us-west-2"
)
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "housing-ai-index"  # Replace with your Pinecone index name
index = pc.Index(index_name)
print(f"Connected to Pinecone index: {index_name}")

# Function to generate embeddings using OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embeddings using OpenAI's API."""
    try:
        response = openai.Embedding.create(input=text, model=model)
        embedding = response["data"][0]["embedding"]  # Access embedding vector correctly
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to chunk text
def chunk_text(content, chunk_size=1000, chunk_overlap=200):
    """Split text into manageable chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(content)

# Function to upload chunks to Pinecone
def upload_to_pinecone(chunks, index):
    """Upload text chunks to Pinecone as vectors."""
    for chunk in chunks:
        try:
            vector = get_embedding(chunk)
            if vector:
                metadata = {"source": "Document Title"}  # Customize metadata
                index.upsert([(str(uuid.uuid4()), vector, metadata)])  # Upload to Pinecone
        except Exception as e:
            print(f"Error uploading chunk: {e}")

# Function to query Pinecone
def query_pinecone(question, index, top_k=5):
    """Retrieve relevant chunks from Pinecone."""
    try:
        query_vector = get_embedding(question)
        if query_vector:
            results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
            return [match["metadata"]["source"] for match in results["matches"]]
        else:
            print("Error: Unable to generate query vector.")
            return []
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# Generate AI response using retrieved context
def generate_response(question, index):
    """Generate a response using OpenAI's GPT model."""
    retrieved_chunks = query_pinecone(question, index)
    context = " ".join(retrieved_chunks)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI specializing in housing policies and regulations."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error querying OpenAI: {e}")
        return "An error occurred while generating the response."

# Example Workflow
if __name__ == "__main__":
    # Example text for ingestion
    document_text = """
    HUD regulations require that rent calculations consider income inclusions and exclusions. 
    Adjusted income is determined by subtracting eligible deductions from the total annual income. 
    HOTMA updates have introduced new thresholds for asset limits.
    """

    # Chunk and upload to Pinecone
    chunks = chunk_text(document_text)
    upload_to_pinecone(chunks, index)
    print(f"Uploaded {len(chunks)} chunks to Pinecone.")

    # Query and get a response
    question = "What are the income exclusions under HOTMA?"
    answer = generate_response(question, index)
    print("AI Response:", answer)
