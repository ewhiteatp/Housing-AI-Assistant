import os
import pinecone
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Retrieve Pinecone API key and environment from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Ensure the API key and environment are set
if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    raise ValueError("Pinecone API key or environment is missing. Please check your .env file.")

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

# Define index name and dimension
INDEX_NAME = "housing-ai"
DIMENSION = 512  # Replace with the correct vector dimension for your use case

# Create or connect to an index
if INDEX_NAME not in pinecone.list_indexes():
    print(f"Creating Pinecone index '{INDEX_NAME}'...")
    pinecone.create_index(name=INDEX_NAME, dimension=DIMENSION)

index = pinecone.Index(INDEX_NAME)

# Function to upsert vectors
def upsert_vectors(vectors: list[tuple[str, list[float]]]):
    """
    Upserts vectors into the Pinecone index.

    Args:
        vectors (list): A list of tuples (id, vector).
    """
    print(f"Upserting {len(vectors)} vectors...")
    index.upsert(vectors=vectors)

# Function to query vectors
def query_vector(vector: list[float], top_k: int = 10):
    """
    Queries the Pinecone index with a given vector.

    Args:
        vector (list): The vector to query.
        top_k (int): The number of top results to return.

    Returns:
        dict: Query results from Pinecone.
    """
    print("Querying the Pinecone index...")
    return index.query(vector=vector, top_k=top_k, include_metadata=True)

# Example usage
if __name__ == "__main__":
    # Test Pinecone connection
    print("Pinecone indexes:", pinecone.list_indexes())

    # Example vector (must match the dimension of the index)
    example_vectors = [("id1", [0.1] * DIMENSION)]
    upsert_vectors(example_vectors)

    # Query the index with an example vector
    query_result = query_vector([0.1] * DIMENSION)
    print("Query result:", query_result)

