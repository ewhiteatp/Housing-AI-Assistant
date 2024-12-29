import openai
import json
import pinecone
from flask import Flask, request, jsonify

app = Flask(__name__)

# Replace with your actual keys
openai.api_key = "sk-proj-..."  # OpenAI API Key
pinecone.init(api_key="pcsk_...", environment="us-west1-gcp")  # Pinecone API Key and Environment

# Initialize Pinecone index
index_name = "my-index"
index = pinecone.Index(index_name)

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data["question"]

        # Generate embeddings with updated API usage
        embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=question
        )
        embedding = embedding_response["data"][0]["embedding"]

        # Query Pinecone
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )

        # Format and return results
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)