import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from flipkart.config import Config 

# --- Configuration (replace with your actual values) ---
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_COLLECTION_NAME = "my_langchain_collection"

# Ensure environment variables are set
if not all([ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN]):
    raise ValueError("Please set ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, and OPENAI_API_KEY environment variables.")

# Initialize embedding model
embeddings = HuggingFaceEndpointEmbeddings(model=Config.EMBEDDING_MODEL)

# Initialize Astra DB Vector Store
vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=ASTRA_DB_COLLECTION_NAME,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
)

# Sample documents
documents = [
    Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "fable"}),
    Document(page_content="Artificial intelligence is a rapidly developing field.", metadata={"source": "tech"}),
    Document(page_content="Astra DB is a serverless, multi-cloud database.", metadata={"source": "database"}),
]

# Add documents to the vector store
print("Adding documents to Astra DB...")
vector_store.add_documents(documents)
print("Documents added.")

# Perform a similarity search
query = "What is Astra DB?"
print(f"\nPerforming similarity search for: '{query}'")
results = vector_store.similarity_search(query, k=1)

# Print results
print("\nSearch Results:")
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")