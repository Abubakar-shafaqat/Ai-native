from qdrant_client import QdrantClient, models
from .config import QDRANT_HOST, QDRANT_API_KEY

QDRANT_COLLECTION_NAME = "book_content"

qdrant_client_instance: QdrantClient = None

def get_qdrant_client() -> QdrantClient:
    global qdrant_client_instance
    if qdrant_client_instance is None:
        qdrant_client_instance = QdrantClient(
            host=QDRANT_HOST,
            api_key=QDRANT_API_KEY,
        )
    return qdrant_client_instance

async def initialize_qdrant_collection():
    client = get_qdrant_client()
    # Check if collection exists, create if not
    collections_response = await client.get_collections()
    existing_collections = [c.name for c in collections_response.collections]

    if QDRANT_COLLECTION_NAME not in existing_collections:
        print(f"Creating Qdrant collection: {QDRANT_COLLECTION_NAME}")
        await client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE), # Assuming OpenAI embeddings
        )
        print(f"Collection {QDRANT_COLLECTION_NAME} created.")
    else:
        print(f"Collection {QDRANT_COLLECTION_NAME} already exists.")

async def close_qdrant_client():
    global qdrant_client_instance
    if qdrant_client_instance:
        # QdrantClient doesn't have an explicit close method for HTTP,
        # but if using gRPC, it might. For now, we just reset the instance.
        qdrant_client_instance = None
        print("Qdrant client instance reset.")
