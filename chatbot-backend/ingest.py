import os
import asyncio
import time
from pathlib import Path
from qdrant_client import models
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from .config import GEMINI_API_KEY
from .qdrant_client_lib import get_qdrant_client, QDRANT_COLLECTION_NAME

# --- Configuration ---
DOCS_PATH = os.path.join(Path(__file__).parent.parent.parent, "docs")
EMBEDDING_MODEL = "models/embedding-001"
MAX_CHUNK_SIZE = 1000  # Max characters per chunk
OVERLAP_SIZE = 200     # Overlap between chunks for context

# --- Gemini Client ---
genai.configure(api_key=GEMINI_API_KEY)

def chunk_text(text: str, max_chunk_size: int, overlap_size: int) -> list[str]:
    """Splits a text into chunks based on character count with overlap."""
    if not text:
        return []
        
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_size
        chunks.append(text[start:end])
        start += max_chunk_size - overlap_size
    return chunks

async def generate_embedding(text: str, max_retries=5, delay=5) -> list[float]:
    """Generates an embedding using the Gemini API with retry logic."""
    for attempt in range(max_retries):
        try:
            # The Gemini API may have rate limits, so we include a retry mechanism
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=text,
                task_type="retrieval_document" # or "retrieval_query"
            )
            return result['embedding']
        except google_exceptions.ResourceExhausted as e:
            print(f"Rate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
        except Exception as e:
            print(f"An unexpected error occurred during embedding generation: {e}")
            return None # Or handle more gracefully
    print("Failed to generate embedding after multiple retries.")
    return None

async def ingest_book_content():
    qdrant_client = get_qdrant_client()
    # Ensure collection exists - updated vector size for Gemini
    await qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
    )
    print(f"Collection {QDRANT_COLLECTION_NAME} ensured/recreated with vector size 768.")

    points = []
    point_id_counter = 0

    for root, _, files in os.walk(DOCS_PATH):
        for file_name in files:
            if file_name.endswith(".md") or file_name.endswith(".mdx"):
                file_path = os.path.join(root, file_name)
                print(f"Processing: {file_path}")
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    chunks = chunk_text(content, MAX_CHUNK_SIZE, OVERLAP_SIZE)
                    for i, chunk in enumerate(chunks):
                        if not chunk.strip(): # Skip empty chunks
                            continue
                            
                        embedding = await generate_embedding(chunk)
                        
                        if embedding is None:
                            print(f"Skipping chunk {i} from {file_name} due to embedding generation failure.")
                            continue

                        # Store metadata for retrieval
                        metadata = {
                            "file_path": os.path.relpath(file_path, DOCS_PATH),
                            "chunk_index": i,
                            "text": chunk # Store text for retrieval
                        }
                        points.append(
                            models.PointStruct(
                                id=point_id_counter,
                                vector=embedding,
                                payload=metadata
                            )
                        )
                        point_id_counter += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    if points:
        print(f"Uploading {len(points)} points to Qdrant collection {QDRANT_COLLECTION_NAME}...")
        # Use async upsert if available or run in executor
        response = await asyncio.to_thread(
            qdrant_client.upsert,
            collection_name=QDRANT_COLLECTION_NAME,
            wait=True,
            points=points
        )
        print(f"Qdrant upsert response: {response}")
    else:
        print("No content chunks generated for ingestion.")

if __name__ == "__main__":
    asyncio.run(ingest_book_content())