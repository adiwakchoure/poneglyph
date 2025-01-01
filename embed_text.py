from fastembed import TextEmbedding
import json
import numpy as np
from minio import Minio
import io
import os
from typing import List, Dict, Iterator
from dotenv import load_dotenv
import uuid
from tqdm import tqdm
import time

# Load environment variables from .env file
load_dotenv()

def load_jsonl(file_path: str) -> Iterator[Dict]:
    """Load documents from a JSONL file."""
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                yield json.loads(line)

def ensure_bucket(minio_client: Minio, bucket_name: str):
    """Ensure bucket exists, create if it doesn't."""
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print(f"Created bucket: {bucket_name}")
        else:
            print(f"Bucket already exists: {bucket_name}")
    except Exception as e:
        print(f"Error with bucket operation: {e}")
        raise

def generate_embeddings(texts: List[str], model: TextEmbedding) -> List[np.ndarray]:
    """Generate embeddings for a list of texts with progress tracking."""
    embeddings = []
    print(f"Generating embeddings for {len(texts)} texts...")
    
    try:
        # Convert generator to list with progress bar
        for i, embedding in enumerate(tqdm(model.embed(texts), total=len(texts), desc="Embedding")):
            embeddings.append(embedding)
            if i > 0 and i % 10 == 0:  # Print progress every 10 documents
                print(f"Processed {i} documents...")
    except Exception as e:
        print(f"Error during embedding generation: {e}")
        raise
    
    return embeddings

def save_document_with_embedding(
    document: Dict,
    embedding: np.ndarray,
    bucket_name: str,
    minio_client: Minio
) -> str:
    """Save a single document with its embedding to MinIO."""
    # Create a document object that includes both text and embedding
    doc_with_embedding = {
        "id": str(uuid.uuid4()),  # Generate a unique ID
        "original_id": document.get("id", None),  # Keep original ID if exists
        "content": document["content"],
        "metadata": {
            "source": document.get("source", "unknown"),
            "url": document.get("url", None),
            "title": document.get("title", None),
            "media_type": document.get("media-type", None),
            "published": document.get("published", None),
            # Add any other metadata from original document
            **document.get("metadata", {})
        },
        "embedding": embedding.tolist(),  # Convert numpy array to list for JSON serialization
        "embedding_model": "intfloat/multilingual-e5-large",
        "embedding_dimension": len(embedding),
        "processed_at": str(os.path.getmtime("data/dataset-100.jsonl"))
    }
    
    # Convert to JSON string
    json_str = json.dumps(doc_with_embedding)
    
    # Create a buffer with the JSON data
    buffer = io.BytesIO(json_str.encode())
    
    # Generate object name using the document ID
    object_name = f"documents/{doc_with_embedding['id']}.json"
    
    # Upload to MinIO
    try:
        minio_client.put_object(
            bucket_name,
            object_name,
            buffer,
            len(json_str),
            content_type='application/json'
        )
        print(f"Saved document {doc_with_embedding['id']} to {bucket_name}/{object_name}")
        return doc_with_embedding['id']
    except Exception as e:
        print(f"Error saving document to MinIO: {e}")
        raise

def main():
    # Get MinIO credentials from environment variables
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    
    if not access_key or not secret_key:
        print("Error: MinIO credentials not found in environment variables")
        return

    print(f"Using MinIO credentials - Access Key: {access_key}")

    # Initialize the embedding model
    try:
        embedding_model = TextEmbedding(
            model_name="intfloat/multilingual-e5-large",
            max_length=512,  # Limit max sequence length
            threads=4  # Use multiple CPU threads
        )
        print("Embedding model initialized successfully")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        return

    # MinIO configuration
    try:
        minio_client = Minio(
            "localhost:9000",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )
        print("MinIO client initialized successfully")
    except Exception as e:
        print(f"Error initializing MinIO client: {e}")
        return

    # Set up bucket
    bucket_name = "embeddings"
    try:
        ensure_bucket(minio_client, bucket_name)
    except Exception as e:
        print(f"Failed to ensure bucket exists: {e}")
        return

    # Process JSONL file
    jsonl_path = "data/dataset-100.jsonl"
    batch_size = 1  # Reduced batch size
    saved_ids = []
    
    try:
        # Count total documents for progress bar
        total_docs = sum(1 for _ in load_jsonl(jsonl_path))
        print(f"Found {total_docs} documents in {jsonl_path}")
        
        # Process documents in batches
        batch_texts = []
        batch_docs = []
        
        for doc in tqdm(load_jsonl(jsonl_path), total=total_docs, desc="Reading documents"):
            batch_texts.append(doc['content'])
            batch_docs.append(doc)
            
            if len(batch_texts) >= batch_size:
                try:
                    # Generate embeddings for batch
                    embeddings = generate_embeddings(batch_texts, embedding_model)
                    
                    # Save each document with its embedding
                    for doc, embedding in zip(batch_docs, embeddings):
                        try:
                            doc_id = save_document_with_embedding(
                                doc,
                                embedding,
                                bucket_name,
                                minio_client
                            )
                            saved_ids.append(doc_id)
                        except Exception as e:
                            print(f"Error processing document: {e}")
                            continue
                    
                    # Clear batches
                    batch_texts = []
                    batch_docs = []
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
        
        # Process remaining documents
        if batch_texts:
            try:
                embeddings = generate_embeddings(batch_texts, embedding_model)
                for doc, embedding in zip(batch_docs, embeddings):
                    try:
                        doc_id = save_document_with_embedding(
                            doc,
                            embedding,
                            bucket_name,
                            minio_client
                        )
                        saved_ids.append(doc_id)
                    except Exception as e:
                        print(f"Error processing document: {e}")
                        continue
            except Exception as e:
                print(f"Error processing final batch: {e}")

    except Exception as e:
        print(f"Error processing JSONL file: {e}")
        return

    print(f"Successfully processed and saved {len(saved_ids)} documents")
    print("First few document IDs:", saved_ids[:5])

if __name__ == "__main__":
    main()