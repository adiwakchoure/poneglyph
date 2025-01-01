from minio import Minio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MinIO client
minio_client = Minio(
    "localhost:9000",
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure=False
)

# List all buckets
print("Available buckets:")
buckets = minio_client.list_buckets()
for bucket in buckets:
    print(f"\nBucket: {bucket.name}")
    print("Contents:")
    
    # List all objects in bucket
    objects = minio_client.list_objects(bucket.name, recursive=True)
    for obj in objects:
        print(f"- {obj.object_name} (Size: {obj.size} bytes)")
