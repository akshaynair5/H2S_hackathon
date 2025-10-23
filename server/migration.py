"""
One-time migration script to add embeddings to existing Firestore articles.
Run this locally or as a Cloud Function after updating the schema.
Dependencies: firebase_admin, sentence_transformers (from vectorDb.py).
"""

import os
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import firebase_admin

load_dotenv()

# Initialize Firestore (reuse from app.py)
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIRESTORE_CREDENTIALS_PATH"))
    firebase_admin.initialize_app(cred)
db = firestore.client(database_id='genai')

# Load embedding model (reuse from vectorDb.py)
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list:
    """Generate embedding for Firestore storage."""
    normalized = text.lower().strip()
    return EMBED_MODEL.encode(normalized).tolist()

def migrate_embeddings():
    """Batch update existing articles with embeddings."""
    docs = db.collection('articles').stream()
    batch = db.batch()
    updated_count = 0
    for doc in docs:
        data = doc.to_dict()
        updates = {}
        if 'embedding' not in data:
            if 'text' in data and data['text'].strip():
                updates['embedding'] = generate_embedding(data['text'])
        updates['verified'] = True  # NEW: Set for all existing
        if updates:
            batch.update(doc.reference, updates)
            updated_count += 1
            print(f"Updated {doc.id} with embedding and verified=True.")
        else:
            print(f"No updates needed for {doc.id}.")
    
    if updated_count > 0:
        batch.commit()
        print(f"Migration complete: Updated {updated_count} documents.")
    else:
        print("No documents needed migration.")

if __name__ == "__main__":
    migrate_embeddings()