import os
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from sentence_transformers import SentenceTransformer
import firebase_admin

load_dotenv()

if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIRESTORE_CREDENTIALS_PATH"))
    firebase_admin.initialize_app(cred)
db = firestore.client(database_id='genai')

EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding(text: str) -> list:
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
        updates['verified'] = True  
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