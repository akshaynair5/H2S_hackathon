import hashlib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
import os

load_dotenv()
# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API")  
INDEX_NAME = "fact-check-cache"
NAMESPACE = "default"
EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Helper Functions
def anon_user_id(fingerprint: str) -> str:
    """Generate anonymous user ID from fingerprint."""
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(fingerprint.encode())
    return digest.finalize().hex()[:16]

def embed_text(text: str) -> list:
    """Convert text to vector embedding."""
    return EMBED_MODEL.encode(text).tolist()

def text_hash(text: str) -> str:
    """Generate unique hash for text."""
    return hashlib.sha256(text.encode()).hexdigest()

def search_feedback(text: str) -> dict:
    """Search Pinecone for exact or similar text feedback with 5+ unique user confirmations."""
    if not text.strip():
        return {"error": "No text provided"}

    vec_id = text_hash(text)
    vector = embed_text(text)

    # Check for exact match first
    exact_match = index.fetch(ids=[vec_id], namespace=NAMESPACE)
    if exact_match.vectors:
        metadata = exact_match.vectors[vec_id].metadata
        if metadata.get("unique_user_count", 0) >= 5:
            return {
                "score": 0.3,  # Mimic ML API format for fake results
                "explanation": metadata.get("explanation", "Community-verified as fake"),
                "details": [{"prediction": "fake"}],
                "source": "cache",
                "text": metadata.get("text", text)  # Return stored text
            }

    # Check for similar texts (cosine similarity > 0.8)
    similar_results = index.query(
        vector=vector,
        top_k=1,
        include_metadata=True,
        namespace=NAMESPACE,
        filter={"unique_user_count": {"$gte": 5}}  # Only reliable results
    )
    if similar_results.matches and similar_results.matches[0].score > 0.8:
        metadata = similar_results.matches[0].metadata
        return {
            "score": 0.3,
            "explanation": metadata.get("explanation", "Community-verified as fake"),
            "details": [{"prediction": "fake"}],
            "source": "cache",
            "text": metadata.get("text", text)  # Return stored text
        }

    return {"status": "no_reliable_match"}

def store_feedback(text: str, explanation: str, sources: list, user_fingerprint: str) -> dict:
    """Store or update feedback in Pinecone, including user-selected text and explanation."""
    if not text.strip() or not explanation:
        return {"error": "Missing text or explanation"}

    vector = embed_text(text)
    vec_id = text_hash(text)
    anon_id = anon_user_id(user_fingerprint)
    timestamp = datetime.utcnow().isoformat()
    
    # Check for existing entry
    existing = index.fetch(ids=[vec_id], namespace=NAMESPACE)
    metadata = {
        "text_hash": vec_id,
        "text": text,  # Store the user-selected text
        "explanation": explanation,  # Store the explanation
        "sources": sources,
        "timestamp": timestamp,
        "ttl_expiry": (datetime.utcnow() + timedelta(days=15)).isoformat(),
        "confirmations": 1,
        "unique_users": [anon_id],
        "unique_user_count": 1
    }

    if existing.vectors:
        # Update existing entry
        old_meta = existing.vectors[vec_id].metadata
        old_confirmations = old_meta.get("confirmations", 0) + 1
        old_users = old_meta.get("unique_users", [])
        if anon_id not in old_users:
            old_users.append(anon_id)
        unique_count = len(old_users)
        metadata.update({
            "text": old_meta.get("text", text),  # Preserve original text
            "explanation": explanation,  # Update with new explanation
            "confirmations": old_confirmations,
            "unique_users": old_users,
            "unique_user_count": unique_count,
            "last_updated": timestamp
        })
        if unique_count < 5:
            return {"status": "vote_recorded", "remaining_for_threshold": 5 - unique_count}
    
    # Store in Pinecone
    index.upsert(vectors=[{"id": vec_id, "values": vector, "metadata": metadata}], namespace=NAMESPACE)
    return {"status": "stored", "confirmations": metadata["confirmations"]}

def cleanup_expired() -> dict:
    """Delete vectors older than 15 days."""
    now = datetime.utcnow()
    expiry = (now - timedelta(days=15)).isoformat()
    expired_results = index.query(
        vector=[0]*384,
        top_k=1000,
        include_metadata=True,
        filter={"ttl_expiry": {"$lt": now.isoformat()}},
        namespace=NAMESPACE
    )
    expired_ids = [m.id for m in expired_results.matches]
    if expired_ids:
        index.delete(ids=expired_ids, namespace=NAMESPACE)
        return {"status": "success", "deleted": len(expired_ids)}
    return {"status": "success", "deleted": 0}