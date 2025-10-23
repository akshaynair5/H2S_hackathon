import hashlib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util  # NEW: Added util for cosine sim
from pinecone import Pinecone, ServerlessSpec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
import os
from typing import Optional
load_dotenv()
# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API")  
INDEX_NAME = "fact-check-cache"
NAMESPACE = "default"
VERIFIED_NAMESPACE = "verified_fakes"
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
    """Convert text to vector embedding (normalize for better similarity)."""
    normalized = text.lower().strip()
    return EMBED_MODEL.encode(normalized).tolist()

def text_hash(text: str, url: Optional[str] = None) -> str:
    """Generate unique hash for text, optionally including URL (for article_id)."""
    content = (url or "") + text
    return hashlib.sha256(content.encode()).hexdigest()

def search_feedback(text: str, article_id: Optional[str] = None) -> dict:
    """Search Pinecone for exact or similar text feedback (now includes model results with threshold 1+)."""
    if not text.strip():
        return {"error": "No text provided"}

    vec_id = article_id or text_hash(text)
    vector = embed_text(text)

    # Exact match (now threshold 1 for system/model)
    exact_match = index.fetch(ids=[vec_id], namespace=NAMESPACE)
    if exact_match.vectors:
        metadata = exact_match.vectors[vec_id].metadata
        if metadata.get("unique_user_count", 0) >= 1:  # NEW: Lowered for model/system
            score = metadata.get("score", 0.5)  # Use stored score
            prediction = metadata.get("prediction", "Unknown")
            return {
                "score": score,
                "explanation": metadata.get("explanation", ""),
                "details": [{"prediction": prediction}],
                "source": "cache",
                "text": metadata.get("text", text),
                "article_id": article_id
            }

    # Similar (cosine >0.85, threshold 1+)
    query_filter = {"unique_user_count": {"$gte": 1}}  # NEW: Lowered
    if article_id:
        query_filter["article_id"] = {"$eq": article_id}
    similar_results = index.query(
        vector=vector,
        top_k=1,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=query_filter
    )
    if similar_results.matches and similar_results.matches[0].score > 0.85:  # NEW: Tighter threshold
        metadata = similar_results.matches[0].metadata
        score = metadata.get("score", 0.5)
        prediction = metadata.get("prediction", "Unknown")
        return {
            "score": score,
            "explanation": metadata.get("explanation", ""),
            "details": [{"prediction": prediction}],
            "source": "cache",
            "text": metadata.get("text", text),
            "article_id": article_id
        }

    return {"status": "no_reliable_match"}

def search_feedback_semantic(text: str, article_id: Optional[str] = None, verified_only: bool = False) -> dict:
    """Semantic search in Pinecone, optionally only verified fakes."""
    if not text.strip():
        return {"error": "No text provided"}

    vector = embed_text(text)
    namespace = VERIFIED_NAMESPACE if verified_only else NAMESPACE
    
    # Query top-k=5, filter verified if in main namespace
    query_filter = {"verified": {"$eq": True}} if not verified_only else {}
    if article_id:
        query_filter["article_id"] = {"$eq": article_id}
    
    similar_results = index.query(
        vector=vector,
        top_k=10,  # Increased for better coverage
        include_metadata=True,
        namespace=namespace,
        filter=query_filter
    )
    
    if similar_results.matches:
        print(f"Pinecone top scores: {[f'{m.score:.3f}' for m in similar_results.matches[:3]]}")  # FIXED: Nested f-string for formatting
        # Pick best (highest cosine > 0.75 for "true semantic")
        best_match = max((m for m in similar_results.matches if m.score > 0.75), key=lambda m: m.score, default=None)
        if best_match:
            metadata = best_match.metadata
            print(f"Pinecone hit! Sim: {best_match.score:.3f}")  # Logging
            return {
                "score": metadata.get("score", 0.5),
                "explanation": metadata.get("explanation", ""),
                "prediction": metadata.get("prediction", "Unknown"),
                "text": metadata.get("text", text),
                "article_id": metadata.get("article_id"),  # Firestore ID
                "source": "cache",
                "similarity": best_match.score,
                "details": [{"prediction": metadata.get("prediction", "Unknown")}]
            }
    
    return {"status": "no_reliable_match"}

def store_feedback(text: str, explanation: str, sources: list, user_fingerprint: str, article_id: Optional[str] = None, score: float = 0.5, prediction: str = "Unknown", verified: bool = True) -> dict:
    """Store/update, now always for model (system) with score/prediction. Supports verified namespace."""
    if not text.strip() or not explanation:
        return {"error": "Missing text or explanation"}

    vector = embed_text(text)
    vec_id = article_id or text_hash(text, url="")
    anon_id = anon_user_id(user_fingerprint)
    timestamp = datetime.utcnow().isoformat()
    namespace = VERIFIED_NAMESPACE if verified else NAMESPACE
    
    existing = index.fetch(ids=[vec_id], namespace=namespace)
    metadata = {
        "article_id": article_id or vec_id,  # Key: Link back to Firestore doc
        "text_hash": vec_id,
        "text": text[:1000],  # Truncate for metadata limits if needed
        "explanation": explanation[:2000],
        "sources": sources,
        "score": score,  # NEW: Store model score
        "prediction": prediction,  # NEW: Store prediction
        "verified": verified,  # NEW
        "timestamp": timestamp,
        "ttl_expiry": (datetime.utcnow() + timedelta(days=15)).isoformat(),
        "confirmations": 1,
        "unique_users": [anon_id],
        "unique_user_count": 1
    }

    if existing.vectors:
        old_meta = existing.vectors[vec_id].metadata
        old_confirmations = old_meta.get("confirmations", 0) + 1
        old_users = old_meta.get("unique_users", [])
        if anon_id not in old_users:
            old_users.append(anon_id)
        unique_count = len(old_users)
        # Average score over updates
        new_score = (old_meta.get("score", 0.5) * old_confirmations + score) / (old_confirmations + 1)
        metadata.update({
            "text": old_meta.get("text", text[:1000]),
            "explanation": explanation[:2000],  # Or merge/append if desired
            "score": new_score,
            "prediction": prediction if prediction != "Unknown" else old_meta.get("prediction", "Unknown"),  # Prefer new if available
            "verified": verified or old_meta.get("verified", False),
            "confirmations": old_confirmations,
            "unique_users": old_users,
            "unique_user_count": unique_count,
            "last_updated": timestamp,
            "article_id": article_id or old_meta.get("article_id")  # Preserve Firestore link
        })
        index.upsert(vectors=[{"id": vec_id, "values": vector, "metadata": metadata}], namespace=namespace)
        return {"status": "updated", "unique_user_count": unique_count}
    
    index.upsert(vectors=[{"id": vec_id, "values": vector, "metadata": metadata}], namespace=namespace)
    return {"status": "stored", "confirmations": 1, "article_id": article_id}

def cleanup_expired() -> dict:
    """Delete vectors older than 15 days."""
    now = datetime.utcnow()
    expiry = (now - timedelta(days=15)).isoformat()
    # Clean both namespaces
    for ns in [NAMESPACE, VERIFIED_NAMESPACE]:
        expired_results = index.query(
            vector=[0]*384,
            top_k=1000,
            include_metadata=True,
            filter={"ttl_expiry": {"$lt": now.isoformat()}},
            namespace=ns
        )
        expired_ids = [m.id for m in expired_results.matches]
        if expired_ids:
            index.delete(ids=expired_ids, namespace=ns)
    return {"status": "success", "deleted": len(expired_ids) if 'expired_ids' in locals() else 0}