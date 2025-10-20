import hashlib
import os
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API")
INDEX_NAME = "fact-check-cache"
NAMESPACE = "default"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------
# Pinecone Setup
# ---------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# ---------------------------
# Utility Functions
# ---------------------------
def anon_user_id(fingerprint: str) -> str:
    """Generate anonymous user ID from fingerprint (SHA256 truncated)."""
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(fingerprint.encode())
    return digest.finalize().hex()[:16]


def embed_text(text: str) -> list:
    """Convert text to vector embedding."""
    return EMBED_MODEL.encode(text, normalize_embeddings=True).tolist()


def text_hash(text: str) -> str:
    """Generate unique deterministic hash for text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def _clean_expired_items():
    """Internal: Efficiently delete expired vectors (older than 15 days)."""
    try:
        now = datetime.utcnow().isoformat()
        expired = index.query(
            vector=[0.0] * 384,
            top_k=200,
            include_metadata=True,
            filter={"ttl_expiry": {"$lt": now}},
            namespace=NAMESPACE,
        )
        expired_ids = [m.id for m in expired.matches]
        if expired_ids:
            index.delete(ids=expired_ids, namespace=NAMESPACE)
            print(f"[Cleanup] Deleted {len(expired_ids)} expired feedback entries.")
    except Exception as e:
        print(f"[Warning] Cleanup failed: {e}")


# ---------------------------
# Core Functions
# ---------------------------

def search_feedback(text: str) -> dict:
    """
    Search Pinecone for exact or similar cached feedback.
    Only returns results verified by 5+ unique users.
    """
    if not text.strip():
        return {"error": "No text provided"}

    vec_id = text_hash(text)
    vector = embed_text(text)

    try:
        # 1️⃣ Exact match
        exact_match = index.fetch(ids=[vec_id], namespace=NAMESPACE)
        if exact_match.vectors:
            meta = exact_match.vectors[vec_id].metadata
            unique_users = meta.get("unique_user_count", 0)
            if unique_users >= 5:
                return {
                    "score": 0.3,
                    "explanation": meta.get("explanation", "Community-verified as fake"),
                    "details": [{"prediction": "fake"}],
                    "source": "cache",
                    "confidence": 0.95,
                    "verified_by": unique_users,
                    "text": meta.get("text", text),
                }

        # 2️⃣ Similar match (cosine similarity > 0.85)
        sim_results = index.query(
            vector=vector,
            top_k=3,
            include_metadata=True,
            namespace=NAMESPACE,
            filter={"unique_user_count": {"$gte": 5}},
        )

        if sim_results.matches:
            top_match = max(sim_results.matches, key=lambda m: m.score)
            if top_match.score > 0.85:
                meta = top_match.metadata
                return {
                    "score": 0.3,
                    "explanation": meta.get("explanation", "Community-verified as fake"),
                    "details": [{"prediction": "fake"}],
                    "source": "cache",
                    "confidence": float(top_match.score),
                    "verified_by": meta.get("unique_user_count", 0),
                    "text": meta.get("text", text),
                }

        return {"status": "no_reliable_match"}

    except Exception as e:
        return {"error": f"Pinecone query failed: {str(e)}"}


def store_feedback(text: str, explanation: str, sources: list, user_fingerprint: str) -> dict:
    """
    Store or update feedback in Pinecone.
    Supports:
      - Aggregating user confirmations
      - Expiry after 15 days
      - Duplicate user detection
    """
    if not text.strip() or not explanation:
        return {"error": "Missing text or explanation"}

    try:
        vec_id = text_hash(text)
        vector = embed_text(text)
        anon_id = anon_user_id(user_fingerprint)
        timestamp = datetime.utcnow().isoformat()
        ttl_expiry = (datetime.utcnow() + timedelta(days=15)).isoformat()

        # Fetch existing record
        existing = index.fetch(ids=[vec_id], namespace=NAMESPACE)
        if existing.vectors:
            meta = existing.vectors[vec_id].metadata
            users = set(meta.get("unique_users", []))
            if anon_id in users:
                # Duplicate feedback by same user — ignore
                return {"status": "duplicate_vote", "message": "Feedback already recorded"}

            # Update entry
            users.add(anon_id)
            meta.update({
                "confirmations": meta.get("confirmations", 0) + 1,
                "unique_users": list(users),
                "unique_user_count": len(users),
                "last_updated": timestamp,
                "ttl_expiry": ttl_expiry,
                "explanation": explanation,  # keep latest
            })

            index.upsert([{"id": vec_id, "values": vector, "metadata": meta}], namespace=NAMESPACE)
            if meta["unique_user_count"] >= 5:
                return {"status": "verified", "message": "Threshold reached (community verified)"}
            else:
                return {
                    "status": "vote_recorded",
                    "remaining_for_threshold": 5 - meta["unique_user_count"],
                    "unique_users": meta["unique_user_count"]
                }

        # New entry
        metadata = {
            "text_hash": vec_id,
            "text": text,
            "explanation": explanation,
            "sources": sources or [],
            "timestamp": timestamp,
            "ttl_expiry": ttl_expiry,
            "confirmations": 1,
            "unique_users": [anon_id],
            "unique_user_count": 1,
        }

        index.upsert([{"id": vec_id, "values": vector, "metadata": metadata}], namespace=NAMESPACE)
        return {"status": "stored", "confirmations": 1}

    except Exception as e:
        return {"error": f"Failed to store feedback: {str(e)}"}


def cleanup_expired() -> dict:
    """Public function to trigger cleanup (manually or via cron)."""
    try:
        _clean_expired_items()
        return {"status": "success"}
    except Exception as e:
        return {"error": str(e)}
