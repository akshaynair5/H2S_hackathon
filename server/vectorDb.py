# vectorDb.py (From new, with minor adaptations for old integration)

import hashlib
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer, util
from pinecone import Pinecone, ServerlessSpec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv
import os
from typing import Optional

# -----------------------------
# CONFIGURATION & SETUP
# -----------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API")
INDEX_NAME = "fact-check-cache"
NAMESPACE = "default"
VERIFIED_NAMESPACE = "verified_fakes"
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone safely
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [i["name"] for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def anon_user_id(fingerprint: str) -> str:
    """Generate anonymous user ID from fingerprint."""
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    digest.update(fingerprint.encode())
    return digest.finalize().hex()[:16]


def embed_text(text: str) -> list:
    """Convert text to normalized vector embedding."""
    normalized = text.lower().strip()
    return EMBED_MODEL.encode(normalized).tolist()


def text_hash(text: str, url: Optional[str] = None) -> str:
    """Generate deterministic hash for text (optionally with URL)."""
    content = (url or "") + text
    return hashlib.sha256(content.encode()).hexdigest()


# -----------------------------
# SEARCH FUNCTIONS
# -----------------------------
def search_feedback(text: str, article_id: Optional[str] = None) -> dict:
    """Search Pinecone for exact or similar text feedback."""
    if not text.strip():
        return {"error": "No text provided"}

    vec_id = article_id or text_hash(text)
    vector = embed_text(text)

    # --- Exact Match ---
    exact_match = index.fetch(ids=[vec_id], namespace=NAMESPACE)
    if exact_match.vectors:
        metadata = exact_match.vectors[vec_id].metadata
        if metadata.get("unique_user_count", 0) >= 1:
            return {
                "score": metadata.get("score", 0.5),
                "explanation": metadata.get("explanation", ""),
                "details": [{"prediction": metadata.get("prediction", "Unknown")}],
                "source": "cache",
                "text": metadata.get("text", text),
                "article_id": article_id,
            }

    # --- Semantic Match ---
    query_filter = {"unique_user_count": {"$gte": 1}}
    if article_id:
        query_filter["article_id"] = {"$eq": article_id}

    similar_results = index.query(
        vector=vector,
        top_k=1,
        include_metadata=True,
        namespace=NAMESPACE,
        filter=query_filter,
    )

    if similar_results.matches and similar_results.matches[0].score > 0.85:
        metadata = similar_results.matches[0].metadata
        return {
            "score": metadata.get("score", 0.5),
            "explanation": metadata.get("explanation", ""),
            "details": [{"prediction": metadata.get("prediction", "Unknown")}],
            "source": "cache",
            "text": metadata.get("text", text),
            "article_id": article_id,
        }

    return {"status": "no_reliable_match"}


def search_feedback_semantic(
    text: str, article_id: Optional[str] = None, verified_only: bool = False
) -> dict:
    """Semantic search in Pinecone, optionally only verified fakes."""
    if not text.strip():
        return {"error": "No text provided"}

    vector = embed_text(text)
    namespace = VERIFIED_NAMESPACE if verified_only else NAMESPACE
    query_filter = {"verified": {"$eq": True}} if not verified_only else {}
    if article_id:
        query_filter["article_id"] = {"$eq": article_id}

    similar_results = index.query(
        vector=vector,
        top_k=10,
        include_metadata=True,
        namespace=namespace,
        filter=query_filter,
    )

    if similar_results.matches:
        top_matches = [f"{m.score:.3f}" for m in similar_results.matches[:3]]
        print(f"[Pinecone] Top scores: {top_matches}")
        best_match = max(
            (m for m in similar_results.matches if m.score > 0.75),
            key=lambda m: m.score,
            default=None,
        )
        if best_match:
            metadata = best_match.metadata
            print(f"[Pinecone Hit] Similarity: {best_match.score:.3f}")
            return {
                "score": metadata.get("score", 0.5),
                "explanation": metadata.get("explanation", ""),
                "prediction": metadata.get("prediction", "Unknown"),
                "text": metadata.get("text", text),
                "article_id": metadata.get("article_id"),
                "source": "cache",
                "similarity": best_match.score,
                "details": [{"prediction": metadata.get("prediction", "Unknown")}],
            }

    return {"status": "no_reliable_match"}


# -----------------------------
# STORE FUNCTION
# -----------------------------
def store_feedback(
    text: str,
    explanation: str,
    sources: list,
    user_fingerprint: str,
    article_id: Optional[str] = None,
    score: float = 0.5,
    prediction: str = "Unknown",
    verified: bool = True,
) -> dict:
    """Store or update a feedback vector with model or user data."""
    if not text.strip() or not explanation:
        return {"error": "Missing text or explanation"}

    vector = embed_text(text)
    vec_id = article_id or text_hash(text, url="")
    anon_id = anon_user_id(user_fingerprint)
    timestamp = datetime.utcnow().isoformat()
    namespace = VERIFIED_NAMESPACE if verified else NAMESPACE

    existing = index.fetch(ids=[vec_id], namespace=namespace)
    metadata = {
        "article_id": article_id or vec_id,
        "text_hash": vec_id,
        "text": text[:1000],
        "explanation": explanation[:2000],
        "sources": sources,
        "score": score,
        "prediction": prediction,
        "verified": verified,
        "timestamp": timestamp,
        "ttl_expiry": (datetime.utcnow() + timedelta(days=15)).isoformat(),
        "confirmations": 1,
        "unique_users": [anon_id],
        "unique_user_count": 1,
    }

    # --- Update existing entry ---
    if existing.vectors:
        old_meta = existing.vectors[vec_id].metadata
        old_confirmations = old_meta.get("confirmations", 0) + 1
        old_users = old_meta.get("unique_users", [])
        if anon_id not in old_users:
            old_users.append(anon_id)
        unique_count = len(old_users)

        new_score = (
            (old_meta.get("score", 0.5) * old_confirmations + score)
            / (old_confirmations + 1)
        )
        metadata.update({
            "text": old_meta.get("text", text[:1000]),
            "score": new_score,
            "prediction": (
                prediction if prediction != "Unknown"
                else old_meta.get("prediction", "Unknown")
            ),
            "verified": verified or old_meta.get("verified", False),
            "confirmations": old_confirmations,
            "unique_users": old_users,
            "unique_user_count": unique_count,
            "last_updated": timestamp,
        })

        index.upsert(
            vectors=[{"id": vec_id, "values": vector, "metadata": metadata}],
            namespace=namespace,
        )
        return {"status": "updated", "unique_user_count": unique_count}

    # --- New entry ---
    index.upsert(
        vectors=[{"id": vec_id, "values": vector, "metadata": metadata}],
        namespace=namespace,
    )
    return {"status": "stored", "confirmations": 1, "article_id": article_id}


# -----------------------------
# CLEANUP FUNCTION
# -----------------------------
def cleanup_expired(days: int = 15) -> dict:
    """Delete vectors older than N days (default: 15)."""
    now = datetime.utcnow()
    deleted_total = 0

    for ns in [NAMESPACE, VERIFIED_NAMESPACE]:
        try:
            expired_results = index.query(
                vector=[0] * 384,
                top_k=1000,
                include_metadata=True,
                filter={"ttl_expiry": {"$lt": now.isoformat()}},
                namespace=ns,
            )
            expired_ids = [m.id for m in expired_results.matches]
            if expired_ids:
                index.delete(ids=expired_ids, namespace=ns)
                deleted_total += len(expired_ids)
                print(f"[Cleanup] Deleted {len(expired_ids)} expired from {ns}")
        except Exception as e:
            print(f"[Cleanup Error in {ns}] {e}")

    return {"status": "success", "deleted": deleted_total}