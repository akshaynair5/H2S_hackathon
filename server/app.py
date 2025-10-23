from flask import Flask, request, jsonify
from visionAi import detect_fake_image
from misinfo_model import detect_fake_text, ask_gemini 
from flask_cors import CORS
from vectorDb import search_feedback, search_feedback_semantic, store_feedback, cleanup_expired, EMBED_MODEL, util
import hashlib
import re  
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json  
from typing import Optional
app = Flask(__name__)
CORS(app)
load_dotenv()
# Firestore client (init once)
if not firebase_admin._apps:
    cred = credentials.Certificate(os.getenv("FIRESTORE_CREDENTIALS_PATH"))
    firebase_admin.initialize_app(cred)
db = firestore.client(database_id='genai')

def generate_id(url, text):
    """Exact SHA256(url + text) for unique article ID."""
    content = (url + text).encode('utf-8')
    return hashlib.sha256(content).hexdigest()

# NEW: Normalization helpers for semantic matching
def normalize_text(text: str) -> str:
    """Light normalization: lower, remove punct, collapse spaces."""
    normalized = re.sub(r'[^\w\s]', ' ', text.lower())
    normalized = re.sub(r'\s+', ' ', normalized.strip())
    return normalized

def generate_normalized_id(url: str, text: str) -> str:
    """SHA256(norm_url + norm_text) for fuzzy matching."""
    norm_url = url.lower() if url else ''
    norm_text = normalize_text(text)
    content = (norm_url + norm_text).encode('utf-8')
    return hashlib.sha256(content).hexdigest()

def generate_embedding(text: str) -> list:
    """Generate embedding for Firestore storage (reused from vectorDb)."""
    normalized = text.lower().strip()
    return EMBED_MODEL.encode(normalized).tolist()

def get_article_doc(article_id):
    """Fetch from Firestore; create if missing."""
    doc_ref = db.collection('articles').document(article_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    return None

def firestore_semantic_search(text: str, min_similarity: float = 0.90, days_back: int = 30) -> Optional[dict]:
    """Client-side semantic search in Firestore: Query recent docs, compute cosine on embeddings."""
    if not text.strip():
        return None
    
    # Query recent docs (limit scan)
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    query = db.collection('articles').where('last_updated', '>=', cutoff).limit(50).stream()  # Tune limit
    candidates = []
    query_embedding = generate_embedding(text)
    
    for doc in query:
        data = doc.to_dict()
        if 'embedding' in data and data['text'].strip():
            stored_emb = data['embedding']
            similarity = util.cos_sim(query_embedding, stored_emb)[0][0].item()  # 1x1 matrix
            if similarity > min_similarity:
                candidates.append({
                    'doc': data,
                    'id': doc.id,
                    'similarity': similarity
                })
    
    if candidates:
        # Pick best (highest sim, then highest score)
        best = max(candidates, key=lambda c: (c['similarity'], c['doc'].get('text_score', 0)))
        print(f"Firestore semantic hit! Best sim: {best['similarity']:.3f} on {best['id']}")
        return {'best': best['doc'], 'best_id': best['id'], 'similarity': best['similarity']}
    
    print("No Firestore semantic hit.")
    return None

# ---------------------------
# UNIFIED ARTICLE DETECTION  - Similar updates, omitted for brevity; apply same pattern
# ---------------------------
# (Use same fuzzy logic as below for /detect_text if needed for articles)

# ---------------------------
# IMAGE DETECTION (Unchanged)
# ---------------------------
@app.route("/detect_image", methods=["POST"])
def detect_image():
    data = request.json
    urls = data.get("urls") or data.get("images") 
    print("Urls is " , urls)
    if not urls:
        return jsonify({"error": "No images provided"}), 400
    if isinstance(urls, str):
        urls = [urls]

    results = detect_fake_image(urls) 
    if not isinstance(results, list):
        results = [results]

    avg_score = sum(r.get("score", 0) for r in results) / len(results)

    response = {
        "score": avg_score,
        "explanation": f"{len(results)} image(s) analyzed",
        "details": results
    }

    return jsonify(response)

# ---------------------------
# TEXT DETECTION (Enhanced with semantic Firestore fuzzy matching + Pinecone verified semantic)
# ---------------------------
@app.route("/detect_text", methods=["POST"])
def detect_text():
    data = request.json 
    original_text = data.get("text", "")
    url = data.get("url", "")  # Optional for better ID
    print(f"User selected text: '{original_text}' - \n")
    print(f"URL IS  '{url}'- \n")

    text = original_text.strip()
    if not text or len(text) < 50:
        return jsonify({"error": "Text too short or missing"}), 400

    # Exact ID
    article_id = generate_id(url, text)
    
    # NEW: Normalized ID for semantic matching
    norm_id = generate_normalized_id(url, text)
    print(f"Exact ID: {article_id}, Norm ID: {norm_id}")

    # Step 1: Exact Firestore match
    cached = get_article_doc(article_id)
    if cached:
        prediction = cached.get("prediction", "Unknown")
        explanation = cached.get("gemini_reasoning", cached.get("text_explanation", ""))
        return jsonify({
            "score": cached.get("text_score", 0.5),
            "prediction": prediction,
            "explanation": explanation,
            "article_id": article_id,
            "source": "firestore_exact",
            "details": [{
                "score": cached.get("text_score", 0.5),
                "prediction": prediction,
                "explanation": explanation,
                "source": "firestore_exact",
                "article_id": article_id
            }]
        })

    # Step 1.5: True semantic Firestore search
    print("Exact miss; trying Firestore semantic search...")
    firestore_semantic = firestore_semantic_search(original_text)
    if firestore_semantic:
        best = firestore_semantic['best']
        best_article_id = firestore_semantic['best_id']
        prediction = best.get("prediction", "Unknown")
        explanation = best.get("gemini_reasoning", best.get("text_explanation", ""))
        print(f"Semantic hit! Using candidate with sim {firestore_semantic['similarity']:.3f}, score {best.get('text_score', 0.5)}")
        # Increment views
        db.collection('articles').document(best_article_id).update({"total_views": firestore.Increment(1)})
        # Personalize if sim <0.95 (not near-exact)
        if firestore_semantic['similarity'] < 0.95:
            personalization_prompt = f"""
            Original: "{original_text}"
            Similar cached: "{best.get('text', '')}", score={best.get('text_score', 0.5)}, pred={prediction}, exp="{explanation}"
            Personalize JSON: {{"score":<0-1>, "prediction":"Fake"/"Real", "explanation":"<reasoning>"}}
            """
            try:
                gemini_resp = ask_gemini(personalization_prompt)
                if gemini_resp.strip().startswith('{'):
                    pers = json.loads(gemini_resp)
                    explanation = pers.get("explanation", explanation)
                    # Store personalized under original ID
                    embedding = generate_embedding(original_text)
                    db.collection('articles').document(article_id).set({
                        "url": url, "text": original_text, "normalized_id": norm_id, "embedding": embedding,
                        "text_score": pers.get("score", best.get("text_score", 0.5)),
                        "prediction": pers.get("prediction", prediction),
                        "gemini_reasoning": explanation, "text_explanation": explanation,
                        "total_views": 1, "total_reports": 1 if pers.get("score", 0.5) < 0.5 else 0,
                        "last_updated": datetime.utcnow(), "type": "text", "verified": True  # Always for cache
                    })
                    store_feedback(original_text, explanation, [], "system", article_id, pers.get("score", 0.5), pers.get("prediction", prediction), verified=True)
                    prediction = pers.get("prediction", prediction)
            except Exception as e:
                print(f"Personalization err: {e}")
        
        return jsonify({
            "score": best.get("text_score", 0.5),
            "prediction": prediction,
            "explanation": explanation,
            "article_id": article_id,
            "source": "firestore_semantic",
            "details": [{"score": best.get("text_score", 0.5), "prediction": prediction, "explanation": explanation, "source": "firestore_semantic", "article_id": best_article_id, "similarity": firestore_semantic['similarity']}]
        })

    print("No semantic Firestore hit; querying Pinecone for semantic similar verified fakes...")
    # NEW: Step 2: Semantic search in Pinecone for verified fakes (broader, verified_only=False)
    semantic_result = search_feedback_semantic(original_text, article_id=article_id, verified_only=False)
    if semantic_result.get("source") == "cache":
        # Fetch full Firestore doc using returned article_id
        cached_doc_id = semantic_result.get("article_id")
        cached_doc = get_article_doc(cached_doc_id)
        if cached_doc:
            cached_text = cached_doc.get("text", original_text)
            cached_score = cached_doc.get("text_score", 0.5)
            cached_prediction = cached_doc.get("prediction", "Unknown")
            cached_explanation = cached_doc.get("gemini_reasoning", cached_doc.get("text_explanation", ""))
            
            # NEW: Personalize with Gemini
            personalization_prompt = f"""
            Original user text to analyze for misinformation/fakeness: "{original_text}"
            
            Similar verified fake from cache:
            - Cached text: "{cached_text}"
            - Cached score (0-1, higher = more likely fake): {cached_score}
            - Cached prediction: {cached_prediction}
            - Cached explanation: {cached_explanation}
            
            Provide a personalized detection for the ORIGINAL user text:
            - Respond in JSON format: {{"score": <float 0-1>, "prediction": "Fake" or "Real", "explanation": "<concise reasoning>"}}
            - Base your analysis on the original text, but incorporate insights from the cached similar content where relevant.
            - Focus on semantic similarities/differences in meaning, claims, or context.
            """
            
            try:
                gemini_response = ask_gemini(personalization_prompt)
                # Parse JSON response
                if gemini_response.strip().startswith('{'):
                    personalized = json.loads(gemini_response)
                    personalized_score = personalized.get("score", cached_score)
                    personalized_prediction = personalized.get("prediction", cached_prediction)
                    personalized_explanation = personalized.get("explanation", cached_explanation)
                else:
                    # Fallback: Use cached, append Gemini summary
                    personalized_score = cached_score
                    personalized_prediction = cached_prediction
                    personalized_explanation = f"{cached_explanation} (Personalized by Gemini: {gemini_response})"
                
                # Store personalized in Firestore (under original ID) + sync to Pinecone
                embedding = generate_embedding(original_text)
                verified = True  # Always for cache hits
                doc_data = {
                    "url": url,
                    "text": original_text,
                    "normalized_id": norm_id,
                    "embedding": embedding,  # NEW
                    "text_score": personalized_score,
                    "prediction": personalized_prediction,
                    "gemini_reasoning": personalized_explanation,  # Dedicated field
                    "text_explanation": personalized_explanation,
                    "total_views": 1,
                    "total_reports": 1 if personalized_score < 0.5 else 0,
                    "last_updated": datetime.utcnow(),
                    "type": "text",
                    "verified": verified  # NEW
                }
                db.collection('articles').document(article_id).set(doc_data)
                
                # Sync to Pinecone
                store_feedback(
                    text=original_text,
                    explanation=personalized_explanation,
                    sources=[],
                    user_fingerprint="system",
                    article_id=article_id,
                    score=personalized_score,
                    prediction=personalized_prediction,
                    verified=verified
                )
                
                return jsonify({
                    "score": personalized_score,
                    "prediction": personalized_prediction,
                    "explanation": personalized_explanation,
                    "article_id": article_id,
                    "source": "personalized_semantic_cache",
                    "details": [{
                        "score": personalized_score,
                        "prediction": personalized_prediction,
                        "explanation": personalized_explanation,
                        "source": "personalized_semantic_cache",
                        "article_id": article_id,
                        "gemini_used": True,
                        "similarity": semantic_result.get("similarity", 0)
                    }]
                })
            except Exception as gemini_err:
                print(f"Gemini personalization failed: {gemini_err}. Falling back to raw cache.")
                # Fallback: Store raw cache
                embedding = generate_embedding(original_text)
                verified = True
                db.collection('articles').document(article_id).set({
                    "url": url,
                    "text": original_text,
                    "normalized_id": norm_id,
                    "embedding": embedding,
                    "text_score": cached_score,
                    "prediction": cached_prediction,
                    "text_explanation": cached_explanation,
                    "total_views": 1,
                    "total_reports": 1 if cached_score < 0.5 else 0,
                    "last_updated": datetime.utcnow(),
                    "type": "text",
                    "verified": verified
                })
                store_feedback(
                    text=original_text,
                    explanation=cached_explanation,
                    sources=[],
                    user_fingerprint="system",
                    article_id=article_id,
                    score=cached_score,
                    prediction=cached_prediction,
                    verified=verified
                )
                return jsonify({
                    **semantic_result,
                    "article_id": article_id,
                    "source": "semantic_cache",
                    "details": [{
                        "score": cached_score,
                        "prediction": cached_prediction,
                        "explanation": cached_explanation,
                        "source": "semantic_cache",
                        "article_id": cached_doc_id
                    }]
                })

    # Step 3: ML model fallback (no cache hit)
    print(f"No semantic cache hit for text: '{text}' - Falling back to ML model")
    model_result = detect_fake_text(original_text)
    text_score = model_result.get("score", 0.5)
    text_prediction = model_result.get("prediction", "Unknown")
    explanation = model_result.get("explanation", "Analysis complete.")

    # Step 4: Cache with normalized_id + embedding + verified
    verified = True  # Always for model results
    embedding = generate_embedding(original_text)
    total_reports = 1 if text_score < 0.5 else 0
    doc_data = {
        "url": url,
        "text": original_text,
        "normalized_id": norm_id,
        "embedding": embedding,  # NEW
        "text_score": text_score,
        "prediction": text_prediction,
        "text_explanation": explanation,
        "total_views": 1,
        "total_reports": total_reports,
        "last_updated": datetime.utcnow(),
        "type": "text",
        "verified": verified  # NEW
    }
    db.collection('articles').document(article_id).set(doc_data)

    # Step 5: Store in Pinecone (always for model results)
    store_feedback(
        original_text, explanation, [], "system", article_id=article_id,
        score=text_score, prediction=text_prediction, verified=verified
    )

    # Response
    response = {
        "score": text_score,
        "prediction": text_prediction,
        "explanation": explanation,
        "article_id": article_id,
        "source": "new_analysis",
        "details": [model_result]
    }
    return jsonify(response)


# ---------------------------
# USER FEEDBACK (Updated: Labels "REAL"/"FAKE", updates Firestore; per advice 6)
# ---------------------------
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    article_id = data.get("article_id", "")  # Required now; fallback to text-based if missing
    text = data.get("text", "")  # Fallback for legacy
    label = data.get("response", "").upper()  # "REAL" or "FAKE" (adapted from original "YES"/"NO")

    if label == "YES":
        label = "FAKE"
    else:
        label = "REAL"
    print("Label is " , label)
    explanation = data.get("explanation", "")
    sources = data.get("sources", [])
    user_fingerprint = request.headers.get("user-fingerprint", "default")
    print(f"""
Collected Data:
  Article ID: {article_id}
  Text: {text}
  Label: {label}
  Explanation: {explanation}
  Sources: {sources}
  User Fingerprint: {user_fingerprint}
""")

    if not article_id and not text.strip() or label not in ["REAL", "FAKE"]:
        return jsonify({"error": "Missing article_id/text or invalid label (use REAL/FAKE)"}), 400

    if article_id:
        # Update Firestore
        increment_reports = 1 if label == "FAKE" else 0
        db.collection('articles').document(article_id).update({
            "total_views": firestore.Increment(1),
            "total_reports": firestore.Increment(increment_reports)
        })
        # Optional: Store detailed feedback
        db.collection('articles').document(article_id).collection('feedbacks').add({
            "label": label,
            "explanation": explanation,
            "user_fingerprint": user_fingerprint,
            "timestamp": datetime.utcnow()
        })

        # Check threshold (advice 6: e.g., >40% flagged)
        doc = get_article_doc(article_id)
        if doc:
            total_views = doc.get('total_views', 1) + 1  # Post-increment
            total_reports = doc.get('total_reports', 0) + increment_reports
            percentage = (total_reports / total_views) * 100
            if percentage > 40:
                db.collection('articles').document(article_id).update({"community_flagged": True})
            return jsonify({"status": "feedback_recorded", "percentage_reported": f"{percentage:.0f}%"})

    # Legacy: If no article_id, use original Pinecone logic (only store FAKE)
    if label != "FAKE":  # Adapted from original "YES" (fake)
        return jsonify({"status": "ignored", "message": "Only FAKE labels are stored in legacy mode"}), 200

    result = store_feedback(text, explanation, sources, user_fingerprint)
    print("Final result is ", result)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result), 200

# ---------------------------
# CLEANUP EXPIRED DATA (Original, unchanged)
# ---------------------------
@app.route("/cleanup_expired", methods=["POST"])
def cleanup_expired_endpoint():
    result = cleanup_expired()
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)