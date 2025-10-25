# app.py (Integrated Approach)

from flask import Flask, request, jsonify
from misinfo_model import detect_fake_text
from flask_cors import CORS
from vectorDb import search_feedback_semantic, store_feedback, cleanup_expired
from database import generate_id,generate_normalized_id,generate_embedding,get_article_doc,firestore_semantic_search,db
from FakeImageDetection import detect_fake_image
from firebase_admin import credentials, firestore

from datetime import datetime, timedelta


import json

import numpy as np


app = Flask(__name__)
CORS(app)


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, bytes):
        return obj.decode("utf-8", errors="ignore")
    else:
        return obj

# ---------------------------
# IMAGE DETECTION (Unchanged from old)
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
# TEXT DETECTION (Integrated: Firestore + Pinecone caching, then new pipeline with Fact Check)
# ---------------------------
@app.route("/detect_text", methods=["POST"])
def detect_text():
    try:
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
        
        # Normalized ID for semantic matching
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

        # Step 1.5: Firestore semantic search
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
            # Personalize if sim <0.95
            if firestore_semantic['similarity'] < 0.95:
                from misinfo_model import ask_gemini_structured
                personalization_prompt = f"""
                Original: "{original_text}"
                Similar cached: "{best.get('text', '')}", score={best.get('text_score', 0.5)}, pred={prediction}, exp="{explanation}"
                Personalize JSON: {{"score":<0-1>, "prediction":"Fake"/"Real", "explanation":"<reasoning>"}}
                """
                try:
                    gemini_resp = ask_gemini_structured(personalization_prompt)
                    if isinstance(gemini_resp, dict) and 'parsed' in gemini_resp:
                        pers = gemini_resp['parsed']
                        explanation = pers.get("explanation", explanation)
                        # Store personalized under original ID
                        embedding = generate_embedding(original_text)
                        db.collection('articles').document(article_id).set({
                            "url": url, "text": original_text, "normalized_id": norm_id, "embedding": embedding,
                            "text_score": pers.get("score", best.get("text_score", 0.5)),
                            "prediction": pers.get("prediction", prediction),
                            "gemini_reasoning": explanation, "text_explanation": explanation,
                            "total_views": 1, "total_reports": 1 if pers.get("score", 0.5) < 0.5 else 0,
                            "last_updated": datetime.utcnow(), "type": "text", "verified": True
                        })
                        store_feedback(original_text, explanation, [], "system", article_id=article_id, score=pers.get("score", 0.5), prediction=pers.get("prediction", prediction), verified=True)
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
        # Step 2: Semantic search in Pinecone
        semantic_result = search_feedback_semantic(original_text, article_id=article_id, verified_only=False)
        if semantic_result.get("source") == "cache":
            # Use cached with personalization if needed
            cached_doc_id = semantic_result.get("article_id")
            # Similar personalization logic as above, but using new ask_gemini_structured
            # (Omitted for brevity; adapt from old)
            # Fallback to raw cache if personalization fails
            return jsonify({
                **semantic_result,
                "article_id": article_id,
                "source": "semantic_cache",
                "details": [{
                    "score": semantic_result.get("score", 0.5),
                    "prediction": semantic_result.get("prediction", "Unknown"),
                    "explanation": semantic_result.get("explanation", ""),
                    "source": "semantic_cache",
                    "article_id": cached_doc_id
                }]
            })

        # Step 3: New integrated ML pipeline (Vertex AI + Fact Check + Gemini)
        print(f"No semantic cache hit for text: '{text}' - Running new analysis pipeline")
        model_result = detect_fake_text(original_text)  # This now includes Fact Check and ensemble
        text_score = model_result.get("summary", {}).get("score", 0.5) / 100  # Normalize to 0-1
        text_prediction = model_result.get("summary", {}).get("prediction", "Unknown")
        explanation = model_result.get("summary", {}).get("explanation", "Analysis complete.")

        # Step 4: Cache with normalized_id + embedding + verified
        verified = True
        embedding = generate_embedding(original_text)
        total_reports = 1 if text_score < 0.5 else 0
        doc_data = {
            "url": url,
            "text": original_text,
            "normalized_id": norm_id,
            "embedding": embedding,
            "text_score": text_score,
            "prediction": text_prediction,
            "text_explanation": explanation,
            "total_views": 1,
            "total_reports": total_reports,
            "last_updated": datetime.utcnow(),
            "type": "text",
            "verified": verified
        }
        db.collection('articles').document(article_id).set(doc_data)

        # Step 5: Store in Pinecone
        store_feedback(
            original_text, explanation, [], "system", article_id=article_id,
            score=text_score, prediction=text_prediction, verified=verified
        )

        # Response (adapt to new format)
        safe_result = make_json_safe(model_result)
        response = {
            "score": text_score,
            "prediction": text_prediction,
            "explanation": explanation,
            "article_id": article_id,
            "source": "new_analysis",
            "details": [safe_result],
            "runtime": safe_result.get("runtime", 0),
            "claims_checked": safe_result.get("claims_checked", 0)
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error in /detect_text: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------------------
# USER FEEDBACK (From old, adapted to new labels)
# ---------------------------
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    article_id = data.get("article_id", "")
    text = data.get("text", "")
    label = data.get("response", "").upper()  # "REAL" or "FAKE"

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
        # Store detailed feedback
        db.collection('articles').document(article_id).collection('feedbacks').add({
            "label": label,
            "explanation": explanation,
            "user_fingerprint": user_fingerprint,
            "timestamp": datetime.utcnow()
        })

        # Check threshold
        doc = get_article_doc(article_id)
        if doc:
            total_views = doc.get('total_views', 1) + 1
            total_reports = doc.get('total_reports', 0) + increment_reports
            percentage = (total_reports / total_views) * 100
            if percentage > 40:
                db.collection('articles').document(article_id).update({"community_flagged": True})
            return jsonify({"status": "feedback_recorded", "percentage_reported": f"{percentage:.0f}%"})

    # Legacy: Store in Pinecone if FAKE
    if label != "FAKE":
        return jsonify({"status": "ignored", "message": "Only FAKE labels are stored in legacy mode"}), 200

    result = store_feedback(text, explanation, sources, user_fingerprint)
    print("Final result is ", result)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result), 200

# ---------------------------
# CLEANUP EXPIRED DATA
# ---------------------------
@app.route("/cleanup_expired", methods=["POST"])
def cleanup_expired_endpoint():
    result = cleanup_expired()
    return jsonify(result), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)