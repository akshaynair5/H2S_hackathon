from flask import Flask, request, jsonify, session, make_response,Response,stream_with_context
from misinfo_model import detect_fake_text
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from vectorDb import search_feedback_semantic, store_feedback, cleanup_expired
from database import generate_id, generate_normalized_id, generate_embedding, get_article_doc, firestore_semantic_search, db
from FakeImageDetection import detect_fake_image
from firebase_admin import credentials, firestore
from tasks import cancel_session_tasks, get_session_tasks 
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import numpy as np
import uuid
import queue
import threading
import json
from translate import translate_to_english



load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY")
CORS(app, supports_credentials=True)



# Global log queue for SSE
log_queues = {}
log_queues_lock = threading.Lock()

# ---------------------------
# RATE LIMITER SETUP
# ---------------------------
def get_user_identifier():
    """Use existing session logic for rate limiting"""
    return (
        request.headers.get("user-fingerprint") or
        request.headers.get("X-Session-ID") or
        request.json.get("session_id") if request.json else None or
        session.get("session_id") or
        get_remote_address()  
    )

limiter = Limiter(
    app=app,
    key_func=get_user_identifier,
    default_limits=["2000 per day", "300 per hour"],  
    storage_uri="memory://",
    headers_enabled=True
)

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "rate_limit_exceeded",
        "message": "Please slow down and try again in a moment.",
        "retry_after_seconds": getattr(e, 'description', 'Unknown')
    }), 429


# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
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


def get_session_id():
    """Extract session ID from various sources"""
    return (
        request.json.get("session_id") if request.json else None or
        request.headers.get("X-Session-ID") or 
        request.headers.get("user-fingerprint") or
        session.get("session_id") or
        str(uuid.uuid4()) 
    )


# ---------------------------
# IMAGE DETECTION 
# ---------------------------
@app.route("/detect_image", methods=["POST"])
@limiter.limit("60 per minute") 
def detect_image():
    data = request.json
    urls = data.get("urls") or data.get("images") 
    print("Urls is", urls)
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



# --------------------------
# Log info 
# ---------------------------



def get_log_queue(session_id):
    """Get or create a log queue for a session"""
    with log_queues_lock:
        if session_id not in log_queues:
            log_queues[session_id] = queue.Queue(maxsize=100)
        return log_queues[session_id]

def cleanup_log_queue(session_id):
    """Remove log queue for session"""
    with log_queues_lock:
        log_queues.pop(session_id, None)

# Add endpoint for log detection
@app.route("/stream_logs/<session_id>", methods=["GET"])
@limiter.exempt
def stream_logs(session_id):
    """Stream logs to frontend via Server-Sent Events"""
    def generate():
        log_queue = get_log_queue(session_id)
        try:
            while True:
                try:
                    # Wait for log message with timeout
                    msg = log_queue.get(timeout=30)
                    if msg == "DONE":
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                        break
                    yield f"data: {json.dumps(msg)}\n\n"
                except queue.Empty:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except GeneratorExit:
            pass
        finally:
            cleanup_log_queue(session_id)
    
    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )



# ---------------------------
# TEXT DETECTION 
# ---------------------------
@app.route("/detect_text", methods=["POST"])
@limiter.limit("30 per minute") 
def detect_text():
    session_id = None
    try:
        data = request.json 
        original_text = data.get("text", "")
        url = data.get("url", "")

        session_id = get_session_id()

        log_queue = get_log_queue(session_id)
        
        # Send initial log
        log_queue.put({"type": "info", "message": "Starting analysis..."})

        print(f"Processing request for session: {session_id}")




       
        
        print(f"User selected text: '{original_text}' - \n")
        print(f"URL IS  '{url}'- \n")
        
        if len(original_text) < 10: 
            return jsonify({"error": "Text too short or missing"}), 400

        if len(original_text) > 1000:  # Adjust threshold as needed
            return jsonify({"error": "Text is too large to analyze. Please provide shorter text (maximum 2 paragraphs)."}), 400
        
        # Translate to English if needed
        try:
            translation_result = translate_to_english(original_text)
            text_for_analysis = translation_result['translated_text']
            detected_language = translation_result['detected_language']
            was_translated = translation_result['was_translated']
    
            if was_translated:
                log_queue.put({
            "type": "info", 
            "message": f"Detected {detected_language}, translated to English for analysis"
        })
        except Exception as e:
            print(f"Translation error: {e}")
            return jsonify({"error": "Failed to translate text. Please try again."}), 500
        
        text = text_for_analysis.strip()
        print(f'Translated text ' , text)
        

        article_id = generate_id(url, text)

        norm_id = generate_normalized_id(url, text)
        print(f"Exact ID: {article_id}, Norm ID: {norm_id}")

        log_queue.put({"type": "info", "message": "Checking cache..."})
        cached = get_article_doc(article_id)
        if cached:
            log_queue.put({"type": "success", "message": "Found in cache!"})
            log_queue.put("DONE")
            prediction = cached.get("prediction", "Unknown")
            explanation = cached.get("gemini_reasoning", cached.get("text_explanation", ""))
            return jsonify({
                "score": cached.get("text_score", 0.5),
                "prediction": prediction,
                "explanation": explanation,
                "article_id": article_id,
                "norm_id":norm_id,
                "source": "firestore_exact",
                "session_id": session_id,  
                "details": [{
                    "score": cached.get("text_score", 0.5),
                    "prediction": prediction,
                    "explanation": explanation,
                    "source": "firestore_exact",
                    "article_id": article_id
                }]
            })

        print("Exact miss; trying Firestore semantic search...")
        log_queue.put({"type": "info", "message": "Running semantic search..."})
        firestore_semantic = firestore_semantic_search(text)

        if firestore_semantic:
            log_queue.put({"type": "success", "message": "Similar article found!"})
            best = firestore_semantic['best']
            best_article_id = firestore_semantic['best_id']
            prediction = best.get("prediction", "Unknown")
            explanation = best.get("gemini_reasoning", best.get("text_explanation", ""))
            print(f"Semantic hit! Using candidate with sim {firestore_semantic['similarity']:.3f}, score {best.get('text_score', 0.5)}")
            
            db.collection('articles').document(best_article_id).update({"total_views": firestore.Increment(1)})
            if firestore_semantic['similarity'] < 0.95:
                from misinfo_model import ask_gemini_structured
                personalization_prompt = f"""
                Original: "{text}"
                Similar cached: "{best.get('text', '')}", score={best.get('text_score', 0.5)}, pred={prediction}, exp="{explanation}"
                Personalize JSON: {{"score":<0-1>, "prediction":"Fake"/"Real", "explanation":"<reasoning>"}}
                """
                try:
                    gemini_resp = ask_gemini_structured(personalization_prompt)
                    if isinstance(gemini_resp, dict) and 'parsed' in gemini_resp:
                        pers = gemini_resp['parsed']
                        explanation = pers.get("explanation", explanation)
                        embedding = generate_embedding(text)
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
                "norm_id":norm_id,
                "source": "firestore_semantic",
                "session_id": session_id,
                "details": [{"score": best.get("text_score", 0.5), "prediction": prediction, "explanation": explanation, "source": "firestore_semantic", "article_id": best_article_id, "similarity": firestore_semantic['similarity']}]
            })

        print("No semantic Firestore hit; querying Pinecone for semantic similar verified fakes...")
        semantic_result = search_feedback_semantic(text, article_id=article_id, verified_only=False)
        if semantic_result.get("source") == "cache":
            cached_doc_id = semantic_result.get("article_id")
            return jsonify({
                **semantic_result,
                "article_id": article_id,
                "source": "semantic_cache",
                "session_id": session_id, 
                "norm_id":norm_id,
                "details": [{
                    "score": semantic_result.get("score", 0.5),
                    "prediction": semantic_result.get("prediction", "Unknown"),
                    "explanation": semantic_result.get("explanation", ""),
                    "source": "semantic_cache",
                    "article_id": cached_doc_id
                }]
            })

        print(f"No semantic cache hit for text: '{text}' - Running new analysis pipeline")
        log_queue.put({"type": "info", "message": "Analyzing with AI models..."})
        log_queue.put({"type": "phase", "message": "Phase 1: Fact checking"})
        model_result = detect_fake_text(text , log_queue)
        log_queue.put({"type": "success", "message": "Analysis complete!"})
        log_queue.put("DONE")
        text_score = model_result.get("summary", {}).get("score", 0.5) / 100
        text_prediction = model_result.get("summary", {}).get("prediction", "Unknown")
        explanation = model_result.get("summary", {}).get("explanation", "Analysis complete.")

        verified = True
        embedding = generate_embedding(text)
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

        store_feedback(
            original_text, explanation, [], "system", article_id=article_id,
            score=text_score, prediction=text_prediction, verified=verified
        )

        safe_result = make_json_safe(model_result)
        response = {
            "score": text_score,
            "prediction": text_prediction,
            "explanation": explanation,
            "article_id": article_id,
            "norm_id":norm_id,
            "source": "new_analysis",
            "session_id": session_id,
            "details": [safe_result],
            "runtime": safe_result.get("runtime", 0),
            "claims_checked": safe_result.get("claims_checked", 0)
        }
        
        return jsonify(response)

    except Exception as e:
        if session_id:  # Only send logs if session_id exists
            log_queue = get_log_queue(session_id)
            log_queue.put({"type": "error", "message": "Analysis failed"})
            log_queue.put("DONE")
        print(f"Error in /detect_text: {e}")
        return jsonify({"error": str(e)}), 500







# ---------------------------
# USER FEEDBACK 
# ---------------------------
@app.route("/submit_feedback", methods=["POST"])
@limiter.limit("100 per minute")
def submit_feedback():
    data = request.json
    article_id = data.get("article_id", "")
    text = data.get("text", "")
    label = data.get("response", "").upper() 

    if label == "YES":
        label = "FAKE"
    else:
        label = "REAL"
    print("Label is", label)
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
        increment_reports = 1 if label == "FAKE" else 0
        db.collection('articles').document(article_id).update({
            "total_views": firestore.Increment(1),
            "total_reports": firestore.Increment(increment_reports)
        })
        db.collection('articles').document(article_id).collection('feedbacks').add({
            "label": label,
            "explanation": explanation,
            "user_fingerprint": user_fingerprint,
            "timestamp": datetime.utcnow()
        })

        doc = get_article_doc(article_id)
        if doc:
            total_views = doc.get('total_views', 1) + 1
            total_reports = doc.get('total_reports', 0) + increment_reports
            percentage = (total_reports / total_views) * 100
            if percentage > 40:
                db.collection('articles').document(article_id).update({"community_flagged": True})
            return jsonify({"status": "feedback_recorded", "percentage_reported": f"{percentage:.0f}%"})

    if label != "FAKE":
        return jsonify({"status": "ignored", "message": "Only FAKE labels are stored in legacy mode"}), 200

    result = store_feedback(text, explanation, sources, user_fingerprint)
    print("Final result is", result)
    if "error" in result:
        return jsonify({"error": result["error"]}), 400
    return jsonify(result), 200


# ---------------------------
# SESSION MANAGEMENT
# ---------------------------
@app.route("/cancel_session", methods=["POST"])
@limiter.exempt 
def cancel_session():
    """Cancel all running tasks for the current session when user exits website"""
    data = request.json or {}

    session_id = (
        data.get("session_id") or 
        request.headers.get("X-Session-ID") or 
        request.headers.get("user-fingerprint") or
        session.get("session_id")
    )
    
    if not session_id:
        return jsonify({
            "error": "No session identifier provided",
            "message": "Please provide session_id in request body or X-Session-ID header"
        }), 400
    
    print(f"Cancelling all tasks for session: {session_id}")
    result = cancel_session_tasks(session_id)
    
    return jsonify({
        "status": "success",
        "session_id": session_id,
        **result
    }), 200


@app.route("/session_tasks", methods=["GET"])
@limiter.exempt 
def session_tasks():
    """Get all active tasks for the current session"""
    session_id = (
        request.args.get("session_id") or
        request.headers.get("X-Session-ID") or 
        request.headers.get("user-fingerprint")
    )
    
    if not session_id:
        return jsonify({"error": "No session identifier"}), 400
    
    active_tasks = get_session_tasks(session_id)
    
    return jsonify({
        "session_id": session_id,
        "active_tasks": active_tasks,
        "count": len(active_tasks)
    }), 200


# ---------------------------
# CLEANUP EXPIRED DATA
# ---------------------------
@app.route("/cleanup_expired", methods=["POST"])
@limiter.exempt 
def cleanup_expired_endpoint():
    result = cleanup_expired()
    return jsonify(result), 200


# ---------------------------
# HEALTH CHECK 
# ---------------------------
@app.route("/health", methods=["GET"])
@limiter.exempt
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }), 200


if __name__ == "__main__":
    app.run(port=5000, debug=True)