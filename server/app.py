from flask import Flask, request, jsonify
from visionAi import detect_fake_image
from misinfo_model import detect_fake_text
from flask_cors import CORS
from vectorDb import search_feedback, store_feedback, cleanup_expired , index
import hashlib

app = Flask(__name__)
CORS(app)

# ---------------------------
# IMAGE DETECTION
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
# TEXT DETECTION
# ---------------------------
# ---------------------------
# TEXT DETECTION
# ---------------------------
@app.route("/detect_text", methods=["POST"])
def detect_text():
    data = request.json 
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    # Check Pinecone for cached feedback first
    cached_result = search_feedback(text)
    if "error" in cached_result:
        print(f"Pinecone search failed for text: '{text}' - Error: {cached_result['error']}")
        return jsonify({"error": cached_result["error"]}), 400
    if cached_result.get("source") == "cache":
        # Determine if it was an exact or similar match
        vec_id = hashlib.sha256(text.encode()).hexdigest()
        exact_match = index.fetch(ids=[vec_id], namespace="default")
        match_type = "exact" if exact_match.vectors else "similar"
        print(f"Found {match_type} match in Pinecone for text: '{text}' - Explanation: {cached_result['explanation']}")
        return jsonify(cached_result)

    # No reliable match found, fallback to ML model
    print(f"No reliable match found in Pinecone for text: '{text}' - Falling back to ML model")
    result = detect_fake_text(text)
    print(jsonify(result))
    return jsonify(result)

# ---------------------------
# USER FEEDBACK
# ---------------------------
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    data = request.json
    text = data.get("text", "")
    explanation = data.get("explanation", "")
    response = data.get("response", "").upper()
    sources = data.get("sources", [])
    user_fingerprint = request.headers.get("user-fingerprint", "default")
    print(f"""
Collected Data:
  Text: {text}
  Explanation: {explanation}
  Response: {response}
  Sources: {sources}
  User Fingerprint: {user_fingerprint}
""")

    if not text.strip() or not explanation or response not in ["YES", "NO"]:
        return jsonify({"error": "Missing or invalid text, explanation, or response"}), 400

    if response != "YES":
        return jsonify({"status": "ignored", "message": "Only YES responses are stored"}), 200

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




# ---------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)