from flask import Flask, request, jsonify
from FakeImageDetection import detect_fake_image
from misinfo_model import detect_fake_text
from flask_cors import CORS
from vectorDb import search_feedback, store_feedback, cleanup_expired , index
import hashlib
import traceback

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

import numpy as np

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
    
@app.route("/detect_text", methods=["POST"])
def detect_text():
    try:
        print("🟢 [START] Incoming /detect_text request")

        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload."}), 400

        text = str(data.get("text", "")).strip()
        if not text:
            return jsonify({"error": "No text provided."}), 400

        if len(text) > 4000:
            print("⚠️ Text too long; truncating to 4000 chars")
            text = text[:4000]

        print(f"[Info] Analyzing new text input: '{text[:80]}...'")
        result = detect_fake_text(text)
        print("✅ detect_fake_text() completed successfully")

        safe_result = make_json_safe(result)
        print("🟢 JSON-safe conversion complete")

        # --- Extract simplified summary ---
        summary = safe_result.get("summary", {})
        score = summary.get("score", 0)
        prediction = summary.get("prediction", "Unknown")
        explanation = summary.get("explanation", "No explanation available.")

        response = jsonify({
            "success": True,
            "source": "analysis",
            "input_text": text,
            "score": score,
            "prediction": prediction,
            "explanation": explanation,
            "runtime": safe_result.get("runtime", 0),
            "claims_checked": safe_result.get("claims_checked", 0),
            "result": safe_result 
        })
        print("✅ Response prepared successfully")
        return response, 200

    except Exception as e:
        print("❌ ERROR inside /detect_text:", e)
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "details": traceback.format_exc(),
        }), 500

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