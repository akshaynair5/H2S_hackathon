from flask import Flask, request, jsonify
from visionAi import detect_fake_image
from misinfo_model import detect_fake_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------------------
# IMAGE DETECTION
# ---------------------------
@app.route("/detect_image", methods=["POST"])
def detect_image():
    data = request.json
    urls = data.get("urls") or data.get("images") 
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
@app.route("/detect_text", methods=["POST"])
def detect_text():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    result = detect_fake_text(text)
    return jsonify(result)

# ---------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
