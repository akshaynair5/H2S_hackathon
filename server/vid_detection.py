from flask import Flask, request, jsonify
from google.cloud import vision
import requests
import io

app = Flask(__name__)
client = vision.ImageAnnotatorClient.from_service_account_file("key.json")

def detect_fake_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        content = io.BytesIO(response.content)

        image = vision.Image(content=content.getvalue())
        result = client.web_detection(image=image)
        annotations = result.web_detection

        detection_summary = {
            "image_url": url,
            "full_matching_images": [],
            "visually_similar_images": [],
            "pages_with_matching_images": []
        }

        if annotations.full_matching_images:
            detection_summary["full_matching_images"] = [img.url for img in annotations.full_matching_images]

        if annotations.visually_similar_images:
            detection_summary["visually_similar_images"] = [img.url for img in annotations.visually_similar_images]

        if annotations.pages_with_matching_images:
            detection_summary["pages_with_matching_images"] = [
                {"title": page.page_title, "url": page.url} for page in annotations.pages_with_matching_images
            ]

        return detection_summary
    except Exception as e:
        return {"error": str(e), "image_url": url}

@app.route("/detect_image", methods=["POST"])
def detect_image():
    data = request.json
    image_urls = data.get("images", [])
    
    results = []
    for url in image_urls:
        results.append(detect_fake_image_from_url(url))

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
