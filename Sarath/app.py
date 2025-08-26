from flask import Flask, request, jsonify
from visionAi import detect_fake_image
from vid_detection import detect_fake_video

app = Flask(__name__)

@app.route("/detect_image", methods=["POST"])
def detect_image():
    data = request.json
    image_url = data["url"]
    # download image from URL before passing
    results = detect_fake_image(image_url)
    return jsonify(results)

@app.route("/detect_video", methods=["POST"])
def detect_video():
    data = request.json
    video_url = data["url"]
    # download video before passing
    results = detect_fake_video(video_url)
    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
