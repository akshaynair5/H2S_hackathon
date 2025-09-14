from google.cloud import vision
import requests
import io

client = vision.ImageAnnotatorClient.from_service_account_file("key.json")

def _detect_single_image(url):
    try:
        response = requests.get(url, timeout=10)
        content = io.BytesIO(response.content)

        image = vision.Image(content=content.getvalue())
        result = client.web_detection(image=image)
        annotations = result.web_detection

        score = 0.0
        explanation_parts = []

        if annotations.full_matching_images:
            score += 0.4
            explanation_parts.append("Exact matches found online (more likely authentic).")

        if annotations.visually_similar_images:
            score += 0.3
            explanation_parts.append("Visually similar images found (likely real).")

        if annotations.pages_with_matching_images:
            score += 0.3
            explanation_parts.append("Pages with matching images detected.")

        if score == 0:
            score = 0.2
            explanation_parts.append("No matches found; could be AI-generated.")

        score = min(1.0, score)

        return {
            "image_url": url,
            "score": score,
            "explanation": " ".join(explanation_parts) if explanation_parts else "Uncertain; no strong signals."
        }

    except Exception as e:
        return {"image_url": url, "score": 0.5, "explanation": f"Error: {str(e)}"}

    try:
        response = requests.get(url, timeout=10)
        content = io.BytesIO(response.content)

        image = vision.Image(content=content.getvalue())
        result = client.web_detection(image=image)
        annotations = result.web_detection

        score = 0.0
        explanation_parts = []

        if annotations.full_matching_images:
            score += 0.2
            explanation_parts.append("Exact matches found online (more likely real).")

        if annotations.visually_similar_images:
            score += 0.2
            explanation_parts.append("Visually similar images found (likely real).")

        if annotations.pages_with_matching_images:
            score += 0.2
            explanation_parts.append("Pages with matching images detected.")

        if not (annotations.full_matching_images or annotations.visually_similar_images or annotations.pages_with_matching_images):
            score = 0.8
            explanation_parts.append("No matches found; could be AI-generated.")

        score = min(1.0, score)

        return {
            "image_url": url,
            "score": score,
            "explanation": " ".join(explanation_parts) if explanation_parts else "Uncertain; no strong signals."
        }

    except Exception as e:
        return {"image_url": url, "score": 0.5, "explanation": f"Error: {str(e)}"}

def detect_fake_image(urls):
    """
    Accepts either:
      - a single image URL (str)
      - a list of image URLs (list[str])
    Returns a dict (if str) or a list of dicts (if list).
    """
    if isinstance(urls, str):
        return _detect_single_image(urls)
    elif isinstance(urls, list):
        return [_detect_single_image(url) for url in urls]
    else:
        raise ValueError("Input must be a string (URL) or a list of URLs.")
