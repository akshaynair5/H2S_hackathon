from google.cloud import vision
import requests
import io
from PIL import Image, ExifTags
from urllib.parse import urlparse
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

# ---------------------------
# Initialize clients
# ---------------------------
client = vision.ImageAnnotatorClient.from_service_account_file("key.json")

extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-base-224")
model = AutoModelForImageClassification.from_pretrained("facebook/convnext-base-224")
model.eval()

CREDIBLE_DOMAINS = ["nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com", "gov"]


# ---------------------------
# Helper: AI image probability
# ---------------------------
def detect_ai_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = extractor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)
    ai_prob = probs[0][1].item() 
    return ai_prob 


# ---------------------------
# Helper: Extract metadata
# ---------------------------
def extract_metadata(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    info = img._getexif()
    metadata = {}
    if info:
        for tag, value in info.items():
            tag_name = ExifTags.TAGS.get(tag, tag)
            metadata[tag_name] = value
    return metadata


# ---------------------------
# Helper: Source credibility
# ---------------------------
def credibility_score(urls):
    score = 0.0
    for u in urls:
        domain = urlparse(u).netloc.lower()
        for cd in CREDIBLE_DOMAINS:
            if cd in domain:
                score += 0.5
    return min(1.0, score)


# ---------------------------
# Core: Single image evaluation
# ---------------------------
def _evaluate_single_image(url):
    explanation_parts = []
    score = 0.0
    try:
        response = requests.get(url, timeout=10)
        image_bytes = response.content

        image = vision.Image(content=image_bytes)
        annotations = client.web_detection(image=image).web_detection

        web_score = 0.0
        matched_pages = [p.url for p in annotations.pages_with_matching_images if p.url] if annotations.pages_with_matching_images else []

        if annotations.full_matching_images:
            web_score += 0.4
            explanation_parts.append("Exact matches found online (likely authentic).")
        if annotations.visually_similar_images:
            web_score += 0.3
            explanation_parts.append("Visually similar images found online.")
        if matched_pages:
            web_score += 0.3
            explanation_parts.append(f"{len(matched_pages)} pages found with matching images.")

        web_score = min(1.0, web_score)

        ai_prob = detect_ai_image(image_bytes)
        explanation_parts.append(f"AI-generated probability: {ai_prob*100:.1f}%")
        ai_score = 1 - ai_prob 

        metadata = extract_metadata(image_bytes)
        if not metadata:
            explanation_parts.append("No metadata found; could be AI-generated or edited.")
        elif any("Photoshop" in str(v) or "AI" in str(v) for v in metadata.values()):
            explanation_parts.append("Metadata suggests editing or AI generation.")
            ai_score = max(0.0, ai_score - 0.2)

        source_score = credibility_score(matched_pages)
        if source_score > 0:
            explanation_parts.append(f"Found on credible sources (score: {source_score*100:.0f}%).")

        final_score = min(1.0, 0.4*web_score + 0.4*ai_score + 0.2*source_score)

        return {
            "image_url": url,
            "score": round(final_score*100),
            "explanation": " ".join(explanation_parts),
            "details": {
                "web_score": web_score,
                "ai_prob": ai_prob,
                "metadata": metadata,
                "source_score": source_score,
                "matched_pages": matched_pages
            }
        }

    except Exception as e:
        return {"image_url": url, "score": 50, "explanation": f"Error processing image: {str(e)}", "details": {}}


# ---------------------------
# Public API: single or list
# ---------------------------
def detect_fake_image(urls):
    if isinstance(urls, str):
        return _evaluate_single_image(urls)
    elif isinstance(urls, list):
        return [_evaluate_single_image(u) for u in urls]
    else:
        raise ValueError("Input must be a string or a list of strings.")