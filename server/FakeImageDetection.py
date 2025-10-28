import os
import base64
import requests
import json
from typing import List, Dict, Any, Union
from google.cloud import vision
from google.api_core.exceptions import GoogleAPICallError, RetryError
from dotenv import load_dotenv
import google.generativeai as genai
import re

load_dotenv()

# ------------------------- Configuration -------------------------
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("IMG_ENDPOINT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = vision.ImageAnnotatorClient()
genai.configure(api_key=GEMINI_API_KEY)


# ------------------------- Helper Functions -------------------------
def _read_image(path: str) -> vision.Image:
    with open(path, "rb") as f:
        content = f.read()
    return vision.Image(content=content)


def _fetch_image_from_url(url: str, save_as: str = "temp_image.jpg") -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    with open(save_as, "wb") as f:
        f.write(response.content)
    return save_as


def _safe_vision_call(func, image: vision.Image, retries=2):
    for attempt in range(retries + 1):
        try:
            return func(image=image)
        except (GoogleAPICallError, RetryError) as e:
            if attempt == retries:
                raise
            print(f"[Warning] API call failed. Retrying ({attempt+1}/{retries})...")


def _strip_markdown_code_block(text: str) -> str:
    text = re.sub(r'^```json\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ------------------------- Vision AI Detection -------------------------
def detect_web_entities(image: vision.Image):
    response = _safe_vision_call(client.web_detection, image)
    if response.error.message:
        raise Exception(f"Web detection error: {response.error.message}")

    web_entities = response.web_detection
    return {
        "entities": [{"description": e.description, "score": e.score} for e in (web_entities.web_entities or [])],
        "exact_matches": [img.url for img in (web_entities.full_matching_images or [])],
        "similar_images": [img.url for img in (web_entities.visually_similar_images or [])],
    }


def detect_labels(image: vision.Image):
    response = _safe_vision_call(client.label_detection, image)
    if response.error.message:
        raise Exception(f"Label detection error: {response.error.message}")
    return [{"description": label.description, "score": label.score} for label in response.label_annotations]


def detect_faces(image: vision.Image):
    response = _safe_vision_call(client.face_detection, image)
    if response.error.message:
        raise Exception(f"Face detection error: {response.error.message}")
    return [{"detection_confidence": face.detection_confidence} for face in response.face_annotations]


def analyze_image(path_or_url: str) -> Dict[str, Any]:
    """Analyzes an image (local path or URL) using Vision AI."""
    if path_or_url.lower().startswith("http"):
        path = _fetch_image_from_url(path_or_url)
    else:
        path = path_or_url

    image = _read_image(path)

    results = {
        "labels": detect_labels(image),
        "faces": detect_faces(image),
        "web": detect_web_entities(image),
    }

    return results


# ------------------------- Vertex AI Prediction -------------------------
def call_vertex_ai_prediction(image_path: str) -> Dict[str, Any]:
    """Send image to Vertex AI endpoint for authenticity prediction."""
    with open(image_path, "rb") as f:
        image_bytes = base64.b64encode(f.read()).decode("utf-8")

    data = {
        "instances": [{"content": image_bytes}],
        "parameters": {"confidenceThreshold": 0.5, "maxPredictions": 5}
    }

    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/{ENDPOINT_ID}:predict"
    )

    token = os.popen("gcloud auth print-access-token").read().strip()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=data)
    try:
        result = resp.json()
        predictions = result.get("predictions", [{}])[0]
        return predictions
    except Exception as e:
        return {"error": f"Failed to parse Vertex AI response: {e}", "raw": resp.text}


# ------------------------- Gemini AI Fallback -------------------------
def call_gemini_detection(image_path: str) -> Dict[str, Any]:
    """Send image to Gemini for authenticity analysis."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        prompt = """
        Analyze the provided image to determine if it is likely AI-generated or real/authentic.
        Consider visual artifacts, inconsistencies, lighting, textures, and other indicators of generation.
        
        Output ONLY a valid JSON object with:
        {
            "ai_probability": <float between 0.0 and 1.0>,
            "verdict": "<Likely AI-generated / Likely Real / Uncertain>",
            "explanation": "<Short reasoning, no emojis, under 200 chars>"
        }
        """

        uploaded_file = genai.upload_file(path=image_path)
        response = model.generate_content([prompt, uploaded_file])
        text = _strip_markdown_code_block(response.text.strip())

        parsed = json.loads(text)
        ai_prob = parsed.get("ai_probability", 0.5)

        if ai_prob > 0.7:
            parsed["verdict"] = "Likely AI-generated"
        elif ai_prob < 0.3:
            parsed["verdict"] = "Likely Real"
        else:
            parsed["verdict"] = "Uncertain"

        return parsed

    except Exception as e:
        return {"error": f"Gemini error: {e}"}


# ------------------------- Combine Vision + Vertex + Gemini -------------------------
def score_ai_likelihood(vision_data: Dict[str, Any], vertex_result: Dict[str, Any]) -> Dict[str, Any]:
    """Combine Vision AI and Vertex AI signals to generate authenticity verdict."""
    explanation = []
    ai_prob = 0.5
    web_info = vision_data.get("web", {})

    if web_info.get("exact_matches"):
        explanation.append("✅ Exact matches found online — likely real.")
        ai_prob -= 0.3
    elif web_info.get("similar_images"):
        explanation.append("⚠️ Similar images found — moderate authenticity.")
        ai_prob -= 0.1
    else:
        explanation.append("❌ No web presence — may be AI-generated.")
        ai_prob += 0.25

    if vision_data.get("faces"):
        explanation.append("✅ Faces detected — consistent with real photos.")
        ai_prob -= 0.1

    if "displayNames" in vertex_result and "confidences" in vertex_result:
        preds = list(zip(vertex_result["displayNames"], vertex_result["confidences"]))
        top_pred, top_conf = preds[0]
        explanation.append(f"🧠 Vertex AI predicts '{top_pred}' with {round(top_conf * 100, 1)}% confidence.")
        if "fake" in top_pred.lower() or "ai" in top_pred.lower():
            ai_prob += top_conf * 0.6
        else:
            ai_prob -= top_conf * 0.6
    else:
        explanation.append("⚠️ Vertex AI result unavailable or malformed.")

    ai_prob = max(0.0, min(1.0, ai_prob))
    verdict = "Likely AI-generated" if ai_prob > 0.7 else "Likely Real" if ai_prob < 0.3 else "Uncertain"

    return {
        "ai_probability": round(ai_prob, 2),
        "verdict": verdict,
        "explanation": " ".join(explanation),
    }


# ------------------------- Core Evaluation -------------------------
def _evaluate_single_image(url_or_path: str) -> Dict[str, Any]:
    try:
        print("Evaluating image:", url_or_path)
        if url_or_path.startswith("http"):
            path = _fetch_image_from_url(url_or_path)
        else:
            path = url_or_path
        
        # Vision AI
        vision_data = analyze_image(path)
        # Vertex AI
        vertex_result = call_vertex_ai_prediction(path)

        # ----------------- Gemini Fallback -----------------
        if not vertex_result or "error" in vertex_result or not vertex_result.get("displayNames"):
            print("[Fallback] Vertex AI returned empty or invalid result. Using Gemini...")
            gemini_result = call_gemini_detection(path)
            ai_prob = gemini_result.get("ai_probability", 0.5)
            verdict = gemini_result.get("verdict", "Uncertain")
            explanation = gemini_result.get("explanation", "Gemini fallback used.")
            final_score = int((1 - ai_prob) * 100)

            return {
                "image_source": url_or_path,
                "score": final_score,
                "verdict": verdict,
                "explanation": explanation,
                "vision_details": vision_data,
                "vertex_ai_result": {"gemini_result": gemini_result},
            }

        # ----------------- Vertex + Vision Normal Path -----------------
        result = score_ai_likelihood(vision_data, vertex_result)
        final_score = int((1 - result["ai_probability"]) * 100)

        return {
            "image_source": url_or_path,
            "score": final_score,
            "verdict": result["verdict"],
            "explanation": result["explanation"],
            "vision_details": vision_data,
            "vertex_ai_result": vertex_result,
        }

    except Exception as e:
        print(f"[detect_fake_image] Error analyzing image: {e}")
        return {
            "image_source": url_or_path,
            "score": 0,
            "verdict": "Error",
            "explanation": f"Error analyzing image: {e}",
            "details": {},
        }


def detect_fake_image(inputs: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if isinstance(inputs, str):
        return _evaluate_single_image(inputs)
    elif isinstance(inputs, list):
        results = [_evaluate_single_image(i) for i in inputs]
        print(f"[detect_fake_image] Processed {len(inputs)} image(s).")
        return results
    else:
        raise ValueError("Input must be a string or a list of strings.")


# ------------------------- Example Usage -------------------------
if __name__ == "__main__":
    test_url = "https://cdn.pixabay.com/photo/2024/05/28/16/39/ai-generated-8794203_1280.png"
    print(json.dumps(detect_fake_image(test_url), indent=2))
