"""
image_verifier.py

Improved image verification module for browser extension.
- Ensemble AI-detection: ConvNeXt classification + CLIP zero-shot similarity to "photo" vs "synthetic".
- Metadata extraction and entropy heuristic.
- Robust Google Vision web_detection usage.
- Credibility scoring from matched pages + configurable credible domains.
- Optional Gemini second-opinion via misinfo_model.ask_gemini (if present).

Dependencies:
- google-cloud-vision
- requests
- pillow
- torch
- transformers
- (optional) misinfo_model.ask_gemini
"""

import io
import os
import math
import logging
from urllib.parse import urlparse
from typing import List, Union, Dict, Any, Optional

import requests
from PIL import Image, ExifTags, TiffImagePlugin

# Hugging Face imports
import torch
from transformers import (
    AutoFeatureExtractor,
    AutoModelForImageClassification,
    CLIPProcessor,
    CLIPModel,
)

# Google Vision imports
from google.cloud import vision

# Optional Gemini helper (user provided file)
try:
    from misinfo_model import ask_gemini
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# ---------- Configuration ----------
# Default HF models (changeable)
DEFAULT_CLIP = "openai/clip-vit-base-patch32"       # CLIP model for zero-shot textual comparisons

# Credible domains (expand as needed)
CREDIBLE_DOMAINS = [
    "nytimes.com", "bbc.com", "cnn.com", "reuters.com", "theguardian.com",
    ".gov", ".edu", "who.int", "un.org"
]

# Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------- Initialize external clients ----------
# Note: caller must ensure GOOGLE_APPLICATION_CREDENTIALS or use from_service_account_file
try:
    vision_client = vision.ImageAnnotatorClient.from_service_account_file("key.json")
except Exception as e:
    logger.warning("Could not initialize Google Vision client from key.json: %s", e)
    # Try using default credentials (if environment configured)
    try:
        vision_client = vision.ImageAnnotatorClient()
    except Exception as e2:
        logger.error("Google Vision client unavailable: %s", e2)
        vision_client = None

_CLIP = None
_CLIP_PROCESSOR = None

ENSEMBLE_MODELS = [
    "dima806/deepfake_vs_real_image_detection",
    "prithivMLmods/Deep-Fake-Detector-v2-Model",
    "NYUAD-ComNets/NYUAD_AI-generated_images_detector",
    
]

_CLASSIFIERS = []
_CLASSIFIER_EXTRACTORS = []

def init_models(
    classifiers=ENSEMBLE_MODELS,
    clip_name=DEFAULT_CLIP,
    device: Optional[str] = None
):
    """
    Initialize multiple HF classifiers into an ensemble,
    and also load a CLIP model once.
    """
    global _CLASSIFIERS, _CLASSIFIER_EXTRACTORS, _CLIP, _CLIP_PROCESSOR

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load classifier ensemble if not already loaded
    if not _CLASSIFIERS:
        for clf_name in classifiers:
            try:
                extractor = AutoFeatureExtractor.from_pretrained(clf_name)
                model = AutoModelForImageClassification.from_pretrained(clf_name).to(device).eval()
                _CLASSIFIER_EXTRACTORS.append(extractor)
                _CLASSIFIERS.append(model)
                logger.info(f"Loaded classifier {clf_name} on {device}")
            except Exception as e:
                logger.error(f"Failed to load {clf_name}: {e}")

    # Load CLIP once
    if _CLIP is None:
        try:
            _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(clip_name)
            _CLIP = CLIPModel.from_pretrained(clip_name).to(device).eval()
            logger.info(f"Loaded CLIP {clip_name} on {device}")
        except Exception as e:
            logger.error(f"Failed to load CLIP {clip_name}: {e}")


import random, numpy as np

def set_seed(seed: int) -> None:
    """
    Set random seeds across Python, NumPy, and Torch
    for reproducibility and stable outputs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Optional AIOrNot API
AIORNOT_ENDPOINT = "https://api.aiornot.com/v2/image/sync"
AIORNOT_KEY = os.getenv("AIORNOT_API_KEY")

# ---------- AIOrNot API ----------
def _aiornot_api_call(image_bytes: bytes) -> Optional[float]:
    """Return AI probability from AIOrNot API, or None if fails."""
    if not AIORNOT_KEY:
        return None
    try:
        files = {"image": ("temp.jpg", image_bytes, "image/jpeg")}
        resp = requests.post(
            AIORNOT_ENDPOINT,
            headers={"Authorization": f"Bearer {AIORNOT_KEY}"},
            files=files,
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("AIOrNot API error %s: %s", resp.status_code, resp.text)
            return None
        data = resp.json()
        # Ensure API returns expected field
        if "ai_probability" in data:
            return float(data["ai_probability"])
        elif "result" in data and "probability" in data["result"]:
            return float(data["result"]["probability"])
        else:
            logger.warning("AIOrNot response missing probability field.")
            return None
    except Exception as e:
        logger.warning("AIOrNot API call failed: %s", e)
        return None

# ---------- Helpers ----------
def robust_get(url: str, timeout: int = 10, max_size_bytes: int = 8 * 1024 * 1024) -> bytes:
    """Download image bytes safely (content-type check, size limit, timeouts)."""
    headers = {"User-Agent": "ImageVerifier/1.0"}
    r = requests.get(url, timeout=timeout, headers=headers, stream=True)
    r.raise_for_status()

    # Ensure response looks like an image
    content_type = r.headers.get("Content-Type", "").lower()
    if not content_type.startswith("image/"):
        raise ValueError(f"URL does not point to an image (Content-Type={content_type})")

    # Enforce size limit if content-length header is present
    content_length = r.headers.get("Content-Length")
    if content_length:
        if int(content_length) > max_size_bytes:
            raise ValueError("Image too large (Content-Length).")

    # Stream with size check
    data = io.BytesIO()
    total = 0
    for chunk in r.iter_content(1024):
        if not chunk:
            break
        total += len(chunk)
        if total > max_size_bytes:
            raise ValueError("Image too large (stream).")
        data.write(chunk)

    image_bytes = data.getvalue()
    print(f"[robust_get] Downloaded {len(image_bytes)} bytes from {url} (Content-Type={content_type})")
    return image_bytes

def extract_metadata(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    raw_exif = getattr(img, "_getexif", lambda: None)() or {}
    metadata = {}

    def make_json_safe(val):
        # Convert IFDRational → float
        if isinstance(val, TiffImagePlugin.IFDRational):
            return float(val)
        # Convert bytes → string
        if isinstance(val, bytes):
            try:
                return val.decode(errors="ignore")
            except Exception:
                return str(val)
        # Convert nested dict/list recursively
        if isinstance(val, dict):
            return {k: make_json_safe(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [make_json_safe(v) for v in val]
        # Fallback: cast to str if not JSON serializable
        try:
            import json
            json.dumps(val)
            return val
        except Exception:
            return str(val)

    for tag, value in raw_exif.items():
        tag_name = ExifTags.TAGS.get(tag, tag)
        metadata[tag_name] = make_json_safe(value)

    return metadata

def image_entropy(image: Image.Image) -> float:
    """Compute an entropy-like measure of the image channels (useful heuristic)."""
    try:
        if image.mode != "L":
            img = image.convert("L")
        else:
            img = image
        hist = img.histogram()
        hist_size = sum(hist)
        if hist_size == 0:
            return 0.0
        probs = [h / hist_size for h in hist if h > 0]
        entropy = -sum(p * math.log2(p) for p in probs)
        return float(entropy)
    except Exception as e:
        logger.debug("Entropy computation failed: %s", e)
        return 0.0

def credibility_score(urls: List[str], credible_domains: List[str] = CREDIBLE_DOMAINS) -> float:
    """Simple credibility scoring: +0.5 per credible domain found, cap at 1.0."""
    score = 0.0
    seen_domains = set()
    for u in urls:
        try:
            domain = urlparse(u).netloc.lower()
            # reduce subdomain to main domain heuristic
            # e.g. www.bbc.co.uk -> bbc.co.uk
            # simpler approach: check if credible substring exists in domain
            for cd in credible_domains:
                if cd.startswith('.'):
                    # exact suffix match for .gov/.edu typed entries
                    if domain.endswith(cd):
                        if cd not in seen_domains:
                            score += 0.5
                            seen_domains.add(cd)
                else:
                    if cd in domain:
                        if cd not in seen_domains:
                            score += 0.5
                            seen_domains.add(cd)
        except Exception:
            continue
    return min(1.0, score)

def detect_ai_image(image_bytes: bytes, device: Optional[str] = None) -> Dict[str, Any]:
    """Run ensemble classifiers + optional AIOrNot API."""
    if not _CLASSIFIERS:
        init_models(device=device)

    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"ensemble_ai_prob": 0.5, "models": [], "verdict": "Uncertain", "error": str(e)}

    probs = []
    model_outputs = []

    # Hugging Face ensemble
    for model, extractor in zip(_CLASSIFIERS, _CLASSIFIER_EXTRACTORS):
        try:
            inputs = extractor(images=pil, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
                p = torch.softmax(logits, dim=-1)[0].cpu().numpy()

            labels = {int(k): v.lower() for k,v in model.config.id2label.items()} if model.config.id2label else {}
            if len(p) == 2 and labels:
                idx_real = [i for i,v in labels.items() if "real" in v]
                idx_fake = [i for i,v in labels.items() if "fake" in v or "ai" in v]
                if idx_real and idx_fake:
                    prob_fake = float(p[idx_fake[0]])
                    prob_real = float(p[idx_real[0]])
                else:
                    prob_fake, prob_real = float(p[1]), float(p[0])
            else:
                prob_fake = float(1.0 - p[np.argmax(p)])
                prob_real = 1 - prob_fake

            probs.append(prob_fake)
            model_outputs.append({
                "model": model.name_or_path,
                "prob_fake": prob_fake,
                "prob_real": prob_real
            })
        except Exception as e:
            model_outputs.append({"model": str(model), "error": str(e)})

    # Ensemble probability
    ensemble_ai_prob = float(np.mean(probs)) if probs else 0.5

    # Try AIOrNot API
    aiornot_prob = _aiornot_api_call(image_bytes)

    # Final combined probability
    if aiornot_prob is not None:
        combined_prob = 0.5 * ensemble_ai_prob + 0.5 * aiornot_prob
    else:
        combined_prob = ensemble_ai_prob

    # Verdict mapping
    if combined_prob > 0.75:
        verdict = "Likely AI-generated"
    elif combined_prob < 0.25:
        verdict = "Likely Real"
    else:
        verdict = "Uncertain"

    return {
        "ensemble_ai_prob": ensemble_ai_prob,
        "aiornot_ai_prob": aiornot_prob,
        "combined_ai_prob": combined_prob,
        "verdict": verdict,
        "models": model_outputs
    }

def _evaluate_single_image(url: str, use_gemini: bool = True) -> Dict[str, Any]:
    """
    Evaluate an image for authenticity using ensemble AI detectors,
    optional AIOrNot API, metadata checks, web evidence, and (optionally) Gemini.
    Returns a score 0–100 (higher = more authentic).
    """
    explanation_parts: List[str] = []

    # ---- Step 1. Fetch & Verify ----
    try:
        image_bytes = robust_get(url)
        pil = Image.open(io.BytesIO(image_bytes))
        pil.verify()
        pil.close()
    except Exception as e:
        return {
            "image_url": url,
            "score": 50,
            "explanation": f"Image fetch/validation failed: {e}",
            "details": {}
        }

    # ---- Step 2. AI detection (ensemble + AIOrNot fallback) ----
    ai_results = detect_ai_image(image_bytes)  # returns ensemble + combined
    ai_prob = float(ai_results.get("combined_ai_prob", 0.5))   # use combined
    real_prob = 1 - ai_prob

    # Explanation for AI prob
    ensemble_prob = ai_results.get("ensemble_ai_prob")
    aiornot_prob = ai_results.get("aiornot_ai_prob")

    if aiornot_prob is not None:
        explanation_parts.append(
            f"Ensemble avg = {ensemble_prob*100:.1f}% AI, "
            f"AIOrNot = {aiornot_prob*100:.1f}% AI → Combined = {ai_prob*100:.1f}% AI."
        )
    else:
        explanation_parts.append(f"Ensemble avg = {ai_prob*100:.1f}% AI (AIOrNot unavailable).")

    if ai_prob < 0.25:
        explanation_parts.append("→ Strongly real.")
    elif ai_prob > 0.75:
        explanation_parts.append("→ Strongly AI-generated.")
    else:
        explanation_parts.append("→ Inconclusive.")

    # ---- Step 3. Metadata analysis ----
    metadata = extract_metadata(image_bytes)
    metadata_factor = 0.0
    if not metadata:
        explanation_parts.append("No EXIF metadata found (neutral).")
    else:
        suspicious = [k for k,v in metadata.items()
                      if any(w in str(v).lower()
                             for w in ["adobe","photoshop","gimp","midjourney","stable","ai"])]
        if suspicious:
            explanation_parts.append(f"⚠ Metadata shows possible editing/AI traces: {suspicious}")
            metadata_factor = -0.15
            ai_prob = min(1.0, ai_prob + 0.1)  # penalize authenticity
        else:
            explanation_parts.append("Metadata present, no suspicious tags.")
            metadata_factor = 0.05

    # ---- Step 4. Web evidence (Google Vision) ----
    web_score, pages = 0.0, []
    if vision_client:
        try:
            gimage = vision.Image(content=image_bytes)
            ann = vision_client.web_detection(image=gimage).web_detection
            pages = [p.url for p in (ann.pages_with_matching_images or []) if p.url]
            if ann.full_matching_images:
                web_score = 0.6
                explanation_parts.append("Exact online matches found → strong credibility.")
            elif ann.visually_similar_images:
                web_score = 0.3
                explanation_parts.append("Visually similar images found online.")
            if pages:
                explanation_parts.append(f"Appears on {len(pages)} sites.")
        except Exception:
            explanation_parts.append("Google Vision lookup failed.")

    # ---- Step 5. Credible sources ----
    source_score = credibility_score(pages)
    if source_score > 0:
        explanation_parts.append(f"Credible domains boost authenticity (+{source_score*100:.0f}%).")

    # ---- Step 6. Weighted authenticity score ----
    authenticity = (
        0.6 * (1 - ai_prob) +
        0.2 * web_score +
        0.15 * source_score +
        metadata_factor
    )
    authenticity = max(0.0, min(1.0, authenticity))
    final_score = int(round(authenticity * 100))

    # Handle ambiguity case
    if 0.45 <= ai_prob <= 0.55 and final_score == 50:
        explanation_parts.append("⚠ Both AI detection and signals inconclusive → Uncertain.")
        final_score = 50

    # ---- Step 7. Gemini reasoning ----
    gemini_text = None
    if use_gemini and HAS_GEMINI:
        try:
            context = (
                f"AI={ai_prob:.2f}, Real={1 - ai_prob:.2f}, "
                f"Web={web_score}, Source={source_score}, Metadata={'yes' if metadata else 'no'}"
            )
            gemini_text = ask_gemini(
                f"Image evidence: {context}. Provide a 1-sentence verdict: Real vs AI-generated vs Uncertain."
            )
            if gemini_text:
                explanation_parts.append("Gemini: " + gemini_text)
        except Exception:
            explanation_parts.append("Gemini lookup failed.")

    # ---- Step 8. Return result ----
    return {
        "image_url": url,
        "score": final_score,
        "explanation": " ".join(explanation_parts),
        "details": {
            "ai_results": ai_results,
            "metadata": metadata,
            "web_score": web_score,
            "source_score": source_score,
            "pages": pages,
            "gemini_text": gemini_text
        }
    }

# ---------- Public API ----------
def detect_fake_image(urls: Union[str, List[str]], use_gemini: bool = True) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Public function:
    - Accepts a single URL or list of URLs.
    - Returns structured results. For list input returns list of dicts in same order.
    """
    if isinstance(urls, str):
        return _evaluate_single_image(urls, use_gemini=use_gemini)
    elif isinstance(urls, list):
        out = []
        for u in urls:
            try:
                out.append(_evaluate_single_image(u, use_gemini=use_gemini))
            except Exception as e:
                out.append({"image_url": u, "score": 50, "explanation": f"Error evaluating image: {e}", "details": {}})
        
        print("\n[detect_fake_image] Input URLs:", urls)

        return out
    else:
        raise ValueError("Input must be a string or a list of strings.")

# ---------- Example usage ----------
if __name__ == "__main__":
    # Simple CLI quick test
    init_models()
