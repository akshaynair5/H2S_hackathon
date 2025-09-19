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
DEFAULT_CLASSIFIER = "facebook/convnext-base-224"   # classification model that you used earlier
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

# HF models - lazy init
_CLASSIFIER = None
_CLASSIFIER_EXTRACTOR = None
_CLIP = None
_CLIP_PROCESSOR = None

def init_models(classifier_name=DEFAULT_CLASSIFIER, clip_name=DEFAULT_CLIP, device: Optional[str]=None):
    """Initialize HF models (call once at process start)."""
    global _CLASSIFIER, _CLASSIFIER_EXTRACTOR, _CLIP, _CLIP_PROCESSOR

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _CLASSIFIER is None:
        _CLASSIFIER_EXTRACTOR = AutoFeatureExtractor.from_pretrained(classifier_name)
        _CLASSIFIER = AutoModelForImageClassification.from_pretrained(classifier_name).to(device)
        _CLASSIFIER.eval()
        logger.info("Loaded classifier %s to %s", classifier_name, device)

    if _CLIP is None:
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(clip_name)
        _CLIP = CLIPModel.from_pretrained(clip_name).to(device)
        _CLIP.eval()
        logger.info("Loaded CLIP %s to %s", clip_name, device)


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

# ---------- AI detection ensemble ----------
def detect_ai_image(image_bytes: bytes, device: Optional[str] = None) -> Dict[str, Any]:
    """
    Ensemble detection:
    - Classifier + CLIP + entropy heuristic
    """
    if _CLASSIFIER is None or _CLIP is None:
        init_models(device=device)

    results = {"classifier_prob_ai": None, "clip_prob_ai": None, "entropy": None}

    try:
        # Ensure image is valid before running models
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.exception("AI detection failed at image decode: %s", e)
        results["combined_ai_prob"] = 0.5
        return results

    try:
        # ---- 1) classifier ----
        inputs = _CLASSIFIER_EXTRACTOR(images=pil, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(_CLASSIFIER.device)
        with torch.no_grad():
            out = _CLASSIFIER(**inputs)
            logits = out.logits
            probs = torch.softmax(logits, dim=1)
            if probs.shape[1] >= 2:
                classifier_ai_prob = float(probs[0][1].item())
            else:
                top_prob = float(probs[0].max().item())
                classifier_ai_prob = max(0.0, 1.0 - top_prob)
        results["classifier_prob_ai"] = classifier_ai_prob

        # ---- 2) CLIP zero-shot ----
        clip_inputs = _CLIP_PROCESSOR(
            text=["a real photograph", "a computer generated image", "an illustration"],
            images=pil,
            return_tensors="pt",
            padding=True
        )
        for k, v in clip_inputs.items():
            if isinstance(v, torch.Tensor):
                clip_inputs[k] = v.to(_CLIP.device)
        with torch.no_grad():
            clip_out = _CLIP(**clip_inputs)
            image_embeds = clip_out.image_embeds
            text_embeds = clip_out.text_embeds
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            sims = (image_embeds @ text_embeds.T).squeeze(0)
            clip_probs = torch.softmax(sims, dim=0).cpu().numpy().tolist()
            clip_ai_prob = float(clip_probs[1])  # index 1 = "computer generated image"
        results["clip_prob_ai"] = clip_ai_prob

        # ---- 3) entropy ----
        ent = image_entropy(pil)
        results["entropy"] = ent

        # ---- combine ----
        norm_entropy = max(0.0, min(1.0, 1.0 - (ent / 8.0)))
        combined = 0.45 * classifier_ai_prob + 0.45 * clip_ai_prob + 0.10 * norm_entropy
        results["combined_ai_prob"] = float(max(0.0, min(1.0, combined)))

    except Exception as e:
        logger.exception("AI detection failed: %s", e)
        results["combined_ai_prob"] = 0.5

    return results

# ---------- Core evaluation ----------
def _evaluate_single_image(url: str, use_gemini: bool = True) -> Dict[str, Any]:
    explanation_parts: List[str] = []
    try:
        image_bytes = robust_get(url)

        # Validate that it’s a real image before proceeding
        try:
            img = Image.open(io.BytesIO(image_bytes))
            img.verify()  # lightweight integrity check
            img.close()
        except Exception as e:
            return {
                "image_url": url,
                "score": 50,
                "explanation": f"Error decoding image: {e}",
                "details": {}
            }

    except Exception as e:
        return {
            "image_url": url,
            "score": 50,
            "explanation": f"Error fetching image: {e}",
            "details": {}
        }
    try:
        image_bytes = robust_get(url)
    except Exception as e:
        return {
            "image_url": url,
            "score": 50,
            "explanation": f"Error fetching image: {e}",
            "details": {}
        }

    # ----------------------------
    # 1) Google Vision web_detection
    # ----------------------------
    pages = []
    web_score = 0.0
    if vision_client is not None:
        try:
            gimage = vision.Image(content=image_bytes)
            annotations = vision_client.web_detection(image=gimage).web_detection
            pages = [p.url for p in annotations.pages_with_matching_images if p.url] if annotations.pages_with_matching_images else []
            exact_matches = annotations.full_matching_images or []
            visually_similar = annotations.visually_similar_images or []

            if exact_matches:
                web_score += 0.45
                explanation_parts.append("Exact matches found online (indicator of authenticity).")
            if visually_similar:
                web_score += 0.25
                explanation_parts.append("Visually similar images appear online.")
            if pages:
                web_score += 0.20
                explanation_parts.append(f"{len(pages)} pages found with matching or similar images.")
            web_score = min(1.0, web_score)
        except Exception as e:
            logger.warning("Vision web_detection failed: %s", e)
            explanation_parts.append("Google Vision web detection failed or is unavailable.")
    else:
        explanation_parts.append("Google Vision client not configured; skipping web checks.")

    # ----------------------------
    # 2) AI-detection ensemble
    # ----------------------------
    ai_results = detect_ai_image(image_bytes)
    ai_prob = float(ai_results.get("combined_ai_prob", 0.5))

    # Check for grey zone (0.45–0.55)
    if 0.45 <= ai_prob <= 0.55:
        explanation_parts.append("AI probability in grey zone — running extra averaging...")
        probs = [ai_prob]
        for seed in [101, 202, 303]:
            set_seed(seed)  # you define set_seed as shown earlier
            extra_results = detect_ai_image(image_bytes)
            probs.append(float(extra_results.get("combined_ai_prob", 0.5)))
        ai_prob = round(sum(probs) / len(probs), 3)  # average 3 extra + original

    # Round to 2 decimals for stability
    ai_prob = round(ai_prob, 2)
    explanation_parts.append(f"AI-generated probability (ensemble): {ai_prob*100:.1f}%")

    # ----------------------------
    # 3) EXIF/metadata signals
    # ----------------------------
    metadata = extract_metadata(image_bytes)
    if not metadata:
        explanation_parts.append("No EXIF metadata found — could be AI-generated, edited, or stripped.")
    else:
        edit_signs = []
        for v in metadata.values():
            try:
                s = str(v).lower()
                if any(x in s for x in ["photoshop", "adobe", "gimp", "luminar", "ai"]):
                    edit_signs.append(str(v))
            except Exception:
                continue
        if edit_signs:
            explanation_parts.append("Metadata indicates possible editing or software presence.")
        else:
            explanation_parts.append("Metadata present, no obvious editing tool tags found.")

    # ----------------------------
    # 4) Source credibility
    # ----------------------------
    source_score = credibility_score(pages)
    if source_score > 0:
        explanation_parts.append(f"Found on credible sources (score: {source_score*100:.0f}%).")

    # ----------------------------
    # 5) Gemini opinion (optional)
    # ----------------------------
    gemini_text = None
    if use_gemini and HAS_GEMINI:
        try:
            prompt = (
                f"Image URL: {url}\n"
                f"Evidence:\n"
                f"- Web matches: {len(pages)} pages\n"
                f"- Google web_score: {web_score:.2f}\n"
                f"- AI ensemble prob: {ai_prob:.3f}\n"
                f"- Source credibility score: {source_score:.2f}\n"
                f"- Metadata keys: {', '.join(list(metadata.keys())[:10]) if metadata else 'None'}\n\n"
                "Based on the above evidence, give a concise judgement: is this likely AI-generated/edited or likely authentic? "
                "Respond in one or two sentences and mention which evidence is most important."
            )
            gemini_text = ask_gemini(prompt)
            if gemini_text and "Error using Gemini" not in gemini_text:
                explanation_parts.append("Gemini opinion: " + gemini_text)
            else:
                explanation_parts.append("Gemini was called but returned an error or fallback.")
        except Exception as e:
            logger.exception("Error calling Gemini: %s", e)
            explanation_parts.append("Gemini check failed.")

    # ----------------------------
    # 6) Final scoring
    # ----------------------------
    metadata_presence = 1.0 if metadata else 0.0

    # Map AI probability into a "realness score" instead of raw (1 - ai_prob)
    if ai_prob < 0.25:
        ai_score = 1.0   # very likely real
    elif ai_prob < 0.50:
        ai_score = 0.7   # leaning real
    elif ai_prob < 0.75:
        ai_score = 0.3   # leaning fake
    else:
        ai_score = 0.0   # very likely fake

    # Weighted authenticity score (tuned)
    authenticity = (
        0.6 * web_score +        # web evidence has high impact
        0.3 * ai_score +         # AI detector influences but less dominant
        0.08 * source_score +    # credibility helps if available
        0.02 * metadata_presence # metadata is weak evidence
    )

    # Clamp if AI detector is very confident it's fake
    if ai_prob >= 0.85:
        authenticity = min(authenticity, 0.2)  # cap at 20% authenticity
    elif ai_prob >= 0.70:
        authenticity = min(authenticity, 0.4)  # cap at 40% authenticity

    # Bias upwards if strong web evidence + low AI probability
    if web_score > 0.8 and ai_prob < 0.3:
        authenticity = max(authenticity, 0.75)

    final_score = int(round(max(0.0, min(1.0, authenticity)) * 100))

    details = {
        "web_score": web_score,
        "ai_results": ai_results,
        "metadata": metadata,
        "source_score": source_score,
        "matched_pages": pages,
        "gemini_text": gemini_text
    }

    logger.info(
        "Scores: ai_prob=%.3f, web_score=%.2f, source=%.2f, meta=%d, final=%.1f%%",
        ai_prob, web_score, source_score, metadata_presence, final_score
    )

    print("\n[_evaluate_single_image] --- Evaluating:", url)

    print("[_evaluate_single_image] Web score:", web_score)
    print("[_evaluate_single_image] AI ensemble prob:", ai_prob)
    print("[_evaluate_single_image] Metadata keys:", list(metadata.keys()))
    print("[_evaluate_single_image] Source credibility score:", source_score)

    print("[_evaluate_single_image] Weighted authenticity (before clamp):", authenticity)
    print("[_evaluate_single_image] Final authenticity score:", final_score)


    return {
        "image_url": url,
        "score": final_score,
        "explanation": " ".join(explanation_parts),
        "details": details
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
