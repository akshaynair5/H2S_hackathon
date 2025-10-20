import os
import re
import html
import time
import json
import requests
from functools import lru_cache
from typing import List, Dict, Any
from urllib.parse import urlparse
from dotenv import load_dotenv
from flask import Flask
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from datetime import datetime
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------- config & init -------------
load_dotenv()
app = Flask(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
GEM_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- Vertex AI config ----------------
PROJECT_ID = "804712050799"
ENDPOINT_ID = "5229109076423606272"
REGION = "us-central1"
PREDICT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"

def get_access_token():
    creds = service_account.Credentials.from_service_account_file(
        "gen-ai-h2s-project-562ce7c50fcf-vertex-ai.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds.token

# ---------------- Embeddings ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=8192)
def get_embedding(text: str):
    return embedder.encode(text, convert_to_tensor=True)

# ---------------- Constants ----------------
CREDIBLE_DOMAINS_SCORE = {
    "bbc.com": 1.0, "reuters.com": 1.0, "nytimes.com": 0.95, "cnn.com": 0.9,
    "theguardian.com": 0.9, "apnews.com": 0.95, "factcheck.org": 1.0,
    "snopes.com": 1.0, "politifact.com": 1.0, "washingtonpost.com": 0.9
}
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CX_ID = os.getenv("GOOGLE_SEARCH_CX")
CLAIM_MIN_LEN = 30
MAX_SEARCH_RESULTS = 5
EMB_SIM_THRESHOLD = 0.40

# ---------------- Utilities ----------------
def retry(func, tries=3, delay=1.0):
    def wrapper(*args, **kwargs):
        for i in range(tries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == tries - 1:
                    raise e
                time.sleep(delay * (2 ** i))
    return wrapper

def simple_sentence_split(text: str) -> List[str]:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s.strip() for s in sents if len(s.strip()) >= CLAIM_MIN_LEN]
    return sents[:3] or [text[:500]]

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def domain_from_url(url: str) -> str:
    m = re.search(r"https?://(www\.)?([^/]+)", url)
    return m.group(2).lower() if m else ""

def domain_score_for_url(url: str) -> float:
    d = domain_from_url(url)
    for dom, score in CREDIBLE_DOMAINS_SCORE.items():
        if dom in d:
            return score
    return 0.0

# ---------------- Gemini helper ----------------
@retry
def ask_gemini_structured(prompt: str) -> Dict[str, Any]:
    try:
        resp = GEM_MODEL.generate_content(prompt)
        text = getattr(resp, "text", str(resp)).strip()
        try:
            parsed = json.loads(text)
            return {"parsed": parsed, "raw_text": text}
        except json.JSONDecodeError:
            match = re.search(r"(\{(?:.|\s)*\})", text)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                    return {"parsed": parsed, "raw_text": text}
                except Exception:
                    pass
            return {"raw_text": text}
    except Exception as e:
        return {"error": str(e)}

# ---------------- Metadata extraction ----------------
def extract_metadata_with_gemini(text: str) -> dict:
    try:
        prompt = f""" Extract structured information from the following news article text. Return only valid JSON with keys: title, text, author, date, source, category. Rules: - Infer 'title' and 'category' from the text. - If 'author' or 'source' is not present, use "Unknown". - If 'date' is missing, use today's date in YYYY-MM-DD. Text: {text} """
        gem_resp = ask_gemini_structured(prompt)
        parsed = gem_resp.get("parsed", {})
        return {
            "title": parsed.get("title") or "Inferred",
            "text": parsed.get("text") or text[:4000],
            "author": parsed.get("author") or "Unknown",
            "date": parsed.get("date") or datetime.now().strftime("%Y-%m-%d"),
            "source": parsed.get("source") or "Unknown",
            "category": parsed.get("category") or "Inferred"
        }
    except Exception:
        return {
            "title": "Inferred",
            "text": text[:4000],
            "author": "Unknown",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "source": "Unknown",
            "category": "Inferred"
        }

# ---------------- Vertex AI ----------------
def predict_with_vertex_ai(metadata: dict) -> dict:
    headers = {"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json"}
    response = requests.post(PREDICT_URL, headers=headers, json={"instances": [metadata]}, timeout=15)
    print(response.text)
    if response.status_code != 200:
        raise Exception(f"Vertex AI Error {response.status_code}: {response.text}")
    return response.json()

def extract_vertex_scores(vertex_result: dict) -> dict:
    """Map Vertex AI output to Real/Fake/Misleading dict."""
    try:
        preds = vertex_result.get("predictions", [{}])[0]
        classes = preds.get("classes", [])
        scores = preds.get("scores", [])
        score_map = {cls.capitalize(): float(score) for cls, score in zip(classes, scores)}
        # Ensure all keys exist
        return {"Real": score_map.get("Real", 0.0),
                "Fake": score_map.get("Fake", 0.0),
                "Misleading": score_map.get("Misleading", 0.0)}
    except Exception:
        return {"Real": 0.7, "Fake": 0.3, "Misleading": 0.0}


def adjusted_ensemble(gem_pred: str, gem_conf: int, vertex_scores: dict, threshold=0.165) -> (str, int):
    """
    Combine Gemini and Vertex AI predictions with soft thresholding.
    Only label as Fake if vertex C_fake - C_real > threshold.
    """
    C_real = vertex_scores.get("Real", 0.7)
    C_fake = vertex_scores.get("Fake", 0.3)

    if C_fake - C_real > threshold:
        vertex_label = "Fake"
        vertex_conf = int((C_fake - C_real) * 100)
    else:
        vertex_label = "Real"
        vertex_conf = int(C_real * 100)

    final_conf = int(0.6 * gem_conf + 0.4 * vertex_conf)

    # Decide final prediction
    if vertex_label == "Fake" and gem_pred == "Fake":
        final_pred = "Fake"
    elif vertex_label == "Real" and gem_pred == "Real":
        final_pred = "Real"
    else:
        final_pred = gem_pred if gem_conf >= vertex_conf else vertex_label

    return final_pred, final_conf

def load_credible_domains() -> List[str]:
    static = ["reuters.com", "bbc.com", "apnews.com", "cnn.com", "nytimes.com",
              "theguardian.com", "npr.org", "aljazeera.com", "bloomberg.com"]
    return static

def corroborate_all_with_google(claims: List[str]) -> Dict[str, Any]:
    """Perform a Google search for each claim and return structured evidence."""
    evidences = []
    CREDIBLE_DOMAINS = load_credible_domains()

    for claim in claims:
        site_filter = " OR ".join([f"site:{d}" for d in CREDIBLE_DOMAINS])
        query = f'"{claim}" {site_filter} (news OR report OR article)'
        try:
            resp = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_KEY,
                    "cx": CX_ID,
                    "q": query,
                    "num": MAX_SEARCH_RESULTS,
                },
                timeout=8
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
        except Exception:
            items = []

        claim_emb = get_embedding(claim)
        claim_evidences = []
        unique_domains = set()

        for it in items:
            link = it.get("link", "")
            domain = urlparse(link).netloc
            snippet = html.unescape(it.get("snippet", ""))[:400]
            NEWS_KEYWORDS = [
                # Common reporting verbs
                "report", "reports", "reported",
                "say", "says", "said",
                "announce", "announces", "announced",
                "state", "states", "stated",
                "claim", "claims", "claimed",
                "confirm", "confirms", "confirmed",
                "according to", "according",
                "told", "revealed", "issued", "released",
                "declared", "denied", "explained", "described",
                "statement from", "spokesperson", "press release",
                "officials", "authorities", "investigation", "evidence"
            ]

            if not any(d in domain for d in CREDIBLE_DOMAINS):
                continue
            if not any(k in snippet.lower() for k in NEWS_KEYWORDS):
                continue

            snippet_emb = get_embedding(snippet)
            similarity = float(util.cos_sim(claim_emb, snippet_emb))
            if similarity < EMB_SIM_THRESHOLD:
                continue

            score = domain_score_for_url(link)
            evidence_score = round(clamp01(0.7 * similarity + 0.3 * score), 3)

            claim_evidences.append({
                "title": it.get("title", ""),
                "link": link,
                "snippet": snippet,
                "domain_score": score,
                "similarity": round(similarity, 3),
                "evidence_score": evidence_score
            })
            unique_domains.add(domain)

        top_evidences = sorted(claim_evidences, key=lambda x: x["evidence_score"], reverse=True)[:3]
        evidences.extend(top_evidences)

    status = "corroborated" if len(set([urlparse(e["link"]).netloc for e in evidences])) >= 2 else \
             "weak" if evidences else "no_results"
    return {"status": status, "evidences": evidences}


# ---------------- Gemini prompt ----------------
def assemble_gemini_prompt_structured(claim: str, evidences: List[Dict[str, Any]], status: str) -> str:
    return f"""
You are an AI fact-checking assistant.

Input claim: \"\"\"{claim}\"\"\"
Corroboration status: {status}
Evidence snippets: {json.dumps(evidences[:5], ensure_ascii=False)}

Instructions:
Return a strict JSON object with the following keys:

- prediction: "Real", "Fake", or "Misleading"
- confidence: integer 0–100
- explanation: 1-2 short sentences max, clear and readable for non-experts.
  - Break multiple points with "|"
  - Start with general plausibility, then note supporting or contradicting evidence
  - Avoid long paragraphs or overly technical phrasing
- evidence: include only the 1-3 most relevant snippets
  - Each snippet max 50 words
  - Specify if it SUPPORTS or CONTRADICTS the claim
- human_summary (optional but recommended): 1 short sentence explaining the claim in simple, everyday language

Example output format:
{{
  "prediction": "Real",
  "confidence": 85,
  "explanation": "The groups mentioned exist and gatherings like described are common | The claim is plausible and consistent with typical public events",
  "evidence": [
    {{"source":"BBC", "link":"https://...", "snippet":"People often gather in cities for protests with music, signs, and flags", "support":"Supports"}}
  ],
  "human_summary": "The claim is plausible and reflects common real-world events."
}}

Return only valid JSON.
"""

def detect_fake_text(text: str) -> dict:
    start_total = time.time()

    # Fix spacing between sentences
    text = re.sub(r"(?<=[a-zA-Z])\.(?=[A-Z])", ". ", text)
    
    # Extract metadata via Gemini
    metadata = extract_metadata_with_gemini(text)
    
    # Vertex AI prediction
    vertex_result = predict_with_vertex_ai(metadata)
    vertex_scores = extract_vertex_scores(vertex_result)

    # Split into claims
    claims = simple_sentence_split(metadata["text"])
    
    # Corroboration from Google
    corroboration_data = corroborate_all_with_google(claims)

    results, overall_scores, preds, explanations = [], [], [], []

    # Parallel Gemini checks for claims
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(
                lambda c: (
                    c,
                    ask_gemini_structured(assemble_gemini_prompt_structured(
                        c, corroboration_data["evidences"], corroboration_data["status"]
                    ))
                ),
                c
            ): c for c in claims
        }

        for future in as_completed(futures):
            claim, gem_resp = future.result()
            parsed = gem_resp.get("parsed", {})
            gem_pred = parsed.get("prediction", "Unknown")
            gem_conf = int(parsed.get("confidence", 70))
            explanation = parsed.get("explanation", "Based on available evidence.")

            final_pred, final_conf = adjusted_ensemble(gem_pred, gem_conf, vertex_scores)

            results.append({
                "claim_text": claim,
                "gemini": parsed,
                "vertex_ai": vertex_result,
                "corroboration": corroboration_data,
                "ensemble": {
                    "final_prediction": final_pred,
                    "final_confidence": final_conf,
                    "combined_score": round(final_conf / 100, 3)
                },
                "explanation": explanation,
                "evidence": corroboration_data["evidences"]
            })

            overall_scores.append(final_conf)
            preds.append(final_pred)
            explanations.append(explanation)

    # -------- Combine results into simplified article-level summary --------
    overall_conf = int(sum(overall_scores) / len(overall_scores)) if overall_scores else 0
    overall_label = max(set(preds), key=preds.count) if preds else "Unknown"
    combined_explanation = " | ".join(explanations[:3]) if explanations else "No detailed explanation available."

    result_summary = {
        "score": overall_conf,
        "prediction": overall_label,
        "explanation": combined_explanation,
    }
    return {
        "summary": result_summary,
        "runtime": round(time.time() - start_total, 2),
        "claims_checked": len(results),
        "raw_details": results
    }