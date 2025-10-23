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
import firebase_admin
from firebase_admin import credentials, firestore
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from vectorDb import (
    embed_text,
    store_feedback,
    search_feedback_semantic,
    pc, INDEX_NAME
)
from migration import generate_embedding
from datetime import datetime, timedelta
import aiohttp
import asyncio

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


# ---------------- GCP credentials (for Vertex AI) ----------------
def get_gcp_credentials():
    """
    Load and refresh service account credentials for GCP (usable by Vertex AI).
    """
    creds = service_account.Credentials.from_service_account_file(
        "gen-ai-h2s-project-562ce7c50fcf-vertex-ai.json",
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds

def get_access_token():
    creds = get_gcp_credentials()
    return creds.token

# ---------------- Firestore init using Firebase Admin ----------------
SERVICE_ACCOUNT_PATH = "gen-ai-h2s-project-562ce7c50fcf-vertex-ai.json"

# Initialize Firebase Admin app
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)

# Firestore client
db = firestore.client(database_id='genai')


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

def get_trusted_score(domain: str) -> float:
    """Return avg_score of domain or 0 if not found"""
    domain = domain.lower().strip()
    doc = db.collection("news_sources").document(domain).get()
    if doc.exists:
        return doc.to_dict().get("avg_score", 0.0)
    return 0.0

def domain_score_for_url(url: str) -> float:
    """
    Return the score for a domain.
    Pulls from Firestore dynamically.
    """
    d = domain_from_url(url)
    score = get_trusted_score(d)
    return score

# Basic TTL cache wrapper for Firestore domain loading
_CACHE_TTL = 300  # seconds (5 min)
_last_cache_time = 0
_cached_domains = []

def invalidate_domain_cache():
    """Force cache invalidation (e.g. after updates)."""
    global _last_cache_time
    _last_cache_time = 0

def load_credible_domains_cached() -> List[str]:
    """Loads credible domains from Firestore with 5-min TTL cache."""
    global _cached_domains, _last_cache_time
    now = time.time()
    if not _cached_domains or (now - _last_cache_time) > _CACHE_TTL:
        _cached_domains = load_credible_domains()  # Your original function
        _last_cache_time = now
    return _cached_domains

def add_or_update_trusted_sources_batch(domain_scores: Dict[str, float]):
    """
    Batch update Firestore for multiple domains at once.
    domain_scores: {domain: new_score, ...}
    """
    batch = db.batch()
    for domain, new_score in domain_scores.items():
        domain = domain.lower().strip()
        doc_ref = db.collection("news_sources").document(domain)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            current_avg = data.get("avg_score", 0.0)
            num_votes = data.get("num_votes", 0)
            updated_avg = round((current_avg * num_votes + new_score) / (num_votes + 1), 3)
            batch.update(doc_ref, {
                "avg_score": updated_avg,
                "num_votes": num_votes + 1,
                "last_updated": datetime.utcnow()
            })
        else:
            batch.set(doc_ref, {
                "avg_score": round(new_score, 3),
                "num_votes": 1,
                "last_updated": datetime.utcnow()
            })
    batch.commit()
    invalidate_domain_cache()

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
@retry
def predict_with_vertex_ai(metadata: dict) -> dict:
    headers = {"Authorization": f"Bearer {get_access_token()}", "Content-Type": "application/json"}
    response = requests.post(PREDICT_URL, headers=headers, json={"instances": [metadata]}, timeout=15)
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

def clear_cache_for_text(text: str) -> bool:
    """
    Deletes cached Pinecone entries semantically matching the input text.
    Used to force a re-analysis if misinformation data has been updated.
    Returns True if a cache entry was found and deleted, else False.
    """
    try:
        # Generate embedding for similarity search
        query_emb = embed_text(text)
        if not query_emb:
            print("[Cache Clear] Could not generate embedding.")
            return False

        # Query top match from Pinecone
        index = pc.Index(INDEX_NAME)
        search = index.query(vector=query_emb, top_k=1, include_metadata=True)

        if not search.matches:
            print("[Cache Clear] No similar cache entry found.")
            return False

        match_id = search.matches[0].id
        index.delete(ids=[match_id])
        print(f"[Cache Clear] Deleted Pinecone entry: {match_id}")
        return True

    except Exception as e:
        print(f"[Cache Clear Error] {e}")
        return False


def adjusted_ensemble(gem_pred: str, gem_conf: int, vertex_scores: dict, threshold=0.165) -> (str, int):
    """
    Combine Gemini and Vertex AI predictions with soft thresholding.
    Handles Real, Fake, Misleading properly.
    """
    C_real = vertex_scores.get("Real", 0.7)
    C_fake = vertex_scores.get("Fake", 0.3)
    C_mis = vertex_scores.get("Misleading", 0.0)

    # Decide vertex label
    max_vertex = max(("Real", C_real), ("Fake", C_fake), ("Misleading", C_mis), key=lambda x: x[1])
    vertex_label, vertex_conf = max_vertex[0], int(max_vertex[1] * 100)

    # Weighted final confidence
    final_conf = int(0.6 * gem_conf + 0.4 * vertex_conf)

    # Ensemble decision rules
    if gem_pred == vertex_label:
        final_pred = gem_pred
    elif "Misleading" in (gem_pred, vertex_label):
        # Prefer Misleading if either predicts it and confidence > 50
        if final_conf > 50:
            final_pred = "Misleading"
        else:
            final_pred = gem_pred if gem_conf >= vertex_conf else vertex_label
    else:
        final_pred = gem_pred if gem_conf >= vertex_conf else vertex_label

    return final_pred, final_conf

def load_credible_domains() -> List[str]:
    """
    Return the list of currently trusted domains from Firestore.
    Only include domains with at least 1 vote.
    """
    docs = db.collection("news_sources").where("num_votes", ">=", 1).stream()
    domains = [doc.id for doc in docs]
    # Fallback to static domains if Firestore empty
    if not domains:
        domains = ["reuters.com", "bbc.com", "apnews.com", "cnn.com", "nytimes.com",
                   "theguardian.com", "npr.org", "aljazeera.com", "bloomberg.com"]
    return domains

def get_domain_bonus(domain: str) -> float:
    """
    Return a very small bonus (2%) if the domain is highly credible (avg_score >= 0.9)
    and has more than 100 votes. Returns 0 otherwise.
    """
    domain = domain.lower().strip().rstrip("/")  # sanitize
    if not domain or domain in ["unknown", ""]:
        return 0.0
    try:
        doc_ref = db.collection("news_sources").document(domain)
        doc = doc_ref.get()
        if doc.exists:
            data = doc.to_dict()
            avg_score = data.get("avg_score", 0.0)
            num_votes = data.get("num_votes", 0)
            if avg_score >= 0.9 and num_votes > 100:
                return 0.02  # very small bonus
        return 0.0
    except Exception as e:
        print(f"⚠️ Domain bonus fetch failed for {domain}: {e}")
        return 0.0

async def fetch_google(session, query):
    try:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_KEY, "cx": CX_ID, "q": query, "num": MAX_SEARCH_RESULTS},
            timeout=10
        ) as resp:
            data = await resp.json()
            return data.get("items", [])
    except Exception:
        return []

async def corroborate_all_with_google_async(claims: List[str]) -> Dict[str, Any]:
    evidences = []
    CREDIBLE_DOMAINS = load_credible_domains_cached()
    domain_updates: Dict[str, float] = {}

    NEWS_KEYWORDS = [
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

    async with aiohttp.ClientSession() as session:
        tasks = []
        for claim in claims:
            site_filter = " OR ".join([f"site:{d}" for d in CREDIBLE_DOMAINS])
            query = f'"{claim}" {site_filter} (news OR report OR article)'
            tasks.append(fetch_google(session, query))
        
        results = await asyncio.gather(*tasks)

    for claim, items in zip(claims, results):
        # original processing logic below remains the same
        claim_emb = get_embedding(claim)
        claim_evidences = []

        for it in items:
            link = it.get("link", "")
            domain = urlparse(link).netloc.lower()
            snippet = html.unescape(it.get("snippet", ""))[:400]

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
            domain_updates[domain] = evidence_score

        top_evidences = sorted(claim_evidences, key=lambda x: x["evidence_score"], reverse=True)[:3]
        evidences.extend(top_evidences)

    if domain_updates:
        try:
            add_or_update_trusted_sources_batch(domain_updates)
        except Exception as e:
            print(f"⚠️ Firestore batch update failed: {e}")

    status = "corroborated" if len(set([urlparse(e["link"]).netloc for e in evidences])) >= 2 else \
             "weak" if evidences else "no_results"

    return {"status": status, "evidences": evidences}

# ---------------- Gemini prompt ----------------

def extract_local_context(claim: str, full_text: str, window: int = 2) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', full_text.strip())
    best_idx = 0
    for i, sent in enumerate(sentences):
        if claim.strip()[:min(30, len(claim))].lower() in sent.lower():
            best_idx = i
            break
    start = max(0, best_idx - window)
    end = min(len(sentences), best_idx + window + 1)
    return " ".join(sentences[start:end])[:1200]

def assemble_gemini_prompt_structured(claim: str, evidences: List[Dict[str, Any]], status: str, full_text: str = "") -> str:
    today_str = datetime.now().strftime("%B %d, %Y")
    local_context = extract_local_context(claim, full_text) if full_text else ""

    context_part = f"The claim appears in the following context:\n\"\"\"{local_context}\"\"\"\n\n" if local_context else ""

    return f"""
You are an AI fact-checking assistant.

{context_part}
Input claim: \"\"\"{claim}\"\"\"
Corroboration status: {status}
Evidence snippets: {json.dumps(evidences[:5], ensure_ascii=False)}
Today's date: {today_str}

Instructions:
- Focus on the local context around the claim to preserve meaning.
- Use evidence snippets to verify factual accuracy.
- Evaluate the claim considering today's date ({today_str}).
- Return a strict JSON object with keys:

- prediction: "Real", "Fake", or "Misleading"
- confidence: integer 0–100
- explanation: 1–2 short, plain sentences | Use "|" to separate reasoning steps
- evidence: 1–3 key snippets (≤50 words each) with a `support` field
- human_summary (optional): plain summary of the claim

Example output:
{{
  "prediction": "Real",
  "confidence": 85,
  "explanation": "The claim matches current verified reports | Context indicates it refers to an ongoing event",
  "evidence": [
    {{"source":"BBC", "link":"https://...", "snippet":"BBC confirms the described protests occurred in Delhi.", "support":"Supports"}}
  ],
  "human_summary": "The claim about protests in Delhi is accurate."
}}

Return only valid JSON.
"""
    
def detect_fake_text(text: str) -> dict:
    """
    Detects misinformation using Vertex AI, Gemini, corroboration,
    integrates vector DB (Pinecone) caching + Firestore embeddings,
    with robust ensemble, domain credibility bonus, and batching.
    """

    start_total = time.time()

    # --- STEP 0: Try retrieving cached result from Pinecone ---
    try:
        cached = search_feedback_semantic(text)
        if cached and "score" in cached:
            if "timestamp" in cached:
                try:
                    cached_time = datetime.fromisoformat(cached["timestamp"])
                    if datetime.utcnow() - cached_time > timedelta(days=7):
                        print("[Cache Expired] Cached result older than 7 days, re-analyzing.")
                        if "clear_cache_for_text" in globals():
                            clear_cache_for_text(text)
                except Exception as e:
                    print(f"[Cache Expiry Check Failed] {e}")

            print("[Cache Hit] Returning result from Pinecone vector DB.")
            return {
                "summary": {
                    "score": cached.get("score", 0),
                    "prediction": cached.get("prediction", "Unknown"),
                    "explanation": cached.get("explanation", "Retrieved from cache."),
                    "source": "cache"
                },
                "runtime": 0,
                "claims_checked": 0,
                "raw_details": []
            }
    except Exception as e:
        print(f"[Cache Lookup Failed] {e}")

    # --- STEP 1: Clean + preprocess text ---
    text = re.sub(r"(?<=[a-zA-Z])\.(?=[A-Z])", ". ", text)

    # --- STEP 2: Metadata extraction (Gemini) ---
    metadata = extract_metadata_with_gemini(text)

    # --- STEP 3: Vertex AI prediction ---
    vertex_result = predict_with_vertex_ai(metadata)
    vertex_scores = extract_vertex_scores(vertex_result)

    # --- STEP 4: Split into claims ---
    claims = simple_sentence_split(metadata["text"])

    # --- STEP 5: Corroboration from Google ---
    corroboration_data = asyncio.run(corroborate_all_with_google_async(claims))

    # --- STEP 6: Parallel Gemini checks for claims ---
    def process_claim(claim: str):
        gem_resp = ask_gemini_structured(
            assemble_gemini_prompt_structured(
                claim,
                corroboration_data["evidences"],
                corroboration_data["status"],
                full_text=metadata["text"]
            )
        )
        parsed = gem_resp.get("parsed", {})
        gem_pred = parsed.get("prediction", "Unknown")
        gem_conf = int(parsed.get("confidence", 70))
        explanation = parsed.get("explanation", "Based on available evidence.")

        final_pred, final_conf = adjusted_ensemble(gem_pred, gem_conf, vertex_scores)

        return {
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
        }, final_conf, final_pred, explanation

    results, overall_scores, preds, explanations = [], [], [], []

    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_claim = {executor.submit(process_claim, claim): claim for claim in claims}
        for future in as_completed(future_to_claim):
            res, final_conf, final_pred, explanation = future.result()
            results.append(res)
            overall_scores.append(final_conf)
            preds.append(final_pred)
            explanations.append(explanation)

    # --- STEP 7: Article-level summary ---
    overall_conf = int(sum(overall_scores) / len(overall_scores)) if overall_scores else 0
    overall_label = max(set(preds), key=preds.count) if preds else "Unknown"
    combined_explanation = " | ".join(explanations[:3]) if explanations else "No detailed explanation available."

    # --- STEP 8: Domain credibility bonus ---
    source_domain = domain_from_url(metadata.get("source", ""))
    domain_bonus = get_domain_bonus(source_domain)
    if domain_bonus > 0:
        bonus_points = int(overall_conf * domain_bonus)
        overall_conf = min(100, overall_conf + bonus_points)
        combined_explanation += f" | Small credibility bonus applied due to trusted source ({source_domain})"

    result_summary = {
        "score": overall_conf,
        "prediction": overall_label,
        "explanation": combined_explanation,
    }

    # --- STEP 9: Store embedding + verification in Firestore (deduplicated) ---
    try:
        embedding = generate_embedding(text)
        if db:
            doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
            db.collection("articles").document(doc_id).set({
                "text": text,
                "embedding": embedding,
                "verified": True,
                "score": overall_conf,
                "label": overall_label,
                "timestamp": firestore.SERVER_TIMESTAMP
            })
    except Exception as e:
        print(f"[Firestore Embedding Storage Failed] {e}")

    # --- STEP 10: Cache final result in Pinecone ---
    try:
        store_feedback(
            text=text,
            explanation=combined_explanation,
            sources=[],
            user_fingerprint="system",
            score=overall_conf / 100,
            prediction=overall_label,
            verified=True
        )
    except Exception as e:
        print(f"[Pinecone Cache Store Failed] {e}")

    return {
        "summary": result_summary,
        "runtime": round(time.time() - start_total, 2),
        "claims_checked": len(results),
        "raw_details": results
    }
