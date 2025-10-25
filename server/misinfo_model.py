# misinfo_model.py (Integrated: New approach + Google Fact Check + Ensemble to LLM)

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
from google.cloud import firestore
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from database import db
from vectorDb import (
    embed_text,
    store_feedback,
    search_feedback_semantic,
    pc, INDEX_NAME
)
from datetime import datetime, timedelta
import aiohttp
import asyncio



# ------------- config & init -------------
load_dotenv()




GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY in environment variables")

genai.configure(api_key=GEMINI_API_KEY)
GEM_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- Vertex AI config ----------------
PROJECT_ID = "804712050799"
ENDPOINT_ID = "3457435204262559744"
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
FACT_CHECK_API_KEY = os.getenv("GEMINI_API_KEY")  # Reuse if needed, or set separate
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
        _cached_domains = load_credible_domains()
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

# ---------------- Google Fact Check API (From old) ----------------
def query_google_fact_check_api(text: str, max_results=5) -> dict:
    """
    Query Google Fact Check Tools API for existing fact-checks.
    """
    try:
        # Extract clean claim for fact-check search
        refine_prompt = f"""
        Extract the core factual claim from this text in 5-15 words.
        Focus on the verifiable statement, remove opinions and context.
        Return ONLY the claim, nothing else.
        
        Text: "{text[:200]}"
        """
        refined_claim = ask_gemini_structured(refine_prompt).get("raw_text", "").strip('"').strip()
        print(f"[Fact Check API] Query: {refined_claim}")
        
        # Call Google Fact Check Tools API
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "key": FACT_CHECK_API_KEY,
            "query": refined_claim,
            "pageSize": max_results,
            "languageCode": "en"
        }
        
        resp = requests.get(url, params=params, timeout=10)
        
        if resp.status_code != 200:
            print(f"[Fact Check API] Error: {resp.status_code}")
            return {
                "status": "api_error",
                "fact_checks": [],
                "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0}
            }
        
        data = resp.json()
        claims = data.get("claims", [])
        
        if not claims:
            print("[Fact Check API] No existing fact-checks found")
            return {
                "status": "no_fact_checks",
                "fact_checks": [],
                "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0}
            }
        
        # Process fact-check results
        fact_checks = []
        ratings_counter = {"false": 0, "true": 0, "mixed": 0, "unknown": 0}
        
        for claim in claims[:max_results]:
            claim_text = claim.get("text", "")
            claim_review = claim.get("claimReview", [])
            
            if claim_review:
                for review in claim_review[:2]:
                    publisher = review.get("publisher", {}).get("name", "Unknown")
                    url = review.get("url", "")
                    rating = review.get("textualRating", "").lower()
                    title = review.get("title", "")
                    
                    # Categorize rating
                    if any(word in rating for word in ["false", "fake", "incorrect", "misleading", "pants on fire"]):
                        rating_category = "false"
                        ratings_counter["false"] += 1
                    elif any(word in rating for word in ["true", "correct", "accurate", "verified"]):
                        rating_category = "true"
                        ratings_counter["true"] += 1
                    elif any(word in rating for word in ["mixed", "partially", "half", "mostly"]):
                        rating_category = "mixed"
                        ratings_counter["mixed"] += 1
                    else:
                        rating_category = "unknown"
                        ratings_counter["unknown"] += 1
                    
                    fact_checks.append({
                        "claim": claim_text[:150],
                        "publisher": publisher,
                        "rating": rating,
                        "rating_category": rating_category,
                        "title": title,
                        "url": url
                    })
                    
                    print(f"[Fact Check] {publisher}: {rating} ({rating_category})")
        
        # Determine overall fact-check consensus
        total = len(fact_checks)
        false_ratio = ratings_counter["false"] / max(total, 1)
        true_ratio = ratings_counter["true"] / max(total, 1)
        
        if false_ratio >= 0.6:
            status = "predominantly_false"
        elif true_ratio >= 0.6:
            status = "predominantly_true"
        elif ratings_counter["mixed"] >= 2:
            status = "mixed_ratings"
        else:
            status = "inconclusive"
        
        print(f"[Fact Check API] Status: {status} (False: {ratings_counter['false']}, True: {ratings_counter['true']}, Mixed: {ratings_counter['mixed']})")
        
        return {
            "status": status,
            "fact_checks": fact_checks,
            "summary": {
                "total": total,
                "false_count": ratings_counter["false"],
                "true_count": ratings_counter["true"],
                "mixed_count": ratings_counter["mixed"]
            }
        }
        
    except Exception as e:
        print(f"[Fact Check API] Exception: {str(e)}")
        return {
            "status": "error",
            "fact_checks": [],
            "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0},
            "error": str(e)
        }

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

    print(f"[Vertex AI] Response: {response.text}")
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


def adjusted_ensemble(gem_pred: str, gem_conf: int, vertex_scores: dict, fact_check_status: str, threshold=0.165) -> (str, int):
    """
    Combine Gemini, Vertex AI, and Fact Check with soft thresholding.
    Fact Check overrides if strong signal.
    """
    C_real = vertex_scores.get("Real", 0.7)
    C_fake = vertex_scores.get("Fake", 0.3)
    C_mis = vertex_scores.get("Misleading", 0.0)

    # Decide vertex label
    max_vertex = max(("Real", C_real), ("Fake", C_fake), ("Misleading", C_mis), key=lambda x: x[1])
    vertex_label, vertex_conf = max_vertex[0], int(max_vertex[1] * 100)

    # Fact Check override
    if fact_check_status == "predominantly_false":
        return "Fake", max(85, gem_conf, vertex_conf)
    elif fact_check_status == "predominantly_true":
        return "Real", max(85, gem_conf, vertex_conf)
    elif fact_check_status == "mixed_ratings":
        return "Misleading", max(60, gem_conf, vertex_conf)

    # Weighted final confidence (Gemini heavy, as it synthesizes Fact Check)
    final_conf = int(0.5 * gem_conf + 0.3 * vertex_conf + 0.2 * (80 if fact_check_status != "no_fact_checks" else 50))

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

    async def extract_dynamic_keywords(claim: str) -> List[str]:
        """Use Gemini to extract meaningful keywords from the claim via ask_gemini_structured."""
        try:
            prompt = f"""
            Extract 5–10 important keywords or named entities from this statement.
            Focus on event-related, geographic, and proper nouns (people, places, incidents).
            Return as a JSON array of strings. No explanation.

            Text: {claim}
            """
            resp = ask_gemini_structured(prompt)
            keywords = []
            if "parsed" in resp:
                keywords = [k.lower() for k in resp["parsed"] if isinstance(k, str) and len(k) > 2]
            elif "raw_text" in resp:
                keywords = [w.strip().lower() for w in re.split(r"[,|]", resp["raw_text"]) if len(w.strip()) > 2]
            return keywords[:10]
        except Exception as e:
            print(f"⚠️ Keyword extraction failed: {e}")
            return []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for claim in claims:
            dynamic_keywords = await extract_dynamic_keywords(claim)
            all_keywords = list(set(NEWS_KEYWORDS + dynamic_keywords))

            # Build site filter ONLY if there are 5+ credible domains
            site_filter = ""
            if len(CREDIBLE_DOMAINS) >= 5:
                site_filter = " OR ".join([f"site:{d}" for d in CREDIBLE_DOMAINS])
            
            keyword_query = " OR ".join(all_keywords[:8])
            query = f'"{claim}" ({keyword_query})'
            if site_filter:
                query += f" {site_filter}"

            print(f"🔍 Google query: {query}")
            tasks.append(fetch_google(session, query))
        
        results = await asyncio.gather(*tasks)

    for claim, items in zip(claims, results):
        claim_emb = get_embedding(claim)
        claim_evidences = []

        for it in items:
            link = it.get("link", "")
            domain = urlparse(link).netloc.lower()
            snippet = html.unescape(it.get("snippet", ""))[:400]

            # Only filter domains if we have 5+ credible domains
            if len(CREDIBLE_DOMAINS) >= 5 and not any(d in domain for d in CREDIBLE_DOMAINS):
                continue

            # Snippet filter now includes dynamic keywords
            if not any(k in snippet.lower() for k in NEWS_KEYWORDS + dynamic_keywords):
                continue

            snippet_emb = get_embedding(snippet)
            similarity = float(util.cos_sim(claim_emb, snippet_emb))

            # Debug info
            print(f"Claim: {claim[:50]}..., snippet: {snippet[:50]}..., similarity: {similarity:.3f}")

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

# ---------------- Updated Gemini prompt (Integrate Fact Check) ----------------

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

def assemble_gemini_prompt_structured(claim: str, evidences: List[Dict[str, Any]], status: str, fact_check_results: dict, full_text: str = "") -> str:
    today_str = datetime.now().strftime("%B %d, %Y")
    local_context = extract_local_context(claim, full_text) if full_text else ""

    context_part = f"The claim appears in the following context:\n\"\"\"{local_context}\"\"\"\n\n" if local_context else ""

    # Format fact-check results
    fact_checks_str = ""
    if fact_check_results["fact_checks"]:
        fact_checks_str = "\n".join([
            f"- {fc['publisher']}: \"{fc['rating']}\" ({fc['rating_category'].upper()}) - {fc['claim'][:100]}"
            for fc in fact_check_results["fact_checks"][:3]
        ])
    else:
        fact_checks_str = "No professional fact-checks found for this specific claim."
    
    fc_summary = fact_check_results["summary"]

    return f"""
You are an AI fact-checking assistant synthesizing ML predictions, search evidence, and professional fact-checks.

{context_part}
Input claim: \"\"\"{claim}\"\"\"
Corroboration status: {status}
Evidence snippets: {json.dumps(evidences[:5], ensure_ascii=False)}
Fact-Check Status: {fact_check_results['status']}
Fact-Check Summary:
{fact_checks_str}
   - Total fact-checks: {fc_summary['total']}
   - Rated FALSE: {fc_summary['false_count']} | TRUE: {fc_summary['true_count']} | MIXED: {fc_summary['mixed_count']}
Today's date: {today_str}

Instructions:
- Prioritize fact-check consensus if available (e.g., predominantly_false → Fake).
- Use evidence snippets and ML context to verify factual accuracy.
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
    Integrated detection: Vertex AI (ML) + Google Fact Check + Corroboration → Gemini synthesis.
    """

    start_total = time.time()

    # --- STEP 0: Try retrieving cached result from Pinecone (already handled in app.py) ---
    # Skip here as app.py handles caching

    # --- STEP 1: Clean + preprocess text ---
    text = re.sub(r"(?<=[a-zA-Z])\.(?=[A-Z])", ". ", text)

    # --- STEP 2: Run Google Fact Check ---
    fact_check_results = query_google_fact_check_api(text)

    # --- STEP 3: Metadata extraction (Gemini) ---
    metadata = extract_metadata_with_gemini(text)

    # --- STEP 4: Vertex AI prediction (ML) ---
    vertex_result = predict_with_vertex_ai(metadata)
    vertex_scores = extract_vertex_scores(vertex_result)

    # --- STEP 5: Split into claims ---
    claims = simple_sentence_split(metadata["text"])

    # --- STEP 6: Corroboration from Google ---
    corroboration_data = asyncio.run(corroborate_all_with_google_async(claims))

    # --- STEP 7: Parallel Gemini checks for claims (with Fact Check in prompt) ---
    def process_claim(claim: str):
        gem_resp = ask_gemini_structured(
            assemble_gemini_prompt_structured(
                claim,
                corroboration_data["evidences"],
                corroboration_data["status"],
                fact_check_results,
                full_text=metadata["text"]
            )
        )
        parsed = gem_resp.get("parsed", {})
        gem_pred = parsed.get("prediction", "Unknown")
        gem_conf = int(parsed.get("confidence", 70))
        explanation = parsed.get("explanation", "Based on available evidence.")

        final_pred, final_conf = adjusted_ensemble(gem_pred, gem_conf, vertex_scores, fact_check_results.get("status", "no_fact_checks"))

        return {
            "claim_text": claim,
            "gemini": parsed,
            "vertex_ai": vertex_scores,  # Simplified
            "fact_check": fact_check_results,
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

    # --- STEP 8: Article-level summary ---
    overall_conf = int(sum(overall_scores) / len(overall_scores)) if overall_scores else 0
    overall_label = max(set(preds), key=preds.count) if preds else "Unknown"
    combined_explanation = " | ".join(explanations[:3]) if explanations else "No detailed explanation available."

    # --- STEP 9: Domain credibility bonus ---
    source_domain = domain_from_url(metadata.get("source", ""))
    domain_bonus = get_domain_bonus(source_domain)
    if domain_bonus > 0:
        bonus_points = int(overall_conf * domain_bonus)
        overall_conf = min(100, overall_conf + bonus_points)
        combined_explanation += f" | Small credibility bonus applied due to trusted source ({source_domain})"

    result_summary = {
        "score": overall_conf,  # 0-100 for app.py
        "prediction": overall_label,
        "explanation": combined_explanation,
    }

    # --- STEP 10: Store embedding + verification in Firestore (deduplicated) ---
    try:
        embedding = [float(x) for x in get_embedding(text).tolist()]  # Convert tensor to list
        if db:
            doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
            db.collection("articles").document(doc_id).set({
                "text": text,
                "embedding": embedding,
                "verified": True,
                "text_score": overall_conf / 100,  # Normalize for old format
                "prediction": overall_label,
                "gemini_reasoning": combined_explanation,
                "text_explanation": combined_explanation,
                "last_updated": datetime.utcnow(),
                "type": "text"
            }, merge=True)
    except Exception as e:
        print(f"[Firestore Embedding Storage Failed] {e}")

    # --- STEP 11: Cache final result in Pinecone ---
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