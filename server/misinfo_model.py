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
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

from database import db
from vectorDb import (
    embed_text,
    store_feedback,
    pc, INDEX_NAME
)
from datetime import datetime, timedelta
import aiohttp
import asyncio




#----------------- Gemini config ----------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ Missing GEMINI_API_KEY in environment variables")
genai.configure(api_key=GEMINI_API_KEY)
GEM_MODEL = genai.GenerativeModel("gemini-2.5-flash")

# ---------------- Vertex AI config ----------------
PROJECT_ID = os.getenv("PROJECT_ID")
ENDPOINT_ID = os.getenv("TEXT_ENDPOINT_ID")
REGION = os.getenv("LOCATION", "us-central1")
PREDICT_URL = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}:predict"
SERVICE_ACCOUNT_PATH = os.getenv("SERVICE_ACCOUNT_PATH")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CX_ID = os.getenv("GOOGLE_SEARCH_CX")
FACT_CHECK_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------- GCP credentials (for Vertex AI) ----------------
def get_gcp_credentials():
    """
    Load and refresh service account credentials for GCP (usable by Vertex AI).
    """
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_PATH,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(Request())
    return creds

def get_access_token():
    creds = get_gcp_credentials()
    return creds.token

# ---------------- Embeddings ----------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

@lru_cache(maxsize=8192)  
def get_embedding(text: str):
    return embedder.encode(text, convert_to_tensor=True)

# ---------------- Constants ----------------

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

_CACHE_TTL = 300 
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

# ---------------- Google Fact Check API ----------------
def query_google_fact_check_api(text: str, max_results=5) -> dict:
    import requests, re

    try:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        candidates = [s for s in sentences if 5 <= len(s.split()) <= 20]
        refined = max(candidates, key=len, default=text[:100]).strip()

        resp = requests.get(
            "https://factchecktools.googleapis.com/v1alpha1/claims:search",
            params={
                "key": FACT_CHECK_API_KEY,
                "query": refined,
                "pageSize": max_results,
                "languageCode": "en"
            },
            timeout=6
        )

        if resp.status_code != 200:
            return {
                "status": "api_error",
                "fact_checks": [],
                "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0}
            }

        claims = resp.json().get("claims", [])
        if not claims:
            return {
                "status": "no_fact_checks",
                "fact_checks": [],
                "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0}
            }

        fact_checks = []
        counters = {"false": 0, "true": 0, "mixed": 0}

        for claim in claims[:max_results]:
            text_snippet = claim.get("text", "")[:150]
            reviews = claim.get("claimReview", [])[:2]

            for r in reviews:
                rating = r.get("textualRating", "").lower()

                if "false" in rating or "fake" in rating or "incorrect" in rating or "misleading" in rating or "pants" in rating:
                    cat = "false"
                elif "true" in rating or "correct" in rating or "accurate" in rating or "verified" in rating:
                    cat = "true"
                elif "mixed" in rating or "partial" in rating or "mostly" in rating or "half" in rating:
                    cat = "mixed"
                else:
                    cat = "unknown"

                if cat in counters:
                    counters[cat] += 1

                fact_checks.append({
                    "claim": text_snippet,
                    "publisher": r.get("publisher", {}).get("name", "Unknown"),
                    "rating": rating,
                    "rating_category": cat,
                    "title": r.get("title", ""),
                    "url": r.get("url", "")
                })

        total = len(fact_checks)
        if not total:
            return {
                "status": "no_fact_checks",
                "fact_checks": [],
                "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0}
            }

        fr = counters["false"] / total
        tr = counters["true"] / total

        if fr >= 0.6:
            status = "predominantly_false"
        elif tr >= 0.6:
            status = "predominantly_true"
        elif counters["mixed"] >= 2:
            status = "mixed_ratings"
        else:
            status = "inconclusive"

        return {
            "status": status,
            "fact_checks": fact_checks,
            "summary": {
                "total": total,
                "false_count": counters["false"],
                "true_count": counters["true"],
                "mixed_count": counters["mixed"]
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "fact_checks": [],
            "summary": {"total": 0, "false_count": 0, "true_count": 0, "mixed_count": 0},
            "error": str(e)
        }

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

@retry
def predict_with_vertex_ai(metadata: dict) -> dict:

    try:
        headers = {
            "Authorization": f"Bearer {get_access_token()}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            PREDICT_URL,
            headers=headers,
            json={"instances": [metadata]},
            timeout=15
        )

        if response.status_code != 200:
            print(f"[Vertex AI] ⚠️ Endpoint returned {response.status_code}: {response.text[:200]}")
            return {"predictions": [{"classes": ["Real", "Fake", "Misleading"], "scores": [0.7, 0.2, 0.1]}]}

        try:
            data = response.json()
        except Exception as e:
            print(f"[Vertex AI] ⚠️ Invalid JSON response: {e}")
            return {"predictions": [{"classes": ["Real", "Fake", "Misleading"], "scores": [0.7, 0.2, 0.1]}]}

        print("[Vertex AI] ✅ Response received successfully")
        return data

    except requests.exceptions.Timeout:
        print("[Vertex AI] ⏰ Timeout — using fallback prediction.")
    except requests.exceptions.ConnectionError:
        print("[Vertex AI] 🌐 Connection error — spot instance may be unavailable.")
    except Exception as e:
        print(f"[Vertex AI] ⚠️ Unexpected error: {e}")

    return {"predictions": [{"classes": ["Real", "Fake", "Misleading"], "scores": [0.7, 0.2, 0.1]}]}


def extract_vertex_scores(vertex_result: dict) -> dict:

    try:
        preds = vertex_result.get("predictions", [{}])[0]
        classes = preds.get("classes", [])
        scores = preds.get("scores", [])
        score_map = {cls.capitalize(): float(score) for cls, score in zip(classes, scores)}

        return {
            "Real": score_map.get("Real", 0.7),
            "Fake": score_map.get("Fake", 0.3),
            "Misleading": score_map.get("Misleading", 0.0)
        }

    except Exception as e:
        print(f"[Vertex AI] ⚠️ Score extraction failed: {e}")
        return {"Real": 0.7, "Fake": 0.3, "Misleading": 0.0}


def clear_cache_for_text(text: str) -> bool:
    """
    Deletes cached Pinecone entries semantically matching the input text.
    Used to force a re-analysis if misinformation data has been updated.
    Returns True if a cache entry was found and deleted, else False.
    """
    try:
        query_emb = embed_text(text)
        if not query_emb:
            print("[Cache Clear] Could not generate embedding.")
            return False

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


def adjusted_ensemble(
    gem_pred: str,
    gem_conf: int,
    vertex_scores: dict,
    fact_check_status: str,
    corroboration_status: str,
    evidence_count: int = 0,
    threshold=0.15
) -> tuple[str, int]:
    """
    ✅ New evidence-first ensemble logic.
    Real-world corroboration > ML models.
    """

    C_real = vertex_scores.get("Real", 0.7)
    C_fake = vertex_scores.get("Fake", 0.3)
    C_mis = vertex_scores.get("Misleading", 0.0)

    # ======================================================
    # 1. Professional Fact-Checks OVERRIDE prediction
    # ======================================================
    if fact_check_status == "predominantly_false":
        return "Fake", min(97, gem_conf + 10)

    if fact_check_status == "predominantly_true":
        boost = 5 if corroboration_status != "corroborated" else 12
        return "Real", min(98, max(gem_conf, 85) + boost)

    if fact_check_status == "mixed_ratings":
        return "Misleading", max(70, gem_conf)

    # ======================================================
    # ✅ NEW OVERRIDE: Strong corroboration beats ML hesitation
    # ======================================================
    if corroboration_status == "corroborated" and evidence_count >= 3:
        override_conf = max(gem_conf, 88 + (evidence_count * 2))  # 3 evidences → 94%, 5 → 98%
        return "Real", min(98, override_conf)

    # ======================================================
    # 2. Vertex AI influence (reduced weight)
    # ======================================================
    strong_vertex_fake = C_fake > 0.45

    if strong_vertex_fake and corroboration_status != "corroborated":
        # Vertex strongly detects fake AND no evidence supporting claim
        return "Fake", max(70, int(C_fake * 100))

    # If vertex AI is mildly suspicious – DO NOT penalize if evidence exists
    vertex_penalty = int(C_fake * 8) if corroboration_status == "no_results" else 0

    # ======================================================
    # 3. Base confidence from Gemini (kept as is)
    # ======================================================
    base_conf = gem_conf - vertex_penalty

    # ======================================================
    # ✅ New Evidence Weighting (MUCH stronger)
    # ======================================================
    if corroboration_status == "corroborated":
        if evidence_count >= 5:
            evidence_boost = 25
        elif evidence_count >= 3:
            evidence_boost = 18
        else:
            evidence_boost = 10

    elif corroboration_status == "weak":
        evidence_boost = 0
        base_conf = max(55, base_conf - 5)

    else:  # "no_results"
        evidence_boost = 0
        base_conf = max(60, base_conf)  # ✅ No penalty anymore

    # ======================================================
    # 4. Combine signals
    # ======================================================
    final_conf = min(100, base_conf + evidence_boost)

    # ======================================================
    # 5. Final Prediction Rules
    # ======================================================
    if gem_pred == "Real":
        return "Real", final_conf

    if gem_pred == "Fake":
        return "Fake", max(final_conf, 70 if corroboration_status == "no_results" else 60)

    if gem_pred == "Misleading":
        return "Misleading", max(final_conf, 60)

    return gem_pred, max(50, final_conf)

def load_credible_domains() -> List[str]:
    """
    Return the list of currently trusted domains from Firestore.
    Only include domains with at least 1 vote.
    """
    docs = db.collection("news_sources").where("num_votes", ">=", 1).stream()
    domains = [doc.id for doc in docs]
    if not domains:
        domains = ["reuters.com", "bbc.com", "apnews.com", "cnn.com", "nytimes.com",
                   "theguardian.com", "npr.org", "aljazeera.com", "bloomberg.com"]
    return domains

def get_domain_bonus(domain: str) -> float:
    """
    Return a very small bonus (2%) if the domain is highly credible (avg_score >= 0.9)
    and has more than 100 votes. Returns 0 otherwise.
    """
    domain = domain.lower().strip().rstrip("/") 
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
                return 0.02 
        return 0.0
    except Exception as e:
        print(f"⚠️ Domain bonus fetch failed for {domain}: {e}")
        return 0.0

async def fetch_google(session, query, num_results=None):
    try:
        limit = int(num_results or MAX_SEARCH_RESULTS)

        params = {
            "key": str(GOOGLE_KEY or ""),
            "cx": str(CX_ID or ""),
            "q": str(query or ""),
            "num": limit
        }

        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params,
            timeout=10
        ) as resp:
            data = await resp.json()
            return data.get("items", [])[:limit]
    except Exception as e:
        print(f"❌ Google fetch failed: {e}")
        return []

async def extract_keywords(claim: str) -> List[str]:
    """
    Reformulate the claim into 3–6 compressed variations preserving meaning.
    Returns an array of alternate phrasing, not keywords.
    """
    try:
        prompt = f"""
Rewrite this claim into 3–6 alternative formulations that preserve meaning.
Each reformulation should be concise, factual, and under 25 words.
Return ONLY a JSON array of strings.

Text: {claim}
"""
        resp = await asyncio.to_thread(ask_gemini_structured, prompt)

        if "error" in resp:
            print(f"⚠️ Reformulation error: {resp['error']}")
            return []

        parsed = resp.get("parsed")

        # Accept JSON list
        if isinstance(parsed, list):
            variations = parsed
        # Sometimes model returns { "rewrites": [...] }
        elif isinstance(parsed, dict):
            variations = parsed.get("rewrites") or parsed.get("summaries") or parsed.get("sentences", [])
        # As fallback parse raw text into JSON-like list
        else:
            raw = resp.get("raw_text", "")
            variations = re.findall(r'"([^"]+)"', raw)

        # Cleanup
        clean = [
            v.strip()
            for v in variations
            if isinstance(v, str) and len(v.strip()) > 5
        ]

        return clean[:6]

    except Exception as e:
        print(f"❌ Reformulation exception: {e}")
        return []


async def summarize_claim(claim: str) -> str:
    """Summarize claim into Google query with fallback."""
    try:
        prompt = f"""
Rewrite this as a concise Google search query (5-10 words).
Return ONLY the query text, no JSON, no quotes, no explanation.

Text: {claim}
"""
        resp = await asyncio.to_thread(ask_gemini_structured, prompt)
        
        if "error" in resp:
            print(f"⚠️ Summarization error: {resp['error']}")
            return claim[:100]
        
        # Check parsed first
        parsed = resp.get("parsed")
        if isinstance(parsed, dict):
            result = parsed.get("query", parsed.get("summary", ""))
            if result:
                return str(result).strip()[:150]
        elif isinstance(parsed, str):
            return parsed.strip()[:150]
        
        # Fallback to raw_text
        raw = resp.get("raw_text", "").strip()
        if raw:
            # Clean JSON artifacts
            clean = re.sub(r'[\{\}":]', '', raw).strip()
            # Remove common prefixes
            clean = re.sub(r'^(query|summary):\s*', '', clean, flags=re.IGNORECASE)
            if len(clean) >= 10:
                return clean[:150]
        
        # Last resort: use original claim
        return claim[:100]
        
    except Exception as e:
        print(f"❌ Summarization exception: {e}")
        return claim[:100]


async def corroborate_all_with_google_async(claims: List[str]) -> Dict[str, Any]:
    """Corroborate claims with Google search - with validation."""
    evidences = []
    CREDIBLE_DOMAINS = load_credible_domains_cached()
    domain_updates: Dict[str, float] = {}
    
    # Extract keywords and summaries in parallel
    keyword_tasks = [extract_keywords(claim) for claim in claims]
    summary_tasks = [summarize_claim(claim) for claim in claims]
    
    keywords_list = await asyncio.gather(*keyword_tasks)
    summaries_list = await asyncio.gather(*summary_tasks)
    
    # Build Google queries with validation
    async with aiohttp.ClientSession() as session:
        tasks = []
        for claim, kws, summary in zip(claims, keywords_list, summaries_list):
            # Validate summary
            if not summary or len(summary) < 10 or "{" in summary or "error" in summary.lower():
                print(f"⚠️ Invalid summary, using claim: {claim[:60]}")
                summary = claim[:100]
            
            # Build query
            if kws and len(kws) > 0:
                kw_str = " ".join(kws[:4])
                # Only add keywords if not already in summary
                if kw_str.lower() not in summary.lower():
                    query = f"{summary} {kw_str}".strip()
                else:
                    query = summary
            else:
                query = summary
            
            print(f"🔍 Google query => {query[:80]}")
            tasks.append(fetch_google(session, query, num_results=10))
        
        results_list = await asyncio.gather(*tasks)
    
    # Process results
    emb_cache = {}
    def get_emb(text):
        if text not in emb_cache:
            emb_cache[text] = get_embedding(text)
        return emb_cache[text]
    
    for claim, items in zip(claims, results_list):
        if not items:
            print(f"⚠️ No Google results for claim: {claim[:60]}")
            continue
        
        print(f"📰 Results retrieved: {len(items)}")
        
        claim_emb = get_emb(claim)
        claim_evidences = []
        
        for it in items[:10]:
            link = it.get("link", "")
            snippet = html.unescape(it.get("snippet", "")).strip()[:400]
            
            # Validate snippet
            if not snippet or len(snippet) < 20:
                continue
            
            snippet_emb = get_emb(snippet)
            similarity = float(util.cos_sim(claim_emb, snippet_emb))
            
            domain = urlparse(link).netloc.lower()
            score = domain_score_for_url(link)
            
            is_credible = domain in CREDIBLE_DOMAINS
            threshold = 0.45 if is_credible else 0.50
            
            if similarity < threshold:
                continue
            
            evidence_score = round(clamp01(0.75 * similarity + 0.25 * score), 3)
            
            claim_evidences.append({
                "title": it.get("title", "No title"),
                "link": link,
                "snippet": snippet,
                "similarity": round(similarity, 3),
                "domain_score": score,
                "evidence_score": evidence_score,
                "is_new_domain": not is_credible
            })
            
            if evidence_score > 0.7:
                domain_updates[domain] = evidence_score
        
        # Add top 3 evidences per claim
        top = sorted(claim_evidences, key=lambda x: x["evidence_score"], reverse=True)[:3]
        evidences.extend(top)
    
    # Update domain scores
    if domain_updates:
        await asyncio.to_thread(add_or_update_trusted_sources_batch, domain_updates)
    
    status = (
        "corroborated" if len(evidences) >= 2 else
        "weak" if evidences else
        "no_results"
    )
    
    return {"status": status, "evidences": evidences}

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

FIRST, ASSESS THE CONTENT TYPE:
───────────────────────────────
Determine if this text contains VERIFIABLE FACTUAL CLAIMS that require fact-checking.

NON-FACTUAL CONTENT (Skip full analysis):
- Personal experiences: "My trip to Japan", "I visited the museum"
- Opinions/feelings: "This movie is amazing", "I think cats are better"
- Questions: "What happened in 2020?", "How do I cook pasta?"
- Creative content: Video titles, song lyrics, fiction, poems
- Instructions/how-tos: "Steps to make coffee"
- Promotional content: "Buy now!", "Best deals today"
- Greetings/casual chat: "Hello everyone", "Thanks for watching"

FACTUAL CONTENT (Requires analysis):
- News events: "Earthquake hits California", "President announces policy"
- Scientific claims: "Study shows coffee prevents cancer"
- Historical statements: "Napoleon died in 1821"
- Statistics: "Unemployment rate dropped to 3%"
- Allegations: "Company accused of fraud"

IF NON-FACTUAL CONTENT DETECTED:
Return JSON with:
- prediction: "Not Applicable"
- confidence: 100
- explanation: "This content does not contain verifiable factual claims requiring fact-checking | [Brief description of content type]"
- evidence: []
- content_type: "personal/opinion/question/creative/promotional/casual"

Example for non-factual:
{{
  "prediction": "Not Applicable",
  "confidence": 100,
  "explanation": "Personal travel content without factual claims | This appears to be a YouTube video title about someone's vacation",
  "evidence": [],
  "content_type": "personal"
}}

IF FACTUAL CONTENT DETECTED:
Proceed with full fact-checking analysis:

Instructions:
- Prioritize fact-check consensus if available (e.g., predominantly_false → Fake).
- Use evidence snippets and ML context to verify factual accuracy.
- Evaluate the claim considering today's date ({today_str}).
- Consider temporal context: old news may be accurate but outdated.
- Return a strict JSON object with keys:

- prediction: "Real", "Fake", or "Misleading"
- confidence: integer 0–100
- explanation: 1–2 short, plain sentences | Use "|" to separate reasoning steps
- evidence: 1–3 key snippets (≤50 words each) with a `support` field
- content_type: "news"
- human_summary (optional): plain summary of the claim

Example for factual content:
{{
  "prediction": "Real",
  "confidence": 85,
  "explanation": "The claim matches current verified reports | Context indicates it refers to an ongoing event",
  "evidence": [
    {{"source":"BBC", "link":"https://...", "snippet":"BBC confirms the described protests occurred in Delhi.", "support":"Supports"}}
  ],
  "content_type": "news",
  "human_summary": "The claim about protests in Delhi is accurate."
}}

SPECIAL CASES:
──────────────
1. **Old News**: If claim is factual but refers to past events (>1 year old):
   - prediction: "Real" (if accurate at the time)
   - Add to explanation: "This refers to a past event from [date]"

2. **Satire/Parody**: If content appears satirical:
   - prediction: "Misleading"
   - explanation: "This appears to be satire or parody content | Not meant as factual reporting"

3. **Mixed Content**: Personal story with factual claims:
   - Focus only on verifiable facts
   - Ignore personal narrative elements

Return ONLY valid JSON. No additional text.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

def quick_initial_assessment(text: str) -> dict:
    """
    Fast lightweight Gemini-only initial impression.
    Returns a short neutral paragraph without predictions or scores.
    """
    prompt = f"""
    You are assisting in misinformation detection, but this is a *quick initial impression*.
    DO NOT claim anything is true or false and DO NOT assign confidence or numeric scores.

    Provide a short paragraph (3–5 sentences max) summarizing:
    - What type of content this text appears to be (news, opinion, speculation)
    - Whether it *sounds* factual or sensational
    - Whether anything seems unverifiable at first glance
    - Ask the user to wait for full fact-check and verification
    - Assume that the date is {datetime.now().strftime("%B %d, %Y")}

    Never assert factual accuracy.

    Text to evaluate:
    \"\"\"{text}\"\"\"
    """

    try:
        resp = GEM_MODEL.generate_content(prompt)
        return {
            "status": "ok",
            "initial_analysis": resp.text.strip(),
        }
    except Exception as e:
        return {
            "status": "error",
            "initial_analysis": "Could not analyze text.",
            "error": str(e)
        }

def detect_fake_text(text: str) -> dict:
    start_total = time.time()
    text = re.sub(r"(?<=[a-zA-Z])\.(?=[A-Z])", ". ", text)

    async def run_parallel_phase1():
        """Run Fact Check + Metadata Extraction concurrently"""
        loop = asyncio.get_running_loop()
        fact_check_task = loop.run_in_executor(None, query_google_fact_check_api, text)
        metadata_task = loop.run_in_executor(None, extract_metadata_with_gemini, text)
        fact_check_results, metadata = await asyncio.gather(fact_check_task, metadata_task)
        return fact_check_results, metadata

    async def run_parallel_phase2(metadata):
        """Run Vertex AI + Corroboration concurrently"""
        claims = simple_sentence_split(metadata["text"])
        loop = asyncio.get_running_loop()
        vertex_task = loop.run_in_executor(None, lambda: extract_vertex_scores(predict_with_vertex_ai(metadata)))
        corroboration_task = corroborate_all_with_google_async(claims)
        vertex_scores, corroboration_data = await asyncio.gather(vertex_task, corroboration_task)
        return vertex_scores, corroboration_data, claims

    async def run_parallel_claim_checks(claims, corroboration_data, fact_check_results, metadata, vertex_scores):
        """Run Gemini claim checks concurrently (in threads for I/O + CPU balance)"""
        loop = asyncio.get_running_loop()

        async def process_claim_async(claim):
            return await loop.run_in_executor(
                None,
                lambda: process_claim_sync(claim, corroboration_data, fact_check_results, metadata, vertex_scores)
            )

        tasks = [process_claim_async(claim) for claim in claims]
        return await asyncio.gather(*tasks)

    def process_claim_sync(claim, corroboration_data, fact_check_results, metadata, vertex_scores):
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

        evidence_count = len([
            e for e in corroboration_data.get("evidences", [])
            if e.get("evidence_score", 0) > 0.6 
        ])
 
        final_pred, final_conf = adjusted_ensemble(
            gem_pred, 
            gem_conf, 
            vertex_scores, 
            fact_check_results.get("status", "no_fact_checks"),
            corroboration_data.get("status", "no_results"),
            evidence_count 
        )
        
        return {
            "claim_text": claim,
            "gemini": parsed,
            "vertex_ai": vertex_scores,
            "fact_check": fact_check_results,
            "corroboration": corroboration_data,
            "ensemble": {
                "final_prediction": final_pred,
                "final_confidence": final_conf,
                "combined_score": round(final_conf / 100, 3)
            },
            "explanation": explanation,
            "evidence": corroboration_data["evidences"],
            "evidence_count": evidence_count 
        }, final_conf, final_pred, explanation

    async def run_parallel_storage(text, overall_conf, overall_label, combined_explanation):
        """Run Firestore + Pinecone storage in parallel"""
        loop = asyncio.get_running_loop()
        firestore_task = loop.run_in_executor(None, lambda: store_in_firestore(text, overall_conf, overall_label, combined_explanation))
        pinecone_task = loop.run_in_executor(None, lambda: store_in_pinecone(text, overall_conf, overall_label, combined_explanation))
        firestore_success, pinecone_success = await asyncio.gather(firestore_task, pinecone_task)
        return firestore_success, pinecone_success

    # --- define storage helpers (same logic, just isolated for async calls) ---
    def store_in_firestore(text, overall_conf, overall_label, combined_explanation):
        try:
            embedding = [float(x) for x in get_embedding(text).tolist()]
            if db:
                doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                db.collection("articles").document(doc_id).set({
                    "text": text,
                    "embedding": embedding,
                    "verified": True,
                    "text_score": overall_conf / 100,
                    "prediction": overall_label,
                    "gemini_reasoning": combined_explanation,
                    "text_explanation": combined_explanation,
                    "last_updated": datetime.utcnow(),
                    "type": "text"
                }, merge=True)
                return True
        except Exception as e:
            print(f"[Firestore Embedding Storage Failed] {e}")
        return False

    def store_in_pinecone(text, overall_conf, overall_label, combined_explanation):
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
            return True
        except Exception as e:
            print(f"[Pinecone Cache Store Failed] {e}")
        return False

    # ===========================
    # MAIN ASYNC EXECUTION LOGIC
    # ===========================
    async def main_pipeline():
        print("[Phase 1] Starting parallel: Fact Check + Metadata Extraction...")
        fact_check_results, metadata = await run_parallel_phase1()
        print(f"[Phase 1] Done in {time.time() - start_total:.2f}s")

        print("[Phase 2] Starting parallel: Vertex AI + Corroboration...")
        vertex_scores, corroboration_data, claims = await run_parallel_phase2(metadata)
        print(f"[Phase 2] Done in {time.time() - start_total:.2f}s")

        print(f"[Phase 3] Processing {len(claims)} claims concurrently...")
        results_raw = await run_parallel_claim_checks(claims, corroboration_data, fact_check_results, metadata, vertex_scores)
        print(f"[Phase 3] Done in {time.time() - start_total:.2f}s")

        results, overall_scores, preds, explanations = [], [], [], []
        for res, conf, pred, exp in results_raw:
            results.append(res)
            overall_scores.append(conf)
            preds.append(pred)
            explanations.append(exp)

        # === Phase 4 (no change) ===
        overall_conf = int(sum(overall_scores) / len(overall_scores)) if overall_scores else 0
        overall_label = max(set(preds), key=preds.count) if preds else "Unknown"
        combined_explanation = " | ".join(explanations[:3]) if explanations else "No detailed explanation available."

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

        # === Phase 5 (parallel storage) ===
        print("[Phase 5] Starting parallel storage: Firestore + Pinecone...")
        firestore_success, pinecone_success = await run_parallel_storage(text, overall_conf, overall_label, combined_explanation)

        total_time = round(time.time() - start_total, 2)
        print(f"[COMPLETE] Total runtime: {total_time}s (Firestore: {firestore_success}, Pinecone: {pinecone_success})")

        return {
            "summary": result_summary,
            "runtime": total_time,
            "claims_checked": len(results),
            "raw_details": results
        }

    return asyncio.run(main_pipeline())