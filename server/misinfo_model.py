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
import sys


from database import db
from vectorDb import (
    embed_text,
    store_feedback,
    pc, INDEX_NAME
)
from datetime import datetime, timedelta
import aiohttp
import asyncio

# -------------------------------------------------
#  PRETTY CONSOLE LOGGING
# -------------------------------------------------
from datetime import datetime
import sys

class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    PURPLE  = "\033[95m"
    BLUE    = "\033[94m"
    GRAY    = "\033[90m"

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")

def log_section(title: str):
    print(f"\n{Colors.BOLD}{Colors.PURPLE}┏{'━' * (len(title) + 4)}┓{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.PURPLE}┃  {title}  ┃{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.PURPLE}┗{'━' * (len(title) + 4)}┛{Colors.RESET}\n")

def log_info(msg: str):
    print(f"{Colors.GRAY}[{_now()}] {Colors.CYAN}ℹ INFO{Colors.RESET}  :: {msg}")

def log_success(msg: str):
    print(f"{Colors.GRAY}[{_now()}] {Colors.GREEN}✓ SUCCESS{Colors.RESET} :: {msg}")

def log_warn(msg: str):
    print(f"{Colors.GRAY}[{_now()}] {Colors.YELLOW}⚠ WARN{Colors.RESET}  :: {msg}")

def log_error(msg: str):
    print(f"{Colors.GRAY}[{_now()}] {Colors.RED}✖ ERROR{Colors.RESET} :: {msg}")

def log_phase(phase: str, elapsed: float = None):
    if elapsed is not None:
        print(f"{Colors.GRAY}[{_now()}] {Colors.BLUE}▶ PHASE{Colors.RESET}  :: {phase}  ({elapsed:.2f}s)")
    else:
        print(f"{Colors.GRAY}[{_now()}] {Colors.BLUE}▶ PHASE{Colors.RESET}  :: {phase}")

def log_debug(msg: str):
    print(f"{Colors.GRAY}[{_now()}] {Colors.GRAY}DEBUG{Colors.RESET} :: {msg}")

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

# ---------------- GCP credentials ----------------
def get_gcp_credentials():
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

SOCIAL_MEDIA_DOMAINS = [
    "x.com", "reddit.com", "instagram.com",
    "facebook.com", "tiktok.com", "linkedin.com", "youtube.com"
]

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
    domain = domain.lower().strip()
    doc = db.collection("news_sources").document(domain).get()
    if doc.exists:
        return doc.to_dict().get("avg_score", 0.0)
    return 0.0

def domain_score_for_url(url: str) -> float:
    d = domain_from_url(url)
    score = get_trusted_score(d)
    return score

_CACHE_TTL = 300 
_last_cache_time = 0
_cached_domains = []

def invalidate_domain_cache():
    global _last_cache_time
    _last_cache_time = 0

def load_credible_domains_cached() -> List[str]:
    global _cached_domains, _last_cache_time
    now = time.time()
    if not _cached_domains or (now - _last_cache_time) > _CACHE_TTL:
        _cached_domains = load_credible_domains()
        _last_cache_time = now
    return _cached_domains

def add_or_update_trusted_sources_batch(domain_scores: Dict[str, float]):
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

# ---------------- Query Concision ----------------
def concise_query(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    
    try:
        prompt = f"""
        Concisely summarize the following text to under {max_chars} characters while preserving the core factual claim and context.
        Do not change the meaning. Return only the concise version, no extra text.
        
        Text: {text}
        """
        resp = GEM_MODEL.generate_content(prompt)
        concised = getattr(resp, "text", str(resp)).strip()
        if len(concised) > max_chars:
            concised = concised[:max_chars]
        log_info(f"Query concised from {len(text)} to {len(concised)} chars")
        return concised
    except Exception as e:
        log_warn(f"Query concision failed: {e}. Using truncation.")
        return text[:max_chars]

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

# ============================================================
#  REDESIGNED COMPREHENSIVE FACT-CHECKING SYSTEM
# ============================================================

def extract_claims_with_llm(text: str) -> List[str]:
    """
    Use LLM to extract verifiable claims from text.
    Returns a list of simple, fact-checkable statements.
    """
    try:
        prompt = f"""Extract all specific, verifiable claims from the following text.
        
Requirements:
- List only simple statements that could be fact-checked
- Each claim should be a standalone statement
- Focus on factual assertions, not opinions
- Return ONLY a JSON array of claims, nothing else

Text: {text}

Output format: ["claim 1", "claim 2", "claim 3"]"""

        response = GEM_MODEL.generate_content(prompt)
        claims_text = response.text.strip()
        
        # Parse JSON response
        if claims_text.startswith('```'):
            claims_text = claims_text.split('```')[1]
            if claims_text.startswith('json'):
                claims_text = claims_text[4:]
        
        claims = json.loads(claims_text.strip())
        log_info(f"Extracted {len(claims)} claims from text")
        return claims
        
    except Exception as e:
        log_error(f"Claim extraction failed: {e}")
        return []


def query_fact_check_for_claim(claim: str, max_results=3) -> dict:
    """
    Query Google Fact Check Tools API for a single claim.
    """
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "key": FACT_CHECK_API_KEY,
            "query": claim,
            "pageSize": max_results,
            "languageCode": "en"
        }
        
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            log_warn(f"Fact-Check API returned HTTP {resp.status_code}")
            return {"found": False, "claim": claim, "fact_checks": []}

        data = resp.json()
        claims_data = data.get("claims", [])
        
        if not claims_data:
            return {"found": False, "claim": claim, "fact_checks": []}

        fact_checks = []
        for claim_item in claims_data[:max_results]:
            for review in claim_item.get("claimReview", [])[:2]:
                fact_checks.append({
                    "claim_text": claim_item.get("text", "")[:150],
                    "publisher": review.get("publisher", {}).get("name", "Unknown"),
                    "rating": review.get("textualRating", ""),
                    "title": review.get("title", ""),
                    "url": review.get("url", "")
                })

        return {
            "found": True,
            "claim": claim,
            "fact_checks": fact_checks
        }
        
    except Exception as e:
        log_error(f"Fact-check query failed for claim '{claim}': {e}")
        return {"found": False, "claim": claim, "fact_checks": [], "error": str(e)}


def analyze_fact_checks_with_llm(claim: str, fact_checks: List[Dict]) -> dict:
    """
    Use LLM to reason about fact-check results and provide verdict.
    """
    try:
        # Format fact-checks for the prompt
        fact_check_summary = "\n".join([
            f"- {fc['publisher']}: {fc['rating']} - {fc['title']}"
            for fc in fact_checks
        ])
        
        prompt = f"""Analyze the following claim and its fact-check results:

Claim: {claim}

Fact-Check Results:
{fact_check_summary}

Based on these fact-checks, provide:
1. A verdict (TRUE/FALSE/MIXED/UNVERIFIED)
2. A brief explanation (2-3 sentences)
3. Confidence level (HIGH/MEDIUM/LOW)

Return ONLY a JSON object with this format:
{{
    "verdict": "TRUE/FALSE/MIXED/UNVERIFIED",
    "explanation": "explanation here",
    "confidence": "HIGH/MEDIUM/LOW"
}}"""

        response = GEM_MODEL.generate_content(prompt)
        result_text = response.text.strip()
        
        # Clean up response
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        
        analysis = json.loads(result_text.strip())
        return {
            "claim": claim,
            "verdict": analysis.get("verdict", "UNVERIFIED"),
            "explanation": analysis.get("explanation", ""),
            "confidence": analysis.get("confidence", "LOW"),
            "fact_checks": fact_checks
        }
        
    except Exception as e:
        log_error(f"LLM analysis failed: {e}")
        return {
            "claim": claim,
            "verdict": "ERROR",
            "explanation": f"Analysis failed: {str(e)}",
            "confidence": "LOW",
            "fact_checks": fact_checks
        }


def comprehensive_fact_check(text: str) -> dict:
    """
    REDESIGNED: Extract claims, check each one, and analyze results.
    
    Process:
    1. Extract claims from text using LLM
    2. Check each claim against Google Fact Check API
    3. For claims with existing fact-checks, use LLM to reason about them
    4. Return comprehensive results with early_exit flag
    """
    try:
        log_info(f"Starting comprehensive fact-check for text: {text[:100]}...")
        
        # Step 1: Extract claims
        claims = extract_claims_with_llm(text)
        if not claims:
            return {
                "status": "no_claims_extracted",
                "fact_checks": [],
                "summary": {
                    "total": 0,
                    "false_count": 0,
                    "true_count": 0,
                    "mixed_count": 0,
                    "unverified_count": 0
                },
                "early_exit": False,
                "claims_analyzed": []
            }
        
        # Step 2 & 3: Check each claim and analyze
        results = []
        early_exit_triggered = False
        counters = {"false": 0, "true": 0, "mixed": 0, "unverified": 0}
        
        for claim in claims:
            log_info(f"Checking claim: {claim}")
            
            # Query fact-check API
            fact_check_result = query_fact_check_for_claim(claim)
            
            if fact_check_result["found"] and fact_check_result["fact_checks"]:
                # Analyze with LLM if fact-checks exist
                analysis = analyze_fact_checks_with_llm(
                    claim, 
                    fact_check_result["fact_checks"]
                )
                
                # Map verdict to category
                verdict = analysis["verdict"].upper()
                if verdict == "TRUE":
                    category = "true"
                    counters["true"] += 1
                elif verdict == "FALSE":
                    category = "false"
                    counters["false"] += 1
                elif verdict == "MIXED":
                    category = "mixed"
                    counters["mixed"] += 1
                else:
                    category = "unverified"
                    counters["unverified"] += 1
                
                # Check for early exit conditions (strong signals)
                if verdict in ["TRUE", "FALSE"] and analysis["confidence"] == "HIGH":
                    early_exit_triggered = True
                
                results.append({
                    "claim": claim,
                    "verdict": verdict,
                    "rating_category": category,
                    "explanation": analysis["explanation"],
                    "confidence": analysis["confidence"],
                    "fact_checks": fact_check_result["fact_checks"]
                })
            else:
                # No existing fact-checks found
                counters["unverified"] += 1
                results.append({
                    "claim": claim,
                    "verdict": "UNVERIFIED",
                    "rating_category": "unverified",
                    "explanation": "No existing fact-checks found for this claim.",
                    "confidence": "LOW",
                    "fact_checks": []
                })
        
        # Step 4: Generate summary
        total = len(results)
        summary = {
            "total": total,
            "false_count": counters["false"],
            "true_count": counters["true"],
            "mixed_count": counters["mixed"],
            "unverified_count": counters["unverified"]
        }
        
        # Determine overall status (matching original structure)
        if total == 0:
            status = "no_fact_checks"
        elif counters["false"] / total >= 0.6:
            status = "predominantly_false"
        elif counters["true"] / total >= 0.6:
            status = "predominantly_true"
        elif counters["mixed"] >= 2:
            status = "mixed_ratings"
        else:
            status = "inconclusive"
        
        log_success(f"Fact-check complete: {status} ({total} claims analyzed)")
        
        return {
            "status": status,
            "fact_checks": results,
            "summary": summary,
            "early_exit": early_exit_triggered,
            "claims_analyzed": results
        }
        
    except Exception as e:
        log_error(f"Comprehensive fact-check failed: {e}")
        return {
            "status": "error",
            "fact_checks": [],
            "summary": {
                "total": 0,
                "false_count": 0,
                "true_count": 0,
                "mixed_count": 0,
                "unverified_count": 0
            },
            "error": str(e),
            "early_exit": False,
            "claims_analyzed": []
        }


# ============================================================
#  REMAINING ORIGINAL FUNCTIONS
# ============================================================

def extract_metadata_with_gemini(text: str) -> dict:
    try:
        prompt = f"""Extract structured information from the following news article text. Return only valid JSON with keys: title, text, author, date, source, category. Rules: - Infer 'title' and 'category' from the text. - If 'author' or 'source' is not present, use "Unknown". - If 'date' is missing, use today's date in YYYY-MM-DD. Text: {text}"""
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
            log_warn(f"Vertex AI endpoint HTTP {response.status_code}")
            return {"predictions": [{"classes": ["Real", "Fake", "Misleading"], "scores": [0.7, 0.2, 0.1]}]}

        data = response.json()
        log_success("Vertex AI response received")
        return data

    except Exception as e:
        log_error(f"Vertex AI error: {e}")
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
        log_error(f"Vertex score extraction failed: {e}")
        return {"Real": 0.7, "Fake": 0.3, "Misleading": 0.0}


def adjusted_ensemble(gem_pred: str, gem_conf: int, vertex_scores: dict, fact_check_status: str, threshold=0.165) -> (str, int):
    """
    Combine Gemini, Vertex AI, and Fact Check with soft thresholding.
    """
    C_real = vertex_scores.get("Real", 0.7)
    C_fake = vertex_scores.get("Fake", 0.3)
    C_mis = vertex_scores.get("Misleading", 0.0)
    max_vertex = max(("Real", C_real), ("Fake", C_fake), ("Misleading", C_mis), key=lambda x: x[1])
    vertex_label, vertex_conf = max_vertex[0], int(max_vertex[1] * 100)

    if fact_check_status == "predominantly_false":
        return "Fake", max(85, gem_conf, vertex_conf)
    elif fact_check_status == "predominantly_true":
        return "Real", max(85, gem_conf, vertex_conf)
    elif fact_check_status == "mixed_ratings":
        return "Misleading", max(60, gem_conf, vertex_conf)

    final_conf = int(0.5 * gem_conf + 0.3 * vertex_conf + 0.2 * (80 if fact_check_status != "no_fact_checks" else 50))

    if gem_pred == vertex_label:
        final_pred = gem_pred
    elif "Misleading" in (gem_pred, vertex_label):
        if final_conf > 50:
            final_pred = "Misleading"
        else:
            final_pred = gem_pred if gem_conf >= vertex_conf else vertex_label
    else:
        final_pred = gem_pred if gem_conf >= vertex_conf else vertex_label

    return final_pred, final_conf

def load_credible_domains() -> List[str]:
    docs = db.collection("news_sources").where("num_votes", ">=", 1).stream()
    domains = [doc.id for doc in docs]
    if not domains:
        domains = [
    # Global/general (previously given)
    "reuters.com", "bbc.com", "apnews.com", "cnn.com", "nytimes.com",
    "theguardian.com", "npr.org", "indiatoday.in", "bloomberg.com",

    # North America (additional)
    "washingtonpost.com", "latimes.com", "wsj.com", "cbc.ca", "globalnews.ca",

    # Europe (additional)
    "thetimes.co.uk", "telegraph.co.uk", "france24.com", "spiegel.de", "elpais.com",

    # Asia (additional)
    "straitstimes.com", "scmp.com", "khaleejtimes.com", "timesofindia.indiatimes.com", "hindustantimes.com",

    # Africa (additional)
    "africanews.com", "news24.com", "allafrica.com", "dailyNATION.co.ke", "timeslive.co.za",

    # South America (additional)
    "clarin.com", "folha.uol.com.br", "elpais.com", "oglobo.globo.com", "lanacion.com.ar",

    # Australia (additional)
    "abc.net.au/news", "smh.com.au", "theaustralian.com.au", "guardian.com/au", "news.com.au"
]

    return domains

def get_domain_bonus(domain: str) -> float:
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
        log_warn(f"Domain bonus fetch failed: {e}")
        return 0.0

async def fetch_google_simple(session, query):
    try:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={"key": GOOGLE_KEY, "cx": CX_ID, "q": query, "num": MAX_SEARCH_RESULTS},
            timeout=10
        ) as resp:
            data = await resp.json()
            return data.get("items", [])
    except Exception as e:
        log_error(f"Google search failed: {e}")
        return []

async def corroborate_with_google_simple(query: str) -> Dict[str, Any]:
    evidences = []
    domain_updates: Dict[str, float] = {}
    
    log_info(f"Google search query → {query[:80]}…")
    
    async with aiohttp.ClientSession() as session:
        items = await fetch_google_simple(session, query)
    
    query_emb = get_embedding(query)
    
    for it in items:
        link = it.get("link", "")
        domain = urlparse(link).netloc.lower()
        
        if any(sm_domain in domain for sm_domain in SOCIAL_MEDIA_DOMAINS):
            log_debug(f"Filtered social media: {domain}")
            continue
        
        snippet = html.unescape(it.get("snippet", ""))[:400]
        
        snippet_emb = get_embedding(snippet)
        similarity = float(util.cos_sim(query_emb, snippet_emb))
        
        if similarity < EMB_SIM_THRESHOLD:
            continue
        
        score = domain_score_for_url(link)
        if score == 0.0:
            score = 0.3
        
        evidence_score = round(clamp01(0.7 * similarity + 0.3 * score), 3)
        
        evidences.append({
            "title": it.get("title", ""),
            "link": link,
            "snippet": snippet,
            "domain_score": score,
            "similarity": round(similarity, 3),
            "evidence_score": evidence_score
        })
        domain_updates[domain] = evidence_score
    
    top_evidences = sorted(evidences, key=lambda x: x["evidence_score"], reverse=True)[:3]
    
    if domain_updates:
        try:
            await asyncio.to_thread(add_or_update_trusted_sources_batch, domain_updates)
            log_success(f"Updated {len(domain_updates)} domain scores")
        except Exception as e:
            log_warn(f"Domain update failed: {e}")
    
    status = (
        "corroborated" if len(set(urlparse(e["link"]).netloc for e in top_evidences)) >= 2
        else "weak" if top_evidences
        else "no_results"
    )
    
    return {"status": status, "evidences": top_evidences}

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

def assemble_gemini_prompt_structured(claim: str, evidences: List[Dict[str, Any]], status: str, fact_check_results: dict, vertex_scores: dict, full_text: str = "") -> str:
    today_str = datetime.now().strftime("%B %d, %Y")
    local_context = extract_local_context(claim, full_text) if full_text else ""

    context_part = f"The claim appears in the following context:\n\"\"\"{local_context}\"\"\"\n\n" if local_context else ""

    # Format fact-checks
    fact_checks_str = ""
    if fact_check_results.get("claims_analyzed"):
        fact_checks_str = "\n".join([
            f"- Claim: {fc['claim'][:100]}\n  Verdict: {fc['verdict']} ({fc['confidence']})\n  {fc['explanation']}"
            for fc in fact_check_results["claims_analyzed"][:3]
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
Fact-Check Analysis:
{fact_checks_str}
   - Total claims analyzed: {fc_summary['total']}
   - Rated FALSE: {fc_summary['false_count']} | TRUE: {fc_summary['true_count']} | MIXED: {fc_summary['mixed_count']} | UNVERIFIED: {fc_summary['unverified_count']}
Vertex AI Scores: Real={vertex_scores['Real']}, Fake={vertex_scores['Fake']}, Misleading={vertex_scores['Misleading']}
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
- Use evidence snippets and Vertex AI scores to verify factual accuracy.
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


# ============================================================
#  MAIN DETECTION FUNCTION WITH INTEGRATED FACT-CHECKING
# ============================================================

def detect_fake_text(text: str ,  log_queue=None) -> dict:
    """
    Main detection function with redesigned comprehensive fact-checking.


    """


    def send_log(type, message):
        """Helper to send logs if queue exists"""
        if log_queue:
            try:
                log_queue.put({"type": type, "message": message}, block=False)
            except:
                pass
    
    start_total = time.time()
    send_log("info", "Processing text...")
    start_total = time.time()
    text = re.sub(r"(?<=[a-zA-Z])\.(?=[A-Z])", ". ", text)
    
    # Step 1: Concise query if needed
    processed_query = concise_query(text, max_chars=200)
    log_info(f"Processing query: {processed_query[:100]}...")

    async def run_parallel_pipelines():
        """
        Run 3 parallel tasks:
        1. Comprehensive Fact Check (redesigned with claim extraction)
        2. Google Search (filtered)
        3. Vertex AI
        """
        send_log("phase", "Running 3 parallel checks")
        loop = asyncio.get_running_loop()
        
        # Shared flag for early exit
        early_exit_event = asyncio.Event()
        
        # Task 1: Comprehensive Fact Check with claim extraction
        async def comprehensive_fact_check_task():
            send_log("info", "Checking against fact-checkers...")
            log_phase("Task 1: Comprehensive Fact Check (with claim extraction)")
            result = await loop.run_in_executor(None, comprehensive_fact_check, processed_query)
            if result.get("early_exit", False):
                send_log("success", "Strong signal detected!")
                log_success("Early exit triggered by Fact Check!")
                early_exit_event.set()
            return result
        
        # Task 2: Google Search (no keyword extraction, filter social media)
        async def google_search_task():
            send_log("info", "Searching for evidence...")
            log_phase("Task 2: Google Search")
            return await corroborate_with_google_simple(processed_query)
        
        # Task 3: Vertex AI
        async def vertex_ai_task():
            send_log("info", "Running AI analysis...")


            log_phase("Task 3: Vertex AI")
            metadata = await loop.run_in_executor(None, extract_metadata_with_gemini, processed_query)
            vertex_result = await loop.run_in_executor(None, predict_with_vertex_ai, metadata)
            return extract_vertex_scores(vertex_result), metadata
        
        # Launch all 3 tasks concurrently
        fact_check_coro = comprehensive_fact_check_task()
        google_search_coro = google_search_task()
        vertex_ai_coro = vertex_ai_task()
        
        # Create tasks with names
        task_to_name = {}
        task1 = asyncio.create_task(fact_check_coro)
        task2 = asyncio.create_task(google_search_coro)
        task3 = asyncio.create_task(vertex_ai_coro)
        
        task_to_name[task1] = "fact_check"
        task_to_name[task2] = "google_search"
        task_to_name[task3] = "vertex_ai"
        
        pending_tasks = {task1, task2, task3}
        
        results = {
            "fact_check_results": None,
            "corroboration_data": None,
            "vertex_scores": None,
            "metadata": None
        }
        
        while pending_tasks:
            done, pending_tasks = await asyncio.wait(
                pending_tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )
            
            for task in done:
                task_name = task_to_name.get(task, "unknown")
                
                try:
                    result = task.result()
                    
                    if task_name == "fact_check":
                        results["fact_check_results"] = result
                        # Check if early exit triggered
                        if result.get("early_exit", False):
                            log_warn("Early exit: Cancelling remaining tasks")
                            # Cancel remaining tasks
                            for remaining_task in pending_tasks:
                                remaining_task.cancel()
                            pending_tasks.clear()
                            break
                    
                    elif task_name == "google_search":
                        results["corroboration_data"] = result
                    
                    elif task_name == "vertex_ai":
                        results["vertex_scores"], results["metadata"] = result
                
                except Exception as e:
                    log_error(f"Task {task_name} failed: {e}")
            
            # Break if early exit triggered
            if early_exit_event.is_set():
                break
        
        # Wait for any remaining tasks to complete or timeout
        if pending_tasks:
            try:
                await asyncio.wait(pending_tasks, timeout=1.0)
            except:
                pass
        
        return results
        
    # Run parallel pipelines
    async def main_pipeline():
        log_phase("Starting 3 parallel tasks", 0)
        
        pipeline_results = await run_parallel_pipelines()
        
        fact_check_results = pipeline_results["fact_check_results"]
        corroboration_data = pipeline_results["corroboration_data"]
        log_info(f'Corroboration Results: {corroboration_data}')
        vertex_scores = pipeline_results["vertex_scores"]
        metadata = pipeline_results["metadata"]
        
        phase1_time = time.time() - start_total
        log_phase("Parallel tasks completed", phase1_time)
        
        # Check if early exit occurred
        early_exit = fact_check_results.get("early_exit", False) if fact_check_results else False
        
        # Handle missing results from cancelled tasks
        if not corroboration_data:
            log_warn("Google search was cancelled or failed")
            corroboration_data = {"status": "no_results", "evidences": []}
        
        if not vertex_scores:
            log_warn("Vertex AI was cancelled or failed")
            vertex_scores = {"Real": 0.7, "Fake": 0.3, "Misleading": 0.0}
        
        if not metadata:
            metadata = extract_metadata_with_gemini(processed_query)
        
        # === Phase 2: LLM Final Reasoning ===
        log_phase("LLM Final Reasoning")
        
        claims = simple_sentence_split(metadata["text"])
        
        # Process claims with LLM
        def process_claim_sync(claim):
            gem_resp = ask_gemini_structured(
                assemble_gemini_prompt_structured(
                    claim,
                    corroboration_data["evidences"],
                    corroboration_data["status"],
                    fact_check_results,
                    vertex_scores,
                    full_text=metadata["text"]
                )
            )
            parsed = gem_resp.get("parsed", {})
            gem_pred = parsed.get("prediction", "Unknown")
            gem_conf = int(parsed.get("confidence", 70))
            explanation = parsed.get("explanation", "Based on available evidence.")
            
            final_pred, final_conf = adjusted_ensemble(
                gem_pred, gem_conf, vertex_scores, fact_check_results.get("status", "no_fact_checks")
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
                "evidence": corroboration_data["evidences"]
            }, final_conf, final_pred, explanation
        
        # Process all claims
        loop = asyncio.get_running_loop()
        claim_tasks = [
            loop.run_in_executor(None, process_claim_sync, claim)
            for claim in claims
        ]
        results_raw = await asyncio.gather(*claim_tasks)
        
        results, overall_scores, preds, explanations = [], [], [], []
        for res, conf, pred, exp in results_raw:
            results.append(res)
            overall_scores.append(conf)
            preds.append(pred)
            explanations.append(exp)
        
        phase2_time = time.time() - start_total
        log_phase("LLM reasoning completed", phase2_time)
        
        # === Phase 3: Aggregate Results ===
        overall_conf = int(sum(overall_scores) / len(overall_scores)) if overall_scores else 0
        overall_label = max(set(preds), key=preds.count) if preds else "Unknown"
        combined_explanation = " | ".join(explanations[:3]) if explanations else "No detailed explanation available."
        
        # Apply domain bonus
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
        
        # === Phase 4: Parallel Storage ===
        log_phase("Parallel storage: Firestore + Pinecone")
        
        async def store_in_firestore_async():
            try:
                embedding = [float(x) for x in get_embedding(text).tolist()]
                if db:
                    doc_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    await loop.run_in_executor(None, lambda: db.collection("articles").document(doc_id).set({
                        "text": text,
                        "embedding": embedding,
                        "verified": True,
                        "text_score": overall_conf / 100,
                        "prediction": overall_label,
                        "gemini_reasoning": combined_explanation,
                        "text_explanation": combined_explanation,
                        "last_updated": datetime.utcnow(),
                        "type": "text"
                    }, merge=True))
                    return True
            except Exception as e:
                log_error(f"Firestore storage failed: {e}")
            return False
        
        async def store_in_pinecone_async():
            try:
                await loop.run_in_executor(None, store_feedback,
                    text,
                    combined_explanation,
                    [],
                    "system",
                    overall_conf / 100,
                    overall_label,
                    True
                )
                return True
            except Exception as e:
                log_error(f"Pinecone storage failed: {e}")
            return False
        
        firestore_success, pinecone_success = await asyncio.gather(
            store_in_firestore_async(),
            store_in_pinecone_async()
        )
        
        total_time = round(time.time() - start_total, 2)
        log_success(f"Complete! Total: {total_time}s | Firestore: {firestore_success} | Pinecone: {pinecone_success}")
        
        return {
            "summary": result_summary,
            "runtime": total_time,
            "claims_checked": len(results),
            "raw_details": results,
            "early_exit": early_exit,
            "fact_check_details": fact_check_results
        }
    
    return asyncio.run(main_pipeline())