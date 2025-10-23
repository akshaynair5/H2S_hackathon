
#Architecture 1 
import concurrent.futures
import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import re
import requests
from bs4 import BeautifulSoup
import html
import json
from datetime import datetime, timedelta
from urllib.parse import urlparse
import hashlib
from collections import Counter

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

tokenizer = AutoTokenizer.from_pretrained("Pulk17/Fake-News-Detection")
model = AutoModelForSequenceClassification.from_pretrained("Pulk17/Fake-News-Detection")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
CX_ID = os.getenv("GOOGLE_SEARCH_CX")


FACT_CHECK_API_KEY = os.getenv("GEMINI_API_KEY")


CREDIBILITY_TIERS = {
    "tier1": {  # Highest credibility - news agencies, fact-checkers
        "domains": [
            "reuters.com", "apnews.com", "bbc.com", "theguardian.com",
            "factcheck.org", "snopes.com", "politifact.com", "fullfact.org",
            "afp.com", "nytimes.com", "washingtonpost.com"
        ],
        "weight": 1.0
    },
    "tier2": {  # High credibility - established news outlets
        "domains": [
            "cnn.com", "npr.org", "pbs.org", "wsj.com", "bloomberg.com",
            "thehindu.com", "ndtv.com", "indiatoday.in", "hindustantimes.com",
            "indiatoday.in", "dw.com", "france24.com"
        ],
        "weight": 0.85
    },
    "tier3": {  # Medium credibility - general news sites
        "domains": [
            "forbes.com", "time.com", "usatoday.com", "independent.co.uk",
            "thetimes.co.uk", "scroll.in", "thewire.in"
        ],
        "weight": 0.7
    }
}

# Known unreliable sources
UNRELIABLE_SOURCES = [
    "naturalnews.com", "infowars.com", "beforeitsnews.com",
    "yournewswire.com", "theonion.com", "clickhole.com"  # Satire sites
]

# ------------------------
# Gemini Helper Function
# ------------------------
def ask_gemini(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns cleaned text.
    """
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)  # Use faster model for efficiency
        output = response.text.strip()
        return output
    except Exception as e:
        return f"Error using Gemini: {str(e)}"


# ------------------------
# NEW: Enhanced Text Analysis Pipeline
# ------------------------
def extract_claims(text: str) -> list:
    """Extract specific factual claims from text using Gemini"""
    prompt = f"""
    Analyze this text and extract ONLY specific, verifiable factual claims.
    Ignore opinions, predictions, or unverifiable statements.
    
    Text: "{text[:500]}"
    
    Return claims as a JSON array, max 5 claims:
    ["claim1", "claim2", ...]
    
    If no verifiable claims, return: []
    """
    try:
        output = ask_gemini(prompt)
        print("Exact Claim " , prompt , "\n")
        claims = json.loads(output)
        return claims if isinstance(claims, list) else []
    except:
        return [text[:150]]  # Fallback to original behavior


def detect_emotional_manipulation(text: str) -> dict:
    """Detect sensationalism and emotional manipulation tactics"""
    prompt = f"""
    Analyze this text for emotional manipulation tactics:
    
    Text: "{text[:300]}"
    
    Check for:
    1. Excessive capitalization/exclamation marks
    2. Extreme emotional language (outrage, fear, shock)
    3. Urgency tactics ("BREAKING", "SHOCKING", "Must see")
    4. Clickbait patterns
    5. Loaded language and bias
    
    Return JSON:
    {{
        "manipulation_score": 0-100,
        "tactics_detected": ["tactic1", "tactic2"],
        "reasoning": "brief explanation"
    }}
    """
    try:
        output = ask_gemini(prompt)
        print("Emotional Manipulation" , output , "\n")
        result = json.loads(output)
        return result
    except:
        # Fallback: simple heuristic detection
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        exclaim_count = text.count('!')
        
        score = min(100, int((caps_ratio * 100 + exclaim_count * 10)))
        return {
            "manipulation_score": score,
            "tactics_detected": ["excessive_punctuation"] if exclaim_count > 3 else [],
            "reasoning": "Detected through pattern analysis"
        }


def assess_source_credibility(url: str) -> dict:
    """Assess credibility based on domain"""
    domain = urlparse(url).netloc.lower().replace("www.", "")
    
    # Check unreliable sources
    if any(unreliable in domain for unreliable in UNRELIABLE_SOURCES):
        return {"tier": "unreliable", "weight": 0.0, "domain": domain}
    
    # Check credibility tiers
    for tier_name, tier_info in CREDIBILITY_TIERS.items():
        if any(cred_domain in domain for cred_domain in tier_info["domains"]):
            return {"tier": tier_name, "weight": tier_info["weight"], "domain": domain}
    
    return {"tier": "unknown", "weight": 0.5, "domain": domain}


def cross_reference_sources(sources: list) -> dict:
    """Analyze agreement and diversity of sources"""
    if not sources:
        return {"agreement_score": 0, "diversity_score": 0, "consensus": "none"}
    
    # Check domain diversity
    domains = [assess_source_credibility(src["link"])["domain"] for src in sources]
    unique_domains = len(set(domains))
    diversity_score = min(100, (unique_domains / len(sources)) * 100)
    
    # Check credibility distribution
    credibility_weights = [assess_source_credibility(src["link"])["weight"] for src in sources]
    avg_credibility = sum(credibility_weights) / len(credibility_weights)
    
    # Check for high-tier source agreement (tier1 + tier2)
    high_tier_count = sum(1 for w in credibility_weights if w >= 0.85)
    
    if high_tier_count >= 3:
        consensus = "strong"
        agreement_score = 90
    elif high_tier_count >= 2:
        consensus = "moderate"
        agreement_score = 70
    elif high_tier_count >= 1:
        consensus = "weak"
        agreement_score = 50
    else:
        consensus = "none"
        agreement_score = 30
    
    return {
        "agreement_score": agreement_score,
        "diversity_score": int(diversity_score),
        "consensus": consensus,
        "high_tier_sources": high_tier_count,
        "avg_credibility": avg_credibility
    }


def check_temporal_consistency(text: str, sources: list) -> dict:
    """Check if claims match the timeline of reported events"""
    prompt = f"""
    Does this text contain time-sensitive claims (dates, recent events, "today", "yesterday")?
    
    Text: "{text[:200]}"
    
    Return JSON:
    {{
        "is_time_sensitive": true/false,
        "temporal_claims": ["claim1", "claim2"],
        "recency_required": true/false
    }}
    """
    try:
        output = ask_gemini(prompt)
        print("Temporal Consistency " , output , "\n")
        temporal_info = json.loads(output)
        
        if temporal_info.get("is_time_sensitive", False):
            # Check if sources are recent (within last 30 days for time-sensitive claims)
            # Note: Google Custom Search API dateRestrict is already set to y1 (1 year)
            # For real implementation, you'd parse publication dates from sources
            return {
                "is_consistent": len(sources) > 0,
                "warning": "Time-sensitive claim - verify date" if not sources else None
            }
        return {"is_consistent": True, "warning": None}
    except:
        return {"is_consistent": True, "warning": None}


def query_google_fact_check_api(text: str, max_results=5) -> dict:
    """
    Query Google Fact Check Tools API for existing fact-checks.
    This pipeline checks if professional fact-checkers have already verified the claim.
    """
    try:
        # Extract clean claim for fact-check search
        refine_prompt = f"""
        Extract the core factual claim from this text in 5-15 words.
        Focus on the verifiable statement, remove opinions and context.
        Return ONLY the claim, nothing else.
        
        Text: "{text[:200]}"
        """
        refined_claim = ask_gemini(refine_prompt).strip('"').strip()
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
                for review in claim_review[:2]:  # Top 2 reviews per claim
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


def corroborate_with_google(query: str, max_results=8):
    """Enhanced Google search with multi-tier source analysis"""
    try:
        # Extract clean search query
        refine_prompt = f"""
        Extract the shortest possible search query (5-10 words max) from this text.
        Focus on the main claim, remove opinions and filler words.
        Return ONLY the search query, nothing else.
        
        Text: "{query}"
        """
        refined_query = ask_gemini(refine_prompt).strip('"').strip()
        print(f"Refined Query: {refined_query}")

        # Primary search
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": CX_ID,
            "q": refined_query,
            "num": max_results,
            "dateRestrict": "y1"  # Last year for recency
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        items = data.get("items", [])
        
        if not items:
            return {
                "status": "no_results",
                "sources": [],
                "explanations": [],
                "cross_reference": {"agreement_score": 0, "consensus": "none"}
            }

        # Process sources
        sources = []
        explanations = []
        
        def process_item(item):
            title = html.unescape(item.get("title", ""))
            link = item.get("link", "")
            snippet = item.get("snippet", "").strip()
            
            if link:
                # Assess credibility
                cred_info = assess_source_credibility(link)
                
                source = {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "credibility_tier": cred_info["tier"],
                    "credibility_weight": cred_info["weight"]
                }
                
                # Generate explanation
                summary_prompt = f"""
                Claim: "{query[:150]}"
                Source snippet: "{snippet}"
                Source credibility: {cred_info['tier']}
                
                In ONE sentence: Does this source support, contradict, or remain neutral about the claim?
                Be specific about what the source says.
                """
                explanation = ask_gemini(summary_prompt)
                print("Explanation " , explanation , "\n")
                exp = {
                    "source": title,
                    "explanation": explanation,
                    "credibility": cred_info["tier"]
                }
                return source, exp
            return None, None

        # Parallel process item explanations
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max_results)) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in items[:max_results]}
            for future in concurrent.futures.as_completed(future_to_item):
                source, exp = future.result()
                if source:
                    sources.append(source)
                if exp:
                    explanations.append(exp)

        # Cross-reference analysis
        cross_ref = cross_reference_sources(sources)
        
        # Filter high-credibility sources
        high_cred_sources = [s for s in sources if s["credibility_weight"] >= 0.85]
        medium_cred_sources = [s for s in sources if 0.7 <= s["credibility_weight"] < 0.85]
        
        # Determine status based on multi-tier analysis
        if len(high_cred_sources) >= 3 and cross_ref["consensus"] == "strong":
            status = "strongly_corroborated"
        elif len(high_cred_sources) >= 2 or (len(high_cred_sources) >= 1 and len(medium_cred_sources) >= 2):
            status = "corroborated"
        elif len(high_cred_sources) == 1 or len(medium_cred_sources) >= 2:
            status = "weakly_corroborated"
        elif any(s["credibility_weight"] == 0.0 for s in sources):
            status = "contradicted"
        else:
            status = "uncorroborated"

        return {
            "status": status,
            "sources": sources,
            "high_credibility_sources": high_cred_sources,
            "explanations": explanations,
            "cross_reference": cross_ref
        }

    except Exception as e:
        print(f"Corroboration error: {str(e)}")
        return {
            "status": "error",
            "sources": [],
            "explanations": [],
            "cross_reference": {"agreement_score": 0, "consensus": "none"},
            "error": str(e)
        }


# ------------------------
# Enhanced Main Detection Function
# ------------------------
def detect_fake_text(text: str):
   
    
    # Pipeline 1: ML Model Inference
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        hf_idx = torch.argmax(probs).item()
        hf_prediction = "Real" if hf_idx == 0 else "Fake"
        hf_score = probs[hf_idx].item()
    
    hf_percent = int(hf_score * 100)
    print(f"[Pipeline 1] ML Model: {hf_prediction} ({hf_percent}%)")

    # Parallel execution of independent pipelines
    def run_fact_check():
        return query_google_fact_check_api(text)

    def run_corroboration():
        return corroborate_with_google(text[:150], max_results=10)

    def run_emotional():
        return detect_emotional_manipulation(text)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        fact_check_future = executor.submit(run_fact_check)
        corroboration_future = executor.submit(run_corroboration)
        emotional_future = executor.submit(run_emotional)

        fact_check_results = fact_check_future.result()
        corroboration = corroboration_future.result()
        manipulation_analysis = emotional_future.result()

    # Temporal depends on corroboration
    temporal_check = check_temporal_consistency(text, corroboration.get("sources", []))

    print(f"[Pipeline 2] Fact Check API Status: {fact_check_results['status']}")
    print(f"[Pipeline 2] Fact Checks Found: {fact_check_results['summary']}")
    print(f"[Pipeline 3] Manipulation Score: {manipulation_analysis['manipulation_score']}")
    print(f"[Pipeline 4] Corroboration Status: {corroboration['status']}")
    print(f"[Pipeline 4] Cross-Reference: {corroboration['cross_reference']}")
    print(f"[Pipeline 5] Temporal Consistency: {temporal_check}")

    # ADAPTIVE WEIGHT CALCULATION based on evidence availability
    has_fact_checks = fact_check_results['status'] not in ["no_fact_checks", "api_error", "error"]
    has_strong_corroboration = corroboration['status'] in ["strongly_corroborated", "corroborated"]
    high_cred_count = len(corroboration.get("high_credibility_sources", []))
    
    # Dynamic weight assignment
    if has_fact_checks:
        # Fact checks found - they get priority but not overwhelming dominance
        fact_check_weight = 45
        search_weight = 30
        manipulation_weight = 15
        temporal_weight = 7
        ml_weight = 1
    else:
        # No fact checks - redistribute weight to other reliable sources
        fact_check_weight = 0
        if has_strong_corroboration and high_cred_count >= 3:
            # Strong search evidence available
            search_weight = 55
            manipulation_weight = 20
            temporal_weight = 15
            ml_weight = 1
        elif has_strong_corroboration:
            # Moderate search evidence
            search_weight = 50
            manipulation_weight = 20
            temporal_weight = 15
            ml_weight = 1
        else:
            # Weak evidence overall - balance all signals
            search_weight = 35
            manipulation_weight = 25
            temporal_weight = 20
            ml_weight = 1
    
    print(f"[Adaptive Weights] FC:{fact_check_weight}% | Search:{search_weight}% | "
          f"Manipulation:{manipulation_weight}% | Temporal:{temporal_weight}% | ML:{ml_weight}%")

    # Enhanced corroboration for claims without fact-checks
    if not has_fact_checks and not has_strong_corroboration:
        print("[Enhanced Mode] Running deeper search analysis...")
        corroboration = corroborate_with_google(text[:150], max_results=15)
        temporal_check = check_temporal_consistency(text, corroboration.get("sources", []))
        high_cred_count = len(corroboration.get("high_credibility_sources", []))
        print(f"[Enhanced] Found {high_cred_count} high-credibility sources")

    # Prepare comprehensive prompt for Gemini
    explanations_str = "\n".join([
        f"- [{exp['credibility'].upper()}] {exp['source']}: {exp['explanation']}" 
        for exp in corroboration.get("explanations", [])
    ])
    
    # Format fact-check results
    fact_checks_str = ""
    if fact_check_results["fact_checks"]:
        fact_checks_str = "\n".join([
            f"- {fc['publisher']}: \"{fc['rating']}\" ({fc['rating_category'].upper()}) - {fc['claim'][:100]}"
            for fc in fact_check_results["fact_checks"][:5]
        ])
    else:
        fact_checks_str = "No professional fact-checks found for this specific claim."
    
    cross_ref = corroboration.get("cross_reference", {})
    fc_summary = fact_check_results["summary"]
    
    gemini_prompt = f"""
You are an expert fact-checker using an adaptive verification system that adjusts based on available evidence.

TEXT TO ANALYZE: "{text[:300]}"

EVIDENCE QUALITY ASSESSMENT:
- Fact-checks available: {"YES" if has_fact_checks else "NO"}
- High-credibility sources: {high_cred_count}
- Source consensus strength: {cross_ref.get('consensus', 'unknown')}

PIPELINE RESULTS (with adaptive weights):

1. PROFESSIONAL FACT-CHECKERS (Weight: {fact_check_weight}%):
{fact_checks_str}
   - Total fact-checks: {fc_summary['total']}
   - Rated FALSE: {fc_summary['false_count']} | TRUE: {fc_summary['true_count']} | MIXED: {fc_summary['mixed_count']}

2. SEARCH EVIDENCE (Weight: {search_weight}%):
{explanations_str}
   - Status: {corroboration['status']}
   - High-credibility sources: {high_cred_count}
   - Source consensus: {cross_ref.get('consensus', 'unknown')}
   - Agreement score: {cross_ref.get('agreement_score', 0)}%

3. EMOTIONAL MANIPULATION (Weight: {manipulation_weight}%):
   - Manipulation score: {manipulation_analysis['manipulation_score']}/100
   - Tactics detected: {', '.join(manipulation_analysis.get('tactics_detected', ['none']))}
   - Reasoning: {manipulation_analysis.get('reasoning', 'N/A')}

4. TEMPORAL CONSISTENCY (Weight: {temporal_weight}%):
   - Is consistent: {temporal_check.get('is_consistent', True)}
   - Warning: {temporal_check.get('warning', 'None')}

5. ML ASSESSMENT (Weight: {ml_weight}%):
   - Prediction: {hf_prediction} ({hf_percent}% confidence)

ADAPTIVE DECISION RULES (apply weights above):
{"[FACT-CHECKS AVAILABLE MODE]" if has_fact_checks else "[NO FACT-CHECKS MODE - rely on search + manipulation + temporal ]"}

When fact-checks exist:
1. If 60%+ rated FALSE → "Fake" (85-95% confidence)
2. If 60%+ rated TRUE → "Real" (85-95% confidence)
3. Otherwise → Combine with search evidence

When NO fact-checks (current situation):
1. If 4+ high-credibility sources agree + manipulation<40 + temporal_ok → "Real" (75-85%)
2. If 3+ high-credibility sources agree + manipulation<50 → "Real" (70-80%)
3. If sources contradict OR manipulation>70 → "Fake" (70-85%)
4. If 2+ high-cred sources + moderate manipulation → "Misleading" (60-75%)
5. If weak evidence + high manipulation(>60) → "Misleading" (55-70%)
6. If agreement_score>75 + low manipulation → trust consensus (70-80%)


OUTPUT FORMAT (strict):
Prediction: <Real/Fake/Misleading>
Confidence: <0-100>%
Explanation: <2-3 sentences explaining the reasoning based on AVAILABLE evidence. If no fact-checks, focus on source credibility, consensus, and red flags. Be specific about what evidence was used.>

DO NOT mention weights, pipelines, or "adaptive mode" in explanation.
Focus on: what credible sources say, level of agreement, manipulation indicators, and temporal/contextual consistency.
"""

    output = ask_gemini(gemini_prompt)
    print(f"[Pipeline 6] Gemini Synthesis: {output[:100]}...")

    # Parse Gemini output
    final_prediction = hf_prediction
    final_score = hf_percent
    final_explanation = "Analysis completed based on available evidence."
    
    for line in output.splitlines():
        line_lower = line.lower().strip()
        if line_lower.startswith("prediction:"):
            final_prediction = line.split(":", 1)[1].strip()
        elif line_lower.startswith("confidence:"):
            match = re.search(r"(\d{1,3})", line)
            if match:
                final_score = min(100, max(0, int(match.group(1))))
        elif line_lower.startswith("explanation:"):
            final_explanation = line.split(":", 1)[1].strip()

    # Adaptive confidence adjustments based on evidence strength
    adjustment = 0
    
    if has_fact_checks:
        # Fact-check based adjustments
        if fact_check_results["status"] == "predominantly_false":
            final_prediction = "Fake"
            adjustment += 15
            final_score = max(final_score, 85)
        elif fact_check_results["status"] == "predominantly_true":
            final_prediction = "Real"
            adjustment += 15
            final_score = max(final_score, 85)
        elif fact_check_results["status"] == "mixed_ratings":
            final_prediction = "Misleading"
            adjustment += 5
    else:
        # No fact-checks - use composite scoring from other pipelines
        composite_score = 0
        
        # Search evidence scoring (0-40 points)
        if corroboration["status"] == "strongly_corroborated":
            composite_score += 40
        elif corroboration["status"] == "corroborated":
            composite_score += 25
        elif corroboration["status"] == "partially_corroborated":
            composite_score += 10
        else:
            composite_score -= 10
        
        # High-credibility source bonus (0-20 points)
        composite_score += min(20, high_cred_count * 5)
        
        # Consensus scoring (0-15 points)
        agreement = cross_ref.get("agreement_score", 0)
        if agreement > 80:
            composite_score += 15
        elif agreement > 60:
            composite_score += 10
        elif agreement > 40:
            composite_score += 5
        
        # Manipulation penalty (-30 to 0 points)
        manip_score = manipulation_analysis["manipulation_score"]
        if manip_score > 70:
            composite_score -= 30
            if "Fake" not in final_prediction:
                final_prediction = "Fake"
        elif manip_score > 50:
            composite_score -= 15
        elif manip_score > 30:
            composite_score -= 5
        
        # Temporal consistency (0-10 points)
        if temporal_check.get('is_consistent', True):
            composite_score += 10
        else:
            composite_score -= 10
        
        # ML model alignment (0-5 points)
        if (hf_prediction == "Real" and composite_score > 30) or \
           (hf_prediction == "Fake" and composite_score < 20):
            composite_score += 5
        
        # Convert composite score to adjustment
        adjustment = int(composite_score / 2)  # Scale to reasonable adjustment range
        
        print(f"[No Fact-Checks] Composite Score: {composite_score} → Adjustment: {adjustment}")
    
    # Additional consensus boost
    if cross_ref.get("agreement_score", 0) > 80 and high_cred_count >= 3:
        adjustment += 10
    
    final_score = min(100, max(35, final_score + adjustment))  # Floor at 35 for uncertainty

    # Confidence cap for weak evidence scenarios
    if not has_fact_checks and high_cred_count < 2 and corroboration["status"] not in ["strongly_corroborated", "corroborated"]:
        final_score = min(final_score, 70)  # Cap confidence when evidence is weak
        print("[Weak Evidence] Capping confidence at 70%")

    print(f"[FINAL] Prediction: {final_prediction}, Score: {final_score}%")
    
    return {
        "prediction": final_prediction,
        "score": final_score,
        "explanation": final_explanation,
        "evidence_quality": {
            "has_fact_checks": has_fact_checks,
            "high_credibility_sources": high_cred_count,
            "source_agreement": cross_ref.get("agreement_score", 0)
        }
    }