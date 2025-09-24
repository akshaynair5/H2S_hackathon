import os, re, html, requests, torch, json
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from flask import Flask
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

load_dotenv()
app = Flask(__name__)

# ------------------------
# API Keys
# ------------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
CX_ID = os.getenv("GOOGLE_SEARCH_CX")

genai.configure(api_key=GEMINI_KEY)

# ------------------------
# Models
# ------------------------
tokenizer = AutoTokenizer.from_pretrained("Pulk17/Fake-News-Detection")  
ml_model = AutoModelForSequenceClassification.from_pretrained("Pulk17/Fake-News-Detection")
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # sentence embedding model for similarity

# ------------------------
# Gemini Helper
# ------------------------
def ask_gemini(prompt: str) -> str:
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error using Gemini: {str(e)}"

# ------------------------
# Google Search Corroboration
# ------------------------
def corroborate_with_google(query: str, max_results=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": GOOGLE_KEY,
        "cx": CX_ID,
        "q": query,
        "num": max_results,
        "dateRestrict": "y1",
    }
    resp = requests.get(url, params=params, timeout=10).json()
    items = resp.get("items", [])
    if not items:
        return {"status": "no_results", "sources": [], "explanations": []}

    # Credible sources database
    credible_domains = [
        "bbc.com","reuters.com","nytimes.com","cnn.com","theguardian.com",
        "apnews.com","factcheck.org","snopes.com","politifact.com","washingtonpost.com"
    ]

    results, explanations, credible_hits = [], [], []
    claim_embedding = embedder.encode(query, convert_to_tensor=True)

    for item in items[:max_results]:
        title = html.unescape(item.get("title", ""))
        link = item.get("link", "")
        snippet = item.get("snippet", "").strip()

        sim_score = util.cos_sim(claim_embedding, embedder.encode(snippet, convert_to_tensor=True)).item()

        source = {"title": title, "link": link, "snippet": snippet, "similarity": round(sim_score,3)}
        results.append(source)

        # Quick domain credibility check
        if any(dom in link for dom in credible_domains):
            credible_hits.append(source)

        # Explanations via Gemini summarization
        summary_prompt = f"""Claim: "{query[:150]}" 
        Snippet: "{snippet}" 
        Task: Summarize in 1 sentence how this snippet supports, contradicts, or is neutral to the claim."""
        explanations.append({"source": title, "explanation": ask_gemini(summary_prompt)})

    # Classification
    if any("fact check" in r["title"].lower() for r in results):
        status = "corroborated (fact-check found)"
    elif len(credible_hits) >= 2:
        status = "corroborated"
    elif len(credible_hits) == 0:
        status = "uncorroborated"
    else:
        status = "weak"

    return {
        "status": status,
        "credible_sources": credible_hits,
        "all_sources": results,
        "explanations": explanations
    }

# ------------------------
# Fake News Detector
# ------------------------
def detect_fake_text(text: str):
    # Step 1: Hugging Face ML Model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = ml_model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        hf_idx = torch.argmax(probs).item()
        hf_prediction = "Real" if hf_idx == 0 else "Fake"
        hf_score = float(probs[hf_idx].item())

    # Step 2: Google Search Corroboration
    corroboration = corroborate_with_google(text[:150])
    explanations_str = "\n".join(
        [f"- {exp['source']}: {exp['explanation']}" for exp in corroboration["explanations"]]
    )

    # Step 3: Gemini Fusion
    gemini_prompt = f"""
    Claim: {text}
    ML Prediction: {hf_prediction} ({hf_score*100:.1f}% confidence)
    Search Evidence:
      {explanations_str}
    Corroboration Status: {corroboration['status']}

    Task:
    - Decide if the claim is Real, Fake, or Misleading.
    - Output: Prediction, Confidence %, Explanation (short).
    """
    gemini_output = ask_gemini(gemini_prompt)

    # Parse Gemini output
    prediction, score, explanation = hf_prediction, int(hf_score*100), gemini_output
    for line in gemini_output.splitlines():
        if line.lower().startswith("prediction:"): prediction = line.split(":",1)[1].strip()
        if line.lower().startswith("confidence:"):
            m = re.search(r"(\d{1,3})", line)
            if m: score = min(100,max(0,int(m.group(1))))
        if line.lower().startswith("explanation:"): explanation = line.split(":",1)[1].strip()

    return {
        "prediction": prediction,
        "score": score,
        "explanation": explanation,
        "ml_raw": {"model_prediction": hf_prediction, "confidence": hf_score},
        "search_evidence": corroboration
    }