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

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

tokenizer = AutoTokenizer.from_pretrained("Pulk17/Fake-News-Detection")
model = AutoModelForSequenceClassification.from_pretrained("Pulk17/Fake-News-Detection")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
CX_ID = os.getenv("GOOGLE_SEARCH_CX") 

# ------------------------
# Gemini Helper Function
# ------------------------
def ask_gemini(prompt: str) -> str:
    """
    Sends a prompt to Gemini and returns cleaned text.
    """
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        output = response.text.strip()
        return output
    except Exception as e:
        return f"Error using Gemini: {str(e)}"


# ------------------------
# Update corroborator with Gemini query shortener
# ------------------------
# def corroborate_with_google(query: str, max_results=5):
#     try:
#         refine_prompt = f"""
#         You are helping a misinformation detection system.
#         Given this text: "{query}"

#         Task:
#         - Extract the shortest possible search query (5–10 words max).
#         - Keep only essential keywords.
#         - Do NOT include formatting, labels, or explanations.
#         Just return the query itself.
#         """
#         refined_query = ask_gemini(refine_prompt)
#         print("Refined Query for Search:", refined_query)


#         url = "https://www.googleapis.com/customsearch/v1"
#         params = {
#             "key": GOOGLE_API_KEY,
#             "cx": CX_ID,
#             "q": refined_query,
#             "num": max_results
#         }
#         resp = requests.get(url, params=params, timeout=10)
#         # print("Request URL:", resp.url)
#         # print("Status Code:", resp.status_code)

#         data = resp.json()
#         # print("Full JSON Response:", data)
#         # print(json.dumps(data , indent=4))

#         items = data.get("items", [])
#         if not items:
#             return {"status": "no_results", "sources": []}

#         sources = []
#         for item in items[:max_results]:
#             title = html.unescape(item.get("title", ""))
#             link = item.get("link", "")
#             if link:
#                 sources.append({"title": title, "link": link})

#         credible_domains = [
#             "bbc.com", "reuters.com", "nytimes.com", "cnn.com",
#             "theguardian.com", "indiatoday.in", "ndtv.com", "thehindu.com","hindustantimes.com"
#         ]
#         credible_hits = [src for src in sources if any(dom in src["link"] for dom in credible_domains)]

#         if len(credible_hits) >= 3:
#             status = "corroborated"
#         elif len(credible_hits) == 0:
#             status = "uncorroborated"
#         else:
#             status = "weak"

#         result = {
#             "status": status,
#             "credible_sources": credible_hits,
#             "all_sources": sources
#         }
        
#         # print("Processed Result:", result)
#         return result

#     except Exception as e:
#         return {"status": "error", "sources": [], "error": str(e)}

def corroborate_with_google(query: str, max_results=5):
    try:
        refine_prompt = f"""
        You are helping a misinformation detection system.
        Given this text: "{query}"

        Task:
        - Extract the shortest possible search query (5–10 words max).
        - Keep only essential keywords.
        - Do NOT include formatting, labels, or explanations.
        Just return the query itself.
        """


        
        print("Refined Query for Search:", query)

        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": GOOGLE_API_KEY,
            "cx": CX_ID,
            "q": query,
            "num": max_results,
            "dateRestrict": "y1"  # NEW: Restrict to results from the last year for recency
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        items = data.get("items", [])
        if not items:
            return {"status": "no_results", "sources": [], "explanations": []}

        sources = []
        explanations = []  # NEW: List to hold per-result explanations
        for item in items[:max_results]:
            title = html.unescape(item.get("title", ""))
            link = item.get("link", "")
            snippet = item.get("snippet", "").strip()  # NEW: Extract snippet
            if link:
                sources.append({"title": title, "link": link, "snippet": snippet})
                
                # NEW: Use Gemini to summarize snippet's relevance as "explanation"

                print("SNIPPET ____" , snippet)
                summary_prompt = f"""
                Given claim: "{query[:150]}"
                And search result snippet: "{snippet}"
                
                Task: Provide a 1-sentence explanation of how this result supports, contradicts, or is neutral to the claim.
                Be concise and objective.
                """
                explanation = ask_gemini(summary_prompt)
                explanations.append({"source": title, "explanation": explanation})

        credible_domains = [
            "bbc.com", "reuters.com", "nytimes.com", "cnn.com",
            "theguardian.com", "indiatoday.in", "ndtv.com", "thehindu.com", "hindustantimes.com",
            "apnews.com", "factcheck.org"  
        ]
        credible_hits = [src for src in sources if any(dom in src["link"] for dom in credible_domains)]

        if len(credible_hits) >= 3:
            status = "corroborated"
        elif len(credible_hits) == 0:
            status = "uncorroborated"
        else:
            status = "weak"

        result = {
            "status": status,
            "credible_sources": credible_hits,
            "all_sources": sources,
            "explanations": explanations  # NEW: Return these for LLM
        }
        return result

    except Exception as e:
        return {"status": "error", "sources": [], "explanations": [], "error": str(e)}

# ------------------------
# Update detect_fake_text to use Gemini separately
# ------------------------
# def detect_fake_text(text: str):

#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
#         hf_idx = torch.argmax(probs).item()
#         hf_prediction = "Real" if hf_idx == 0 else "Fake"
      
#         hf_score = probs[hf_idx].item()

#     hf_percent = int(hf_score * 100)
#     print(f"HF PREDICTION '{hf_prediction}' HF SCORE - '{hf_score}' ")

#     gemini_prompt = f"""
#     You are the voice of a misinformation detection browser extension.

#     Input:
#     - Model result: "{hf_prediction}" with {hf_percent}% confidence.
#     - Text to analyze: {text}

#     Task:
#     - Decide if the text is Real, Fake, or Misleading.
#     - Give ONE confidence percentage (0–100).
#     - Provide a short explanation (1–2 sentences).
#     - Speak naturally, as if the extension itself is responding.

#     Format strictly as:
#     Prediction: <Real/Fake/Misleading>
#     Confidence: <number>%
#     Explanation: <short natural explanation>
#     """
#     output = ask_gemini(gemini_prompt)
#     print(f"LLM output is '{output}' ")

#     final_prediction, final_score, final_explanation = hf_prediction, hf_percent, output

#     for line in output.splitlines():
#         if line.lower().startswith("prediction:"):
#             final_prediction = line.split(":", 1)[1].strip()
#         elif line.lower().startswith("confidence:"):
#             match = re.search(r"(\d{1,3})", line)
#             if match:
#                 final_score = min(100, max(0, int(match.group(1))))
#         elif line.lower().startswith("explanation:"):
#             final_explanation = line.split(":", 1)[1].strip()

#     if len(final_explanation.split(".")) > 2:
#         final_explanation = ".".join(final_explanation.split(".")[:2]).strip()

#     corroboration = corroborate_with_google(text[:150])
#     print("Corroboration result:", corroboration)
#     if corroboration["status"] == "corroborated":
#         final_score = min(100, final_score + 10)
#         final_explanation += " Multiple credible outlets are reporting this."
#     elif corroboration["status"] == "uncorroborated":
#         final_score = max(0, final_score - 15)
#         final_explanation += " No major sources appear to report this claim."
#     elif corroboration["status"] == "weak":
#         final_explanation += " Found limited reports from smaller outlets."

#     return {
#         "prediction": final_prediction,
#         "score": final_score,
#         "explanation": final_explanation
#     }

def detect_fake_text(text: str):
    # ML inference using Hugging Face model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        hf_idx = torch.argmax(probs).item()
        hf_prediction = "Real" if hf_idx == 0 else "Fake"
        hf_score = probs[hf_idx].item()
    
    hf_percent = int(hf_score * 100)
    print(f"HF PREDICTION '{hf_prediction}' HF SCORE - '{hf_score}' ")

    # Call corroborate_with_google to get search evidence
    corroboration = corroborate_with_google(text[:150])
    print("Corroboration result:", corroboration)

    # Format search explanations for Gemini prompt
    explanations_str = "\n".join([f"- {exp['source']}: {exp['explanation']}" for exp in corroboration.get("explanations", [])])

    # Gemini prompt to weigh ML and search evidence
    gemini_prompt = f"""
    You are the voice of a misinformation detection browser extension.

    Input:
    - ML result: "{hf_prediction}" with {hf_percent}% confidence.
    - Text to analyze: {text}
    - Search evidence (give this high importance for recent news):
      {explanations_str}
    - Corroboration status: {corroboration['status']}

    Task:
    - Decide if the text is Real, Fake, or Misleading, prioritizing search evidence over ML if they conflict.
    - Give ONE confidence percentage (0–100), boosting if evidence aligns, reducing if contradicts.
    - Provide a short explanation (1–2 sentences), referencing key search insights.
    - Speak naturally, as if the extension itself is responding.

    Format strictly as:
    Prediction: <Real/Fake/Misleading>
    Confidence: <number>%
    Explanation: <short natural explanation>
    """
    output = ask_gemini(gemini_prompt)
    print(f"LLM output is '{output}' ")

    # Parse Gemini output
    final_prediction, final_score, final_explanation = hf_prediction, hf_percent, output
    for line in output.splitlines():
        if line.lower().startswith("prediction:"):
            final_prediction = line.split(":", 1)[1].strip()
        elif line.lower().startswith("confidence:"):
            match = re.search(r"(\d{1,3})", line)
            if match:
                final_score = min(100, max(0, int(match.group(1))))
        elif line.lower().startswith("explanation:"):
            final_explanation = line.split(":", 1)[1].strip()

    # Return final result
    return {
        "prediction": final_prediction,
        "score": final_score,
        "explanation": final_explanation
    }   






    