# reasoning_generator.py
import os
import requests
import google.generativeai as genai

# Configure Gemini API key (from your environment variable)
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def fetch_fact_check_data(claim: str):
    """
    Try retrieving verified fact-check information using Google's Fact Check Tools API.
    If unavailable, return an internal reasoning fallback.
    """
    try:
        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {"query": claim, "key": os.getenv("FACTCHECK_API_KEY")}
        resp = requests.get(url, params=params, timeout=8)
        data = resp.json()

        if "claims" in data and data["claims"]:
            review = data["claims"][0]["claimReview"][0]
            source = review["publisher"]["name"]
            rating = review.get("textualRating", "N/A")
            title = review.get("title", "")
            review_url = review.get("url", "")
            return f"According to {source}, the claim '{claim}' is rated '{rating}'. {title}. More details: {review_url}"
    except Exception as e:
        print("Fact-check API error:", e)

    return f"No verified fact-check found for '{claim}'. Generating reasoning internally."


def generate_reasoning(claim: str, context: str = ""):
    """
    Generate a 100-word reasoning paragraph using Gemini API.
    It takes a claim and optional context (like fact-check data) and produces an explanation.
    """
    prompt = f"""
Claim: "{claim}"
Verified context or information: {context}

Write a clear, factual explanation (~100 words) of why this claim may be false or misleading.
Avoid speculation and focus on evidence-based reasoning.
Make it easy to understand for a general audience.
"""

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate reasoning. Error: {e}"


def get_explanation_for_fake(claim: str):
    """
    Full pipeline: try getting fact-check data first,
    then generate a reasoning paragraph using Gemini.
    """
    context = fetch_fact_check_data(claim)
    reasoning = generate_reasoning(claim, context)
    return reasoning