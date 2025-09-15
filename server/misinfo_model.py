import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import re

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load HuggingFace model once
tokenizer = AutoTokenizer.from_pretrained("Pulk17/Fake-News-Detection")
model = AutoModelForSequenceClassification.from_pretrained("Pulk17/Fake-News-Detection")

def detect_fake_text(text: str):
    # --- HuggingFace prediction ---
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        hf_idx = torch.argmax(probs).item()
        hf_prediction = "Real" if hf_idx == 0 else "Fake"
        hf_score = probs[hf_idx].item()

    hf_percent = int(hf_score * 100)

    # --- Gemini refinement prompt ---
    prompt = f"""
    You are the voice of a misinformation detection browser extension.

    Input:
    - Model result: "{hf_prediction}" with {hf_percent}% confidence.
    - Text to analyze: {text}

    Your job:
    - Decide if the text is Real, Fake, or Misleading.
    - Give ONE confidence percentage (0–100).
    - Provide a short explanation (1–2 sentences).
    - Speak naturally, as if the extension itself is responding.
    - Do NOT output multiple scores, technical details, or extra text.

    Format your response clearly:
    Prediction: <Real/Fake/Misleading>
    Confidence: <number between 0 and 100>%
    Explanation: <short natural explanation>
    """

    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        output = response.text.strip()

        # Defaults
        final_prediction, final_score, final_explanation = hf_prediction, hf_percent, output

        # --- Parse Gemini output ---
        for line in output.splitlines():
            if line.lower().startswith("prediction:"):
                final_prediction = line.split(":", 1)[1].strip()
            elif line.lower().startswith("confidence:"):
                match = re.search(r"(\d{1,3})", line)
                if match:
                    final_score = min(100, max(0, int(match.group(1))))
            elif line.lower().startswith("explanation:"):
                final_explanation = line.split(":", 1)[1].strip()

        # Fallback: if explanation is too long, shorten
        if len(final_explanation.split(".")) > 2:
            final_explanation = ".".join(final_explanation.split(".")[:2]).strip()

    except Exception as e:
        final_prediction, final_score, final_explanation = (
            "Unknown",
            50,
            f"Could not analyze text (error: {str(e)})."
        )

    return {
        "prediction": final_prediction,
        "score": final_score,  # percentage
        "explanation": f"{final_explanation}"
    }