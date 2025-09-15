import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask
import os

load_dotenv()

app = Flask(__name__)
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def detect_fake_text(text: str):
    """
    Calls Gemini API to check correctness / validity of provided text.
    Returns a dict with prediction, score, and explanation.
    """

    prompt = f"""
    You are a misinformation detection system.
    Analyze the following text and return:
    - Whether it's likely true, false, or misleading.
    - A confidence score between 0 and 1.
    - A short explanation (1–2 sentences).

    Text: {text}
    """
    print(api_key)
    try:
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        output = response.text.strip()
        prediction = "Unknown"
        score = 0.5
        explanation = output

        for line in output.splitlines():
            if "Prediction:" in line:
                prediction = line.split(":", 1)[1].strip()
            elif "Confidence:" in line:
                try:
                    score = float(line.split(":", 1)[1].strip())
                except:
                    pass
            elif "Explanation:" in line:
                explanation = line.split(":", 1)[1].strip()

        return {
            "prediction": prediction,
            "score": score,
            "explanation": explanation
        }

    except Exception as e:
        return {"error": str(e)}
