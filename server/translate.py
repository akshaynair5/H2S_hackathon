from google.cloud import translate_v2 as translate
import os
from dotenv import load_dotenv

# Load .env file (optional if credentials are set in env)
load_dotenv()


key_path = os.getenv("FIREBASE_KEY_PATH", "gen-ai-h2s-project-562ce7c50fcf-vertex-ai-fact-check.json")
if key_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


def translate_to_english(text_to_check: str) -> dict:
    """
    Detects the language of a text and translates it to English 
    if it is not already English.
    
    Args:
        text_to_check: The input string to check and translate if needed.
    
    Returns:
        A dictionary with:
        - 'original_text': The original input text
        - 'detected_language': Language code (e.g., 'en', 'es', 'ml')
        - 'translated_text': English translation (or original if already English)
        - 'was_translated': Boolean indicating if translation occurred
    """
    translate_client = translate.Client()
    
    # Detect the language
    detection = translate_client.detect_language(text_to_check)
    detected_lang = detection['language']
    
    # Check if already English
    if detected_lang == 'en':
        print(f"Detected language is English. No translation needed.")
        return {
            'original_text': text_to_check,
            'detected_language': detected_lang,
            'translated_text': text_to_check,
            'was_translated': False
        }
    else:
        # Translate to English
        print(f"Detected language: '{detected_lang}'. Translating to English.")
        translation = translate_client.translate(
            text_to_check,
            target_language='en'
        )
        return {
            'original_text': text_to_check,
            'detected_language': detected_lang,
            'translated_text': translation['translatedText'],
            'was_translated': True
        }
    
# if __name__ == "__main__":
#     result = translate_to_english("ഇരട്ട എഞ്ചിൻ സർക്കാർ ഒരു വിട്ടുവീഴ്ചയും ചെയ്യില്ല’ ബിഹാറിലെ എൻഡിഎ പ്രകടനപത്രികയെ പ്രശംസിച്ച് മോദി")
#     print("Result:", result)
        