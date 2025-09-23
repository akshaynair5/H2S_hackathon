from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def detect_fake_text(text, model_name="Pulk17/Fake-News-Detection"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        idx = torch.argmax(probs).item()
        return {
            "prediction": "Real" if idx == 0 else "Fake",
            "confidence": float(probs[idx])
        }

# Test
result1 = detect_fake_text("Scientists confirm a cure for aging found in lab mice")
result2 = detect_fake_text("Scientists confirm a cure for aging found in lab mice", 
                           "jy46604790/Fake-News-Bert-Detect")

print(result1)
print(result2)
