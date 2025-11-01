import time
import pandas as pd
import requests
import json
from queue import Queue
from urllib.parse import quote  


BACKEND_URL = "http://127.0.0.1:5000"


test_queries = [
    # Recent claims (2024-2025)
    "Donald Trump said that he would be a dictator only on day one of his presidency",
    
    "Elon Musk's net worth increased by $170 billion following Donald Trump's 2024 election victory",
    
    "A viral video shows California Governor Gavin Newsom admitting that voter ID laws prevent election fraud",
    
    "FEMA was confiscating Starlink terminals from hurricane relief volunteers in North Carolina",
    
    # Mid-range claims (2020-2023)
    "COVID-19 vaccines contain microchips that allow the government to track people",
    
    "President Joe Biden fell asleep during a meeting with Israeli Prime Minister Benjamin Netanyahu in July 2023",
    
    "Ivermectin is an effective treatment for COVID-19 and the FDA is suppressing this information",
    
    # Older claims (2015-2019)
    "Donald Trump mocked a disabled New York Times reporter during his 2016 presidential campaign",
    
    "Alexandria Ocasio-Cortez said that the world will end in 12 years if climate change is not addressed",
    
    "A photograph shows President Barack Obama refusing to salute the American flag during the national anthem",

    "In October 2025, U.S. Immigration and Customs Enforcement agents deployed tear gas in a Chicago neighborhood while children headed to a Halloween parade there.",
    "In October 2025, actor Millie Bobby Brown donated $1 million to the LGBTQ+ community, saying, \"I hope they find a cure.\"",
    "In late October 2025, posts shared on social media showed a genuine news release announcing that U.S. President Donald Trump was selling pieces of the demolished White House East Wing on his website."

]

results = []

print("Starting API simulation tests (mimicking extension)...")
print(f"Ensure app.py is running on {BACKEND_URL}\n")

for idx, query in enumerate(test_queries, 1):
    if not query.strip():
        
        continue
    
    print(f"Running simulated extension test {idx}: {query[:100]}...")
    
    log_queue = Queue()
    start_time = time.time()
    
    try:
    
        payload = {
            "text": query,
            "url": "https://example.com/article",  
            "session_id": f"test-session-{idx}"  
        }
        
        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": payload["session_id"]
        }
        
        response = requests.post(
            f"{BACKEND_URL}/detect_text",
            json=payload,
            headers=headers,
            timeout=120  
        )
        
        end_time = time.time()
        runtime = round(end_time - start_time, 2)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} - {response.text}")
        
        data = response.json()
        
        
        prediction = data.get("prediction", "Unknown")
        explanation = data.get("explanation", "No explanation available")
        runtime_from_api = data.get("runtime", runtime)
        norm_id = data.get("norm_id", "N/A")  
        
        
        log_url = f"{BACKEND_URL}/stream_logs/{payload['session_id']}"
        log_resp = requests.get(log_url, stream=True, timeout=5)
        logs = []  
        
        print(f"✅ Simulated test {idx} completed in {runtime}s")
        print(f"   Prediction: {prediction}")
        print(f"   Reasoning: {explanation[:200]}...")
        print(f"   Norm ID: {norm_id}\n")  
        
        results.append({
            'Test_Case': idx,
            'User_Query': query,
            'Model_Reasoning': explanation,
            'Total_Time': runtime_from_api,
            'Model_Prediction': prediction,
            'Human_Verified_Prediction': '',  
            'Norm_ID': norm_id,  
            'Logs_Summary': json.dumps(logs[-5:]) if logs else 'API logs not streamed'
        })
        
    except Exception as e:
        print(f"❌ Simulated test {idx} failed: {str(e)}")
        


df = pd.DataFrame(results)
df.to_csv('misinfo_api_sim_results.csv', index=False, encoding='utf-8')
print(f"\n📊 API sim results saved to 'misinfo_api_sim_results.csv'")
print(df.head())