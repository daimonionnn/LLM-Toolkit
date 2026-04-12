#!/usr/bin/env python3
import urllib.request
import json
import time

URL = "http://127.0.0.1:8080/completion"
BASE_WORDS = ["The", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog.", "Here", "is", "some", "more", "text", "to", "fill", "up", "the", "context", "window."]

print("Testing llama-server performance on http://127.0.0.1:8080 ...\n")
print(f"{'Context Size':<15} | {'Prefill (Prompt)':<20} | {'Generation (Decode)'}")
print("-" * 65)

# Test different context lengths
for target_words in [128, 1024, 4096]:
    # Construct a dummy prompt of the target length
    prompt_words = (BASE_WORDS * (target_words // len(BASE_WORDS) + 1))[:target_words]
    prompt = " ".join(prompt_words)
    
    payload = json.dumps({
        "prompt": prompt,
        "n_predict": 50,    # Generate 50 tokens to measure decode speed
        "temperature": 0.0  # Make it deterministic
    }).encode('utf-8')
    
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode())
            
            t = result.get("timings", {})
            if t:
                prefill = t.get("prompt_per_second", 0)
                gen = t.get("predicted_per_second", 0)
                ctx = t.get("prompt_n", 0)
                
                print(f"{ctx} tokens".ljust(15) + f"| {prefill:.2f} t/s".ljust(23) + f"| {gen:.2f} t/s")
            else:
                print(f"Error: No timings returned for {target_words} words.")
    except urllib.error.URLError as e:
        print(f"Connection failed: {e}. Is the server running on port 8080?")
        break
