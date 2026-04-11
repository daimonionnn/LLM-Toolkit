import urllib.request
import urllib.parse
import json
import sys

def search_github(query):
    url = 'https://api.github.com/search/issues?q=' + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        for item in data.get('items', [])[:5]:
            print(f"Title: {item['title']}\nURL: {item['html_url']}\nState: {item['state']}\nBody Snippet: {item['body'][:200] if item['body'] else ''}...\n---")
    except Exception as e:
        print("Error:", e)

search_github(sys.argv[1])
