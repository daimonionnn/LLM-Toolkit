import urllib.request
import urllib.parse
import json
import sys

def search_github(query):
    url = f"https://api.github.com/search/issues?q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    print(f"Searching: {url}")
    try:
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))
        
        for item in data.get('items', [])[:5]:
            print(f"Title: {item['title']}\nURL: {item['html_url']}\nState: {item['state']}")
            if item.get('body'):
                print(f"Body: {item['body'][:300]}")
            print("---")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    search_github(sys.argv[1])
