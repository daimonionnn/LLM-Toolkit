import urllib.request
import urllib.parse
import re
import json
import sys

def search_ddg(query):
    url = 'https://html.duckduckgo.com/html/'
    data = urllib.parse.urlencode({'q': query}).encode('utf-8')
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    try:
        response = urllib.request.urlopen(req)
        html = response.read().decode('utf-8')
        pattern = r'<a class="result__snippet[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
        results = []
        for url, snippet in matches[:10]:
            snippet = re.sub(r'<[^>]+>', '', snippet).strip()
            if url.startswith('/l/?'):
                m = re.search(r'uddg=([^&]+)', url)
                if m: url = urllib.parse.unquote(m.group(1))
            results.append({"url": url, "snippet": snippet})
        print(json.dumps(results, indent=2))
    except Exception as e:
        print("Error:", e)

search_ddg(sys.argv[1])
