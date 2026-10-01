import urllib.request
import json
try:
    print('Calling /screen...')
    req = urllib.request.Request('http://127.0.0.1:8000/screen', data=b'{"input_dir": "./resumes", "use_llm": false}', headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print('Candidates:', len(data.get('ranked_candidates', [])))
except Exception as e:
    print(e)
