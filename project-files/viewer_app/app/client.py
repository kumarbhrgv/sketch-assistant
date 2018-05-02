import requests
payload = {'text': 'draw bird'}
r = requests.get('http://localhost:5000/', params=payload)
print(r.text)
