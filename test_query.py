import requests

url = "http://127.0.0.1:8000/query"
payload = {"query": "Explain Section 125 CrPC regarding maintenance of wife."}

response = requests.post(url, json=payload)
print("Status code:", response.status_code)
print("Response text:\n", response.text)
