import requests

resp = requests.post("http://localhost:5000", files={'file': open('dataename.jpg', 'rb')})

print(resp.text)