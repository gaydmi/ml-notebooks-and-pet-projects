import requests

url = 'http://127.0.0.1:5000'
path = '/predict'
myobj = {'text': 'Hello, my name is Name'}

x = requests.post(url + path, data = myobj)
print(x.text)