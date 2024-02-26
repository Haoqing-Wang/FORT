import json
js = json.load(open('name.json', 'r'))
d = {}
for _ in js:
    d[int(_['id'])] = _['name']
print(d)
js2 = json.load(open('name.json', 'r'))