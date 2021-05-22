import json

data = json.loads(open("origin/train.json").read())

rel_map = {}
for inp in data:
    r = inp["relation"]
    if r not in rel_map and r != "no_relation": rel_map[r] = len(rel_map)

open("relation_map.json","w").write(json.dumps(rel_map, indent="\t"))
