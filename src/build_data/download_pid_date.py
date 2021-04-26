import requests
import os, sys
from tqdm import tqdm
import random
import json

data_path = os.environ["BIORE_DATA_ROOT"]
pid_list = []

for line in open(data_path + "/new_ctd/alignment_CTD_PTC.merge_rel.pubtator"):
    l = line.strip().split("|t|")
    if len(l) > 1:
        pid_list.append(l[0])
        

for line in open(data_path + "/new_ctd/alignment_CTD_PTC.NULL.pubtator"):
    l = line.strip().split("|t|")
    if len(l) > 1:
        pid_list.append(l[0])

pid_list = list(set(pid_list))

fout = open(data_path + "/new_ctd/pid2year.map2","w")
for pmid in tqdm(pid_list):
    fail_time = 0
    while fail_time < 5:
        try:
            json_text = requests.get(f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids={pmid}", timeout=10).text
            year = json.loads(json_text)["year"]
            year = int(year)
            fout.write(f'{pmid} {year}\n')
            break
        except:
            fail_time += 1
    if fail_time == 5:
        fout.write(f'{pmid} failed\n')
    fout.flush()
fout.close()
