import os, sys
import gzip
import requests
from tqdm import tqdm
data_path = os.environ["BIORE_DATA_ROOT"]
import random
import json
random.seed(1234)

pubtator_pmids = []
for line_ in gzip.open(data_path+"/new_ctd/origin/bioconcepts2pubtatorcentral.gz","rb"):
    try:
        line = line_.decode('ascii')
        pmid = line.strip("\n").split("\t")[0]
        pubtator_pmids.append(pmid)
    except:
        print(line_)
        sys.stdout.flush()
pubtator_pmids = list(set(pubtator_pmids))
random.shuffle(pubtator_pmids)
print("ptc loaded")
sys.stdout.flush()

ctd_pmids = []
for line_ in gzip.open(f"{data_path}/new_ctd/origin/CTD_chem_gene_ixns.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 11:
        pmids = l[-1].split("|")
        for pmid in pmids:
            ctd_pmids.append(pmid)

for line_ in gzip.open(f"{data_path}/new_ctd/origin/CTD_chemicals_diseases.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 10:
        pmids = l[-1].split("|")
        for pmid in pmids:
            ctd_pmids.append(pmid)

for line_ in gzip.open(f"{data_path}/new_ctd/origin/CTD_genes_diseases.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 9:
        pmids = l[-1].split("|")
        for pmid in pmids:
            ctd_pmids.append(pmid)
ctd_pmids = set(ctd_pmids)
print("ctd loaded")
sys.stdout.flush()

fout = open(data_path + "/alignment_CTD_PTC.NULL.pubtator","w")
for pmid in tqdm(list(pubtator_pmids)):
    if pmid not in ctd_pmids:
            try:
                json_text = requests.get(f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?pmids={pmid}", timeout=10).text
                year = json.loads(json_text)["year"]
                year = int(year)
            except:
                continue
            if year > 2019: 
                # when this data is created (March 2021), CTD does not fully consider papers from year 2020 and later, 
                # therefore we choose to ignore papers after 2019 to try to avoid false NULL. 
                continue
            try:
                text = requests.get(f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids={pmid}", timeout=10).text.strip("\n")
            except:
                continue
            entity_type_set = set([])
            for line in text.split("\n")[2:]:
                etype = line.strip("\n").split("\t")[-2]
                if etype in ["Chemical","Gene", "Disease"]:
                    entity_type_set.add(etype)
            if len(entity_type_set) >= 2: # generate harder negatives constraining there to be at least two types of entities
                fout.write(text + "\n\n")
            #except:
            #pass
fout.close()
