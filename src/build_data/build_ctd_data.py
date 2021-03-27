import os, sys
import gzip
import requests
from tqdm import tqdm
data_path = os.environ["BIORE_DATA_ROOT"]
import random
random.seed(1234)

# download CTD annotation and pubtator annotation
if not os.path.exists(f"{data_path}/origin/CTD_chem_gene_ixns.tsv.gz"):
    os.system("wget http://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz -P {data_path}/origin/")
if not os.path.exists(f"{data_path}/origin/CTD_chemicals_diseases.tsv.gz"):
    os.system("wget http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz -P {data_path}/origin/")
if not os.path.exists(f"{data_path}/origin/CTD_genes_diseases.tsv.gz"):
    os.system("wget http://ctdbase.org/reports/CTD_genes_diseases.tsv.gz -P {data_path}/origin/")

pmid2rels = {}

for line_ in gzip.open(f"{data_path}/origin/CTD_chem_gene_ixns.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 11 and l[-1] != "":
        # MeSH identifier
        # NCBI Gene identifier
        # relations
        # pubmed ids
        chemid, geneid, relations, pmids = "MESH:" + l[1], l[4], l[-2].split("|"), l[-1].split("|")
        for pmid in pmids:
            if pmid not in pmid2rels: pmid2rels[pmid] = []
            for rel in relations:
                pmid2rels[pmid].append((chemid, geneid, "chem_gene:" + rel))

for line_ in gzip.open(f"{data_path}/origin/CTD_chemicals_diseases.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 10 and l[-1] != "" and l[7] == "":
        # MeSH identifier
        # MeSH or OMIM identifier 
        chemid, diseaseid, relations, pmids = "MESH:" + l[1], l[4], l[5].split("|"), l[-1].split("|")
        for pmid in pmids:
            if pmid not in pmid2rels: pmid2rels[pmid] = []
            for rel in relations:
                pmid2rels[pmid].append((chemid, diseaseid, "chem_disease:" + rel))

for line_ in gzip.open(f"{data_path}/origin/CTD_genes_diseases.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 9 and l[-1] != "" and l[6] == "":
        # NCBI Gene identifier
        # MeSH or OMIM identifier 
        geneid, diseaseid, relations, pmids = l[1], l[3], l[4].split("|"), l[-1].split("|")
        for pmid in pmids:
            if pmid not in pmid2rels: pmid2rels[pmid] = []
            for rel in relations:
                pmid2rels[pmid].append((geneid, diseaseid, "gene_disease:" + rel))

print(f"CTD loaded, containing {len(pmid2rels)} pmids")
sys.stdout.flush()

fout = open(f"{data_path}/alignment_CTD_PTC.pubtator","w")
fout_not_match = open(f"{data_path}/not_aligned_CTD_PTC.pubtator","w")
fout_error = open(f"{data_path}/failed_pmid_list","w")
for pmid, triples in tqdm(list(pmid2rels.items())):
    try:
        text = requests.get(f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids={pmid}", timeout=10).text.strip("\n")
        #text = requests.get(f"https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/pubtator?pmids={pmid}&concepts=gene,disease,chemical").text.strip()
    except:
        fout_error.write(f'{pmid}\n')
    entity_set = set([])
    for line in text.split("\n")[2:]:
        eid = line.strip("\n").split("\t")[-1]
        entity_set.add(eid)
    
    matched_rels = []
    for s, o, r in list(set(triples)):
        if s in entity_set and o in entity_set:
            matched_rels.append([s, o, r])
    if len(matched_rels) > 0:
        fout.write(text + "\n")
        for s, o, r in matched_rels:
            fout.write(f"{pmid}\t{r}\t{s}\t{o}\n")
        fout.write("\n")
    else:
        fout_not_match.write(text + "\n")
        for s, o, r in list(set(triples)):
            fout_not_match.write(f"{pmid}\t{r}\t{s}\t{o}\n")
        fout_not_match.write("\n")
fout.close()
fout_not_match.close()
fout_error.close()
