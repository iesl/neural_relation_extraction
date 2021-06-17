import os, sys
import gzip
import requests
from tqdm import tqdm
data_path = os.environ["BIORE_DATA_ROOT"]
import random
random.seed(1234)

# download CTD annotation and pubtator annotation
if not os.path.exists(f"{data_path}/new_ctd/origin/CTD_chem_gene_ixns.tsv.gz"):
    os.system("wget http://ctdbase.org/reports/CTD_chem_gene_ixns.tsv.gz -P {data_path}/new_ctd/origin/")
if not os.path.exists(f"{data_path}/new_ctd/origin/CTD_chemicals_diseases.tsv.gz"):
    os.system("wget http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz -P {data_path}/new_ctd/origin/")
if not os.path.exists(f"{data_path}/new_ctd/origin/CTD_genes_diseases.tsv.gz"):
    os.system("wget http://ctdbase.org/reports/CTD_genes_diseases.tsv.gz -P {data_path}/new_ctd/origin/")

pmid2rels = {}

relname2interaction = {"binding": "binds", 
                       "metabolic processing": "metabolism", 
                       "cotreatment": "co-treated", 
                       "response to substance": "susceptibility"}

def interaction_parser(text, e1, e2, relations):
    # input the interaction description, e1, e2, and the relation list
    # output a relation list if exist otherwise []

    def parse_window(text):
        # recursively output all windows
        windows = []
        text_this_layer = ""
        leftpos = text.find("[")
        rightpos = text.find("]")
        text_ = text[:]
        while leftpos != -1 and rightpos >= leftpos:
            text_this_layer += text_[:leftpos]
            windows_, text_ = parse_window(text_[(leftpos+1):])
            for i, win in enumerate(windows_[::-1]):
                if "co-treated" in win:
                    text_this_layer += win.replace("co-treated", "")
                    del windows_[-int(i+1)]
            windows.extend(windows_)
            leftpos = text_.find('[')
            rightpos = text_.find(']')
        text_this_layer += text_[:rightpos]
        if text_this_layer != "":
            windows.append(text_this_layer)
        return windows, text_[(rightpos+1):]
        
    
    rels = []
    texts, _ = parse_window(text)
    for t in texts:
        if e1 in t and e2 in t:
            for rel in relations:
                r = rel.replace("increases^", "increased ")
                r = r.replace("decreases^", "decreased ")
                r = r.replace("affects^","")
                if r in relname2interaction:
                    r = relname2interaction[r]
                if r in t:
                    rels.append(rel)
    return rels

count_total_chem_gene = 0
count_binary_chem_gene = 0
for line_ in gzip.open(f"{data_path}/new_ctd/origin/CTD_chem_gene_ixns.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 11 and l[-1] != "":
        # MeSH identifier
        # NCBI Gene identifier
        # relations
        # pubmed ids
        chemname, chemid, genename, geneid, description, relations, pmids = l[0], "MESH:" + l[1], l[3], l[4], l[8], l[9].split("|"), l[10].split("|")
        parsed_relations = interaction_parser(description, chemname, genename, relations)
        count_total_chem_gene += len(relations)
        count_binary_chem_gene += len(parsed_relations)
        if len(parsed_relations) == 0:
            continue
        for pmid in pmids:
            if pmid not in pmid2rels: pmid2rels[pmid] = {}
            for rel in parsed_relations:
                if (chemid, geneid, "chem_gene:" + rel) not in pmid2rels[pmid]:
                    pmid2rels[pmid][(chemid, geneid, "chem_gene:" + rel)] = []
                pmid2rels[pmid][(chemid, geneid, "chem_gene:" + rel)].append(description)
print(f"\ncount_total_chem_gene = {count_total_chem_gene}; count_binary_chem_gene = {count_binary_chem_gene}\n")

for line_ in gzip.open(f"{data_path}/new_ctd/origin/CTD_chemicals_diseases.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 10 and l[-1] != "" and l[7] == "": # l[7] == "" means it is not inferred but explicitly expressed
        # MeSH identifier
        # MeSH or OMIM identifier 
        chemid, diseaseid, relations, pmids = "MESH:" + l[1], l[4], l[5].split("|"), l[-1].split("|")
        for pmid in pmids:
            if pmid not in pmid2rels: pmid2rels[pmid] = {}
            for rel in relations:
                if (chemid, diseaseid, "chem_disease:" + rel) not in pmid2rels[pmid]:
                    pmid2rels[pmid][(chemid, diseaseid, "chem_disease:" + rel)] = []

for line_ in gzip.open(f"{data_path}/new_ctd/origin/CTD_genes_diseases.tsv.gz", "rb"):
    line = line_.decode('ascii')
    l = line.strip("\n").split("\t")
    if line[0] != "#" and len(l) == 9 and l[-1] != "" and l[6] == "": # l[6] == "" means it is not inferred but explicitly expressed
        # NCBI Gene identifier
        # MeSH or OMIM identifier 
        geneid, diseaseid, relations, pmids = l[1], l[3], l[4].split("|"), l[-1].split("|")
        for pmid in pmids:
            if pmid not in pmid2rels: pmid2rels[pmid] = {}
            for rel in relations:
                if (geneid, diseaseid, "gene_disease:" + rel) not in pmid2rels[pmid]:
                    pmid2rels[pmid][(geneid, diseaseid, "gene_disease:" + rel)] = []

print(f"CTD loaded, containing {len(pmid2rels)} pmids")
sys.stdout.flush()

fout = open(f"{data_path}/new_ctd/alignment_CTD_PTC.pubtator","w")
fout_not_match = open(f"{data_path}/new_ctd/not_aligned_CTD_PTC.pubtator","w")
fout_error = open(f"{data_path}/new_ctd/failed_pmid_list","w")
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
    for (s, o, r), desc in list(triples.items()):
        if s in entity_set and o in entity_set:
            matched_rels.append([s, o, r, desc])
    if len(matched_rels) > 0:
        fout.write(text + "\n")
        for s, o, r, desc in matched_rels:
            fout.write(f"{pmid}\t{r}\t{s}\t{o}\t{' // '.join(desc)}\n")
        fout.write("\n")
    else:
        fout_not_match.write(text + "\n")
        for (s, o, r), desc in list(triples.items()):
            fout_not_match.write(f"{pmid}\t{r}\t{s}\t{o}\t{' // '.join(desc)}\n")
        fout_not_match.write("\n")
fout.close()
fout_not_match.close()
fout_error.close()
