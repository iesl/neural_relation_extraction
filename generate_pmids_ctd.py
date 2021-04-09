import gzip
pubmids = []
for line in gzip.open("data/origin/CTD_chem_gene_ixns.tsv.gz","rb"):
    line = line.decode('ascii')
    if line[0] != "#" and len(line.strip("\n").split("\t")) == 11:
        l = line.strip("\n").split("\t")
        pmids = l[-1].split("|")
        for pmid in pmids:
            pubmids.append(pmid)

for line in gzip.open("data/origin/CTD_chemicals_diseases.tsv.gz","rb"):
    line = line.decode('ascii')
    if line[0] != "#" and len(line.strip("\n").split("\t")) == 10:
        l = line.strip("\n").split("\t")
        pmids = l[-1].split("|")
        for pmid in pmids:
            pubmids.append(pmid)

for line in gzip.open("data/origin/CTD_genes_diseases.tsv.gz","rb"):
    line = line.decode('ascii')
    if line[0] != "#" and len(line.strip("\n").split("\t")) == 9:
        l = line.strip("\n").split("\t")
        pmids = l[-1].split("|")
        for pmid in pmids:
            pubmids.append(pmid)

pubmids = set(pubmids)
print(len(pubmids))
