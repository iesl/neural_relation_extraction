import os, sys
from tqdm import tqdm
data_path = os.environ["BIORE_DATA_ROOT"]
max_length = int(sys.argv[1])

pid = ""
abstract = ""
title = ""
skip = False
count_articles_before = 0
count_articles_after = 0
fout = open(data_path + "/alignment_CTD_PTC.prune_maxlength.pubtator","w")
for line in open(data_path + "/alignment_CTD_PTC.pubtator"):

    if "|t|" in line:
        pid, title = line.strip().split("|t|")[:2]
        count_articles_before += 1
        continue
    if "|a|" in line:
        abstract = line.strip().split("|a|")[1]
        if len((title + abstract).split()) > max_length:
            skip = True
        else:
            skip = False
            fout.write(f"{pid}|a|{title}\n")
            fout.write(f"{pid}|a|{abstract}\n")
            count_articles_after += 1
        continue
    if skip == False:
        fout.write(line)
print(f"{count_articles_after}/{count_articles_before} articles left")